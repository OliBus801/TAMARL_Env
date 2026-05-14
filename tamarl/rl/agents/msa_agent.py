"""MSA (Method of Successive Averages) Agent — Agnostic to aggregation level.

Implements MSA for Dynamic Traffic Assignment. The agent maintains a
probability distribution over K candidate routes for each of its B
parameter blocs. At sampling time, the distribution is projected to the
N vehicles via aggregation_indices. The update uses the Time-Dependent
(TD) oracle to evaluate all K paths and updates the distribution MSA-style.

Sélection d'action (get_actions_batched)
  1. Projection probs [B, K] → [N, K] via aggregation_indices.
  2. Masquage et renormalisation.
  3. Tirage multinomial → retourne [N] actions.

Mise à jour (update)
  - TD oracle évalue les K chemins pour chaque "leg" individuel → best_k [N].
  - La distribution cible est calculée comme la proportion de véhicules
    par bloc préférant chaque chemin.
  - Blended Update : probs ← (1 - α) * probs + α * target_distribution.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator


class MSAAgent:
    r"""MSA (Method of Successive Averages) bandit agent.

    Route probabilities evolve as:
        p_{n+1}(k) = (1 - α_n) * p_n(k)  +  α_n * Target_n(k)

    where ``Target_n(k)`` is the proportion of legs in the block for which `k`
    is the best-response path (minimum TD cost), and ``α_n`` decays exponentially.

    The distribution is initialized in "All-or-Nothing" (AoN) mode: 100%
    probability on the free-flow shortest path.

    Args:
        env:         The wrapper instance (AgentLevelWrapper or ODLevelWrapper).
                     Used to access candidate_routes, first_edges, and the live DNL.
        k_paths:     Number of candidate routes per OD pair (K).
        device:      Torch device string.
        seed:        Optional RNG seed for reproducibility.
        alpha_max:   Maximum value for the MSA step size.
        alpha_min:   Minimum value for the MSA step size.
        alpha_decay: Exponential decay rate for alpha.
    """

    def __init__(
        self,
        env,
        k_paths: int,
        device: str = "cpu",
        seed: Optional[int] = None,
        alpha_max: float = 1.0,
        alpha_min: float = 0.05,
        alpha_decay: float = 0.01,
    ):
        self._env = env
        self.k_paths = k_paths
        self.device = device
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay

        if seed is not None:
            torch.manual_seed(seed)

        # B = num_od_pairs si mode OD, sinon num_envs (TotalLegs)
        is_od_mode = hasattr(env, "num_od_pairs")
        if is_od_mode:
            self.num_models = env.num_od_pairs
        else:
            self.num_models = env.num_envs

        # num_envs = N (total legs)
        self.num_envs = env.num_envs

        # Compteur d'épisodes
        self.episode = 0

        # ── Time-Dependent Evaluator ──────────────────────────────────
        self._evaluator = TimeDependentEvaluator.from_wrapper(env)

        # ── Initialisation All-or-Nothing ─────────────────────────────
        # On assume que l'action 0 est le plus court chemin en flux libre.
        # On initialise donc self.probs à 100% sur l'action 0 pour tous les blocs.
        self.probs = torch.zeros(
            (self.num_models, k_paths),
            device=device,
            dtype=torch.float32,
        )
        self.probs[:, 0] = 1.0

    # ── Rétrocompatibilité ───────────────────────────────────────────────
    @property
    def num_agents(self) -> int:
        return self.num_models

    # ──────────────────────────────────────────────────────────────────
    #  Bandit API
    # ──────────────────────────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Échantillonne une action par leg selon la distribution MSA courante.

        Args:
            obs:                  [N, K] — ignoré (mode blind).
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] mapping legs → blocs B. Si None,
                                  identité (mode agent_level, N = B).

        Returns:
            actions: [N] tenseur long.
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Projection probs [B, K] → [N, K]
        p = self.probs[aggregation_indices]  # [N, K]

        # 2. Masquage et renormalisation
        p = p * masks.float()
        row_sums = p.sum(dim=1, keepdim=True).clamp(min=1e-10)
        p = p / row_sums

        # 3. Tirage multinomial
        actions = torch.multinomial(p, num_samples=1).squeeze(1)  # [N]
        return actions

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Mise à jour MSA via l'oracle TD sur tous les K chemins.

        Calcule la proportion cible et applique la formule de blend.

        Args:
            actions:              [N] — actions prises (ignorées ici).
            rewards:              [N] — récompenses reçues (ignorées ici).
            aggregation_indices:  [N] mapping legs → blocs B.
        """
        self.episode += 1
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * math.exp(-self.alpha_decay * self.episode)

        dnl = self._env.bandit.dnl
        if not dnl.collect_link_tt:
            # L'évaluateur TD requiert les données par intervalle.
            return

        N = self.num_envs
        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # ── Oracle TD : évaluation de tous les K chemins ──────────────
        _, best_k = self._evaluator.evaluate(
            dnl=dnl,
            departure_times=self._env.bandit.scenario.departure_times,
        )

        # ── Calcul de la Cible (Proportion Target) ───────────────────
        # On calcule la proportion de véhicules dans chaque bloc qui a
        # chaque chemin k comme meilleur.
        flat_idx = aggregation_indices * self.k_paths + best_k  # [N]
        flat_size = self.num_models * self.k_paths

        vote_counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        vote_counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        target = vote_counts.reshape(self.num_models, self.k_paths)  # [B, K]
        
        # Normalisation pour obtenir les proportions
        bloc_sizes = target.sum(dim=1, keepdim=True).clamp(min=1e-10)
        target = target / bloc_sizes

        # ── MSA blend: p ← (1 - α) * p + α * target ──────────────────
        self.probs = (1.0 - alpha) * self.probs + alpha * target

    # ──────────────────────────────────────────────────────────────────
    #  Housekeeping
    # ──────────────────────────────────────────────────────────────────

    def end_episode(self) -> None:
        pass

    def __repr__(self) -> str:
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * math.exp(-self.alpha_decay * self.episode)
        return (
            f"MSAAgent(B={self.num_models}, N_legs={self.num_envs}, "
            f"K={self.k_paths}, episode={self.episode}, α={alpha:.4f})"
        )
