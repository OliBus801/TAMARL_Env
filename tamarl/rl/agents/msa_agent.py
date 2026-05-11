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
  - Si mode OD (B < N) : agrégation des best_k via un vote majoritaire par
    (bloc, action) via scatter_add_ pour obtenir le best_k représentatif
    de chaque bloc.
  - MSA blend : probs ← (1 - α) * probs + α * target_onehot [B, K].
"""
from __future__ import annotations

from typing import Optional

import torch

from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator


class MSAAgent:
    r"""MSA (Method of Successive Averages) bandit agent.

    Route probabilities evolve as:
        p_{n+1}(k) = (1 - α_n) * p_n(k)  +  α_n * 𝟙[k == k*_n]

    where ``k*_n`` is the best-response path (minimum TD cost) and
    ``α_n = 1 / n`` is the MSA step size.

    The distribution is initialised uniformly over K paths.

    Args:
        env:       The wrapper instance (AgentLevelWrapper or ODLevelWrapper).
                   Used to access candidate_routes, first_edges, and the live DNL.
        k_paths:   Number of candidate routes per OD pair (K).
        device:    Torch device string.
        seed:      Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        env,
        k_paths: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self._env = env
        self.k_paths = k_paths
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # B = num_od_pairs si mode OD, sinon num_envs (TotalLegs)
        if hasattr(env, "num_od_pairs"):
            self.num_models = env.num_od_pairs
        else:
            self.num_models = env.num_envs

        # num_envs = N (total legs)
        self.num_envs = env.num_envs

        # ── Distribution de probabilités [B, K] ──────────────────────
        # Initialisée uniformément sur les K chemins.
        self.probs = torch.full(
            (self.num_models, k_paths),
            1.0 / k_paths,
            device=device,
            dtype=torch.float32,
        )

        # Compteur d'épisodes (1-indexé : α_1 = 1.0 → AoN complet au 1er épisode)
        self.episode = 0

        # ── Time-Dependent Evaluator ──────────────────────────────────
        self._evaluator = TimeDependentEvaluator.from_wrapper(env)

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

        Les récompenses sont ignorées au profit de l'évaluation TD qui fournit
        le best_k [N] pour chaque leg. En mode OD (B < N), le best_k est
        agrégé par vote majoritaire par bloc via scatter_add_ avant d'appliquer
        le blend MSA sur la distribution [B, K].

        Args:
            actions:              [N] — actions prises (ignorées ici).
            rewards:              [N] — récompenses reçues (ignorées ici).
            aggregation_indices:  [N] mapping legs → blocs B.
        """
        self.episode += 1
        alpha = 1.0 / self.episode  # Pas MSA

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
        # best_k: [N] — best path index per leg

        # ── Agrégation best_k : [N] → [B] via vote majoritaire ───────
        # Pour chaque (bloc, chemin), on compte combien de legs ont choisi
        # ce chemin comme meilleur. Le bloc retient le chemin le plus voté.
        flat_idx = aggregation_indices * self.k_paths + best_k  # [N]
        flat_size = self.num_models * self.k_paths

        vote_counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        vote_counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        vote_matrix = vote_counts.reshape(self.num_models, self.k_paths)  # [B, K]
        best_k_per_bloc = torch.argmax(vote_matrix, dim=1)  # [B]

        # ── Build the AoN one-hot target distribution [B, K] ─────────
        target = torch.zeros_like(self.probs)
        target.scatter_(1, best_k_per_bloc.unsqueeze(1), 1.0)

        # ── MSA blend: p ← (1 - α) * p + α * target ──────────────────
        self.probs = (1.0 - alpha) * self.probs + alpha * target

    # ──────────────────────────────────────────────────────────────────
    #  Housekeeping
    # ──────────────────────────────────────────────────────────────────

    def end_episode(self) -> None:
        """Aucun bookkeeping supplémentaire requis."""
        pass

    def __repr__(self) -> str:
        alpha = 1.0 / self.episode if self.episode > 0 else 1.0
        return (
            f"MSAAgent(B={self.num_models}, N_legs={self.num_envs}, "
            f"K={self.k_paths}, episode={self.episode}, α={alpha:.4f})"
        )
