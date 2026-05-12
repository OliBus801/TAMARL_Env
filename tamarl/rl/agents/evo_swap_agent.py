"""Evolutionary Swap Agent (EvoSwapAgent).

Implements an evolutionary greedy algorithm with inertia. 
A proportion epsilon of the population (mutation) re-evaluates their path choice 
based on the time-dependent travel time estimates from the previous episode.
The remaining population keeps their previous path choice (inertia).

The mutation rate epsilon decays over time.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tamarl.envs.components.time_dependent_evaluator import TimeDependentEvaluator


class EvoSwapAgent:
    """Evolutionary Swap Agent with inertia.

    Args:
        env:         The wrapper instance (AgentLevelWrapper or ODLevelWrapper).
        k_paths:     Number of candidate routes per OD pair (K).
        device:      Torch device string.
        seed:        Optional RNG seed for reproducibility.
        epsilon_max:   Maximum mutation rate.
        epsilon_min:   Minimum mutation rate.
        epsilon_decay: Exponential decay rate for epsilon.
    """

    def __init__(
        self,
        env,
        k_paths: int,
        device: str = "cpu",
        seed: Optional[int] = None,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.01,
    ):
        self._env = env
        self.k_paths = k_paths
        self.device = device
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if seed is not None:
            torch.manual_seed(seed)

        # num_envs = N (total legs)
        self.num_envs = env.num_envs
        
        # Agnostic to aggregation: we might be in OD mode or Agent mode.
        # But we maintain actions at the vehicle level (N) to preserve inertia.
        is_od_mode = hasattr(env, "num_od_pairs")
        if is_od_mode:
            self.num_models = env.num_od_pairs
        else:
            self.num_models = env.num_envs

        # Compteur d'épisodes
        self.episode = 0
        self.epsilon = epsilon_max

        # ── Time-Dependent Evaluator ──────────────────────────────────
        self._evaluator = TimeDependentEvaluator.from_wrapper(env)

        # ── Actions Initiales ─────────────────────────────────────────
        # Initialisation All-or-Nothing (Action 0 = FFTT shortest path)
        self.current_actions = torch.zeros(
            (self.num_envs,),
            device=device,
            dtype=torch.long,
        )

    # ── Rétrocompatibilité ───────────────────────────────────────────────
    @property
    def num_agents(self) -> int:
        return self.num_models

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retourne les actions mémorisées (Inertie)."""
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
            
        # On s'assure que self.current_actions a la bonne taille (cas dynamique ?)
        # Normalement N est constant dans le bandit DTA.
        return self.current_actions

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Mise à jour évolutionnaire : mutation d'une partie de la population."""
        self.episode += 1
        
        dnl = self._env.bandit.dnl
        if not dnl.collect_link_tt:
            # L'évaluateur TD requiert les données par intervalle.
            # On applique quand même le decay pour rester cohérent.
            self._decay_epsilon()
            return

        # 1. Évaluation des meilleures alternatives (Best Response)
        # best_k : [N]
        _, best_k = self._evaluator.evaluate(
            dnl=dnl,
            departure_times=self._env.bandit.scenario.departure_times,
        )

        # 2. Création du masque de mutation
        # On tire aléatoirement qui va changer de chemin (Mutation)
        mutate_mask = torch.rand(self.num_envs, device=self.device) < self.epsilon

        # 3. Application de la mutation
        self.current_actions = torch.where(mutate_mask, best_k, self.current_actions)

        # 4. Decay de l'epsilon
        self._decay_epsilon()

    def _decay_epsilon(self) -> None:
        """Applique la décroissance exponentielle à epsilon."""
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * self.episode)

    def end_episode(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"EvoSwapAgent(N={self.num_envs}, K={self.k_paths}, "
            f"episode={self.episode}, ε={self.epsilon:.4f})"
        )
