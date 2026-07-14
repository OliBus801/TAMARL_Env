"""Random agent — Agnostic to aggregation level.

Agent sans état qui sélectionne une route uniforme aléatoire parmi les
routes valides pour chaque véhicule. Agnostique au niveau d'agrégation :
il reçoit aggregation_indices [N] mais ne l'utilise pas (pas de paramètres).
"""

from __future__ import annotations

from typing import Optional

import torch


class RandomAgent:
    """Sélecteur de route uniforme aléatoire.

    À chaque pas, chaque véhicule choisit une route aléatoire parmi les
    K chemins candidates valides pour sa paire OD.

    Compatible avec l'interface get_actions_batched (mode vectorisé).

    Args:
        num_agents:  Nombre de véhicules N (conservé pour rétrocompatibilité).
        k:           Nombre de routes candidates K.
        seed:        Graine aléatoire pour la reproductibilité.
    """

    def __init__(self, num_agents: int, k: int, seed: int | None = None):
        self.num_agents = num_agents
        self.k = k
        if seed is not None:
            torch.manual_seed(seed)

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Échantillonne une route valide uniformément pour chaque véhicule.

        Args:
            obs:                  [N, K] — ignoré.
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] — ignoré (agent sans paramètres).

        Returns:
            actions: [N] tenseur long.
        """
        # Distribution uniforme sur les routes valides
        valid_probs = masks.float()
        row_sums = valid_probs.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0).squeeze(1)
        if zero_rows.any():
            valid_probs[zero_rows] = 1.0
            row_sums = valid_probs.sum(dim=1, keepdim=True)
        valid_probs = valid_probs / row_sums.clamp(min=1e-6)
        return torch.multinomial(valid_probs, 1).squeeze(1)

    def act(self) -> torch.Tensor:
        """Rétrocompatibilité : tirage uniforme sans masque."""
        return torch.randint(0, self.k, (self.num_agents,))

    def __repr__(self) -> str:
        return f"RandomAgent(N={self.num_agents}, K={self.k})"
