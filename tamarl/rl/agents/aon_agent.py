"""AON (All-Or-Nothing) agent — Agnostic to aggregation level.

Agent sans état qui sélectionne toujours la route 0 (plus court chemin
en temps de parcours libre). Agnostique au niveau d'agrégation : il reçoit
un mapping aggregation_indices [N] mais n'en a pas besoin car sa politique
est purement basée sur les masques d'action (sans paramètres internes).
"""

from typing import Optional

import torch


class AONAgent:
    """Agent All-Or-Nothing vectorisé pour la sélection de routes.

    Suppose que les chemins sont triés par temps de trajet libre, donc
    l'index 0 est toujours le plus court chemin en réseau vide.

    Agnostique au niveau d'agrégation : fonctionne identiquement en mode
    agent_level, od_pair, ou tout autre mode.
    """

    def __init__(self, seed: int | None = None):
        pass

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sélectionne la route 0 (plus court chemin libre) pour tous les véhicules.

        Args:
            obs:                  [N, K] — ignoré.
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] — ignoré (agent sans paramètres).

        Returns:
            actions: [N] tenseur long (principalement des 0).
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=masks.device, dtype=torch.long)

        # Action par défaut : route 0 (plus court chemin en free-flow)
        actions = torch.zeros(N, device=masks.device, dtype=torch.long)

        # Filet de sécurité : si route 0 invalide, prendre la première route valide
        is_route_zero_invalid = ~masks[:, 0].bool()
        if is_route_zero_invalid.any():
            fallback_actions = torch.argmax(masks[is_route_zero_invalid].int(), dim=1)
            actions[is_route_zero_invalid] = fallback_actions

        return actions

    def __repr__(self) -> str:
        return "AONAgent (All-Or-Nothing — Top 1 Path)"
