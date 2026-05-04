import math
import torch
import numpy as np
from typing import Dict, Optional

class UCBAgent:
    """
    Agent UCB (Upper Confidence Bound) vectorisé pour la sélection de routes.
    Formulé pour MINIMISER un coût (temps de trajet).
    """

    def __init__(self, num_agents: int, k_paths: int, c_exploration: float = 100.0, device: str = 'cpu'):
        self.c = c_exploration
        self.device = device
        
        # État interne : Compteur d'utilisation des routes par chaque agent [A, K]
        self.action_counts = torch.zeros((num_agents, k_paths), device=self.device, dtype=torch.float32)
        
        # Compteur global d'épisodes (le 't' dans l'équation UCB)
        self.episode_t = 1 

    # ── Batched API ───────────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
    ) -> torch.Tensor:
        
        K_agents = deciding_indices.numel()
        if K_agents == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        # 1. Extraire les comptes d'actions spécifiques aux agents qui décident
        counts = self.action_counts[deciding_indices]
        
        # 2. Calcul du terme d'exploration UCB
        # On ajoute 1e-6 pour éviter la division par zéro sur les routes jamais prises
        ln_t = math.log(self.episode_t) if self.episode_t > 1 else 0.0
        exploration_term = self.c * torch.sqrt(ln_t / (counts + 1e-6))
        
        # 3. Calcul du coût optimiste
        # obs contient les coûts historiques (mu_k) fournis par le Wrapper
        optimistic_costs = obs.clone().float() - exploration_term
        
        # 4. Masquage des routes invalides (Padding de Yen)
        optimistic_costs[~masks.bool()] = float('inf')
        
        # 5. Sélection : argmin pour minimiser le coût optimiste
        actions = torch.argmin(optimistic_costs, dim=1)
        
        # 6. Mise à jour de l'état interne (Indexation avancée PyTorch = 0 boucle for)
        self.action_counts[deciding_indices, actions] += 1
        
        return actions

    # ── Gestion du temps ──────────────────────────────────────────────

    def end_episode(self):
        """À appeler obligatoirement dans train.py à la fin de chaque épisode."""
        self.episode_t += 1

    def __repr__(self) -> str:
        return f"UCBAgent(c={self.c:.1f}, episode={self.episode_t})"
