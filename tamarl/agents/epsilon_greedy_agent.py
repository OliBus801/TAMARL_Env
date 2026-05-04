import numpy as np
import torch
from typing import Dict, Optional

class EpsilonGreedyAgent:
    """
    Agent Epsilon-Greedy vectorisé pour la modélisation des choix de routes.
    Assume que l'observation fournie contient les coûts à minimiser (temps de trajet).
    """

    def __init__(
        self, 
        epsilon_start: float = 1.0, 
        epsilon_end: float = 0.05, 
        epsilon_decay: float = 0.995, 
        seed: Optional[int] = None
    ):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    # ── Batched API ───────────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sélectionne les actions (routes) pour tous les agents en une seule passe tensorielle.
        """
        K_agents = deciding_indices.numel()
        if K_agents == 0:
            return torch.empty(0, device=deciding_indices.device, dtype=torch.long)

        device = deciding_indices.device

        # 1. EXPLOITATION : Trouver la route avec le coût minimal
        # On convertit les masques booléens pour mettre les routes invalides à l'infini
        masked_costs = obs.clone().float()
        masked_costs[~masks.bool()] = float('inf')
        best_actions = torch.argmin(masked_costs, dim=1)

        # 2. EXPLORATION : Choisir une route valide au hasard
        valid_probs = masks.float()
        # Clamp pour éviter une division par zéro si un agent n'a aucune action valide
        valid_probs = valid_probs / valid_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        random_actions = torch.multinomial(valid_probs, 1).squeeze(1)

        # 3. SÉLECTION EPSILON-GREEDY
        # Générer un masque d'exploration basé sur le epsilon actuel
        explore_mask = torch.rand(K_agents, device=device) < self.epsilon
        
        # Combiner les deux tenseurs : si True -> random, si False -> best
        actions = torch.where(explore_mask, random_actions, best_actions)

        return actions

    # ── Dict API (Legacy PettingZoo) ──────────────────────────────────

    def get_actions(
        self, 
        observations: Dict[str, np.ndarray], 
        infos: Dict[str, dict]
    ) -> Dict[str, int]:
        """Interface classique pour compatibilité avec les anciens environnements."""
        actions = {}
        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue
                
            obs = observations.get(agent_id)
            if obs is None:
                continue
                
            valid_indices = np.where(mask > 0)[0]
            
            if self.rng.random() < self.epsilon:
                # Exploration
                actions[agent_id] = int(self.rng.choice(valid_indices))
            else:
                # Exploitation : argmin sur les actions valides
                masked_obs = np.full_like(obs, np.inf, dtype=np.float32)
                masked_obs[valid_indices] = obs[valid_indices]
                actions[agent_id] = int(np.argmin(masked_obs))

        return actions

    # ── Gestion de l'exploration ──────────────────────────────────────

    def decay_epsilon(self):
        """Diminue le epsilon à la fin de chaque épisode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def __repr__(self) -> str:
        return f"EpsilonGreedyAgent(ε={self.epsilon:.4f}, decay={self.epsilon_decay})"
