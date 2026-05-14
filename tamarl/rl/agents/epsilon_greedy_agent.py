"""Epsilon-Greedy bandit agent — Agnostic to aggregation level.

L'agent maintient des Q-values internes de taille [B, K], où B est le
nombre de "blocs" de paramètres (B = N agents indépendants, ou B = M paires
OD en parameter sharing). Il ne sait PAS à quel niveau il opère : c'est le
mapping_tensor [N] (soit arange(N), soit od_indices) fourni par train_bandit
qui gère la correspondance entre les N véhicules et les B blocs.

Sélection d'action (get_actions_batched)
  1. Projection des Q-values : Q_exp = self.q_values[mapping_tensor]  → [N, K]
  2. Exploitation via argmax avec tie-breaking sur Q_exp.
  3. Exploration : tirage uniforme parmi les chemins valides.
  4. Décision epsilon-greedy → retourne [N] actions.

Mise à jour des poids (update)
  - Calcul de la mise à jour EMA individuelle pour les N véhicules.
  - Agrégation vers [B, K] via scatter_add_ (moyenne par bloc).
  - Mise à jour de self.q_values[B, K] uniquement sur les entrées actives.
"""
import torch
from typing import Optional


class EpsilonGreedyAgent:
    r"""Agent Epsilon-Greedy vectorisé, agnostique au niveau d'agrégation.

    Args:
        num_models:     Nombre de blocs de paramètres B (= N agents,
                        M paires OD, ou 1 pour le mode centralisé).
        k_paths:        Nombre de routes candidates K.
        epsilon_start:  Taux d'exploration initial.
        epsilon_end:    Taux d'exploration minimum.
        epsilon_decay:  Facteur multiplicatif de décroissance par épisode.
        alpha:          Taux d'apprentissage pour la mise à jour EMA.
        device:         Dispositif PyTorch ('cpu' ou 'cuda').
        seed:           Graine aléatoire pour la reproductibilité.
    """

    def __init__(
        self,
        num_agents: int,       # B (nombre de blocs de paramètres)
        k_paths: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        alpha: float = 0.1,
        device: str = "cpu",
        seed: Optional[int] = None,
        # Alias moderne : num_models prend le dessus sur num_agents si fourni
        num_models: Optional[int] = None,
    ):
        # Résolution de l'alias : num_models est prioritaire
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Q-values [B, K] — initialisées avec un petit bruit pour briser la symétrie
        self.q_values = (
            torch.randn((num_agents, k_paths), device=device, dtype=torch.float32) * 1e-6
        )

    # ── Propriété de rétrocompatibilité ──────────────────────────────────
    @property
    def num_agents(self) -> int:
        """Alias rétrocompatible vers num_models."""
        return self.num_models

    # ── Sélection d'actions ──────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sélectionne une action pour chacun des N véhicules.

        Args:
            obs:                  [N, K] observations (non utilisé — mode blind).
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] mapping de chaque véhicule vers son bloc
                                  de paramètres. Si None, on suppose N = B et
                                  on utilise torch.arange(N).

        Returns:
            actions: [N] tenseur long d'indices de routes sélectionnées.
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        # Résoudre le mapping : par défaut, identité (mode agent_level)
        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Projection des Q-values [B, K] → [N, K]
        q_exp = self.q_values[aggregation_indices]

        # 2. Exploitation : argmax avec tie-breaking aléatoire
        masked_q = q_exp.clone()
        masked_q[~masks.bool()] = -float("inf")
        tie_breaker = torch.rand_like(masked_q) * 1e-6
        best_actions = torch.argmax(masked_q + tie_breaker, dim=1)  # [N]

        # 3. Exploration : tirage uniforme parmi les routes valides
        valid_probs = masks.float()
        valid_probs = valid_probs / valid_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        random_actions = torch.multinomial(valid_probs, 1).squeeze(1)  # [N]

        # 4. Décision epsilon-greedy
        explore_mask = torch.rand(N, device=self.device) < self.epsilon
        actions = torch.where(explore_mask, random_actions, best_actions)  # [N]

        return actions

    # ── Mise à jour des poids ────────────────────────────────────────────

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Met à jour les Q-values via EMA, agrégé par bloc de paramètres.

        La mise à jour EMA est calculée d'abord pour les N véhicules, puis
        les delta-updates sont moyennés par bloc B via scatter_add_ avant
        d'être appliqués à self.q_values.

        Args:
            actions:              [N] actions prises.
            rewards:              [N] récompenses reçues (temps de trajet < 0).
            aggregation_indices:  [N] mapping véhicules → blocs.
        """
        N = actions.shape[0]
        if N == 0:
            return

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # Q-value actuelle pour chaque (véhicule, action)
        current_q = self.q_values[aggregation_indices, actions]  # [N]

        # Delta de mise à jour EMA individuel pour chaque véhicule
        delta = self.alpha * (rewards.float() - current_q)  # [N]

        # ── Agrégation vers [B, K] via scatter_add_ ──────────────────
        # On calcule la somme et le compte par (bloc, action) pour moyenner.
        flat_idx = aggregation_indices * self.k_paths + actions  # [N]
        flat_size = self.num_models * self.k_paths

        delta_sum = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        delta_sum.scatter_add_(0, flat_idx, delta)
        counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        # Moyenne par bloc actif
        active = counts > 0
        mean_delta = torch.zeros_like(delta_sum)
        mean_delta[active] = delta_sum[active] / counts[active]

        # Application à self.q_values [B, K]
        self.q_values += mean_delta.reshape(self.num_models, self.k_paths)

    # ── Gestion de l'exploration ─────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Décroissance multiplicative de epsilon après chaque épisode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyAgent(B={self.num_models}, K={self.k_paths}, "
            f"ε={self.epsilon:.4f}, α={self.alpha})"
        )
