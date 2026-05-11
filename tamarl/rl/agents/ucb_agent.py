"""UCB (Upper Confidence Bound) bandit agent — Agnostic to aggregation level.

L'agent maintient des Q-values et des compteurs internes de taille [B, K].
Il est agnostique au niveau d'agrégation : c'est aggregation_indices [N]
fourni par train_bandit qui gère la correspondance véhicules ↔ blocs.

Sélection d'action (get_actions_batched)
  1. Projection [B, K] → [N, K] via aggregation_indices.
  2. Score UCB = Q_projected + c * sqrt(ln(t_projected) / (counts_projected + ε))
  3. Argmax avec tie-breaking → retourne [N] actions.

Mise à jour des poids (update)
  - Agrégation des récompenses et compteurs vers [B, K] via scatter_add_.
  - Mise à jour de la moyenne incrémentale : Q = Q + (R_mean - Q) / N_total.
  - Incrément du compteur de temps global t_per_model [B].
"""
import torch
from typing import Optional


class UCBAgent:
    r"""Agent UCB vectorisé, agnostique au niveau d'agrégation.

    Le bonus d'exploration est :
        UCB(a) = Q(a) + c * sqrt( ln(t) / n(a) )

    Args:
        num_models:      Nombre de blocs de paramètres B.
        k_paths:         Nombre de routes candidates K.
        c_exploration:   Constante d'exploration c.
        device:          Dispositif PyTorch.
        num_agents:      Alias rétrocompatible pour num_models.
    """

    def __init__(
        self,
        num_agents: int,       # B (nombre de blocs de paramètres)
        k_paths: int,
        c_exploration: float = 100.0,
        device: str = "cpu",
        # Alias moderne
        num_models: Optional[int] = None,
    ):
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.c = c_exploration
        self.device = device

        # Compteurs d'utilisation [B, K]
        self.action_counts = torch.zeros(
            (num_agents, k_paths), device=device, dtype=torch.float32
        )

        # Q-values estimées [B, K] — bruit initial pour briser la symétrie
        self.q_values = (
            torch.randn((num_agents, k_paths), device=device, dtype=torch.float32) * 1e-6
        )

        # Temps global par bloc [B] — initialisé à 1 pour éviter ln(0)
        self.t_per_model = torch.ones(num_agents, device=device, dtype=torch.float32)

    # ── Rétrocompatibilité ───────────────────────────────────────────────
    @property
    def num_agents(self) -> int:
        return self.num_models

    # ── Sélection d'actions ──────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sélectionne l'action avec le score UCB le plus élevé.

        Args:
            obs:                  [N, K] (non utilisé — mode blind).
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] mapping véhicules → blocs. Si None,
                                  identité (mode agent_level).

        Returns:
            actions: [N] tenseur long.
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Projection [B, K] → [N, K]
        counts = self.action_counts[aggregation_indices]   # [N, K]
        q_vals = self.q_values[aggregation_indices]        # [N, K]
        t = self.t_per_model[aggregation_indices]          # [N]

        # 2. Terme d'exploration UCB
        ln_t = torch.log(t).unsqueeze(1)  # [N, 1]
        exploration_term = self.c * torch.sqrt(ln_t / (counts + 1e-6))  # [N, K]

        # 3. Score UCB total
        ucb_scores = q_vals + exploration_term  # [N, K]

        # 4. Masquage des routes invalides
        ucb_scores = ucb_scores.masked_fill(~masks.bool(), -float("inf"))

        # 5. Argmax avec tie-breaking aléatoire
        tie_breaker = torch.rand_like(ucb_scores) * 1e-6
        actions = torch.argmax(ucb_scores + tie_breaker, dim=1)  # [N]

        return actions

    # ── Mise à jour des poids ────────────────────────────────────────────

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Met à jour les Q-values et les compteurs via moyenne incrémentale.

        Les N récompenses sont d'abord moyennées par bloc (B) via scatter_add_,
        puis intégrées dans les Q-values par mise à jour incrémentale :
            Q_new = Q_old + (R_mean - Q_old) / n_total

        Args:
            actions:              [N] indices des routes choisies.
            rewards:              [N] récompenses reçues.
            aggregation_indices:  [N] mapping véhicules → blocs.
        """
        N = actions.shape[0]
        if N == 0:
            return

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # ── Agrégation vers [B, K] via scatter_add_ ──────────────────
        flat_idx = aggregation_indices * self.k_paths + actions  # [N]
        flat_size = self.num_models * self.k_paths

        reward_sum = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        counts_new = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        reward_sum.scatter_add_(0, flat_idx, rewards.float())
        counts_new.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        # Reshape vers [B, K]
        reward_sum_bk = reward_sum.reshape(self.num_models, self.k_paths)
        counts_new_bk = counts_new.reshape(self.num_models, self.k_paths)
        active_mask = counts_new_bk > 0  # [B, K]

        # Récompense moyenne par (bloc, action) actif
        mean_reward = torch.zeros_like(reward_sum_bk)
        mean_reward[active_mask] = reward_sum_bk[active_mask] / counts_new_bk[active_mask]

        # ── Mise à jour incrémentale de la moyenne : Q = Q + (R - Q) / n ─
        # n_total après cette mise à jour
        self.action_counts += counts_new_bk
        n_total = self.action_counts  # [B, K]

        # Mise à jour uniquement sur les (bloc, action) actifs
        q_old = self.q_values[active_mask]
        n = n_total[active_mask]
        r = mean_reward[active_mask]
        self.q_values[active_mask] = q_old + (r - q_old) / n

        # ── Incrément du temps global par bloc ────────────────────────
        # Un bloc est actif si au moins une de ses actions a été utilisée
        bloc_active = counts_new_bk.sum(dim=1) > 0  # [B]
        self.t_per_model[bloc_active] += 1

    # ── Housekeeping ─────────────────────────────────────────────────────

    def end_episode(self) -> None:
        """Optionnel : synchronisation à la fin de l'épisode."""
        pass

    def __repr__(self) -> str:
        return f"UCBAgent(B={self.num_models}, K={self.k_paths}, c={self.c:.1f})"
