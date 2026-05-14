"""Thompson Sampling agent — Agnostic to aggregation level.

L'agent maintient des distributions a posteriori Normales indépendantes
de taille [B, K] (B blocs de paramètres, K routes candidates). Il est
agnostique au niveau d'agrégation : aggregation_indices [N] gère la
correspondance véhicules ↔ blocs.

Sélection d'action (get_actions_batched)
  1. Projection des paramètres a posteriori : means, stds → [N, K]
  2. Tirage Thompson : échantillonner N(mean, std) pour chaque (véhicule, route)
  3. Masquage + argmax greedy → retourne [N] actions.

Mise à jour des poids (update)
  - L'update conjoint (Normal-Normal) est calculé individuellement pour chaque
    véhicule puis agrégé (moyenné) par bloc via scatter_add_.
  - Mise à jour des paramètres a posteriori self.means et self.stds [B, K].
"""
import torch
from typing import Optional


class ThompsonSamplingAgent:
    r"""Thompson Sampling vectorisé avec prior conjugué Normal-Normal.

    Les récompenses sont supposées être des temps de trajet négatifs.

    Mise à jour conjuguée :
        1/σ²_new = 1/σ²_old + 1/σ²_env
        μ_new    = σ²_new * (μ_old/σ²_old + r/σ²_env)

    Args:
        num_models:   Nombre de blocs de paramètres B.
        k_paths:      Nombre de routes candidates K.
        prior_mean:   [B, K] tenseur de moyennes a priori (optionnel).
        prior_std:    Écart-type a priori initial (prior plat par défaut).
        env_std:      [B] écart-type du bruit environnemental par bloc.
        device:       Dispositif PyTorch.
        seed:         Graine aléatoire.
        num_agents:   Alias rétrocompatible pour num_models.
    """

    def __init__(
        self,
        num_agents: int,       # B (nombre de blocs de paramètres)
        k_paths: int,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: float = 10000.0,
        env_std: Optional[torch.Tensor] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        # Alias moderne
        num_models: Optional[int] = None,
    ):
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.device = device

        # ── Moyennes a posteriori [B, K] ─────────────────────────────
        if prior_mean is not None:
            if not isinstance(prior_mean, torch.Tensor):
                prior_mean = torch.tensor(prior_mean, device=device, dtype=torch.float32)
            self.means = prior_mean.to(device).float()
        else:
            self.means = torch.zeros(
                (num_models, k_paths), device=device, dtype=torch.float32
            )

        # ── Écarts-types a posteriori [B, K] (croyance) ─────────────
        self.stds = torch.full(
            (num_agents, k_paths), prior_std, device=device, dtype=torch.float32
        )

        # ── Variance environnementale [B, 1] ─────────────────────────
        if env_std is not None:
            if not isinstance(env_std, torch.Tensor):
                env_std = torch.tensor(env_std, device=device, dtype=torch.float32)
            if env_std.dim() == 1:
                env_std = env_std.unsqueeze(1)  # [B, 1]
            self.env_var = env_std.to(device).float() ** 2  # [B, 1]
        else:
            self.env_var = torch.full(
                (num_agents, 1), 100.0 ** 2, device=device, dtype=torch.float32
            )

        if seed is not None:
            torch.manual_seed(seed)

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
        """Échantillonne depuis la distribution a posteriori et agit greedy.

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

        # 1. Projection des paramètres a posteriori [B, K] → [N, K]
        m = self.means[aggregation_indices]  # [N, K]
        s = self.stds[aggregation_indices]   # [N, K]

        # 2. Tirage Thompson
        samples = torch.normal(mean=m, std=s)  # [N, K]

        # 3. Masquage des routes invalides
        samples = samples.masked_fill(~masks.bool(), -float("inf"))

        # 4. Argmax greedy avec tie-breaking aléatoire
        tie_breaker = torch.rand_like(samples) * 1e-6
        actions = torch.argmax(samples + tie_breaker, dim=1)  # [N]

        return actions

    # ── Mise à jour des poids ────────────────────────────────────────────

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Mise à jour bayésienne conjuguée des distributions a posteriori.

        Les récompenses de N véhicules sont agrégées (moyennées) par (bloc,
        action) via scatter_add_ avant d'effectuer la mise à jour conjuguée
        Normal-Normal sur les paramètres [B, K].

        Args:
            actions:              [N] indices des routes choisies.
            rewards:              [N] récompenses reçues (temps de trajet < 0).
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
        counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        reward_sum.scatter_add_(0, flat_idx, rewards.float())
        counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        # Reshape vers [B, K]
        reward_sum_bk = reward_sum.reshape(self.num_models, self.k_paths)
        counts_bk = counts.reshape(self.num_models, self.k_paths)
        active_mask = counts_bk > 0  # [B, K]

        # Récompense moyenne par (bloc, action) actif
        mean_reward = torch.zeros_like(reward_sum_bk)
        mean_reward[active_mask] = reward_sum_bk[active_mask] / counts_bk[active_mask]

        # ── Mise à jour conjuguée Normal-Normal ───────────────────────
        # On opère uniquement sur les (bloc, action) actifs
        # env_var est [B, 1] → on expand pour [B, K]
        env_var_bk = self.env_var.expand_as(self.means)  # [B, K]

        old_mean = self.means   # [B, K]
        old_var = self.stds ** 2  # [B, K]
        ev = env_var_bk           # [B, K]

        # 1/σ²_new = 1/σ²_old + 1/σ²_env
        new_var = 1.0 / ((1.0 / old_var) + (1.0 / ev))

        # μ_new = σ²_new * (μ_old/σ²_old + r/σ²_env)
        new_mean = new_var * ((old_mean / old_var) + (mean_reward / ev))

        # Mise à jour uniquement sur les entrées actives (évite de "changer"
        # les distributions non observées ce pas de temps)
        self.means[active_mask] = new_mean[active_mask]
        self.stds[active_mask] = torch.sqrt(new_var[active_mask])

    def __repr__(self) -> str:
        return (
            f"ThompsonSamplingAgent(B={self.num_models}, K={self.k_paths}, "
            f"mean_belief_std={self.stds.mean().item():.1f}, "
            f"avg_env_std={torch.sqrt(self.env_var).mean().item():.1f})"
        )
