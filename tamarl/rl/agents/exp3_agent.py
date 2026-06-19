"""Exp3 bandit agent — Agnostic to aggregation level.

L'agent maintient des poids internes de taille [B, K] (B = nombre de blocs
de paramètres). Il ne contient aucune logique conditionnelle "si OD / si
agent". C'est le mapping_tensor [N] (aggregation_indices) fourni par
train_bandit qui détermine la correspondance entre véhicules et blocs.

Sélection d'action (get_actions_batched)
  1. Projection des poids : w_exp = self.weights[aggregation_indices] → [N, K]
  2. Distribution de probabilités Hedge (Softmax) mixée avec uniforme.
  3. Sauvegarde des probabilités d'échantillonnage pour l'update.
  4. Tirage via torch.multinomial → retourne [N] actions.

Mise à jour des poids (update)
  - Calcul de l'estimateur sans biais : cost_hat = cost_norm / p_action [N]
  - Agrégation vers [B, K] via scatter_add_ (moyenne par bloc).
  - Mise à jour multiplicative de self.weights[B, K] sur les blocs actifs.
"""
import torch
import torch.nn.functional as F
from typing import Optional


class Exp3Agent:
    r"""Exp3 (Exponential-weight for Exploration and Exploitation) agent.

    Conçu pour les bandits adversariaux (information partielle : seul le coût
    de l'action choisie est observé). Utilise l'estimateur sans biais :
        cost_hat = cost_norm / p_action.

    Opère avec des poids bruts (mise à jour multiplicative) pour garantir
    des mises à jour réactives.

    Args:
        num_models:  Nombre de blocs de paramètres B.
        k_paths:     Nombre de routes candidates K.
        eta:         Taux d'apprentissage (learning rate).
        gamma:       Taux de mélange avec la distribution uniforme (exploration).
        rho:         Facteur de normalisation des coûts (ex: 2 * max FFTT).
        device:      Dispositif PyTorch.
        seed:        Graine aléatoire.
    """

    def __init__(
        self,
        num_agents: int,       # B (nombre de blocs de paramètres)
        k_paths: int,
        eta: float = 0.01,
        gamma: float = 0.05,
        rho: float = 1.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        # Alias moderne
        num_models: Optional[int] = None,
    ):
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.eta = eta
        self.gamma = gamma
        self.rho = rho
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Poids internes [B, K], initialisés à 1.0 selon Exp3 standard
        self.weights = torch.ones(
            (num_agents, k_paths), device=device, dtype=torch.float32
        )

        # Probabilités du dernier tirage [N, K] — sauvegardées pour l'update
        self._last_probs: Optional[torch.Tensor] = None
        # Aggregation_indices utilisés lors du dernier tirage — nécessaires pour l'update
        self._last_agg_idx: Optional[torch.Tensor] = None

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
        """Échantillonne des actions selon la distribution Hedge + uniforme.

        P(i) = (1 - gamma) * (w_i / sum_w) + gamma * (1 / k_valid)

        Args:
            obs:                  [N, K] (non utilisé — mode blind).
            masks:                [N, K] masque booléen des routes valides.
            aggregation_indices:  [N] mapping véhicules → blocs B. Si None,
                                  identité (N = B, mode agent_level).

        Returns:
            actions: [N] tenseur long.
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Projection des poids [B, K] → [N, K]
        w = self.weights[aggregation_indices]  # [N, K]

        # Zéro pour les routes invalides
        w = w.masked_fill(~masks.bool(), 0.0)

        # 2. Distribution Hedge (normalisation par la somme des poids)
        sum_w = w.sum(dim=1, keepdim=True)
        zero_w_rows = (sum_w == 0).squeeze(1)
        p_hedge = w / (sum_w + 1e-10)  # [N, K]
        if zero_w_rows.any():
            p_hedge[zero_w_rows] = 1.0 / self.k_paths

        # 3. Mélange avec distribution uniforme sur les chemins valides
        k_valid = masks.float().sum(dim=1, keepdim=True)
        zero_mask_rows = (k_valid == 0).squeeze(1)
        p_uniform = masks.float()
        if zero_mask_rows.any():
            p_uniform[zero_mask_rows] = 1.0
            k_valid = p_uniform.sum(dim=1, keepdim=True)
        p_uniform = p_uniform / (k_valid + 1e-8)  # [N, K]

        probs = (1.0 - self.gamma) * p_hedge + self.gamma * p_uniform  # [N, K]

        # 4. Sauvegarde pour l'update
        self._last_probs = probs
        self._last_agg_idx = aggregation_indices

        # 5. Échantillonnage
        actions = torch.multinomial(probs, 1).squeeze(1)  # [N]
        return actions

    # ── Mise à jour des poids ────────────────────────────────────────────

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Mise à jour multiplicative Exp3 : w[a] *= exp(-eta * cost_hat / p_a).

        Le signal d'apprentissage (cost_hat) est calculé individuellement pour
        les N véhicules, puis moyenné par bloc (B) via scatter_add_ avant
        d'être appliqué aux poids.

        Args:
            actions:              [N] indices des routes choisies.
            rewards:              [N] récompenses (temps de trajet négatif).
            aggregation_indices:  [N] mapping véhicules → blocs B.
        """
        if self._last_probs is None:
            return

        N = actions.shape[0]
        if N == 0:
            return

        # Strategic Ignorance: filter out unstarted legs
        valid_mask = kwargs.get('valid_mask')
        if valid_mask is not None:
            # _last_probs is [N_full, K], so we must filter it BEFORE filtering actions
            if self._last_probs is not None:
                self._last_probs = self._last_probs[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            if aggregation_indices is not None:
                aggregation_indices = aggregation_indices[valid_mask]
            elif self._last_agg_idx is not None:
                self._last_agg_idx = self._last_agg_idx[valid_mask]
            N = actions.shape[0]
            if N == 0:
                return

        if aggregation_indices is None:
            if self._last_agg_idx is not None:
                aggregation_indices = self._last_agg_idx
            else:
                aggregation_indices = torch.arange(N, device=self.device)

        # Coûts normalisés (rewards = -travel_time → costs = travel_time)
        costs_norm = (-rewards.float()) / self.rho  # [N]

        vehicle_indices = torch.arange(N, device=self.device)

        # Probabilité de l'action choisie pour chaque véhicule
        p_a = self._last_probs[vehicle_indices, actions]  # [N]

        # Estimateur sans biais : cost_hat = cost_norm / p_a
        cost_hat = costs_norm / (p_a + 1e-8)  # [N]

        # ── Agrégation vers [B, K] via scatter_add_ ──────────────────
        flat_idx = aggregation_indices * self.k_paths + actions  # [N]
        flat_size = self.num_models * self.k_paths

        cost_sum = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        cost_sum.scatter_add_(0, flat_idx, cost_hat)
        counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        # Moyenne du coût estimé par (bloc, action) actif
        active = counts > 0
        mean_cost = torch.zeros_like(cost_sum)
        mean_cost[active] = cost_sum[active] / counts[active]

        # ── Mise à jour multiplicative des poids [B, K] ───────────────
        multipliers = torch.ones_like(cost_sum)
        multipliers[active] = torch.exp(-self.eta * mean_cost[active])
        self.weights *= multipliers.reshape(self.num_models, self.k_paths)

        # ── Renormalisation périodique (prévient overflow/underflow) ──
        if torch.any(self.weights > 1e10) or torch.any(self.weights < 1e-10):
            sum_w = self.weights.sum(dim=1, keepdim=True)
            self.weights = self.weights * (self.k_paths / (sum_w + 1e-10))

    def __repr__(self) -> str:
        return (
            f"Exp3Agent(B={self.num_models}, K={self.k_paths}, "
            f"eta={self.eta:.4f}, gamma={self.gamma:.4f}, rho={self.rho:.1f})"
        )
