"""Epsilon-Greedy bandit agent — Agnostic to aggregation level.

The agent maintains internal Q-values of size [B, K], where B is the
number of parameter "blocks" (B = N independent agents, or B = M OD pairs
in parameter sharing). It does NOT know at which level it operates: it is the
mapping_tensor [N] (either arange(N) or od_indices) provided by train_bandit
that manages the mapping between the N vehicles and the B blocks.

Action Selection (get_actions_batched)
  1. Q-values projection: Q_exp = self.q_values[mapping_tensor]  → [N, K]
  2. Exploitation via argmax with tie-breaking on Q_exp.
  3. Exploration: uniform sampling among valid paths.
  4. Epsilon-greedy decision → returns [N] actions.

Weights Update (update)
  - Calculation of individual EMA update for the N vehicles.
  - Aggregation to [B, K] via scatter_add_ (average per block).
  - Update of self.q_values[B, K] only on active entries.
"""

from typing import Optional

import torch


class EpsilonGreedyAgent:
    r"""Vectorized Epsilon-Greedy Agent, agnostic to the aggregation level.

    Args:
        num_models:     Number of parameter blocks B (= N agents,
                        M OD pairs, or 1 for centralized mode).
        k_paths:        Number of candidate paths K.
        prior_mean:     [B, K] tensor of prior means (optional).
        epsilon_start:  Initial exploration rate.
        epsilon_end:    Minimum exploration rate.
        epsilon_decay:  Multiplicative decay factor per episode.
        alpha:          Learning rate for EMA update.
        device:         PyTorch device ('cpu' or 'cuda').
        seed:           Random seed for reproducibility.
    """

    def __init__(
        self,
        num_agents: int,  # B (number of parameter blocks)
        k_paths: int,
        prior_mean: torch.Tensor | None = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        alpha: float = 0.1,
        device: str = "cpu",
        seed: int | None = None,
        # Alias moderne : num_models prend le dessus sur num_agents si fourni
        num_models: int | None = None,
    ):
        # Alias resolution: num_models takes priority
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.epsilon = epsilon_start
        self.epsilon_end = min(epsilon_start, epsilon_end)
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Q-values [B, K] — initialized with small noise to break symmetry
        self.q_values = (
            torch.randn((self.num_models, k_paths), device=device, dtype=torch.float32) * 1e-6
        )
        if prior_mean is not None:
            if not isinstance(prior_mean, torch.Tensor):
                prior_mean = torch.tensor(prior_mean, device=device, dtype=torch.float32)
            self.q_values += prior_mean.to(device).float()

    # ── Backward compatibility property ──────────────────────────────────
    @property
    def num_agents(self) -> int:
        """Backward compatible alias to num_models."""
        return self.num_models

    # ── Action Selection ─────────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Selects an action for each of the N vehicles.

        Args:
            obs:                  [N, K] observations (unused — blind mode).
            masks:                [N, K] boolean mask of valid paths.
            aggregation_indices:  [N] mapping of each vehicle to its parameter
                                  block. If None, it assumes N = B and
                                  uses torch.arange(N).

        Returns:
            actions: [N] long tensor of selected path indices.
        """
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        # Resolve mapping: by default, identity (agent_level mode)
        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Q-values projection [B, K] → [N, K]
        q_exp = self.q_values[aggregation_indices]

        # 2. Exploitation: argmax with random tie-breaking
        masked_q = q_exp.clone()
        masked_q[~masks.bool()] = -float("inf")
        tie_breaker = torch.rand_like(masked_q) * 1e-6
        best_actions = torch.argmax(masked_q + tie_breaker, dim=1)  # [N]

        # 3. Exploration: uniform sampling among valid paths
        valid_probs = masks.float()
        valid_sums = valid_probs.sum(dim=1, keepdim=True)
        zero_uniform_rows = (valid_sums == 0).squeeze(1)
        if zero_uniform_rows.any():
            valid_probs[zero_uniform_rows] = 1.0
            valid_sums = valid_probs.sum(dim=1, keepdim=True)
        valid_probs = valid_probs / valid_sums.clamp(min=1e-6)
        random_actions = torch.multinomial(valid_probs, 1).squeeze(1)  # [N]

        # 4. Epsilon-greedy decision
        explore_mask = torch.rand(N, device=self.device) < self.epsilon
        actions = torch.where(explore_mask, random_actions, best_actions)  # [N]

        return actions

    # ── Weights Update ───────────────────────────────────────────────────

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Updates Q-values via EMA, aggregated by parameter block.

        The EMA update is calculated first for the N vehicles, then
        the delta-updates are averaged by block B via scatter_add_ before
        being applied to self.q_values.

        Args:
            actions:              [N] taken actions.
            rewards:              [N] received rewards (travel times < 0).
            aggregation_indices:  [N] mapping vehicles → blocks.
        """
        N = actions.shape[0]
        if N == 0:
            return

        # Strategic Ignorance: filter out unstarted legs
        valid_mask = kwargs.get("valid_mask")
        if valid_mask is not None:
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            if aggregation_indices is not None:
                aggregation_indices = aggregation_indices[valid_mask]
            N = actions.shape[0]
            if N == 0:
                return

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # Current Q-value for each (vehicle, action)
        current_q = self.q_values[aggregation_indices, actions]  # [N]

        # Individual EMA update delta for each vehicle
        delta = self.alpha * (rewards.float() - current_q)  # [N]

        # ── Aggregation to [B, K] via scatter_add_ ──────────────────
        # We calculate the sum and the count per (block, action) to average.
        flat_idx = aggregation_indices * self.k_paths + actions  # [N]
        flat_size = self.num_models * self.k_paths

        delta_sum = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        delta_sum.scatter_add_(0, flat_idx, delta)
        counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        # Average per active block
        active = counts > 0
        mean_delta = torch.zeros_like(delta_sum)
        mean_delta[active] = delta_sum[active] / counts[active]

        # Apply to self.q_values [B, K]
        self.q_values += mean_delta.reshape(self.num_models, self.k_paths)

    # ── Exploration Management ───────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Multiplicative decay of epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyAgent(B={self.num_models}, K={self.k_paths}, "
            f"ε={self.epsilon:.4f}, α={self.alpha})"
        )
