"""Replicator Dynamics bandit agent.

Implements Replicator Dynamics for Dynamic Traffic Assignment.
The agent maintains a probability distribution over K candidate routes
for each of its B parameter blocks.

Action Selection (get_actions_batched):
  1. Projection of probabilities via aggregation_indices.
  2. Epsilon noise addition (decaying over time) and masking.
  3. Multinomial sampling.

Weights Update (update):
  - Calculate average utility u_k = exp(beta * reward) for chosen routes.
  - Apply mean substitution for unchosen routes.
  - Update probabilities using RD equation: p_k(t+1) = p_k(t) * u_k / u_bar
"""

from typing import Optional

import torch


class ReplicatorDynamicsAgent:
    r"""Vectorized Replicator Dynamics Agent.

    Args:
        num_models:     Number of parameter blocks B (= N agents,
                        M OD pairs, or 1 for centralized mode).
        k_paths:        Number of candidate paths K.
        beta:           Temperature / cost sensitivity parameter.
        prior_mean:     [B, K] tensor of prior means (negative FFTT).
        epsilon_start:  Initial exploration noise rate.
        epsilon_end:    Minimum exploration noise rate.
        epsilon_decay:  Multiplicative decay factor per episode.
        device:         PyTorch device ('cpu' or 'cuda').
        seed:           Random seed for reproducibility.
    """

    def __init__(
        self,
        num_agents: int,
        k_paths: int,
        beta: float = 0.1,
        prior_mean: torch.Tensor | None = None,
        epsilon_start: float = 0.05,
        epsilon_end: float = 1e-8,
        epsilon_decay: float = 0.99,
        device: str = "cpu",
        seed: int | None = None,
        num_models: int | None = None,
    ):
        if num_models is not None:
            num_agents = num_models

        self.num_models = num_agents
        self.k_paths = k_paths
        self.beta = beta
        self.epsilon = epsilon_start
        self.epsilon_end = min(epsilon_start, epsilon_end)
        self.epsilon_decay = epsilon_decay
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        if prior_mean is not None:
            if not isinstance(prior_mean, torch.Tensor):
                prior_mean = torch.tensor(prior_mean, device=device, dtype=torch.float32)
            # Initialize with Softmax over beta * prior_mean (prior_mean is negative FFTT)
            prior_mean = prior_mean.to(device).float()
            self.rd_probs = torch.softmax(self.beta * prior_mean, dim=1)
            # Fallback to uniform if all values are -inf (yielding nan)
            nan_rows = torch.isnan(self.rd_probs).any(dim=1)
            if nan_rows.any():
                self.rd_probs[nan_rows] = 1.0 / k_paths
        else:
            self.rd_probs = (
                torch.ones((self.num_models, k_paths), device=device, dtype=torch.float32) / k_paths
            )

    @property
    def num_agents(self) -> int:
        return self.num_models

    @property
    def entropy(self) -> float:
        """Returns the mean entropy of the RD probabilities across all blocks."""
        # Add small epsilon to avoid log(0)
        p = self.rd_probs.clamp(min=1e-10)
        ent = -(p * torch.log(p)).sum(dim=1).mean()
        return float(ent.item())

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        N = masks.shape[0]
        if N == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        if aggregation_indices is None:
            aggregation_indices = torch.arange(N, device=self.device)

        # 1. Projection probs [B, K] → [N, K]
        p_rd = self.rd_probs[aggregation_indices]

        # 2. Masking valid paths
        p_rd = p_rd * masks.float()

        # Fallback if a row becomes all zeros
        row_sums = p_rd.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0).squeeze(1)
        if zero_rows.any():
            # If the mask is all zeros, we fallback to uniform over all actions (all ones)
            # otherwise we fallback to uniform over the valid paths (masks)
            mask_sums = masks[zero_rows].sum(dim=1, keepdim=True)
            fallback_probs = masks[zero_rows].float()
            all_zero_masks = (mask_sums == 0).squeeze(1)
            if all_zero_masks.any():
                fallback_probs[all_zero_masks] = 1.0

            p_rd[zero_rows] = fallback_probs
            row_sums = p_rd.sum(dim=1, keepdim=True)

        p_rd = p_rd / row_sums.clamp(min=1e-10)

        # 3. Add exploration noise
        p_uniform = masks.float()
        p_uniform_sums = p_uniform.sum(dim=1, keepdim=True)
        zero_uniform_rows = (p_uniform_sums == 0).squeeze(1)
        if zero_uniform_rows.any():
            p_uniform[zero_uniform_rows] = 1.0
            p_uniform_sums = p_uniform.sum(dim=1, keepdim=True)

        p_uniform = p_uniform / p_uniform_sums.clamp(min=1e-10)

        p_final = (1.0 - self.epsilon) * p_rd + self.epsilon * p_uniform

        # 4. Multinomial sampling
        actions = torch.multinomial(p_final, 1).squeeze(1)

        return actions

    def update(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        aggregation_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        N = actions.shape[0]
        if N == 0:
            return

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

        # Compute sum and count of rewards per (block, action)
        flat_idx = aggregation_indices * self.k_paths + actions
        flat_size = self.num_models * self.k_paths

        sum_rewards = torch.zeros(flat_size, device=self.device, dtype=torch.float32)
        counts = torch.zeros(flat_size, device=self.device, dtype=torch.float32)

        sum_rewards.scatter_add_(0, flat_idx, rewards.float())
        counts.scatter_add_(0, flat_idx, torch.ones(N, device=self.device))

        sum_rewards = sum_rewards.reshape(self.num_models, self.k_paths)
        counts = counts.reshape(self.num_models, self.k_paths)

        # Calculate mean reward for chosen routes
        active = counts > 0
        mean_rewards = torch.zeros_like(sum_rewards)
        mean_rewards[active] = sum_rewards[active] / counts[active]

        # Calculate utility for chosen routes: u_k = exp(beta * mean_reward)
        u_k = torch.zeros_like(mean_rewards)
        # To avoid overflow, we could subtract max before exp, but RD is translation invariant if we scale properly.
        # u_k = exp(beta * (reward - baseline)). Actually RD: u_k = exp(beta * reward).
        # If rewards are large negative (e.g. -100 to -1000), beta * reward can underflow.
        # Let's normalize by subtracting the max mean_reward per block before exp.
        # This keeps the ratios identical.
        max_rewards = torch.zeros_like(mean_rewards)
        max_rewards[active] = mean_rewards[active]
        max_rewards[~active] = -float("inf")
        block_max = max_rewards.max(dim=1, keepdim=True).values.clamp(min=-1e6)

        normalized_mean_rewards = mean_rewards - block_max
        u_k[active] = torch.exp(self.beta * normalized_mean_rewards[active])

        # Mean substitution for unchosen routes
        total_counts = counts.sum(dim=1, keepdim=True).clamp(min=1e-10)
        u_bar_observed = (u_k * counts).sum(dim=1, keepdim=True) / total_counts

        # Only apply mean substitution for blocks that actually had vehicles
        active_blocks = (total_counts > 1e-10).expand_as(u_k)
        unchosen = (~active) & active_blocks
        u_k[unchosen] = u_bar_observed.expand_as(u_k)[unchosen]

        # Replicator Dynamics update
        active_block_mask = (total_counts > 1e-10).squeeze(1)

        u_bar_rd = (self.rd_probs * u_k).sum(dim=1, keepdim=True).clamp(min=1e-10)

        new_probs = self.rd_probs.clone()
        new_probs[active_block_mask] = (
            self.rd_probs[active_block_mask] * u_k[active_block_mask]
        ) / u_bar_rd[active_block_mask]

        # Normalize to prevent drift
        new_probs = new_probs / new_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)

        # Guard against NaNs
        nan_mask = torch.isnan(new_probs).any(dim=1)
        if nan_mask.any():
            new_probs[nan_mask] = 1.0 / self.k_paths

        self.rd_probs = new_probs

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"ReplicatorDynamicsAgent(B={self.num_models}, K={self.k_paths}, "
            f"β={self.beta}, ε={self.epsilon:.4e})"
        )
