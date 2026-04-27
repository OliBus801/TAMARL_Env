"""Random agent policy for the DTA Markov Game environment."""

import numpy as np
import torch
from typing import Dict, Optional


class RandomAgent:
    """Simple policy that samples uniformly from valid (masked) actions.
    
    Compatible with both the PettingZoo dict API and the batched tensor API.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ── Batched API ───────────────────────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select random valid actions for all deciding agents (vectorised).

        Args:
            obs:               [K, obs_dim]  (unused, kept for API compat)
            masks:             [K, max_deg]  action masks (int8, 1=valid)
            deciding_indices:  [K]           agent indices
            deterministic:     bool          (unused, kept for API compat)

        Returns:
            actions: [K] tensor of action indices
        """
        K = deciding_indices.numel()
        if K == 0:
            return torch.empty(0, device=deciding_indices.device, dtype=torch.long)

        # Uniform probability over valid actions, then sample
        probs = masks.float()
        n_valid = probs.sum(dim=1, keepdim=True).clamp(min=1)
        probs = probs / n_valid
        return torch.multinomial(probs, 1).squeeze(1)

    # ── Dict API (legacy) ────────────────────────────────────────────

    def get_actions(
        self, 
        observations: Dict[str, np.ndarray], 
        infos: Dict[str, dict],
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Select random valid actions for all agents with action masks."""
        actions = {}
        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is not None and mask.sum() > 0:
                valid_indices = np.where(mask > 0)[0]
                actions[agent_id] = int(self.rng.choice(valid_indices))
        return actions

