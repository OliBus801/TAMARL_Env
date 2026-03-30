"""Random agent policy for the DTA Markov Game environment."""

import numpy as np
from typing import Dict, Optional


class RandomAgent:
    """Simple policy that samples uniformly from valid (masked) actions.
    
    Compatible with the PettingZoo ParallelEnv API.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def get_actions(
        self, 
        observations: Dict[str, np.ndarray], 
        infos: Dict[str, dict]
    ) -> Dict[str, int]:
        """Select random valid actions for all agents with action masks.
        
        Args:
            observations: dict of agent_id → observation array
            infos: dict of agent_id → info dict containing 'action_mask'
            
        Returns:
            Dict of agent_id → selected action index
        """
        actions = {}
        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is not None and mask.sum() > 0:
                valid_indices = np.where(mask > 0)[0]
                actions[agent_id] = int(self.rng.choice(valid_indices))
            # If no valid actions (agent not deciding), skip
        return actions
