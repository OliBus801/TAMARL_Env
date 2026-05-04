"""Random agent for the one-shot bandit DTA environment.

Each vehicle independently samples a uniform random route index
from {0, ..., K-1}.  Compatible with VehicleLevelWrapper
(gymnasium.vector.VectorEnv).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class RandomAgent:
    """Uniform random route selector for the VehicleLevelWrapper.

    At each step, every vehicle picks a random path index uniformly
    from the K candidate routes available for its OD pair.

    Args:
        num_agents: Number of vehicles (A).
        k: Number of candidate routes per OD pair.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, num_agents: int, k: int, seed: Optional[int] = None):
        self.num_agents = num_agents
        self.k = k
        self.rng = np.random.default_rng(seed)

    def act(self) -> np.ndarray:
        """Sample a random route for every vehicle.

        Returns:
            actions: [A] int array with values in {0, ..., K-1}.
        """
        return self.rng.integers(0, self.k, size=self.num_agents)
