"""Gymnasium single-agent wrapper for SB3 compatibility.

Wraps the DTAMarkovGameEnv (PettingZoo ParallelEnv) into a standard
Gymnasium Env using parameter sharing: one NN policy controls all agents.

The wrapper internally iterates over all agents, presenting their individual
observations one-at-a-time to the SB3 algorithm. Each ``step()`` corresponds
to one agent making one decision. When all agents in a macro-step have acted,
the underlying simulator is advanced.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv


class DTASingleAgentWrapper(gym.Env):
    """Turn a PettingZoo ParallelEnv into a Gymnasium Env via parameter sharing.

    The environment exposes one agent decision at a time. Internally it
    maintains a queue of pending decisions. When the queue is exhausted the
    simulator advances to the next macro-step producing a new batch of
    decisions.

    This wrapper implements ``action_masks()`` as required by
    ``sb3_contrib.MaskablePPO``.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, parallel_env: DTAMarkovGameEnv):
        super().__init__()
        self.parallel_env = parallel_env

        # Spaces — same for all agents in parameter-sharing mode
        sample_agent = parallel_env.possible_agents[0]
        self.observation_space = parallel_env.observation_space(sample_agent)
        self.action_space = parallel_env.action_space(sample_agent)

        # Internal state
        self._obs_dict: Dict[str, np.ndarray] = {}
        self._info_dict: Dict[str, dict] = {}
        self._pending_agents: list[str] = []
        self._current_agent: Optional[str] = None
        self._current_mask: Optional[np.ndarray] = None

        # Accumulate actions until all pending agents have acted
        self._collected_actions: Dict[str, int] = {}

        # Track accumulated reward across all agent sub-steps in a macro-step
        self._macro_rewards: Dict[str, float] = {}
        self._episode_done = False

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        obs_dict, info_dict = self.parallel_env.reset(seed=seed)

        self._obs_dict = obs_dict
        self._info_dict = info_dict
        self._episode_done = False
        self._macro_rewards = {}
        self._collected_actions = {}

        # Build pending queue: only agents with valid action masks
        self._build_pending_queue()

        # If no agent is deciding yet (shouldn't happen), return zeros
        if self._current_agent is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {"action_mask": np.zeros(self.action_space.n, dtype=np.int8)}

        obs = self._obs_dict.get(
            self._current_agent,
            np.zeros(self.observation_space.shape, dtype=np.float32),
        )
        info = {"action_mask": self._current_mask}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Process one agent's action.

        If there are still pending agents in this macro-step, buffer the
        action and move to the next agent without advancing the simulator.
        Once all pending agents have acted, call parallel_env.step() and
        build the next pending queue.
        """
        if self._episode_done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"action_mask": np.zeros(self.action_space.n, dtype=np.int8)}

        # Record action for the current agent
        if self._current_agent is not None:
            self._collected_actions[self._current_agent] = int(action)

        # More agents pending in this macro-step?
        if self._pending_agents:
            self._current_agent = self._pending_agents.pop(0)
            self._current_mask = self._get_mask(self._current_agent)
            obs = self._obs_dict.get(
                self._current_agent,
                np.zeros(self.observation_space.shape, dtype=np.float32),
            )
            # Intermediate sub-step: no reward, not done
            return obs, 0.0, False, False, {"action_mask": self._current_mask}

        # All pending agents have acted → advance the simulator
        obs_dict, rewards, terminations, truncations, info_dict = (
            self.parallel_env.step(self._collected_actions)
        )

        # Accumulate rewards from this macro-step
        total_reward = sum(rewards.values()) / max(len(rewards), 1)

        self._obs_dict = obs_dict
        self._info_dict = info_dict
        self._collected_actions = {}

        # Check if episode is done
        if not self.parallel_env.agents:
            self._episode_done = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, float(total_reward), True, False, {
                "action_mask": np.zeros(self.action_space.n, dtype=np.int8)
            }

        # Build next pending queue
        self._build_pending_queue()

        if self._current_agent is None:
            # No deciding agents but episode not done — advance again
            # This can happen if all remaining agents are traveling
            # We "skip" by returning current state with zero reward
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, float(total_reward), False, False, {
                "action_mask": np.ones(self.action_space.n, dtype=np.int8)
            }

        obs = self._obs_dict.get(
            self._current_agent,
            np.zeros(self.observation_space.shape, dtype=np.float32),
        )
        info = {"action_mask": self._current_mask}
        return obs, float(total_reward), False, False, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask (required by MaskablePPO)."""
        if self._current_mask is not None:
            return self._current_mask
        return np.ones(self.action_space.n, dtype=np.int8)

    def close(self):
        self.parallel_env.close()

    # ── Internal helpers ───────────────────────────────────────────────

    def _build_pending_queue(self):
        """Build the queue of agents that have valid action masks."""
        deciding = []
        for agent_id in self.parallel_env.agents:
            info = self._info_dict.get(agent_id, {})
            mask = info.get("action_mask")
            if mask is not None and mask.sum() > 0:
                deciding.append(agent_id)

        if deciding:
            self._current_agent = deciding[0]
            self._current_mask = self._get_mask(self._current_agent)
            self._pending_agents = deciding[1:]
        else:
            self._current_agent = None
            self._current_mask = None
            self._pending_agents = []

    def _get_mask(self, agent_id: str) -> np.ndarray:
        """Get action mask for a specific agent."""
        info = self._info_dict.get(agent_id, {})
        mask = info.get("action_mask")
        if mask is not None:
            return mask.astype(np.int8)
        return np.ones(self.action_space.n, dtype=np.int8)
