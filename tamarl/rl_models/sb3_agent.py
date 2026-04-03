"""SB3 agent wrapper for the TAMARL training loop.

Adapts Stable-Baselines3 algorithms (PPO, DQN, A2C) to the same interface
used by RandomAgent, QLearningAgent, and MSAAgent: namely ``get_actions()``
and ``update()``.

Internally uses parameter sharing — one NN handles all agents.  Action
masking is supported natively for PPO (via MaskablePPO) and manually for
DQN and A2C (invalid actions masked to -inf before argmax / softmax).
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import torch

# Suppress SB3 import warnings about optional deps
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from stable_baselines3 import DQN, A2C
    from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
    from sb3_contrib import MaskablePPO

from tamarl.envs.sb3_gymnasium_wrapper import DTASingleAgentWrapper
from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv


class SB3Agent:
    """Wrapper that adapts SB3 algorithms to the TAMARL training loop.

    Supported algorithms: ``'ppo'``, ``'dqn'``, ``'a2c'``.

    This agent uses **parameter sharing**: a single neural network is shared
    across all agents. Each agent's observation is fed through the same
    policy to obtain an action, with action masking applied.

    Args:
        algorithm: one of ``'ppo'``, ``'dqn'``, ``'a2c'``
        env: the ``DTAMarkovGameEnv`` instance (PettingZoo ParallelEnv)
        learning_rate: learning rate for the optimizer
        gamma: discount factor
        net_arch: list of hidden layer sizes (e.g. [64, 64])
        device: ``'cpu'`` or ``'cuda'``
        seed: random seed
        batch_size: minibatch size for training
        buffer_size: replay buffer size (DQN only)
        n_steps: rollout length (PPO/A2C only)
        verbose: SB3 verbosity level
    """

    SUPPORTED = ("ppo", "dqn", "a2c")

    def __init__(
        self,
        algorithm: str,
        env: DTAMarkovGameEnv,
        learning_rate: float = 3e-4,
        gamma: float = 1.0,
        net_arch: Optional[list] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        n_steps: int = 128,
        verbose: int = 0,
    ):
        algorithm = algorithm.lower()
        if algorithm not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose from {self.SUPPORTED}"
            )

        self.algorithm_name = algorithm
        self.env = env
        self.device_str = device

        if net_arch is None:
            net_arch = [64, 64]

        # Wrap the parallel env into a single-agent Gymnasium env
        self._gym_env = DTASingleAgentWrapper(env)

        policy_kwargs = dict(net_arch=net_arch)

        if algorithm == "ppo":
            self.model = MaskablePPO(
                "MlpPolicy",
                self._gym_env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=seed,
                verbose=verbose,
            )
        elif algorithm == "dqn":
            self.model = DQN(
                "MlpPolicy",
                self._gym_env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=seed,
                verbose=verbose,
                # Start learning after a few steps
                learning_starts=50,
                # Exploration schedule
                exploration_fraction=0.5,
                exploration_final_eps=0.05,
            )
        elif algorithm == "a2c":
            self.model = A2C(
                "MlpPolicy",
                self._gym_env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=seed,
                verbose=verbose,
            )

        # Track transitions for manual training
        self._prev_obs: Dict[str, np.ndarray] = {}
        self._prev_action: Dict[str, int] = {}
        self._accumulated_reward: Dict[str, float] = {}

        # For periodic SB3 updates
        self._step_count = 0
        self._update_every = n_steps  # PPO/A2C update every n_steps
        self._transitions_buffer: list = []

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        infos: Dict[str, dict],
    ) -> Dict[str, int]:
        """Select actions for all deciding agents using the shared policy.

        Action masking is applied:
        - PPO: via MaskablePPO's native ``action_masks`` parameter
        - DQN: Q-values of masked actions set to -inf before argmax
        - A2C: logits of masked actions set to -inf before sampling
        """
        actions = {}

        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue

            obs = observations.get(agent_id)
            if obs is None:
                continue

            action = self._predict_with_mask(obs, mask)
            actions[agent_id] = action

            # Store for update
            self._prev_obs[agent_id] = obs
            self._prev_action[agent_id] = action
            if agent_id not in self._accumulated_reward:
                self._accumulated_reward[agent_id] = 0.0

        return actions

    def _predict_with_mask(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Get action from the model with action masking applied."""
        if self.algorithm_name == "ppo":
            # MaskablePPO supports action_masks natively
            action, _ = self.model.predict(
                obs, action_masks=mask.astype(bool),
                deterministic=False,
            )
            return int(action)

        elif self.algorithm_name == "dqn":
            # Get Q-values and mask invalid actions
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.model.device)

            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor)

            # Mask invalid actions with -inf
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.model.device)
            q_values[0, ~mask_tensor] = float("-inf")

            # Epsilon-greedy with masking
            if np.random.random() < self.model.exploration_rate:
                valid_idxs = np.where(mask > 0)[0]
                return int(np.random.choice(valid_idxs))
            else:
                return int(q_values.argmax(dim=1).item())

        elif self.algorithm_name == "a2c":
            # Get logits and mask invalid actions
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.model.device)

            with torch.no_grad():
                features = self.model.policy.extract_features(
                    obs_tensor,
                    self.model.policy.pi_features_extractor,
                )
                latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                logits = self.model.policy.action_net(latent_pi)

            # Mask invalid actions with -inf
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.model.device)
            logits[0, ~mask_tensor] = float("-inf")

            # Sample from masked distribution
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            return int(action)

        raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

    def update(
        self,
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        infos: Dict[str, dict],
    ):
        """Collect transitions and trigger model updates periodically."""
        for agent_id, reward in rewards.items():
            if agent_id not in self._prev_obs:
                continue

            self._accumulated_reward[agent_id] = (
                self._accumulated_reward.get(agent_id, 0.0) + reward
            )

            done = terminations.get(agent_id, False) or truncations.get(
                agent_id, False
            )
            info = infos.get(agent_id, {})
            mask_next = info.get("action_mask")
            is_deciding = mask_next is not None and mask_next.sum() > 0

            if not done and not is_deciding:
                continue

            # Store transition
            obs_next = observations.get(agent_id)
            if obs_next is None:
                obs_next = np.zeros_like(self._prev_obs[agent_id])

            accum_r = self._accumulated_reward[agent_id]

            self._transitions_buffer.append(
                {
                    "obs": self._prev_obs[agent_id],
                    "action": self._prev_action[agent_id],
                    "reward": accum_r,
                    "obs_next": obs_next,
                    "done": done,
                    "mask": mask_next if mask_next is not None else np.ones(
                        self.env.dnl.max_out_degree, dtype=np.int8
                    ),
                }
            )

            self._accumulated_reward[agent_id] = 0.0
            self._step_count += 1

            if done:
                self._prev_obs.pop(agent_id, None)
                self._prev_action.pop(agent_id, None)
                self._accumulated_reward.pop(agent_id, None)

        # Trigger model update when buffer is large enough
        if len(self._transitions_buffer) >= self._update_every:
            self._train_from_buffer()

    def _train_from_buffer(self):
        """Train models from collected transitions using manual gradient steps."""
        if not self._transitions_buffer:
            return

        if self.algorithm_name == "dqn":
            self._train_dqn()
        elif self.algorithm_name in ("ppo", "a2c"):
            self._train_on_policy()

        self._transitions_buffer = []

    def _train_dqn(self):
        """Train DQN from buffered transitions using manual gradient steps."""
        if not self._transitions_buffer:
            return

        policy = self.model.policy
        policy.set_training_mode(True)

        obs_array = np.array([t["obs"] for t in self._transitions_buffer])
        actions_array = np.array([t["action"] for t in self._transitions_buffer])
        rewards_array = np.array([t["reward"] for t in self._transitions_buffer])
        dones_array = np.array([t["done"] for t in self._transitions_buffer])
        next_obs_array = np.array([t["obs_next"] for t in self._transitions_buffer])
        masks_array = np.array([t["mask"] for t in self._transitions_buffer])

        obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=self.model.device)
        actions_t = torch.as_tensor(actions_array, dtype=torch.long, device=self.model.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards_array, dtype=torch.float32, device=self.model.device)
        dones_t = torch.as_tensor(dones_array, dtype=torch.float32, device=self.model.device)
        next_obs_t = torch.as_tensor(next_obs_array, dtype=torch.float32, device=self.model.device)
        masks_t = torch.as_tensor(masks_array, dtype=torch.bool, device=self.model.device)

        # Sample a minibatch if buffer is larger than batch size
        n = len(self._transitions_buffer)
        batch_size = min(n, self.model.batch_size)
        indices = np.random.choice(n, batch_size, replace=False)

        obs_batch = obs_t[indices]
        actions_batch = actions_t[indices]
        rewards_batch = rewards_t[indices]
        dones_batch = dones_t[indices]
        next_obs_batch = next_obs_t[indices]
        masks_batch = masks_t[indices]

        with torch.no_grad():
            # Target Q-values from target network
            next_q_values = self.model.q_net_target(next_obs_batch)
            # Mask invalid actions in next state
            next_q_values[~masks_batch] = float("-inf")
            next_q_max = next_q_values.max(dim=1)[0]
            # Handle cases where all actions are masked (terminal states)
            next_q_max = torch.where(
                torch.isinf(next_q_max), torch.zeros_like(next_q_max), next_q_max
            )
            target_q = rewards_batch + self.model.gamma * next_q_max * (1.0 - dones_batch)

        # Current Q-values
        current_q_all = self.model.q_net(obs_batch)
        current_q = current_q_all.gather(1, actions_batch).squeeze(-1)

        # Huber loss (smooth L1)
        loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)

        self.model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.q_net.parameters(), 10.0)
        self.model.policy.optimizer.step()

        # Soft-update target network
        tau = 0.005
        for param, target_param in zip(
            self.model.q_net.parameters(), self.model.q_net_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Decay exploration rate
        self.model.exploration_rate = max(
            self.model.exploration_final_eps,
            self.model.exploration_rate * 0.999,
        )

        policy.set_training_mode(False)

    def _train_on_policy(self):
        """Train PPO/A2C from buffered transitions using manual gradient steps."""
        if not self._transitions_buffer:
            return

        policy = self.model.policy
        policy.set_training_mode(True)

        obs_array = np.array([t["obs"] for t in self._transitions_buffer])
        actions_array = np.array([t["action"] for t in self._transitions_buffer])
        rewards_array = np.array([t["reward"] for t in self._transitions_buffer])
        dones_array = np.array([t["done"] for t in self._transitions_buffer])
        next_obs_array = np.array([t["obs_next"] for t in self._transitions_buffer])

        obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=self.model.device)
        actions_t = torch.as_tensor(actions_array, dtype=torch.long, device=self.model.device)
        rewards_t = torch.as_tensor(rewards_array, dtype=torch.float32, device=self.model.device)
        dones_t = torch.as_tensor(dones_array, dtype=torch.float32, device=self.model.device)
        next_obs_t = torch.as_tensor(next_obs_array, dtype=torch.float32, device=self.model.device)

        # Compute returns (simple Monte Carlo or 1-step TD)
        with torch.no_grad():
            next_values = policy.predict_values(next_obs_t).squeeze(-1)
            returns = rewards_t + self.model.gamma * next_values * (1.0 - dones_t)

            values = policy.predict_values(obs_t).squeeze(-1)
            advantages = returns - values

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        if self.algorithm_name == "ppo":
            # PPO-style update
            for _ in range(4):  # n_epochs
                # Re-evaluate actions
                _values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t)
                # We do a simple policy gradient update
                policy_loss = -(log_prob * advantages.detach()).mean()
                value_loss = torch.nn.functional.mse_loss(_values.squeeze(-1), returns.detach())
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.0 * entropy_loss

                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()

        elif self.algorithm_name == "a2c":
            _values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t)
            policy_loss = -(log_prob * advantages.detach()).mean()
            value_loss = torch.nn.functional.mse_loss(_values.squeeze(-1), returns.detach())
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()

        policy.set_training_mode(False)

    def decay_epsilon(self):
        """No-op — SB3 handles exploration internally."""
        pass

    @property
    def epsilon(self) -> float:
        """Return exploration rate for logging (DQN only)."""
        if self.algorithm_name == "dqn":
            return self.model.exploration_rate
        return 0.0

    def __repr__(self) -> str:
        algo = self.algorithm_name.upper()
        return (
            f"SB3Agent(algorithm={algo}, "
            f"lr={self.model.learning_rate}, "
            f"γ={self.model.gamma})"
        )
