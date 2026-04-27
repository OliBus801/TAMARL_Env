"""SB3 agent wrapper for the TAMARL training loop.

Adapts Stable-Baselines3 algorithms (PPO, DQN, A2C) to the same interface
used by RandomAgent, QLearningAgent, and MSAAgent: namely ``get_actions()``
and ``update()``.

Internally uses **full parameter sharing** — a single neural network handles
ALL agents regardless of their OD pair.  Each agent's observation (which
includes its current node, destination, etc.) is fed through the shared
policy to obtain an action.  Action masking is supported natively for PPO
(via MaskablePPO) and manually for DQN and A2C (invalid actions masked to
-inf before argmax / softmax).
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

    This agent uses **full parameter sharing**: a single neural network is
    shared across ALL agents.  Each agent's observation is fed through the
    same policy to obtain an action, with action masking applied.
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
        # Legacy kwarg — kept for backward compat, ignored
        od_pairs: Optional[np.ndarray] = None,
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

        # Single shared model for ALL agents
        self._transitions_buffer: list = []

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
                learning_starts=50,
                exploration_initial_eps=1.0,
                exploration_fraction=0.5,
                exploration_final_eps=0.05,
            )
            # SB3 sets exploration_rate=0.0 by default and only
            # updates it during .learn(), which we never call.
            # Manually initialise to the configured starting value.
            self.model.exploration_rate = self.model.exploration_initial_eps
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

        # Track transitions for manual training (batched)
        self._has_prev_t = torch.zeros(env.dnl.num_agents, dtype=torch.bool, device=device)
        self._prev_obs_t: Optional[torch.Tensor] = None    # [N, obs_dim]
        self._prev_actions_t = torch.full((env.dnl.num_agents,), -1, dtype=torch.long, device=device)
        self._accumulated_reward_t = torch.zeros(
            env.dnl.num_agents, device=device, dtype=torch.float32
        )

        # For periodic SB3 updates
        self._step_count = 0
        self._update_every = n_steps if n_steps > 0 else 1024

        # Legacy dict-based tracking (for PettingZoo compat)
        self._prev_obs: Dict[str, np.ndarray] = {}
        self._prev_action: Dict[str, int] = {}
        self._accumulated_reward: Dict[str, float] = {}

    # ══════════════════════════════════════════════════════════════════
    #  BATCHED API  (tensors in, tensors out)
    # ══════════════════════════════════════════════════════════════════

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select actions for all deciding agents in a single batch forward pass.

        Args:
            obs:               [K, obs_dim]     observations for deciding agents
            masks:             [K, max_deg]     action masks (int8, 1=valid)
            deciding_indices:  [K]              agent indices

        Returns:
            actions: [K] tensor of action indices
        """
        K = deciding_indices.numel()
        if K == 0:
            return torch.empty(0, device=obs.device, dtype=torch.long)

        obs_f = obs.to(self.model.device, dtype=torch.float32)
        masks_b = masks.to(self.model.device, dtype=torch.bool)

        if self.algorithm_name == "ppo":
            actions = self._predict_batch_ppo(obs_f, masks_b, self.model, deterministic)
        elif self.algorithm_name == "dqn":
            actions = self._predict_batch_dqn(obs_f, masks_b, self.model, deterministic)
        elif self.algorithm_name == "a2c":
            actions = self._predict_batch_a2c(obs_f, masks_b, self.model, deterministic)

        # Store for update
        if self._prev_obs_t is None:
            self._prev_obs_t = torch.zeros((self.env.dnl.num_agents, obs.shape[1]), device=obs.device)
            
        self._prev_obs_t[deciding_indices] = obs.detach()
        self._prev_actions_t[deciding_indices] = actions.detach()
        self._has_prev_t[deciding_indices] = True

        return actions

    def _predict_batch_ppo(self, obs: torch.Tensor, masks: torch.Tensor, model, deterministic: bool = False) -> torch.Tensor:
        """Batch PPO prediction using MaskablePPO's policy internals."""
        policy = model.policy
        policy.set_training_mode(False)

        with torch.no_grad():
            # Get action distribution from policy
            features = policy.extract_features(obs, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)                  # [K, n_actions]

            # Apply action masks: set invalid to -inf
            logits[~masks] = float("-inf")

            if deterministic:
                actions = logits.argmax(dim=1)                     # [K]
            else:
                # Sample from categorical distribution
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()                            # [K]

        return actions

    def _predict_batch_dqn(self, obs: torch.Tensor, masks: torch.Tensor, model, deterministic: bool = False) -> torch.Tensor:
        """Batch DQN prediction with epsilon-greedy and masking."""
        K = obs.shape[0]

        with torch.no_grad():
            q_values = model.q_net(obs)                       # [K, n_actions]
            q_values[~masks] = float("-inf")

        # Epsilon-greedy: decide which agents explore
        if deterministic:
            explore = torch.zeros(K, dtype=torch.bool, device=obs.device)
        else:
            explore = torch.rand(K, device=obs.device) < model.exploration_rate
            
        greedy_actions = q_values.argmax(dim=1)                     # [K]

        # For exploring agents, sample uniformly from valid actions
        if explore.any():
            n_valid = masks.sum(dim=1).float()                      # [K]
            # Uniform random among valid actions
            rand_probs = masks.float() / n_valid.unsqueeze(1).clamp(min=1)
            random_actions = torch.multinomial(rand_probs, 1).squeeze(1)
            greedy_actions[explore] = random_actions[explore]

        return greedy_actions

    def _predict_batch_a2c(self, obs: torch.Tensor, masks: torch.Tensor, model, deterministic: bool = False) -> torch.Tensor:
        """Batch A2C prediction with masked sampling."""
        policy = model.policy
        policy.set_training_mode(False)

        with torch.no_grad():
            features = policy.extract_features(obs, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)                  # [K, n_actions]

            logits[~masks] = float("-inf")
            
            if deterministic:
                actions = logits.argmax(dim=1)                     # [K]
            else:
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(1)       # [K]

        return actions

    def update_batched(
        self,
        obs_new: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        masks_new: torch.Tensor,
        deciding_new: torch.Tensor,
        active_old: torch.Tensor,
    ):
        """Collect transitions from batched step and trigger training.

        Args:
            obs_new:       [N_active_new, obs_dim]  new observations
            rewards:       [N_active_old]            rewards for prev-active agents
            terminated:    [N_active_old]            bool mask
            truncated:     [N_active_old]            bool mask
            masks_new:     [K', max_deg]             new action masks
            deciding_new:  [K']                      newly deciding indices
            active_old:    [N_active_old]             previous active indices
        """
        if not self._has_prev_t.any():
            return

        # Accumulate rewards for prev-active agents
        has_prev_mask = self._has_prev_t[active_old]
        valid_active_old = active_old[has_prev_mask]
        valid_rewards = rewards[has_prev_mask]
        self._accumulated_reward_t[valid_active_old] += valid_rewards

        # Determine done status for agents
        done_mask = terminated | truncated
        is_done_global = torch.zeros(self.env.dnl.num_agents, dtype=torch.bool, device=active_old.device)
        is_done_global[active_old] = done_mask

        is_deciding_global = torch.zeros(self.env.dnl.num_agents, dtype=torch.bool, device=active_old.device)
        if deciding_new.numel() > 0:
            is_deciding_global[deciding_new] = True

        should_store = self._has_prev_t & (is_done_global | is_deciding_global)

        if should_store.any():
            store_idx = torch.nonzero(should_store).squeeze(-1)
            store_obs = self._prev_obs_t[store_idx]
            store_act = self._prev_actions_t[store_idx]
            store_reward = self._accumulated_reward_t[store_idx]
            store_done = is_done_global[store_idx]

            # Get next obs for these agents
            obs_dim = store_obs.shape[1]
            obs_next = torch.zeros((store_idx.numel(), obs_dim), device=store_obs.device)

            if deciding_new.numel() > 0:
                new_active = self.env._active_indices
                if new_active.numel() > 0 and obs_new.numel() > 0:
                    for j in range(store_idx.numel()):
                        aidx = store_idx[j].item()
                        pos = (new_active == aidx).nonzero(as_tuple=True)[0]
                        if pos.numel() > 0:
                            obs_next[j] = obs_new[pos[0]]

            # Get masks for next state
            max_deg = self.env.dnl.max_out_degree
            masks_next = torch.ones((store_idx.numel(), max_deg), device=store_obs.device, dtype=torch.int8)

            if deciding_new.numel() > 0:
                for j in range(store_idx.numel()):
                    pos = (deciding_new == store_idx[j]).nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        masks_next[j] = masks_new[pos[0]]

            # Store transitions into single shared buffer
            for j in range(store_idx.numel()):
                scaled_reward = float(store_reward[j].item()) / float(self.env._max_steps)
                self._transitions_buffer.append({
                    "obs": store_obs[j].cpu().numpy(),
                    "action": int(store_act[j].item()),
                    "reward": scaled_reward,
                    "obs_next": obs_next[j].cpu().numpy(),
                    "done": bool(store_done[j].item()),
                    "mask": masks_next[j].cpu().numpy(),
                })
                self._step_count += 1
                
                # Reset state
                aidx = store_idx[j].item()
                self._accumulated_reward_t[aidx] = 0.0
                if store_done[j]:
                    self._has_prev_t[aidx] = False
                    self._prev_actions_t[aidx] = -1

        # Trigger model update when buffer is large enough
        if len(self._transitions_buffer) >= self._update_every:
            self._train_from_buffer()

    # ══════════════════════════════════════════════════════════════════
    #  LEGACY DICT API  (for PettingZoo compat)
    # ══════════════════════════════════════════════════════════════════

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        infos: Dict[str, dict],
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Select actions (PettingZoo dict interface)."""
        actions = {}

        for agent_id, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue

            obs = observations.get(agent_id)
            if obs is None:
                continue

            action = self._predict_with_mask(obs, mask, self.model, deterministic)
            actions[agent_id] = action

            self._prev_obs[agent_id] = obs
            self._prev_action[agent_id] = action
            if agent_id not in self._accumulated_reward:
                self._accumulated_reward[agent_id] = 0.0

        return actions

    def _predict_with_mask(self, obs: np.ndarray, mask: np.ndarray, model, deterministic: bool = False) -> int:
        """Get action from the model with action masking applied (single agent)."""
        if self.algorithm_name == "ppo":
            action, _ = model.predict(
                obs, action_masks=mask.astype(bool),
                deterministic=deterministic,
            )
            return int(action)

        elif self.algorithm_name == "dqn":
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = obs_tensor.to(model.device)

            with torch.no_grad():
                q_values = model.q_net(obs_tensor)

            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
            q_values[0, ~mask_tensor] = float("-inf")

            if not deterministic and np.random.random() < model.exploration_rate:
                valid_idxs = np.where(mask > 0)[0]
                return int(np.random.choice(valid_idxs))
            else:
                return int(q_values.argmax(dim=1).item())

        elif self.algorithm_name == "a2c":
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = obs_tensor.to(model.device)

            with torch.no_grad():
                features = model.policy.extract_features(
                    obs_tensor,
                    model.policy.pi_features_extractor,
                )
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)

            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
            logits[0, ~mask_tensor] = float("-inf")

            if deterministic:
                action = int(logits.argmax(dim=1).item())
            else:
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
        """Collect transitions and trigger model updates periodically (dict API)."""
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

            obs_next = observations.get(agent_id)
            if obs_next is None:
                obs_next = np.zeros_like(self._prev_obs[agent_id])

            accum_r = self._accumulated_reward[agent_id]
            scaled_reward = accum_r / float(self.env._max_steps)

            self._transitions_buffer.append(
                {
                    "obs": self._prev_obs[agent_id],
                    "action": self._prev_action[agent_id],
                    "reward": scaled_reward,
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

        if len(self._transitions_buffer) >= self._update_every:
            self._train_from_buffer()

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING  (shared between dict and batched APIs)
    # ══════════════════════════════════════════════════════════════════

    def _train_from_buffer(self):
        """Train the single shared model from collected transitions."""
        if not self._transitions_buffer:
            return

        if self.algorithm_name == "dqn":
            self._train_dqn(self.model, self._transitions_buffer)
            # Off-policy: keep replay buffer up to max size
            max_size = self.model.buffer_size
            if len(self._transitions_buffer) > max_size:
                self._transitions_buffer = self._transitions_buffer[-max_size:]
                
        elif self.algorithm_name in ("ppo", "a2c"):
            self._train_on_policy(self.model, self._transitions_buffer)
            # On-policy: clear rollout buffer completely
            self._transitions_buffer.clear()

    def _train_dqn(self, model, buffer):
        """Train DQN from buffered transitions."""

        policy = model.policy
        policy.set_training_mode(True)

        obs_array = np.array([t["obs"] for t in buffer])
        actions_array = np.array([t["action"] for t in buffer])
        rewards_array = np.array([t["reward"] for t in buffer])
        dones_array = np.array([t["done"] for t in buffer])
        next_obs_array = np.array([t["obs_next"] for t in buffer])
        masks_array = np.array([t["mask"] for t in buffer])

        obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=model.device)
        actions_t = torch.as_tensor(actions_array, dtype=torch.long, device=model.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards_array, dtype=torch.float32, device=model.device)
        dones_t = torch.as_tensor(dones_array, dtype=torch.float32, device=model.device)
        next_obs_t = torch.as_tensor(next_obs_array, dtype=torch.float32, device=model.device)
        masks_t = torch.as_tensor(masks_array, dtype=torch.bool, device=model.device)

        n = len(buffer)
        batch_size = min(n, model.batch_size)
        indices = np.random.choice(n, batch_size, replace=False)

        obs_batch = obs_t[indices]
        actions_batch = actions_t[indices]
        rewards_batch = rewards_t[indices]
        dones_batch = dones_t[indices]
        next_obs_batch = next_obs_t[indices]
        masks_batch = masks_t[indices]

        with torch.no_grad():
            next_q_values = model.q_net_target(next_obs_batch)
            next_q_values[~masks_batch] = float("-inf")
            next_q_max = next_q_values.max(dim=1)[0]
            next_q_max = torch.where(
                torch.isinf(next_q_max), torch.zeros_like(next_q_max), next_q_max
            )
            target_q = rewards_batch + model.gamma * next_q_max * (1.0 - dones_batch)

        current_q_all = model.q_net(obs_batch)
        current_q = current_q_all.gather(1, actions_batch).squeeze(-1)

        loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)

        model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), 10.0)
        model.policy.optimizer.step()

        tau = 0.005
        for param, target_param in zip(
            model.q_net.parameters(), model.q_net_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        policy.set_training_mode(False)

    def _train_on_policy(self, model, buffer):
        """Train PPO/A2C from buffered transitions."""

        policy = model.policy
        policy.set_training_mode(True)

        obs_array = np.array([t["obs"] for t in buffer])
        actions_array = np.array([t["action"] for t in buffer])
        rewards_array = np.array([t["reward"] for t in buffer])
        dones_array = np.array([t["done"] for t in buffer])
        next_obs_array = np.array([t["obs_next"] for t in buffer])
        masks_array = np.array([t["mask"] for t in buffer])

        obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=model.device)
        actions_t = torch.as_tensor(actions_array, dtype=torch.long, device=model.device)
        rewards_t = torch.as_tensor(rewards_array, dtype=torch.float32, device=model.device)
        dones_t = torch.as_tensor(dones_array, dtype=torch.float32, device=model.device)
        next_obs_t = torch.as_tensor(next_obs_array, dtype=torch.float32, device=model.device)
        masks_t = torch.as_tensor(masks_array, dtype=torch.bool, device=model.device)

        with torch.no_grad():
            next_values = policy.predict_values(next_obs_t).squeeze(-1)
            returns = rewards_t + model.gamma * next_values * (1.0 - dones_t)
            values = policy.predict_values(obs_t).squeeze(-1)
            advantages = returns - values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.algorithm_name == "ppo":
            # Baseline probabilities for the PPO clipping ratio
            with torch.no_grad():
                _, old_log_prob, _ = policy.evaluate_actions(obs_t, actions_t, action_masks=masks_t)
                
            for _ in range(4):
                _values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t, action_masks=masks_t)
                
                # PPO Clipped Surrogate Objective
                ratio = torch.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                value_loss = torch.nn.functional.mse_loss(_values.squeeze(-1), returns.detach())
                entropy_loss = -entropy.mean()
                
                # Entropy coefficient at 0.01 allows the network to explore
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()

        elif self.algorithm_name == "a2c":
            _values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t)
            policy_loss = -(log_prob * advantages.detach()).mean()
            value_loss = torch.nn.functional.mse_loss(_values.squeeze(-1), returns.detach())
            entropy_loss = -entropy.mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()

        policy.set_training_mode(False)

    def decay_epsilon(self):
        """Decay epsilon for the DQN model using a multiplicative schedule.

        Called once per episode from the training loop.  Uses the model's
        own ``exploration_initial_eps`` / ``exploration_final_eps`` as bounds
        and applies a multiplicative decay factor (0.995 per episode) so
        that exploration is gradually reduced over training.
        """
        if self.algorithm_name != "dqn":
            return
        decay_factor = 0.995
        self.model.exploration_rate = max(
            self.model.exploration_final_eps,
            self.model.exploration_rate * decay_factor,
        )

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
            f"sharing=full, "
            f"lr={self.model.learning_rate}, "
            f"γ={self.model.gamma})"
        )
