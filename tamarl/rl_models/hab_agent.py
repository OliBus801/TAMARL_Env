"""Hierarchical Advisory Behavior (HAB) Agent for DTA Markov Game.

Implements a three-tier architecture:
  1. **TimeAdvisor** — adjusts t_critical (max travel time budget).
  2. **SizeAdvisor** — adjusts |Π|_critical (max path-set size).
  3. **ConstrainedMainAgent** — selects paths from the restricted set A†.

All three use tabular Q-learning with adaptive learning rates.
The orchestrator ``HABAgent`` coordinates them via ``get_actions_batched``
and ``update_hab``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════
#  Helper: adaptive alpha with numerical stability
# ═══════════════════════════════════════════════════════════════════

def _adaptive_alpha(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    """Compute (1 + exp(-x)) / (1 - exp(-x)), clamped to [lo, hi]."""
    x = max(x, 1e-8)  # avoid division by zero
    e = np.exp(-x)
    val = (1.0 - e) / (1.0 + e + 1e-12)
    return float(np.clip(val, lo, hi))


def _hab_reward(delta_improve: float, eta_sign: float) -> float:
    """Compute the HAB advisor/main-agent reward function.

    r = (1 - exp(-η · log(|Δc| + 1))) / (1 + exp(-η · log(|Δc| + 1)))
    or the advisor variant with swapped signs.
    """
    log_val = np.log(abs(delta_improve) + 1.0)
    exponent = -eta_sign * log_val
    e = np.exp(np.clip(exponent, -20, 20))
    return float((1.0 - e) / (1.0 + e + 1e-12))


# ═══════════════════════════════════════════════════════════════════
#  Advisor base
# ═══════════════════════════════════════════════════════════════════

class _AdvisorBase:
    """Shared logic for Time and Size advisors.

    State discretisation: edge densities are binned into 3 levels
    (free < 0.5, resistance 0.5–0.8, jam > 0.8).  The state key is the
    tuple of per-edge bin indices.

    Action: (direction ∈ {-1, +1}, magnitude index).
    Encoded as a single integer:  action = dir_idx * n_magnitudes + mag_idx.
    """

    FREE_THRESH = 0.5
    RESIST_THRESH = 0.8

    def __init__(
        self,
        n_edges: int,
        n_magnitudes: int = 5,
        kappa_max: float = 1.0,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_edges = n_edges
        self.n_magnitudes = n_magnitudes
        self.kappa_max = kappa_max
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 2 directions × n_magnitudes
        self.n_actions = 2 * n_magnitudes
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self._prev_state: Optional[Tuple] = None
        self._prev_action: int = -1
        self._has_prev = False

    # ── state discretisation ──────────────────────────────────────

    def _discretise_density(self, density: np.ndarray) -> Tuple:
        """Bin edge densities into (free=0, resistance=1, jam=2)."""
        bins = np.zeros(len(density), dtype=np.int8)
        bins[density >= self.FREE_THRESH] = 1
        bins[density >= self.RESIST_THRESH] = 2
        return tuple(bins.tolist())

    # ── action decode ─────────────────────────────────────────────

    def _decode_action(self, action: int) -> Tuple[int, float]:
        """Decode integer action to (direction ∈ {-1,+1}, value)."""
        dir_idx = action // self.n_magnitudes
        mag_idx = action % self.n_magnitudes
        direction = -1 if dir_idx == 0 else 1
        # Value is linearly spaced in (0, kappa_max]
        value = self.kappa_max * (mag_idx + 1) / self.n_magnitudes
        return direction, value

    # ── epsilon-greedy select ─────────────────────────────────────

    def select_action(self, state: Tuple, rng: np.random.Generator, deterministic: bool = False) -> int:
        if not deterministic and rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        q = self.q_table[state]
        return int(np.argmax(q))

    # ── Q-update (caller supplies alpha and reward) ───────────────

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        alpha: float,
        done: bool = False,
    ):
        q = self.q_table[state]
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        q[action] += alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ═══════════════════════════════════════════════════════════════════
#  TimeAdvisor
# ═══════════════════════════════════════════════════════════════════

class TimeAdvisor(_AdvisorBase):
    """Adjusts t_critical — the maximum travel-time budget per OD pair.

    α_i = clamp((1+exp(-Δυ)) / (1-exp(-Δυ)))
    where Δυ = |t_critical - t̄_actual|.
    """

    def __init__(self, n_edges: int, **kwargs):
        super().__init__(n_edges, **kwargs)
        # t_critical per OD pair (starts at a large default)
        self.t_critical: Dict[Tuple[int, int], float] = defaultdict(lambda: 1e6)

    def apply_action(self, od_key: Tuple[int, int], action: int):
        direction, value = self._decode_action(action)
        self.t_critical[od_key] = max(
            0.0, self.t_critical[od_key] + direction * value
        )

    def compute_alpha(self, od_key: Tuple[int, int], mean_tt: float) -> float:
        delta = abs(self.t_critical[od_key] - mean_tt)
        return _adaptive_alpha(delta)


# ═══════════════════════════════════════════════════════════════════
#  SizeAdvisor
# ═══════════════════════════════════════════════════════════════════

class SizeAdvisor(_AdvisorBase):
    """Adjusts |Π|_critical — the number of paths kept in the action set.

    The upper bound for each OD pair is the total number of enumerated paths,
    **not** a fixed ``top_k`` CLI parameter.  The Size Advisor learns to
    shrink or expand |Π|_critical within [1, n_paths_for_od].

    α_i = clamp((1+exp(-log(Δξ+1))) / (1-exp(-log(Δξ+1))))
    where Δξ = ||Π|_critical - (|Π| - Σζ_p)|.
    """

    def __init__(
        self,
        n_edges: int,
        n_paths_per_od: Dict[Tuple[int, int], int],
        **kwargs,
    ):
        super().__init__(n_edges, **kwargs)
        self._n_paths_per_od = n_paths_per_od
        # |Π|_critical per OD (starts at the full path count — use all)
        self.pi_critical: Dict[Tuple[int, int], int] = {
            od: n for od, n in n_paths_per_od.items()
        }

    def apply_action(self, od_key: Tuple[int, int], action: int):
        direction, value = self._decode_action(action)
        n_max = self._n_paths_per_od.get(od_key, 1)
        new_val = self.pi_critical.get(od_key, n_max) + int(
            direction * max(1, round(value))
        )
        self.pi_critical[od_key] = max(1, min(n_max, new_val))

    def compute_alpha(
        self, od_key: Tuple[int, int], n_valid: int, n_total: int
    ) -> float:
        effective = n_valid
        delta = abs(self.pi_critical.get(od_key, n_total) - effective)
        log_delta = np.log(delta + 1.0)
        return _adaptive_alpha(log_delta)


# ═══════════════════════════════════════════════════════════════════
#  ConstrainedMainAgent
# ═══════════════════════════════════════════════════════════════════

class ConstrainedMainAgent:
    """Path-based main agent with softmax policy over restricted action set.

    Q-table is keyed by (destination,).  Each entry has ``max_paths``
    Q-values, where ``max_paths`` is the largest path count across all
    OD pairs (so the array is fixed-size).

    Action selection uses softmax (Boltzmann) over Q-values of the
    restricted path set A†.  **No ε-greedy** — exploration comes from
    the stochastic nature of the softmax distribution itself.
    """

    def __init__(
        self,
        max_paths: int,
        gamma: float = 0.99,
        temperature: float = 1.0,
    ):
        self.max_paths = max_paths
        self.gamma = gamma
        self.temperature = temperature

        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(max_paths, dtype=np.float64)
        )

        # Per-agent episode state
        self._prev_state: List[Optional[Tuple]] = []
        self._prev_action: np.ndarray = np.array([], dtype=np.int64)

    def init_agents(self, n_agents: int):
        self._prev_state = [None] * n_agents
        self._prev_action = np.full(n_agents, -1, dtype=np.int64)
        self._has_prev = np.zeros(n_agents, dtype=bool)

    def reset_episode(self):
        n = len(self._prev_state)
        self._prev_state = [None] * n
        self._prev_action[:] = -1
        self._has_prev[:] = False

    def _state_key(self, obs: np.ndarray) -> Tuple:
        return (int(obs[1]),)  # (destination,)

    def select_actions_softmax(
        self,
        obs_batch: np.ndarray,
        masks: np.ndarray,
        agent_ids: np.ndarray,
        restricted_masks: np.ndarray,
    ) -> np.ndarray:
        """Select actions via softmax over Q-values on restricted paths.

        Args:
            obs_batch: [K, obs_dim]
            masks: [K, max_paths] env-level path masks (1=exists)
            agent_ids: [K] agent indices
            restricted_masks: [K, max_paths] HAB-restricted masks (1=allowed)

        Returns:
            actions: [K] chosen path indices.
        """
        K = obs_batch.shape[0]
        actions = np.zeros(K, dtype=np.int64)

        for i in range(K):
            aid = int(agent_ids[i])
            combined_mask = masks[i] * restricted_masks[i]
            valid = np.where(combined_mask > 0)[0]

            state = self._state_key(obs_batch[i])

            if len(valid) == 0:
                # Fallback: use env mask only (advisors too strict)
                valid = np.where(masks[i] > 0)[0]
                if len(valid) == 0:
                    continue

            q = self.q_table[state][valid]
            # Softmax (Boltzmann distribution)
            q_shifted = q - q.max()
            exp_q = np.exp(q_shifted / max(self.temperature, 1e-6))
            probs = exp_q / (exp_q.sum() + 1e-12)

            choice_local = np.random.choice(len(valid), p=probs)
            actions[i] = valid[choice_local]

            self._prev_state[aid] = state
            self._prev_action[aid] = actions[i]
            self._has_prev[aid] = True

        return actions


# ═══════════════════════════════════════════════════════════════════
#  HABAgent  (Orchestrator)
# ═══════════════════════════════════════════════════════════════════

class HABAgent:
    """Hierarchical Advisory Behavior agent orchestrator.

    Coordinates TimeAdvisor, SizeAdvisor, and ConstrainedMainAgent.
    Exposes ``get_actions_batched`` for the training loop and
    ``update_hab`` for post-episode learning.

    The number of available paths per OD pair is determined entirely
    by the environment's ``paths_per_od``.  The Size Advisor dynamically
    adjusts how many of those paths are kept (|Π|_critical ∈ [1, n_paths]).
    There is **no** external ``top_k`` parameter.

    Args:
        n_agents: total agent count.
        n_edges: number of edges in the network.
        od_pairs: [N, 2] numpy array.
        paths_per_od: mapping (o,d) → list of edge-index paths.
        edge_endpoints: [E, 2] numpy array.
        ff_times: [E] numpy array of free-flow travel times (in seconds).
        dt: simulation timestep (seconds).
        gamma: discount factor for all sub-agents.
        slack_weight: ϖ for the main agent reward.
        margin: relative margin for inefficient-path detection (Eq. 25).
        epsilon_start/end/decay: exploration schedule for advisors only.
        seed: RNG seed.
    """

    def __init__(
        self,
        n_agents: int,
        n_edges: int,
        od_pairs: np.ndarray,
        paths_per_od: Dict[Tuple[int, int], List[List[int]]],
        edge_endpoints: np.ndarray = None,
        ff_times: np.ndarray = None,
        dt: float = 1.0,
        gamma: float = 0.99,
        slack_weight: float = 0.1,
        margin: float = 0.2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.n_edges = n_edges
        self.gamma = gamma
        self.slack_weight = slack_weight
        self.margin = margin
        self.formulation = "path-based"
        self.rng = np.random.default_rng(seed)

        self._od_pairs = od_pairs.copy()
        self._paths_per_od = paths_per_od
        self._agent_od_keys = [
            (int(od_pairs[i, 0]), int(od_pairs[i, 1]))
            for i in range(n_agents)
        ]

        # Derive per-OD path counts and global max
        self._n_paths_per_od: Dict[Tuple[int, int], int] = {
            od: len(paths) for od, paths in paths_per_od.items()
        }
        self._max_paths = max(self._n_paths_per_od.values()) if self._n_paths_per_od else 1

        # Pre-compute free-flow path costs as fallback for episode 1
        self._ff_path_costs: Dict[Tuple[int, int], Dict[int, float]] = {}
        if ff_times is not None:
            for od_key, paths in paths_per_od.items():
                costs = {}
                for p_idx, path_edges in enumerate(paths):
                    costs[p_idx] = float(
                        sum(ff_times[e] for e in path_edges)
                    ) if path_edges else 0.0
                self._ff_path_costs[od_key] = costs

        # Sub-agents
        adv_kw = dict(
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        self.time_advisor = TimeAdvisor(n_edges, **adv_kw)
        self.size_advisor = SizeAdvisor(
            n_edges, n_paths_per_od=self._n_paths_per_od, **adv_kw,
        )
        self.main_agent = ConstrainedMainAgent(self._max_paths, gamma=gamma)
        self.main_agent.init_agents(n_agents)

        # CostTracker (lazy import to avoid circular deps)
        from tamarl.envs.components.cost_tracker import CostTracker
        self.cost_tracker = CostTracker(
            od_pairs=od_pairs,
            paths_per_od=paths_per_od,
            edge_endpoints=edge_endpoints,
            dt=dt,
        )

        # Per-episode bookkeeping
        self._chosen_path_idx = np.full(n_agents, 0, dtype=np.int64)
        self._prev_density_state: Optional[Tuple] = None
        self._prev_time_actions: Dict[Tuple[int, int], int] = {}
        self._prev_size_actions: Dict[Tuple[int, int], int] = {}
        self._prev_path_costs: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None

        # Exploration tracking (advisors only — main agent uses softmax)
        self.epsilon = epsilon_start
        self.total_updates = 0

    # ── Reset ─────────────────────────────────────────────────────

    def reset_episode(self):
        self.main_agent.reset_episode()
        self._chosen_path_idx[:] = 0
        self._prev_density_state = None
        self._prev_time_actions.clear()
        self._prev_size_actions.clear()

    def prepare_for_eval(self, env):
        """Prepare for deterministic evaluation by setting advisors to greedy choices."""
        if self._prev_density_state is None:
            return

        # Re-select advisor actions greedily based on the final density of the last episode
        for od_key in self._n_paths_per_od.keys():
            # Time Advisor
            if od_key in self._prev_time_actions:
                ta_prev = self._prev_time_actions[od_key]
                dir_prev, val_prev = self.time_advisor._decode_action(ta_prev)
                self.time_advisor.t_critical[od_key] = max(0.0, self.time_advisor.t_critical[od_key] - dir_prev * val_prev)
            
            ta_greedy = self.time_advisor.select_action(self._prev_density_state, self.rng, deterministic=True)
            self.time_advisor.apply_action(od_key, ta_greedy)

            # Size Advisor
            # (Approximation: we don't strictly revert pi_critical because we didn't save the pre-action state,
            # but we just apply the greedy action from the current state to ensure pure exploitation)
            sa_greedy = self.size_advisor.select_action(self._prev_density_state, self.rng, deterministic=True)
            self.size_advisor.apply_action(od_key, sa_greedy)

    # ── Estimated path costs ──────────────────────────────────────

    def _get_estimated_costs(
        self, od_key: Tuple[int, int],
    ) -> Dict[int, float]:
        """Return estimated cost per path index for an OD pair.

        Uses previous episode's dynamic costs if available,
        otherwise falls back to free-flow travel times.
        """
        if self._prev_path_costs and od_key in self._prev_path_costs:
            return self._prev_path_costs[od_key]
        return self._ff_path_costs.get(od_key, {})

    # ── Batched action selection ──────────────────────────────────

    def get_actions_batched(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        deciding_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select path actions for all deciding agents.

        For each agent the restricted action set A†_i is built as follows:

        1. **Time Constraint**: Remove all paths whose estimated cost
           exceeds ``t_critical`` for the agent's OD pair.
        2. **Size Constraint**: Among the remaining paths, sort by
           estimated cost (ascending) and keep only the best
           ``pi_critical`` paths.
        3. **Softmax selection**: The Main Agent samples from a Boltzmann
           distribution over Q-values of the paths in A†_i.
        """
        K = deciding_indices.numel()
        if K == 0:
            return torch.empty(0, device=deciding_indices.device, dtype=torch.long)

        device = deciding_indices.device
        obs_np = obs.cpu().numpy()
        masks_np = masks.cpu().numpy()
        agent_ids = deciding_indices.cpu().numpy()
        mask_width = masks_np.shape[1]  # env mask dimension (may differ from _max_paths)

        # Build restricted masks from advisor constraints
        restricted = np.zeros((K, mask_width), dtype=np.int8)

        for i in range(K):
            aid = int(agent_ids[i])
            od_key = self._agent_od_keys[aid]
            n_paths = self._n_paths_per_od.get(od_key, 0)
            if n_paths == 0:
                continue

            estimated_costs = self._get_estimated_costs(od_key)

            # ── 1. Time Constraint: keep paths with cost ≤ t_critical ──
            t_crit = self.time_advisor.t_critical[od_key]
            time_valid = []
            for p_idx in range(n_paths):
                cost = estimated_costs.get(p_idx, 0.0)
                if cost <= t_crit:
                    time_valid.append((p_idx, cost))

            # If time constraint eliminates everything, keep all paths
            # (advisors will learn this was too strict)
            if not time_valid:
                time_valid = [
                    (p_idx, estimated_costs.get(p_idx, 0.0))
                    for p_idx in range(n_paths)
                ]

            # ── 2. Size Constraint: sort by cost, keep best pi_critical ──
            pi_crit = self.size_advisor.pi_critical.get(od_key, n_paths)
            time_valid.sort(key=lambda x: x[1])  # ascending cost
            kept = time_valid[:pi_crit]

            for p_idx, _ in kept:
                restricted[i, p_idx] = 1

        # Main agent selects via softmax (no ε-greedy)
        actions_np = self.main_agent.select_actions_softmax(
            obs_np, masks_np, agent_ids, restricted,
        )

        # Record chosen paths
        for i in range(K):
            self._chosen_path_idx[int(agent_ids[i])] = actions_np[i]

        return torch.from_numpy(actions_np).to(device)

    # ── Path recording for env ────────────────────────────────────

    def set_chosen_path(self, agent_idx: int, action: int):
        self._chosen_path_idx[agent_idx] = action

    # ── Post-episode update ───────────────────────────────────────

    def update_hab(self, env) -> Dict[str, float]:
        """Run the full HAB post-episode update cycle.

        1. Compute OD metrics and cost improvements.
        2. Update advisors (reward + adaptive α + Q-update).
        3. Update main agent Q-values.

        Args:
            env: the DTAMarkovGameEnv after the episode.

        Returns:
            Dict of diagnostic stats for logging.
        """
        dnl = env.dnl

        # 1. Metrics
        od_metrics = self.cost_tracker.get_od_metrics(dnl)
        improvements = self.cost_tracker.compute_cost_improvement(od_metrics)
        path_validity = self.cost_tracker.detect_inefficient_paths(
            dnl, margin=self.margin
        )
        current_path_costs = self.cost_tracker.get_current_path_costs(dnl)

        # Current density state
        density = env.get_network_density().cpu().numpy()
        density_state = self.time_advisor._discretise_density(density)

        # 2. Update advisors per OD
        advisor_rewards = []
        for od_key, metrics in od_metrics.items():
            delta_improve = improvements.get(od_key, 0.0)
            eta_sign = 1.0 if delta_improve >= 0 else -1.0
            adv_reward = _hab_reward(delta_improve, eta_sign)
            advisor_rewards.append(adv_reward)

            n_total = self._n_paths_per_od.get(od_key, 1)

            # ── Time Advisor ──
            if od_key in self._prev_time_actions and self._prev_density_state is not None:
                alpha_t = self.time_advisor.compute_alpha(
                    od_key, metrics.mean_travel_time
                )
                self.time_advisor.update(
                    self._prev_density_state,
                    self._prev_time_actions[od_key],
                    adv_reward,
                    density_state,
                    alpha=alpha_t,
                )

            # Select new action
            ta = self.time_advisor.select_action(density_state, self.rng)
            self.time_advisor.apply_action(od_key, ta)
            self._prev_time_actions[od_key] = ta

            # ── Size Advisor ──
            n_valid, _ = path_validity.get(od_key, (n_total, n_total))
            if od_key in self._prev_size_actions and self._prev_density_state is not None:
                alpha_s = self.size_advisor.compute_alpha(
                    od_key, n_valid, n_total
                )
                self.size_advisor.update(
                    self._prev_density_state,
                    self._prev_size_actions[od_key],
                    adv_reward,
                    density_state,
                    alpha=alpha_s,
                )

            sa = self.size_advisor.select_action(density_state, self.rng)
            self.size_advisor.apply_action(od_key, sa)
            self._prev_size_actions[od_key] = sa

        # 3. Update main agent Q-values
        delta_base, delta_slack = self.cost_tracker.calculate_slack_reward(
            dnl, self._chosen_path_idx,
            prev_path_costs=self._prev_path_costs,
            slack_weight=self.slack_weight,
        )
        main_updates = 0
        for i in range(self.n_agents):
            if not self.main_agent._has_prev[i]:
                continue
            od_key = self._agent_od_keys[i]
            mu_i = od_metrics[od_key].cost_variance if od_key in od_metrics else 0.0
            alpha_m = _adaptive_alpha(mu_i)

            delta_c = delta_base[i] + delta_slack[i]
            eta = 1.0 if delta_c >= 0 else -1.0
            reward = _hab_reward(delta_c, eta)

            state = self.main_agent._prev_state[i]
            action = int(self.main_agent._prev_action[i])
            if state is not None:
                self.main_agent.q_table[state][action] += alpha_m * (
                    reward - self.main_agent.q_table[state][action]
                )
                main_updates += 1

        self.total_updates += main_updates

        # 4. Store for next episode
        self._prev_density_state = density_state
        self.cost_tracker.store_episode_costs(od_metrics)
        self._prev_path_costs = current_path_costs

        # Stats
        avg_adv_r = float(np.mean(advisor_rewards)) if advisor_rewards else 0.0

        # Debug print: critical values (requested by user)
        t_vals = list(self.time_advisor.t_critical.values())
        pi_vals = list(self.size_advisor.pi_critical.values())
        avg_t = t_vals[-1] if t_vals else 0.0
        avg_pi = pi_vals[-1] if pi_vals else 0.0
        print(f"DEBUG: t_critical={avg_t:.2f}, |Π|_critical={avg_pi:.2f}")

        return {
            'hab_advisor_reward': avg_adv_r,
            'hab_main_updates': main_updates,
            'hab_q_size': len(self.main_agent.q_table),
        }

    # ── Epsilon decay ─────────────────────────────────────────────

    def decay_epsilon(self):
        self.time_advisor.decay_epsilon()
        self.size_advisor.decay_epsilon()
        self.epsilon = self.time_advisor.epsilon

    @property
    def q_table_size(self) -> int:
        return len(self.main_agent.q_table)

    def __repr__(self) -> str:
        return (
            f"HABAgent(max_paths={self._max_paths}, γ={self.gamma}, "
            f"ε_adv={self.epsilon:.4f}, "
            f"q_states={self.q_table_size}, updates={self.total_updates})"
        )
