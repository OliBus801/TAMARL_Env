"""Profiler for the TAMARL training loop.

Instruments the key sections of run_episode to identify bottlenecks.
"""
import argparse
import time
import numpy as np
import torch
from collections import defaultdict

from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv
from tamarl.envs.scenario_loader import load_scenario
from tamarl.rl_models.sb3_agent import SB3Agent
from tamarl.rl_models.random_agent import RandomAgent


class Timer:
    """Simple cumulative timer for profiling."""
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._stack = []

    def start(self, name):
        self._stack.append((name, time.perf_counter()))

    def stop(self):
        name, t0 = self._stack.pop()
        elapsed = time.perf_counter() - t0
        self.times[name] += elapsed
        self.counts[name] += 1
        return elapsed

    def report(self, total_time=None):
        if total_time is None:
            total_time = sum(self.times.values())
        print(f"\n{'='*70}")
        print(f"  PROFILING RESULTS  (total wall time: {total_time:.3f}s)")
        print(f"{'='*70}")
        # Sort by time descending
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = t / total_time * 100 if total_time > 0 else 0
            cnt = self.counts[name]
            avg = t / cnt if cnt > 0 else 0
            print(f"  {name:<40s}  {t:8.4f}s  ({pct:5.1f}%)  "
                  f"[{cnt:6d} calls, {avg*1000:8.3f}ms/call]")
        print(f"{'='*70}")


def run_profiled_episode(env, agent, timer):
    """Run a single episode with detailed profiling."""
    timer.start("total_episode")

    timer.start("env.reset")
    obs, infos = env.reset()
    timer.stop()  # env.reset

    cumulative_rewards = {}
    n_decisions = 0
    is_learner = hasattr(agent, 'update')
    n_macro_steps = 0

    while env.agents:
        n_macro_steps += 1

        # --- get_actions ---
        timer.start("agent.get_actions")
        actions = agent.get_actions(obs, infos)
        timer.stop()
        n_decisions += len(actions)

        # --- env.step ---
        timer.start("env.step_total")
        obs, rewards, terminations, truncations, infos = env.step(actions)
        timer.stop()

        # --- reward accumulation ---
        timer.start("reward_accumulation")
        for agent_id, r in rewards.items():
            cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + r
        timer.stop()

        # --- agent.update ---
        if is_learner:
            timer.start("agent.update")
            agent.update(obs, rewards, terminations, truncations, infos)
            timer.stop()

    timer.stop()  # total_episode
    return n_macro_steps, n_decisions


def run_profiled_env_step(env, agent, timer):
    """Run a single episode with profiling INSIDE env.step and get_actions."""
    timer.start("total_episode")

    # Reset
    timer.start("env.reset")
    obs, infos = env.reset()
    timer.stop()

    cumulative_rewards = {}
    is_learner = hasattr(agent, 'update')
    n_macro_steps = 0
    n_decisions = 0
    n_deciding_agents_total = 0

    # Detailed profiling of SB3 get_actions
    is_sb3 = isinstance(agent, SB3Agent)

    while env.agents:
        n_macro_steps += 1
        n_active = len(env.agents)

        # ================================================================
        # Profile get_actions in detail (for SB3)
        # ================================================================
        if is_sb3:
            timer.start("get_actions.total")
            actions = {}
            n_deciding_this_step = 0

            for agent_id, info in infos.items():
                mask = info.get("action_mask")
                if mask is None or mask.sum() == 0:
                    continue
                obs_i = obs.get(agent_id)
                if obs_i is None:
                    continue

                n_deciding_this_step += 1

                timer.start("get_actions.predict_single")
                action = agent._predict_with_mask(obs_i, mask)
                timer.stop()

                actions[agent_id] = action

                # Bookkeeping
                timer.start("get_actions.bookkeeping")
                agent._prev_obs[agent_id] = obs_i
                agent._prev_action[agent_id] = action
                if agent_id not in agent._accumulated_reward:
                    agent._accumulated_reward[agent_id] = 0.0
                timer.stop()

            n_deciding_agents_total += n_deciding_this_step
            timer.stop()  # get_actions.total
        else:
            timer.start("agent.get_actions")
            actions = agent.get_actions(obs, infos)
            timer.stop()

        n_decisions += len(actions)

        # ================================================================
        # Profile env.step internals
        # ================================================================
        timer.start("env.step.apply_actions")
        if actions:
            env._action_mgr.apply_actions(actions)
        timer.stop()

        timer.start("env.step.advance_dnl")
        n_ticks = env._advance_to_next_decisions()
        timer.stop()

        timer.start("env.step.compute_rewards")
        current_agents = list(env.agents) + [a for a in set(cumulative_rewards) - set(env.agents)]
        # Actually use the env step logic
        rewards_set = set(list(obs.keys()) + list(actions.keys()))
        rewards = env._rewarder.compute_step_rewards(rewards_set, n_ticks)
        timer.stop()

        timer.start("env.step.terminations")
        terminations = {}
        truncations = {}
        current_agents_list = list(env.agents)
        if current_agents_list:
            active_indices = [int(a.split("_")[-1]) for a in current_agents_list]
            active_tensor = torch.tensor(active_indices, device=env.dnl.device, dtype=torch.long)
            statuses = env.dnl.status[active_tensor].cpu().numpy()
            for i, agent_id in enumerate(current_agents_list):
                terminations[agent_id] = bool(statuses[i] == 3)
                truncations[agent_id] = False
        if env.dnl.current_step >= env._max_steps:
            for agent_id in current_agents_list:
                if not terminations.get(agent_id, False):
                    truncations[agent_id] = True
        timer.stop()

        # Remove finished agents
        timer.start("env.step.remove_agents")
        env.agents = [
            a for a in current_agents_list
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]
        timer.stop()

        # Build observations
        timer.start("env.step.build_observations")
        deciding = env._scheduler.get_deciding_agents()
        observations_new = env._obs_builder.build_observations(deciding)
        timer.stop()

        timer.start("env.step.non_deciding_obs")
        if env.agents:
            agent_indices = np.array([int(a.split("_")[-1]) for a in env.agents])
            idx_tensor = torch.from_numpy(agent_indices).to(env.dnl.device)
            statuses_remaining = env.dnl.status[idx_tensor].cpu().numpy()
            needs_obs_mask = np.array([a not in observations_new for a in env.agents])
            valid_mask = needs_obs_mask & ((statuses_remaining == 1) | (statuses_remaining == 2))
            if valid_mask.any():
                valid_idx_tensor = torch.from_numpy(agent_indices[valid_mask]).to(env.dnl.device)
                curr_edges = env.dnl.current_edge[valid_idx_tensor]
                nodes = env.dnl.edge_endpoints[curr_edges, 1].cpu().numpy()
                c_legs = env.dnl.current_leg[valid_idx_tensor]
                dests = env.dnl.destinations[valid_idx_tensor, c_legs].cpu().numpy()
                norm_time = env.dnl.current_step / env._max_steps
                valid_agent_ids = [env.agents[i] for i in np.where(valid_mask)[0]]
                for i, agent_id in enumerate(valid_agent_ids):
                    obs_arr = np.zeros(env._obs_builder.obs_size, dtype=np.float32)
                    obs_arr[0] = float(nodes[i])
                    obs_arr[1] = float(dests[i])
                    obs_arr[2] = norm_time
                    observations_new[agent_id] = obs_arr
        timer.stop()

        timer.start("env.step.build_infos")
        infos_new = {}
        action_masks = env._action_mgr.get_action_masks(deciding)
        for agent_id in list(set(list(terminations.keys()) + env.agents)):
            info = {}
            if agent_id in action_masks:
                info["action_mask"] = action_masks[agent_id]
            else:
                info["action_mask"] = np.zeros(env.dnl.max_out_degree, dtype=np.int8)
            idx = int(agent_id.split("_")[-1])
            info["curr_leg"] = int(env.dnl.current_leg[idx].item())
            infos_new[agent_id] = info
        timer.stop()

        obs = observations_new
        infos = infos_new

        # reward accumulation
        for agent_id, r in rewards.items():
            cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + r

        # agent.update
        if is_learner:
            timer.start("agent.update")
            agent.update(obs, rewards, terminations, truncations, infos)
            timer.stop()

    timer.stop()  # total_episode
    return n_macro_steps, n_decisions, n_deciding_agents_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile training loop")
    parser.add_argument("--scenario", default="tamarl/data/scenarios/ortuzar_willumsen/")
    parser.add_argument("--population", default="200")
    parser.add_argument("--agent", default="ppo", choices=["random", "ppo", "dqn", "a2c"])
    parser.add_argument("--max_steps", type=int, default=7200)
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed profiling inside env.step and get_actions")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Training Loop Profiler")
    print(f"{'='*70}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Population: {args.population}")
    print(f"  Agent: {args.agent}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*70}\n")

    # Create env
    env = DTAMarkovGameEnv(
        scenario_path=args.scenario,
        population_filter=args.population,
        timestep=1.0,
        max_steps=args.max_steps,
        device="cpu",
    )

    print(f"  Env: {env.dnl.num_agents} agents, {env.dnl.num_edges} edges, "
          f"{env.dnl.num_nodes} nodes, max_out_degree={env.dnl.max_out_degree}")

    # Create agent
    if args.agent in ("ppo", "dqn", "a2c"):
        agent = SB3Agent(
            algorithm=args.agent,
            env=env,
            device="cpu",
        )
    else:
        agent = RandomAgent()

    print(f"  Agent: {agent}\n")

    # Run profiled episode
    timer = Timer()

    if args.detailed:
        print("  Running DETAILED profiling (env.step + get_actions internals)...\n")
        n_macro, n_dec, n_deciding = run_profiled_env_step(env, agent, timer)
        print(f"\n  Macro-steps: {n_macro}")
        print(f"  Total decisions: {n_dec}")
        print(f"  Total deciding agents across all steps: {n_deciding}")
        print(f"  Avg deciding agents per macro-step: {n_deciding/n_macro:.1f}")
    else:
        print("  Running HIGH-LEVEL profiling...\n")
        n_macro, n_dec = run_profiled_episode(env, agent, timer)
        print(f"\n  Macro-steps: {n_macro}")
        print(f"  Total decisions: {n_dec}")

    total = timer.times.get("total_episode", sum(timer.times.values()))
    timer.report(total)

    env.close()
