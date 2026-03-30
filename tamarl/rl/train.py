"""Basic RL training runner for the DTA Markov Game environment.

Runs episodes with a given policy and logs per-episode statistics.
"""

import argparse
import time
import numpy as np
from typing import Optional

from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv
from tamarl.envs.components.metrics import compute_tstt, compute_mean_travel_time, compute_arrival_rate
from tamarl.rl_models.random_agent import RandomAgent


def run_episode(env: DTAMarkovGameEnv, agent: RandomAgent, verbose: bool = False):
    """Run a single episode and return stats.
    
    Returns:
        dict with keys: tstt, mean_travel_time, arrival_rate, total_reward, 
                        episode_length, n_decisions
    """
    obs, infos = env.reset()
    
    total_rewards = {}
    n_decisions = 0
    
    while env.agents:
        actions = agent.get_actions(obs, infos)
        n_decisions += len(actions)
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent_id, r in rewards.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r
        
        if verbose and env.dnl.current_step % 100 == 0:
            en_route = ((env.dnl.status == 1) | (env.dnl.status == 2)).sum().item()
            arrived = (env.dnl.status == 3).sum().item()
            print(f"  tick {env.dnl.current_step} | en_route={en_route} | arrived={arrived} | deciding={len(actions)}")
    
    # Compute final metrics
    tstt = compute_tstt(env.dnl)
    mean_tt = compute_mean_travel_time(env.dnl)
    arr_rate = compute_arrival_rate(env.dnl)
    mean_reward = np.mean(list(total_rewards.values())) if total_rewards else 0.0
    
    return {
        'tstt': tstt,
        'mean_travel_time': mean_tt,
        'arrival_rate': arr_rate,
        'mean_reward': mean_reward,
        'episode_length': env.dnl.current_step,
        'n_decisions': n_decisions,
    }


def train(
    scenario_path: str,
    population_filter: Optional[str] = None,
    n_episodes: int = 5,
    max_steps: int = 3600,
    timestep: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """Run the training loop with a random agent.
    
    Args:
        scenario_path: path to scenario folder
        population_filter: substring to match population file
        n_episodes: number of episodes to run
        max_steps: maximum ticks per episode
        timestep: simulation timestep
        device: 'cpu' or 'cuda'
        seed: random seed
        verbose: print per-tick progress
    """
    print(f"{'='*60}")
    print(f"DTA Markov Game - Training Runner")
    print(f"{'='*60}")
    print(f"Scenario: {scenario_path}")
    print(f"Population filter: {population_filter}")
    print(f"Episodes: {n_episodes} | Max steps: {max_steps}")
    print(f"Device: {device} | Seed: {seed}")
    print(f"{'='*60}\n")
    
    env = DTAMarkovGameEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=timestep,
        max_steps=max_steps,
        device=device,
        seed=seed,
    )
    
    agent = RandomAgent(seed=seed)
    
    print(f"Environment created: {env.dnl.num_agents} agents, "
          f"{env.dnl.num_edges} edges, {env.dnl.num_nodes} nodes, "
          f"max_out_degree={env.dnl.max_out_degree}")
    print()
    
    all_stats = []
    for ep in range(n_episodes):
        t0 = time.time()
        stats = run_episode(env, agent, verbose=verbose)
        elapsed = time.time() - t0
        
        all_stats.append(stats)
        
        print(f"Episode {ep+1}/{n_episodes} | "
              f"TSTT={stats['tstt']:.1f}s | "
              f"Mean TT={stats['mean_travel_time']:.1f}s | "
              f"Arrival={stats['arrival_rate']*100:.1f}% | "
              f"Mean Reward={stats['mean_reward']:.1f} | "
              f"Ticks={stats['episode_length']} | "
              f"Decisions={stats['n_decisions']} | "
              f"Wall={elapsed:.2f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary over {n_episodes} episodes:")
    tstt_vals = [s['tstt'] for s in all_stats]
    tt_vals = [s['mean_travel_time'] for s in all_stats]
    arr_vals = [s['arrival_rate'] for s in all_stats]
    print(f"  TSTT:      {np.mean(tstt_vals):.1f} ± {np.std(tstt_vals):.1f}")
    print(f"  Mean TT:   {np.mean(tt_vals):.1f} ± {np.std(tt_vals):.1f}")
    print(f"  Arrival:   {np.mean(arr_vals)*100:.1f}% ± {np.std(arr_vals)*100:.1f}%")
    print(f"{'='*60}")
    
    env.close()
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTA Markov Game Training Runner")
    parser.add_argument("--scenario", required=True, help="Path to scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter (e.g. '100')")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=3600, help="Max ticks per episode")
    parser.add_argument("--timestep", type=float, default=1.0, help="Simulation timestep (s)")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-tick progress")
    
    args = parser.parse_args()
    
    train(
        scenario_path=args.scenario,
        population_filter=args.population,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        timestep=args.timestep,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )
