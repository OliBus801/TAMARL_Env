import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.pomdp_wrapper import POMDPWrapper
from tamarl.rl.wandb_logger import WandbLogger


# --- Agents (IPPO, MAPPO) ---
class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class PPOBase:
    def __init__(self, actor_dim, critic_dim, action_dim, device="cpu", lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, epochs=4, batch_size=256):
        self.actor = MLPNetwork(actor_dim, 64, action_dim).to(device)
        self.critic = MLPNetwork(critic_dim, 64, 1).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def get_action_and_value(self, actor_obs, critic_obs, action=None):
        logits = self.actor(actor_obs)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(critic_obs)

    def update(self, rollouts, final_rewards):
        if len(rollouts) == 0:
            return {}
            
        b_actor_obs = torch.stack([r['actor_obs'] for r in rollouts])
        b_critic_obs = torch.stack([r['critic_obs'] for r in rollouts])
        b_actions = torch.stack([r['action'] for r in rollouts])
        b_logprobs = torch.stack([r['log_prob'] for r in rollouts])
        b_values = torch.stack([r['value'] for r in rollouts]).squeeze(-1)
        
        # Calculate advantages
        b_rewards = torch.tensor([final_rewards[r['agent']] for r in rollouts], dtype=torch.float32, device=self.device)
        
        b_returns = b_rewards
        b_advantages = b_returns - b_values
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        dataset_size = len(b_actor_obs)
        b_inds = np.arange(dataset_size)
        
        for epoch in range(self.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    b_actor_obs[mb_inds], b_critic_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()
                
        return {
            "loss/policy": pg_loss.item(),
            "loss/value": v_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/total": loss.item()
        }

class IPPOAgent(PPOBase):
    def __init__(self, obs_dim, action_dim, device="cpu"):
        super().__init__(obs_dim, obs_dim, action_dim, device=device)

class MAPPOAgent(PPOBase):
    def __init__(self, obs_dim, state_dim, action_dim, device="cpu"):
        super().__init__(obs_dim, state_dim, action_dim, device=device)


# --- Training loop ---

def _compute_stats(env: POMDPWrapper, rewards: np.ndarray, wall_time: float, n_decisions: int) -> dict:
    tt_matrix = env.dnl.leg_metrics[:, :, 1]
    tt_obs = tt_matrix[env.aw._leg_agent_idx, env.aw._leg_leg_idx].cpu().numpy()
    tt = tt_obs
    tstt = float(tt.sum())
    mean_tt = float(tt.mean()) if tt.size > 0 else 0.0
    arrival_rate = float((env.dnl.status >= 3).float().mean().cpu())
    
    stats = {
        "tstt": tstt,
        "mean_travel_time": mean_tt,
        "arrival_rate": arrival_rate,
        "mean_reward": float(rewards.mean()) if rewards.size > 0 else 0.0,
        "episode_length": int(env.dnl.current_step),
        "n_decisions": n_decisions,
        "tt_std": float(np.std(tt)) if tt.size > 0 else 0.0,
        "tt_min": float(np.min(tt)) if tt.size > 0 else 0.0,
        "tt_max": float(np.max(tt)) if tt.size > 0 else 0.0,
        "tt_median": float(np.median(tt)) if tt.size > 0 else 0.0,
        "tt_p90": float(np.percentile(tt, 90)) if tt.size > 0 else 0.0,
        "tt_p95": float(np.percentile(tt, 95)) if tt.size > 0 else 0.0,
        "reward_std": float(np.std(rewards)) if rewards.size > 0 else 0.0,
        "wall_time": wall_time,
    }
    return stats

def _aggregate_stats(window: list[dict]) -> dict[str, float]:
    keys = [
        "tstt", "mean_travel_time", "arrival_rate", "mean_reward", "episode_length", "n_decisions", "tt_p95", "wall_time"
    ]
    agg = {}
    for k in keys:
        vals = [s[k] for s in window if k in s]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg


def train(
    scenario_path: str,
    population_filter: str | None = None,
    n_episodes: int = 5,
    max_steps: int = 86400,
    stuck_threshold: int = 10,
    timestep: float = 1.0,
    scale_factor: float = 1.0,
    device: str = "cpu",
    seed: int | None = None,
    # Agent
    agent_type: str = "ippo",
    top_k_paths: int = 3,
    # Logging
    log_interval: int = 100,
    wandb_enabled: bool = False,
    wandb_project: str = "tamarl",
    wandb_tags: list | None = None,
    wandb_agent: str | None = None,
    **kwargs
):
    # Enable CUDA if available and requested
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    scenario_name = os.path.basename(scenario_path.rstrip("/"))
    scenario_id = f"{scenario_name}-{population_filter}" if population_filter else scenario_name

    config = {
        "scenario_path": scenario_path,
        "population_filter": population_filter,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "stuck_threshold": stuck_threshold,
        "timestep": timestep,
        "scale_factor": scale_factor,
        "device": device,
        "seed": seed,
        "log_interval": log_interval,
        "agent_type": agent_type,
        "top_k_paths": top_k_paths,
        "wandb_agent": wandb_agent,
    }

    print(f"{'=' * 65}")
    print("  POMDP Deep MARL — Training Runner")
    print(f"{'=' * 65}")
    print(f"  Scenario:      {scenario_path}")
    print(f"  Population:    {population_filter}")
    print(f"  Agent:         {agent_type}")
    print(f"  Episodes:      {n_episodes} | Max steps: {max_steps} | Stuck threshold: {stuck_threshold}")
    print(f"  Log interval:  every {log_interval} episodes")
    print(f"  Device:        {device} | Seed: {seed}")
    print(f"  Top-k Paths:   {top_k_paths} | Scale Factor: {scale_factor}")
    print(f"{'=' * 65}\n")

    logger = WandbLogger(
        project=wandb_project,
        run_name=f"{agent_type}-marl-seed{seed}",
        scenario_id=scenario_id,
        agent_type=wandb_agent if wandb_agent is not None else agent_type,
        config=config,
        enabled=wandb_enabled,
        tags=wandb_tags,
    )

    # Base Env
    bandit_env = DTABanditEnv(
        scenario_path=scenario_path,
        population_filter=population_filter,
        timestep=timestep,
        scale_factor=scale_factor,
        max_steps=max_steps,
        stuck_threshold=stuck_threshold,
        device=device,
        seed=seed,
    )
    
    agent_wrapper = AgentLevelWrapper(bandit_env, top_k=top_k_paths)
    env = POMDPWrapper(agent_wrapper)

    num_edges = len(bandit_env.scenario.edge_static)
    state_dim = num_edges
    obs_dim = 1 + top_k_paths * 2
    action_dim = top_k_paths

    if agent_type == "mappo":
        agent = MAPPOAgent(obs_dim, state_dim, action_dim, device)
    elif agent_type == "ippo":
        agent = IPPOAgent(obs_dim, action_dim, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    logger.log_config({
        "num_agents": bandit_env.num_agents,
        "num_edges": bandit_env.scenario.num_edges,
        "num_nodes": bandit_env.scenario.num_nodes,
        "bandit_top_k": top_k_paths,
    })

    all_stats = []
    window_stats = []

    pbar = tqdm(range(n_episodes), desc="Training", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        is_log_step = (ep == 0) or ((ep + 1) % log_interval == 0) or (ep == n_episodes - 1)
        
        obs_tensor, active_indices, _, done = env.reset()
        episode_rollout = []
        n_decisions = 0
        
        t0 = time.time()
        inner_pbar = None
        if n_episodes == 1:
            inner_pbar = tqdm(total=max_steps, desc=f"Episode {ep+1} Steps", leave=False)
            last_t = 0
            
        while not done:
            if inner_pbar:
                if env.t - last_t >= 3600:
                    inner_pbar.update(env.t - last_t)
                    last_t = env.t
            
            # Format inputs
            S_batch = env.dnl.edge_occupancy.float().unsqueeze(0).repeat(len(active_indices), 1)
            critic_obs = S_batch if agent_type == "mappo" else obs_tensor
            
            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs_tensor, critic_obs)
            
            actions_np = actions.cpu().numpy()
            n_decisions += len(actions_np)
            
            obs_tensor_next, next_active_indices, rewards, done = env.step(actions_np, active_indices)
            
            for i, idx in enumerate(active_indices):
                episode_rollout.append({
                    "agent": idx,
                    "actor_obs": obs_tensor[i],
                    "critic_obs": critic_obs[i],
                    "action": actions[i],
                    "log_prob": log_probs[i],
                    "value": values[i]
                })
                
            obs_tensor = obs_tensor_next
            active_indices = next_active_indices

        if inner_pbar:
            inner_pbar.update(max_steps - last_t)
            inner_pbar.close()

        wall_time = time.time() - t0
        
        # We now have rewards from the environment
        stats = _compute_stats(env, rewards.numpy(), wall_time, n_decisions)
        all_stats.append(stats)
        window_stats.append(stats)
        
        postfix = {
            "Mean TT": f"{stats['mean_travel_time']:.0f}",
            "Arr": f"{stats['arrival_rate'] * 100:.0f}%",
        }
        pbar.set_postfix(postfix)
        
        # PPO Update
        update_metrics = agent.update(episode_rollout, rewards.numpy())
        
        # Log
        if is_log_step:
            agg = _aggregate_stats(window_stats)

            tqdm.write(f"\n  ┌─ Episodes {ep + 1 - len(window_stats) + 1}–{ep + 1} ({len(window_stats)} eps) ───────────────────────")
            tqdm.write(f"  │ TSTT:       {agg['tstt_mean']:.0f} ± {agg['tstt_std']:.0f}")
            tqdm.write(f"  │ Mean TT:    {agg['mean_travel_time_mean']:.1f} ± {agg['mean_travel_time_std']:.1f}")
            tqdm.write(f"  │ p95:        {agg.get('tt_p95_mean', 0):.1f} ± {agg.get('tt_p95_std', 0):.1f}")
            tqdm.write(f"  │ Arrival:    {agg['arrival_rate_mean'] * 100:.1f}%")
            tqdm.write(f"  │ Length:     {agg['episode_length_mean']:.1f} ± {agg['episode_length_std']:.1f}")
            tqdm.write(f"  │ Reward:     {agg['mean_reward_mean']:.1f} ± {agg['mean_reward_std']:.1f}")
            tqdm.write("  └─────────────────────────────────────────────")

            wandb_metrics = {
                "metrics/TSTT": agg["tstt_mean"],
                "metrics/Mean TT": agg["mean_travel_time_mean"],
                "metrics/Arrival Rate": agg["arrival_rate_mean"],
                "metrics/Mean Reward": agg["mean_reward_mean"],
            }
            wandb_metrics.update(update_metrics)

            for k, v in agg.items():
                if k not in ["tstt_mean", "mean_travel_time_mean", "arrival_rate_mean", "mean_reward_mean"]:
                    nice_name = k.replace("_", " ").title().replace("Tstt", "TSTT").replace("Tt ", "TT ")
                    wandb_metrics[f"charts/{nice_name}"] = v

            logger.log_episode(ep, wandb_metrics)
            window_stats = []

    print(f"\n{'=' * 65}")
    print(f"  Summary ({n_episodes} episodes)")
    print(f"{'=' * 65}")
    def _fmt(key):
        vals = [s[key] for s in all_stats if key in s]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"
    print(f"  TSTT:             {_fmt('tstt')}")
    print(f"  Mean TT:          {_fmt('mean_travel_time')}")
    print(f"  TT p95:           {_fmt('tt_p95')}")
    print(f"  Arrival:          {np.mean([s['arrival_rate'] for s in all_stats]) * 100:.1f}%")
    print(f"  Episode Length:   {_fmt('episode_length')}")
    print(f"  Mean Reward:      {_fmt('mean_reward')}")
    print(f"{'=' * 65}")

    tstt_vals = [s["tstt"] for s in all_stats]
    logger.log_summary({
        "metrics/TSTT Mean": float(np.mean(tstt_vals)),
        "charts/TSTT Std": float(np.std(tstt_vals)),
        "metrics/Arrival Rate Mean": float(np.mean([s["arrival_rate"] for s in all_stats])),
        "metrics/Mean Reward Mean": float(np.mean([s["mean_reward"] for s in all_stats])),
    })
    logger.finish()

# --- Config parsing ---

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)

    kwargs: dict = {}
    sc = cfg.get("scenario", {})
    if "path" in sc: kwargs["scenario_path"] = sc["path"]
    if "population_filter" in sc: kwargs["population_filter"] = sc["population_filter"]
    if "scale_factor" in sc: kwargs["scale_factor"] = float(sc["scale_factor"])

    tr = cfg.get("training", {})
    _map = {
        "agent": "agent_type", "episodes": "n_episodes", "max_steps": "max_steps",
        "stuck_threshold": "stuck_threshold", "timestep": "timestep", "device": "device",
        "seed": "seed", "top_k_paths": "top_k_paths"
    }
    for jk, kk in _map.items():
        if jk in tr: kwargs[kk] = tr[jk]

    log = cfg.get("logging", {})
    if "log_interval" in log: kwargs["log_interval"] = log["log_interval"]
    if "wandb_enabled" in log: kwargs["wandb_enabled"] = log["wandb_enabled"]
    if "wandb_project" in log: kwargs["wandb_project"] = log["wandb_project"]
    if "wandb_tags" in log: kwargs["wandb_tags"] = log["wandb_tags"]
    if "wandb_agent" in log: kwargs["wandb_agent"] = log["wandb_agent"]

    return kwargs

def _parse_wandb_tags(tags_str: str) -> list[str]:
    if not tags_str: return []
    tags_str = tags_str.strip()
    if (tags_str.startswith("[") and tags_str.endswith("]")):
        try: return [str(t).strip() for t in json.loads(tags_str.replace("'", '"'))]
        except: tags_str = tags_str[1:-1]
    return [t.strip().strip('"').strip("'").strip() for t in tags_str.split(",") if t.strip()]

def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--population", default=None)
    parser.add_argument("--agent", default=None, choices=["ippo", "mappo"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--stuck_threshold", type=int, default=None)
    parser.add_argument("--timestep", type=float, default=None)
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--top_k_paths", "--k", type=int, default=None, dest="top_k_paths")
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_tags", default=None)
    parser.add_argument("--wandb_agent", default=None)
    return parser

_CLI_TO_KWARGS = {
    "scenario": "scenario_path", "population": "population_filter", "agent": "agent_type",
    "episodes": "n_episodes", "max_steps": "max_steps", "stuck_threshold": "stuck_threshold",
    "timestep": "timestep", "scale_factor": "scale_factor", "device": "device",
    "seed": "seed", "log_interval": "log_interval", "top_k_paths": "top_k_paths",
    "wandb": "wandb_enabled", "wandb_project": "wandb_project", "wandb_tags": "wandb_tags",
    "wandb_agent": "wandb_agent",
}

if __name__ == "__main__":
    parser = _build_parser()
    args, unknown = parser.parse_known_args()

    import inspect
    sig = inspect.signature(train)
    kwargs = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

    if args.config:
        kwargs.update(load_config(args.config))

    args_dict = vars(args)
    for cli_name, kwarg_name in _CLI_TO_KWARGS.items():
        cli_val = args_dict.get(cli_name)
        if cli_val is not None:
            if cli_name == "wandb_tags" and isinstance(cli_val, str):
                cli_val = _parse_wandb_tags(cli_val)
            kwargs[kwarg_name] = cli_val

    if "scenario_path" not in kwargs or kwargs["scenario_path"] is None:
        parser.error("--scenario is required")

    train(**kwargs)
