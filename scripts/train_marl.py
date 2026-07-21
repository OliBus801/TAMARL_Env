import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.pomdp_wrapper import POMDPWrapper

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
        """
        rollouts is a list of dicts: {'agent': idx, 'actor_obs': tensor, 'critic_obs': tensor, 'action': tensor, 'log_prob': tensor, 'value': tensor}
        final_rewards is a tensor of shape [num_agents] containing the episode reward for each agent.
        Since rewards only arrive at the end of the episode, we assign the final reward to the single action taken by each agent.
        For multi-leg, it would be distributed. Here we assume 1 action per agent per episode.
        """
        if len(rollouts) == 0:
            return {}
            
        b_actor_obs = torch.stack([r['actor_obs'] for r in rollouts])
        b_critic_obs = torch.stack([r['critic_obs'] for r in rollouts])
        b_actions = torch.stack([r['action'] for r in rollouts])
        b_logprobs = torch.stack([r['log_prob'] for r in rollouts])
        b_values = torch.stack([r['value'] for r in rollouts]).squeeze(-1)
        
        # Calculate advantages
        b_rewards = torch.tensor([final_rewards[r['agent']] for r in rollouts], dtype=torch.float32, device=self.device)
        
        # Simple advantage since it's a 1-step episode per agent effectively
        # Advantage = Return - Value. Return = Reward since no future steps.
        b_returns = b_rewards
        b_advantages = b_returns - b_values
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO Update
        dataset_size = len(b_actor_obs)
        b_inds = np.arange(dataset_size)
        
        clipfracs = []
        
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
                
                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
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
                
                # Entropy loss
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
        # IPPO: critic takes local observation
        super().__init__(obs_dim, obs_dim, action_dim, device=device)

class MAPPOAgent(PPOBase):
    def __init__(self, obs_dim, state_dim, action_dim, device="cpu"):
        # MAPPO: critic takes global state
        super().__init__(obs_dim, state_dim, action_dim, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="data/scenarios/SiouxFalls")
    parser.add_argument("--population", type=str, default=None, help="Population file filter (e.g. '100')")
    parser.add_argument("--algo", type=str, default="mappo", choices=["ippo", "mappo"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=36000)
    parser.add_argument("--k", type=int, default=3, help="Number of paths in the agent's action space")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="TorchDNL-MARL", help="W&B project name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="W&B run tags (comma-separated or JSON list string)")
    args = parser.parse_args()
    
    if args.wandb:
        tags = None
        if args.wandb_tags:
            if args.wandb_tags.startswith("[") and args.wandb_tags.endswith("]"):
                import json
                try:
                    tags = json.loads(args.wandb_tags)
                except Exception:
                    tags = [t.strip() for t in args.wandb_tags.split(",")]
            else:
                tags = [t.strip() for t in args.wandb_tags.split(",")]
        wandb.init(project=args.wandb_project, config=vars(args), tags=tags)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Base Environment
    bandit_env = DTABanditEnv(
        scenario_path=args.scenario,
        device=device,
        max_steps=args.max_steps,
        population_filter=args.population
    )
    
    # 2. Setup Agent Wrapper (for routes)
    agent_wrapper = AgentLevelWrapper(bandit_env, top_k=args.k)
    
    # 3. Setup POMDP Wrapper
    env = POMDPWrapper(agent_wrapper)
    
    # 4. Setup Agent
    num_edges = len(bandit_env.scenario.edge_static)
    state_dim = num_edges
    obs_dim = 1 + args.k * 2 # t_norm + FFTT (k) + Occ (k, mock)
    action_dim = args.k
    
    if args.algo == "mappo":
        agent = MAPPOAgent(obs_dim, state_dim, action_dim, device)
    elif args.algo == "ippo":
        agent = IPPOAgent(obs_dim, action_dim, device)
    
    # 5. Training Loop
    for ep in range(args.episodes):
        obs, active_indices, rewards, done = env.reset()
        
        episode_rollout = []
        
        last_reported_time = 0
        pbar = tqdm(total=args.max_steps, desc=f"Episode {ep+1}/{args.episodes}")
        
        while not done:
            if env.t - last_reported_time >= 3600:
                pbar.update(env.t - last_reported_time)
                last_reported_time = env.t
                
            # Formatting obs based on POMDPWrapper return
            obs_tensor, S = env._get_obs(active_indices)
            S_batch = S.unsqueeze(0).repeat(len(active_indices), 1)
            
            if args.algo == "mappo":
                critic_obs = S_batch
            else:
                critic_obs = obs_tensor
                
            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs_tensor, critic_obs)
                
            actions_np = actions.cpu().numpy()
            
            # Step environment
            _, next_active_indices, rewards, done = env.step(actions_np, active_indices)
            
            # Store trajectory info
            for i, idx in enumerate(active_indices):
                episode_rollout.append({
                    "agent": idx,
                    "actor_obs": obs_tensor[i],
                    "critic_obs": critic_obs[i],
                    "action": actions[i],
                    "log_prob": log_probs[i],
                    "value": values[i]
                })
                
            active_indices = next_active_indices
            
        pbar.update(args.max_steps - last_reported_time)
        pbar.close()
                
        # Final rewards arrived
        mean_tt = -rewards.mean().item()
        print(f"Episode {ep+1} complete, Mean Travel Time: {mean_tt:.2f}")
        
        # Update Agent
        update_metrics = agent.update(episode_rollout, rewards)
        
        log_dict = {"Mean Travel Time": mean_tt, "Episode": ep}
        log_dict.update(update_metrics)
        if args.wandb:
            wandb.log(log_dict)
        
if __name__ == "__main__":
    main()
