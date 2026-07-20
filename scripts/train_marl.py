import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.pomdp_wrapper import POMDPWrapper

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class IPPOAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.actor = MLPNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = MLPNetwork(obs_dim, 64, 1).to(device) # Decentralized critic
        self.device = device
    def get_action(self, obs):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)
    def get_value(self, obs):
        return self.critic(obs)
    def update(self, rollouts):
        pass

class MAPPOAgent:
    def __init__(self, obs_dim, state_dim, action_dim, device="cpu"):
        self.actor = MLPNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = MLPNetwork(state_dim, 64, 1).to(device) # Centralized critic
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.device = device

    def get_action(self, obs):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, state):
        return self.critic(state)
        
    def update(self, rollouts):
        # Placeholder for MAPPO update using GAE
        # rollouts: list of (obs, state, action, log_prob, reward, value)
        pass

class DQNAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.q_net = MLPNetwork(obs_dim, 64, action_dim).to(device)
        self.device = device
    def get_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return torch.randint(0, self.q_net.net[-1].out_features, (obs.shape[0],), device=self.device), None
        return self.q_net(obs).argmax(dim=-1), None
    def get_value(self, state):
        return None
    def update(self, replay_buffer):
        pass

class QMIXAgent:
    def __init__(self, obs_dim, state_dim, action_dim, num_agents, device="cpu"):
        self.agent_q = MLPNetwork(obs_dim, 64, action_dim).to(device)
        self.mixer = MLPNetwork(state_dim, 64, num_agents).to(device) # Simplistic mixer placeholder
        self.device = device
    def get_action(self, obs, epsilon=0.1):
        if np.random.rand() < epsilon:
            return torch.randint(0, self.agent_q.net[-1].out_features, (obs.shape[0],), device=self.device), None
        return self.agent_q(obs).argmax(dim=-1), None
    def get_value(self, state):
        return None
    def update(self, replay_buffer):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="data/scenarios/SiouxFalls")
    parser.add_argument("--algo", type=str, default="mappo", choices=["ippo", "mappo", "dqn", "qmix"])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    wandb.init(project="TorchDNL-MARL", config=vars(args))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Base Environment
    bandit_env = DTABanditEnv(
        scenario_path=args.scenario,
        device=device,
        max_steps=36000
    )
    
    # 2. Setup Agent Wrapper (for routes)
    agent_wrapper = AgentLevelWrapper(bandit_env, top_k=3)
    
    # 3. Setup POMDP Wrapper
    env = POMDPWrapper(agent_wrapper)
    
    # 4. Setup Agent
    num_edges = len(bandit_env.scenario.edge_static)
    state_dim = num_edges
    obs_dim = 1 + 3 * 2 # t_norm + FFTT (3) + Occ (3, mock)
    action_dim = 3
    
    agent = MAPPOAgent(obs_dim, state_dim, action_dim, device)
    
    # 5. Training Loop
    for ep in range(args.episodes):
        obs, active_indices, rewards, done = env.reset()
        
        episode_rollout = []
        
        while not done:
            # Active agents take actions
            S = env._get_obs(active_indices)[1]
            S_batch = S.unsqueeze(0).repeat(len(active_indices), 1) # Same state for all
            
            with torch.no_grad():
                actions, log_probs = agent.get_action(obs[0] if isinstance(obs, tuple) else obs)
                values = agent.get_value(S_batch)
                
            actions_np = actions.cpu().numpy()
            
            # Step environment
            obs, active_indices, rewards, done = env.step(actions_np, active_indices)
            
            # Store trajectory info
            for i, idx in enumerate(active_indices):
                episode_rollout.append({
                    "agent": idx,
                    "obs": (obs[0] if isinstance(obs, tuple) else obs)[i] if obs is not None else None,
                    "state": S_batch[i],
                    "action": actions[i],
                    "log_prob": log_probs[i],
                    "value": values[i]
                })
                
        mean_tt = -rewards.mean().item()
        print(f"Episode {ep}, Mean Travel Time: {mean_tt}")
        
        wandb.log({"Mean Travel Time": mean_tt, "Episode": ep})
        
        # In a real setup, we would compute GAE here and call agent.update(episode_rollout)

if __name__ == "__main__":
    main()
