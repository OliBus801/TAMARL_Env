import numpy as np
import torch
from torch_geometric.data import Data

from tamarl.envs.dta_env import DynamicTrafficAssignmentEnv


def build_line_env():
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    data = Data(num_nodes=3, edge_index=edge_index, edge_attr=edge_attr)
    env = DynamicTrafficAssignmentEnv(data, origins=[0], destinations=[2], max_steps=5)
    return env


def test_reset_and_spaces():
    env = build_line_env()
    obs, infos = env.reset(seed=42)
    assert env.agents == ["agent_0"]
    assert "agent_0" in obs
    agent_obs = obs["agent_0"]
    assert agent_obs["current_node"] == 0
    assert agent_obs["destination_node"] == 2
    assert env.observation_space("agent_0").contains(agent_obs)
    assert env.action_space("agent_0").n == env.network_meta.max_out_degree


def test_step_reaches_destination():
    env = build_line_env()
    obs, _ = env.reset()
    actions = {"agent_0": 0}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert env._current_nodes["agent_0"] == 1
    # second move to destination
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert terminations["agent_0"] is True


def test_parallel_rewards_and_flows():
    edge_index = torch.tensor([[0, 0], [1, 2]])
    edge_attr = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    data = Data(num_nodes=3, edge_index=edge_index, edge_attr=edge_attr)
    env = DynamicTrafficAssignmentEnv(
        data,
        origins=[0, 0],
        destinations=[1, 2],
        max_steps=3,
        alpha=0.0,
        beta=1.0,
    )
    obs, _ = env.reset()
    actions = {"agent_0": 0, "agent_1": 1}
    _, rewards, _, _, _ = env.step(actions)
    # with alpha=0, reward should be -ff_time
    assert np.isclose(rewards["agent_0"], -1.0)
    assert np.isclose(rewards["agent_1"], -1.0)

    # second step, only agent_1 moves
    actions = {"agent_1": 1}
    _, rewards, _, _, _ = env.step(actions)
    # agent_1 already reached its destination and should receive zero reward
    assert np.isclose(rewards["agent_1"], 0.0)


