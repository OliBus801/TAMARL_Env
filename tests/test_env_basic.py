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
    assert type(env.data) == Data
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
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
    edge_attr = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    data = Data(num_nodes=4, edge_index=edge_index, edge_attr=edge_attr)
    env = DynamicTrafficAssignmentEnv(
        data,
        origins=[0, 0, 0],
        destinations=[1, 2, 1],
        max_steps=3,
        alpha=0.15,
        beta=4.0,
    )
    obs, _ = env.reset()
    actions = {"agent_0": 0, "agent_1": 1, "agent_2": 2}
    _, rewards, terminations, truncations, _ = env.step(actions)
    # with alpha=0.15, reward should be -ff_time*(1.15)
    assert np.isclose(rewards["agent_0"], -1.15)
    assert np.isclose(rewards["agent_1"], -1.15)

    # agent_0 and agent_1 should both have arrived at their destinations
    assert terminations["agent_0"] is True
    assert terminations["agent_1"] is True
    assert terminations["agent_2"] is False

    # agent_2 should be at node_1, but has reached a dead-end so should be truncated and penalized
    assert truncations["agent_0"] is False
    assert truncations["agent_1"] is False
    assert truncations["agent_2"] is True
    assert rewards["agent_2"] < -10

def test_action_masking():
    edge_index = torch.tensor([[0, 0, 1], [1, 3, 2]])
    edge_attr = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    data = Data(num_nodes=4, edge_index=edge_index, edge_attr=edge_attr)
    env = DynamicTrafficAssignmentEnv(
        data,
        origins=[0, 1, 3],
        destinations=[2, 2, 3],
        max_steps=3,
        alpha=0.15,
        beta=4.0,
    )
    obs, _ = env.reset()

    # agent_0 at node_0 has two possible actions (to node_1 or node_3)
    assert obs["agent_0"]["action_mask"].tolist() == [1, 1]
    # agent_1 at node_1 has one possible action (to node_2)
    assert obs["agent_1"]["action_mask"].tolist() == [1, 0]
    # agent_2 at node_3 has no possible actions (already at destination)
    assert obs["agent_2"]["action_mask"].tolist() == [0, 0]


