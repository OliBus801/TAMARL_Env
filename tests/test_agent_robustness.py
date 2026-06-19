import torch
import pytest
from tamarl.rl.agents.replicator_dynamics_agent import ReplicatorDynamicsAgent
from tamarl.rl.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from tamarl.rl.agents.random_agent import RandomAgent
from tamarl.rl.agents.exp3_agent import Exp3Agent
from tamarl.rl.agents.msa_agent import MSAAgent


class MockEnv:
    def __init__(self, num_envs=2, num_od_pairs=2):
        self.num_envs = num_envs
        self.num_od_pairs = num_od_pairs
        self.K = 3
        self._device = "cpu"
        
        # Mock attributes for TimeDependentEvaluator
        self.leg_to_agent = [(0, 0), (1, 1)]
        self.routes_flat_csr = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.routes_offsets_csr = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        self.first_edges_all_legs = torch.tensor([0, 0], dtype=torch.long)
        self.od_indices_all_legs = torch.tensor([0, 1], dtype=torch.long)
        
        class MockBandit:
            def __init__(self):
                class MockScenario:
                    def __init__(self):
                        self.departure_times = torch.zeros(num_envs)
                self.dnl = None
                self.scenario = MockScenario()
                self.link_tt_interval = 300.0
        self.bandit = MockBandit()


def test_replicator_dynamics_nan_prior_fallback():
    # 2 parameter blocks, 3 paths
    prior_mean = torch.tensor([
        [-10.0, -20.0, -30.0],
        [-float('inf'), -float('inf'), -float('inf')]  # All -inf row (e.g. disconnected)
    ])
    
    agent = ReplicatorDynamicsAgent(
        num_agents=2,
        k_paths=3,
        prior_mean=prior_mean,
        beta=0.1,
        epsilon_start=0.1,
        device="cpu"
    )
    
    # Assert no NaNs in rd_probs
    assert not torch.isnan(agent.rd_probs).any()
    # Row 1 should be softmaxed properly
    expected_row0 = torch.softmax(0.1 * torch.tensor([-10.0, -20.0, -30.0]), dim=0)
    assert torch.allclose(agent.rd_probs[0], expected_row0, atol=1e-5)
    # Row 1 should fallback to uniform
    assert torch.allclose(agent.rd_probs[1], torch.tensor([1/3, 1/3, 1/3]))


def test_replicator_dynamics_all_zero_masks():
    agent = ReplicatorDynamicsAgent(
        num_agents=2,
        k_paths=3,
        epsilon_start=0.1,
        device="cpu"
    )
    
    obs = torch.zeros((2, 3))
    # One row has valid masks, one row is all False (all-zero mask)
    masks = torch.tensor([
        [True, False, False],
        [False, False, False]
    ])
    
    # Selecting actions should succeed without CUDA multinomial/nan errors
    actions = agent.get_actions_batched(obs, masks)
    assert actions.shape == (2,)
    assert actions[0] == 0  # Only index 0 was valid
    assert actions[1] in [0, 1, 2]  # Fallback to uniform selection


def test_epsilon_greedy_all_zero_masks():
    agent = EpsilonGreedyAgent(
        num_agents=2,
        k_paths=3,
        epsilon_start=1.0,  # Force exploration
        device="cpu"
    )
    
    obs = torch.zeros((2, 3))
    masks = torch.tensor([
        [True, False, False],
        [False, False, False]
    ])
    
    actions = agent.get_actions_batched(obs, masks)
    assert actions.shape == (2,)
    assert actions[0] == 0
    assert actions[1] in [0, 1, 2]


def test_random_agent_all_zero_masks():
    agent = RandomAgent(num_agents=2, k=3)
    
    obs = torch.zeros((2, 3))
    masks = torch.tensor([
        [True, False, False],
        [False, False, False]
    ])
    
    actions = agent.get_actions_batched(obs, masks)
    assert actions.shape == (2,)
    assert actions[0] == 0
    assert actions[1] in [0, 1, 2]


def test_exp3_agent_all_zero_masks():
    agent = Exp3Agent(
        num_agents=2,
        k_paths=3,
        gamma=0.1,
        device="cpu"
    )
    
    obs = torch.zeros((2, 3))
    masks = torch.tensor([
        [True, False, False],
        [False, False, False]
    ])
    
    actions = agent.get_actions_batched(obs, masks)
    assert actions.shape == (2,)
    assert actions[0] == 0
    assert actions[1] in [0, 1, 2]


def test_msa_agent_all_zero_masks():
    env = MockEnv(num_envs=2, num_od_pairs=2)
    agent = MSAAgent(
        env=env,
        k_paths=3,
        device="cpu"
    )
    
    obs = torch.zeros((2, 3))
    masks = torch.tensor([
        [True, False, False],
        [False, False, False]
    ])
    
    actions = agent.get_actions_batched(obs, masks)
    assert actions.shape == (2,)
    assert actions[0] == 0
    assert actions[1] in [0, 1, 2]


def test_wrapper_warning_no_paths():
    import numpy as np
    from tamarl.envs.components.route_utils import build_routes_csr
    
    paths_dict = {
        (0, 1): [[1, 2]],
        (2, 3): []  # Empty path list
    }
    unique_od = np.array([[0, 1], [2, 3]])
    
    # Check build_routes_csr's output
    flat, offsets, masks, fftt = build_routes_csr(paths_dict, unique_od, top_k=3)
    
    # Check that mask for (2, 3) is all False
    assert not masks[1].any()
    
    # Simulate the check inside the wrappers:
    od_has_no_paths = ~masks.any(axis=1)
    inverse_od_np = np.array([0, 1, 1, 0]) # 4 legs, 2 map to the second OD pair
    no_path_legs = int(np.sum(od_has_no_paths[inverse_od_np]))
    assert no_path_legs == 2
