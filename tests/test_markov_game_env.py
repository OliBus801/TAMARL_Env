"""Integration tests for the DTA Markov Game environment."""

import pytest
import numpy as np
from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv
from tamarl.rl_models.random_agent import RandomAgent
from tamarl.envs.components.metrics import compute_tstt, compute_arrival_rate


SCENARIO_PATH = "tamarl/data/scenarios/grid_world/3x3_uni"
POPULATION = "100"


@pytest.fixture
def env():
    e = DTAMarkovGameEnv(
        scenario_path=SCENARIO_PATH,
        population_filter=POPULATION,
        max_steps=3600,
        device="cpu",
        seed=42,
    )
    yield e
    e.close()


def test_env_creates(env):
    """Test that the environment creates successfully."""
    assert env.dnl is not None
    assert env.dnl.num_agents == 100
    assert env.dnl.num_edges == 12
    assert env.dnl.rl_mode is True


def test_reset_returns_obs(env):
    """Test that reset returns valid observations for all agents."""
    obs, infos = env.reset()
    
    assert len(obs) == 100
    assert len(infos) == 100
    
    for agent_id in env.possible_agents:
        assert agent_id in obs
        assert agent_id in infos
        assert "action_mask" in infos[agent_id]
        
        o = obs[agent_id]
        assert o.shape == (env._obs_builder.obs_size,)
        assert o.dtype == np.float32


def test_random_episode_completes(env):
    """Test that a full episode with random actions completes."""
    agent = RandomAgent(seed=42)
    obs, infos = env.reset()
    
    step_count = 0
    while env.agents:
        actions = agent.get_actions(obs, infos)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        
        if step_count > 5000:
            break
    
    # All agents should have arrived
    arr_rate = compute_arrival_rate(env.dnl)
    assert arr_rate == 1.0, f"Only {arr_rate*100:.1f}% of agents arrived"


def test_action_masking_valid(env):
    """Test that action masks have at least one valid action for deciding agents."""
    obs, infos = env.reset()
    agent = RandomAgent(seed=42)
    
    for _ in range(10):
        if not env.agents:
            break
        
        actions = agent.get_actions(obs, infos)
        
        # Every deciding agent should have at least one valid action
        for agent_id, action in actions.items():
            mask = infos[agent_id]["action_mask"]
            assert mask.sum() > 0, f"{agent_id} has no valid actions"
        
        obs, rewards, terminations, truncations, infos = env.step(actions)


def test_rewards_negative(env):
    """Test that all step rewards are <= 0."""
    obs, infos = env.reset()
    agent = RandomAgent(seed=42)
    
    for _ in range(20):
        if not env.agents:
            break
        
        actions = agent.get_actions(obs, infos)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent_id, r in rewards.items():
            assert r <= 0.0, f"{agent_id} got positive reward {r}"


def test_tstt_positive(env):
    """Test that TSTT is positive after episode completes."""
    agent = RandomAgent(seed=42)
    obs, infos = env.reset()
    
    while env.agents:
        actions = agent.get_actions(obs, infos)
        obs, rewards, terminations, truncations, infos = env.step(actions)
    
    tstt = compute_tstt(env.dnl)
    assert tstt > 0, f"TSTT should be positive, got {tstt}"
