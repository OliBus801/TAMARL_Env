import torch

from tamarl.core.network_loading import (
    compute_link_travel_times,
    compute_step_rewards,
    update_cumulative_flows,
)


def test_update_cumulative_flows():
    flows = torch.zeros(3)
    edges_step1 = torch.tensor([0, 1, 1])
    flows = update_cumulative_flows(flows, edges_step1, num_edges=3)
    assert torch.allclose(flows, torch.tensor([1.0, 2.0, 0.0]))

    edges_step2 = torch.tensor([2, 2, 2])
    flows = update_cumulative_flows(flows, edges_step2, num_edges=3)
    assert torch.allclose(flows, torch.tensor([1.0, 2.0, 3.0]))


def test_compute_link_travel_times():
    flows = torch.tensor([0.0, 2.0])
    ff_time = torch.tensor([1.0, 2.0])
    capacity = torch.tensor([1.0, 2.0])
    alpha = 0.15
    beta = 4.0
    times = compute_link_travel_times(flows, ff_time, capacity, alpha, beta)
    expected = torch.tensor([1.0, 2.0 * (1 + 0.15 * (1.0**4))])
    assert torch.allclose(times, expected)


def test_compute_step_rewards():
    link_times = torch.tensor([1.0, 3.0, 2.5])
    edges_chosen = torch.tensor([0, 2, 1])
    rewards = compute_step_rewards(edges_chosen, link_times)
    expected = torch.tensor([-1.0, -2.5, -3.0])
    assert torch.allclose(rewards, expected)
