"""Pure tensor operations for dynamic network loading.
Currently includes BPR travel time calculations based on each time step's link flows. (pretty basic.)"""
from __future__ import annotations

import torch

def update_cumulative_flows(
    flows: torch.Tensor, edges_chosen: torch.Tensor, num_edges: int
) -> torch.Tensor:
    """Update cumulative flows given chosen edges for a step.

    Args:
        flows: Current cumulative flows shaped [num_edges].
        edges_chosen: Tensor of edge ids chosen this step.
        num_edges: Total number of edges in the network.

    Returns:
        Updated cumulative flows tensor.
    """

    step_flow = torch.bincount(edges_chosen, minlength=num_edges).to(flows.dtype)
    return flows + step_flow


def compute_link_travel_times(
    flows: torch.Tensor,
    ff_time: torch.Tensor,
    capacity: torch.Tensor,
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """Compute BPR-based travel times for each link."""

    alpha_t = torch.as_tensor(alpha, dtype=flows.dtype, device=flows.device)
    beta_t = torch.as_tensor(beta, dtype=flows.dtype, device=flows.device)
    ratio = flows / capacity
    return ff_time * (1 + alpha_t * torch.pow(ratio, beta_t))


def compute_step_rewards(
    edges_chosen: torch.Tensor, link_travel_times: torch.Tensor
) -> torch.Tensor:
    """Return per-agent rewards for chosen edges."""

    return -link_travel_times[edges_chosen]


__all__ = [
    "update_cumulative_flows",
    "compute_link_travel_times",
    "compute_step_rewards",
]
