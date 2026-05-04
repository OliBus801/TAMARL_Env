"""DTABanditEnv — One-shot DTA simulation engine.

This environment treats the DNL simulator as a black box:
  1. Receives a complete paths tensor [A, MaxPathLen].
  2. Runs the simulation to completion in a tight internal loop.
  3. Returns negative travel times (rewards) for each agent.

No per-node RL decisions are made — this is a *bandit* formulation
where the entire route is decided upfront.
"""
from __future__ import annotations

from typing import Optional

import torch

from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.envs.scenario_loader import load_scenario


class DTABanditEnv:
    """One-shot DTA simulation: paths in → negative travel times out.

    The DNL runs in *paths* mode (non-RL): agents follow pre-computed
    edge sequences with no mid-route decisions.  The environment
    simply orchestrates reset → simulate → extract metrics.

    Attributes:
        dnl: The underlying TorchDNLMATSim engine.
        scenario: The parsed ScenarioData.
        num_agents: Number of agents (vehicles).
        device: Torch device string.
    """

    def __init__(
        self,
        scenario_path: str,
        population_filter: Optional[str] = None,
        timestep: float = 1.0,
        scale_factor: float = 1.0,
        max_steps: int = 36000,
        device: str = "cpu",
        seed: Optional[int] = None,
        stuck_threshold: int = 10,
        track_events: bool = False,
    ):
        self._scenario_path = scenario_path
        self._max_steps = max_steps
        self._device = device
        self._seed = seed

        # Load scenario (network + population)
        self.scenario = load_scenario(
            scenario_path,
            population_filter=population_filter,
            timestep=timestep,
            scale_factor=scale_factor,
        )

        self.num_agents = self.scenario.num_agents

        # Store scenario parameters for DNL re-instantiation
        self._timestep = timestep
        self._stuck_threshold = stuck_threshold
        self._track_events = track_events
        self.collect_link_tt = False

        # DNL will be created/configured in reset()
        self.dnl: Optional[TorchDNLMATSim] = None

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, paths: torch.Tensor) -> None:
        """Configure the DNL with the given paths and prepare for step().

        Args:
            paths: [A, MaxPathLen] int tensor of edge indices, padded
                   with -1.  Each row is a full route for one agent.
                   Multi-leg routes use -2 as leg separator.
        """
        assert paths.shape[0] == self.num_agents, (
            f"paths has {paths.shape[0]} agents, expected {self.num_agents}"
        )

        paths_device = paths.to(self._device, dtype=torch.int32)

        # (Re-)create the DNL in paths mode (non-RL)
        self.dnl = TorchDNLMATSim(
            edge_static=self.scenario.edge_static,
            paths=paths_device,
            device=self._device,
            departure_times=self.scenario.departure_times,
            edge_endpoints=self.scenario.edge_endpoints,
            first_edges=self.scenario.first_edges,
            destinations=self.scenario.destinations,
            act_end_times=self.scenario.act_end_times,
            act_durations=self.scenario.act_durations,
            num_legs=self.scenario.num_legs,
            dt=self._timestep,
            seed=self._seed,
            stuck_threshold=self._stuck_threshold,
            track_events=self._track_events,
        )

        self.dnl.num_nodes = self.scenario.num_nodes

        if self.collect_link_tt:
            self.dnl.collect_link_tt = True
            self.dnl.max_intervals = 100
            self.dnl.interval_tt_sum = torch.zeros(
                (self.dnl.max_intervals, self.dnl.num_edges),
                device=self.dnl.device,
                dtype=torch.float32,
            )
            self.dnl.interval_tt_count = torch.zeros(
                (self.dnl.max_intervals, self.dnl.num_edges),
                device=self.dnl.device,
                dtype=torch.float32,
            )

    def step(self) -> torch.Tensor:
        """Run the simulation to completion and return rewards.

        Returns:
            rewards: [A] float tensor of *negative* travel times.
                     Lower (more negative) = longer trip = worse.
        """
        if self.dnl is None:
            raise RuntimeError("Must call reset(paths) before step().")

        # Tight simulation loop — no Python-side interruptions
        while self.dnl.current_step < self._max_steps:
            # Early exit when every agent is done (status ≥ 3)
            if (self.dnl.status >= 3).all().item():
                break
            self.dnl.step()

        # Extract per-agent travel time from leg_metrics
        # leg_metrics shape: [A, MaxLegs, 2] → [:, :, 1] is travel time
        # For the one-shot bandit, we sum travel times across all legs
        travel_times = self.dnl.leg_metrics[:, :, 1].sum(dim=1)  # [A]

        # Reward = negative travel time (RL wants to maximise → minimise TT)
        rewards = -travel_times

        return rewards

    def get_travel_times(self) -> torch.Tensor:
        """Return raw (positive) per-agent travel times after step().

        Returns:
            travel_times: [A] float tensor in simulation steps.
        """
        if self.dnl is None:
            raise RuntimeError("Must call reset(paths) then step() first.")
        return self.dnl.leg_metrics[:, :, 1].sum(dim=1)

    def get_metrics(self) -> dict:
        """Return summary metrics from the last simulation run."""
        if self.dnl is None:
            raise RuntimeError("No simulation has been run yet.")
        return self.dnl.get_metrics()
