"""Compare DNL workload profiles for fixed-route benchmark and bandit training.

This script answers a question that cProfile alone does not: when
``torchdnl.step`` is slower in the RL loop, are the internal batches larger?

It instruments a ``TorchDNL`` instance at each tick and records:

* per-phase timings for nodes_A, flow_update, links_B, and demand_C;
* workload sizes before each tick, such as spatial candidates and demand queue;
* every ``torch.argsort`` call made inside a DNL phase, including input size.

The fixed-route benchmark run uses the route strings as ``paths``.
The training run uses ``DTABanditEnv`` plus the selected bandit wrapper and
profiles the ``TorchDNL`` instance created inside ``bandit.reset(paths)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

np = None
torch = None
TorchDNL = None
DTABanditEnv = None
AgentLevelWrapper = None
ODLevelWrapper = None
CentralizedLevelWrapper = None
RandomAgent = None
parse_network = None
load_config = None


def _load_runtime_deps() -> None:
    """Import heavy runtime dependencies only after argparse has run."""
    global np, torch, TorchDNL, DTABanditEnv, AgentLevelWrapper
    global ODLevelWrapper, CentralizedLevelWrapper, RandomAgent, parse_network
    global load_config

    if torch is not None:
        return

    import numpy as _np
    import torch as _torch

    from tamarl.core.torchdnl import TorchDNL as _TorchDNL
    from tamarl.envs.agent_level_wrapper import AgentLevelWrapper as _AgentLevelWrapper
    from tamarl.envs.centralized_level_wrapper import (
        CentralizedLevelWrapper as _CentralizedLevelWrapper,
    )
    from tamarl.envs.dta_bandit_env import DTABanditEnv as _DTABanditEnv
    from tamarl.envs.od_level_wrapper import ODLevelWrapper as _ODLevelWrapper
    from tamarl.envs.scenario_loader import parse_network as _parse_network
    from tamarl.rl.agents.random_agent import RandomAgent as _RandomAgent
    from tamarl.rl.train_bandit import load_config as _load_config

    np = _np
    torch = _torch
    TorchDNL = _TorchDNL
    DTABanditEnv = _DTABanditEnv
    AgentLevelWrapper = _AgentLevelWrapper
    ODLevelWrapper = _ODLevelWrapper
    CentralizedLevelWrapper = _CentralizedLevelWrapper
    RandomAgent = _RandomAgent
    parse_network = _parse_network
    load_config = _load_config


PHASES = ("nodes_A", "flow_update", "links_B", "demand_C")
WORKLOAD_KEYS = (
    "active_agents",
    "en_route_agents",
    "buffer_total",
    "spatial_total",
    "nodes_A_candidates",
    "rl_decision_waiters",
    "links_B_candidates",
    "demand_C_exiters",
    "demand_C_waiting",
)


def _time_to_sec(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported time value: {value!r}")
    h, m, s = (int(float(p)) for p in parts)
    return h * 3600 + m * 60 + s


def _locate_scenario_files(
    root_folder: str,
    population_filter: str | None,
) -> tuple[str, str]:
    files = [f for f in os.listdir(root_folder) if f.endswith(".xml")]
    net_candidates = [f for f in files if "network" in f.lower()]
    pop_candidates = [f for f in files if "population" in f.lower() or "plans" in f.lower()]

    if population_filter:
        token_matches = []
        for candidate in pop_candidates:
            tokens = candidate.replace("-", "_").replace(".", "_").split("_")
            if population_filter in tokens:
                token_matches.append(candidate)
        pop_candidates = token_matches or [p for p in pop_candidates if population_filter in p]

    if not net_candidates:
        raise FileNotFoundError(f"No network XML found in {root_folder}")
    if not pop_candidates:
        raise FileNotFoundError(
            f"No population/plans XML found in {root_folder}"
            + (f" matching {population_filter!r}" if population_filter else "")
        )

    route_pops = [p for p in pop_candidates if "route" in p.lower()]
    population_name = route_pops[0] if route_pops else pop_candidates[0]
    return (
        os.path.join(root_folder, net_candidates[0]),
        os.path.join(root_folder, population_name),
    )


def _parse_population_paths(
    population_file: str,
    link_id_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    """Parse agent plans into full fixed paths for benchmark-mode DNL."""
    agents: list[dict[str, Any]] = []
    skipped_legs = 0

    context = ET.iterparse(population_file, events=("end",))
    for _, elem in context:
        if elem.tag != "person":
            continue

        selected_plan = None
        for child in elem:
            if child.tag == "plan":
                if child.get("selected") == "yes":
                    selected_plan = child
                    break
                if selected_plan is None:
                    selected_plan = child

        if selected_plan is None:
            elem.clear()
            continue

        first_dep_time = 0
        first_act = True
        person_legs: list[list[int]] = []
        act_end_times: list[int] = []
        act_durations: list[int] = []

        for plan_element in list(selected_plan):
            if plan_element.tag in ("act", "activity"):
                end_time = plan_element.get("end_time")
                duration = plan_element.get("duration")

                act_end = _time_to_sec(end_time) if end_time else -1
                act_dur = _time_to_sec(duration) if duration else -1

                if first_act:
                    if act_end >= 0:
                        first_dep_time = act_end
                    elif act_dur >= 0:
                        first_dep_time = act_dur
                    first_act = False
                else:
                    act_end_times.append(act_end)
                    act_durations.append(act_dur)

            elif plan_element.tag == "leg" and plan_element.get("mode") == "car":
                route_tag = plan_element.find("route")
                route_text = (
                    route_tag.text.strip() if route_tag is not None and route_tag.text else None
                )
                if not route_text:
                    skipped_legs += 1
                    continue

                path_indices: list[int] = []
                valid_path = True
                for link_id in route_text.split():
                    edge_idx = link_id_to_idx.get(link_id)
                    if edge_idx is None:
                        valid_path = False
                        break
                    path_indices.append(edge_idx)

                if valid_path and path_indices:
                    person_legs.append(path_indices)
                else:
                    skipped_legs += 1

        if person_legs:
            boundaries = len(person_legs) - 1
            agents.append(
                {
                    "dep_time": first_dep_time,
                    "legs": person_legs,
                    "act_end_times": act_end_times[:boundaries],
                    "act_durations": act_durations[:boundaries],
                }
            )

        elem.clear()

    if skipped_legs:
        print(f"  Skipped {skipped_legs} population legs without valid fixed paths.")
    if not agents:
        raise ValueError("No valid fixed-route agents found in population file.")
    return agents


def _pack_fixed_route_agents(
    agents: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_agents = len(agents)
    max_path_len = max(
        sum(len(leg) for leg in agent["legs"]) + len(agent["legs"]) - 1 for agent in agents
    )
    max_acts = max(len(agent["act_end_times"]) for agent in agents)

    paths = torch.full((num_agents, max_path_len), -1, dtype=torch.int32)
    departure_times = torch.tensor([agent["dep_time"] for agent in agents], dtype=torch.int32)
    act_end_times = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    act_durations = torch.full((num_agents, max_acts), -1, dtype=torch.int32)
    num_legs = torch.zeros(num_agents, dtype=torch.int32)

    for agent_idx, agent in enumerate(agents):
        ptr = 0
        num_legs[agent_idx] = len(agent["legs"])
        for leg_idx, leg in enumerate(agent["legs"]):
            leg_tensor = torch.tensor(leg, dtype=torch.int32)
            paths[agent_idx, ptr : ptr + len(leg)] = leg_tensor
            ptr += len(leg)
            if leg_idx < len(agent["legs"]) - 1:
                paths[agent_idx, ptr] = -2
                ptr += 1

        if agent["act_end_times"]:
            n_acts = len(agent["act_end_times"])
            act_end_times[agent_idx, :n_acts] = torch.tensor(
                agent["act_end_times"], dtype=torch.int32
            )
            act_durations[agent_idx, :n_acts] = torch.tensor(
                agent["act_durations"], dtype=torch.int32
            )

    return paths, departure_times, act_end_times, act_durations, num_legs


def build_fixed_route_dnl(args: argparse.Namespace) -> TorchDNL:
    network_file, population_file = _locate_scenario_files(args.scenario, args.population)
    print(f"  Benchmark network:    {network_file}")
    print(f"  Benchmark population: {population_file}")

    _, edges_data, link_id_to_idx = parse_network(
        network_file,
        scale_factor=args.scale_factor,
        timestep=args.timestep,
    )
    agents = _parse_population_paths(population_file, link_id_to_idx)

    edge_static = torch.tensor([edge["attr"] for edge in edges_data], dtype=torch.float32)
    edge_endpoints = torch.tensor(
        [[edge["u"], edge["v"]] for edge in edges_data], dtype=torch.int32
    )
    paths, departure_times, act_end_times, act_durations, num_legs = _pack_fixed_route_agents(
        agents
    )

    return TorchDNL(
        edge_static=edge_static,
        paths=paths,
        device=args.device,
        departure_times=departure_times,
        edge_endpoints=edge_endpoints,
        act_end_times=act_end_times,
        act_durations=act_durations,
        num_legs=num_legs,
        stuck_threshold=args.stuck_threshold,
        dt=args.timestep,
        seed=args.seed,
        track_events=args.track_events,
        collect_link_tt=args.collect_link_tt,
        link_tt_interval=args.link_tt_interval,
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


@dataclass
class ArgsortRecord:
    phase: str
    size: int
    seconds: float


class DNLWorkloadProfiler:
    """Collects timing and workload data from one DNL run."""

    def __init__(self, name: str):
        self.name = name
        self.current_phase: str | None = None
        self.wall_time = 0.0
        self.step_times: list[float] = []
        self.phase_times: dict[str, list[float]] = defaultdict(list)
        self.section_times: dict[str, list[float]] = defaultdict(list)
        self.workloads: dict[str, list[int]] = defaultdict(list)
        self.argsort_records: list[ArgsortRecord] = []
        self.final_status_counts: dict[str, int] = {}
        self.extra: dict[str, Any] = {}

    def record_snapshot(self, dnl: TorchDNL) -> None:
        status = dnl.status
        ready = dnl.wakeup_time <= dnl.current_step
        buffer_ready = (status == 2) & ready
        spatial_ready = (status == 1) & ready
        waiting_ready = (status == 0) & ready

        values = {
            "active_agents": int((status != 3).sum().item()),
            "en_route_agents": int(((status == 1) | (status == 2)).sum().item()),
            "buffer_total": int((status == 2).sum().item()),
            "spatial_total": int((status == 1).sum().item()),
            "nodes_A_candidates": int((buffer_ready & (dnl.next_edge != -1)).sum().item()),
            "rl_decision_waiters": int((buffer_ready & (dnl.next_edge == -1)).sum().item()),
            "links_B_candidates": int(spatial_ready.sum().item()),
            "demand_C_exiters": int((status == 4).sum().item()),
            "demand_C_waiting": int(waiting_ready.sum().item()),
        }
        for key, value in values.items():
            self.workloads[key].append(value)

    def time_phase(self, phase: str, fn) -> None:
        self.current_phase = phase
        t0 = time.perf_counter()
        try:
            fn()
        finally:
            self.phase_times[phase].append(time.perf_counter() - t0)
            self.current_phase = None

    def record_section_time(self, section: str, seconds: float) -> None:
        self.section_times[section].append(seconds)

    def record_argsort(self, size: int, seconds: float) -> None:
        self.argsort_records.append(
            ArgsortRecord(self.current_phase or "outside_dnl_phase", size, seconds)
        )

    def record_final_status(self, dnl: TorchDNL) -> None:
        status = dnl.status
        self.final_status_counts = {
            "waiting_or_activity": int((status == 0).sum().item()),
            "traveling": int((status == 1).sum().item()),
            "buffer": int((status == 2).sum().item()),
            "done": int((status == 3).sum().item()),
            "exiter": int((status == 4).sum().item()),
        }

    def summarize(self) -> dict[str, Any]:
        argsort_by_phase: dict[str, dict[str, dict[str, float]]] = {}
        for phase in sorted({record.phase for record in self.argsort_records}):
            phase_records = [r for r in self.argsort_records if r.phase == phase]
            argsort_by_phase[phase] = {
                "sizes": _summary(r.size for r in phase_records),
                "seconds": _summary(r.seconds for r in phase_records),
            }

        return {
            "name": self.name,
            "wall_time_s": self.wall_time,
            "dnl_steps": len(self.step_times),
            "step_times_s": _summary(self.step_times),
            "phase_times_s": {phase: _summary(self.phase_times.get(phase, [])) for phase in PHASES},
            "section_times_s": {
                section: _summary(values) for section, values in sorted(self.section_times.items())
            },
            "workloads": {key: _summary(self.workloads.get(key, [])) for key in WORKLOAD_KEYS},
            "argsort": argsort_by_phase,
            "argsort_total_s": float(sum(r.seconds for r in self.argsort_records)),
            "argsort_calls": len(self.argsort_records),
            "final_status_counts": self.final_status_counts,
            "extra": self.extra,
        }


class ArgsortPatch:
    """Temporarily records torch.argsort calls for one profiler."""

    def __init__(self, profiler: DNLWorkloadProfiler):
        self.profiler = profiler
        self._orig_argsort = torch.argsort

    def __enter__(self):
        def wrapped(input_tensor, *args, **kwargs):
            t0 = time.perf_counter()
            result = self._orig_argsort(input_tensor, *args, **kwargs)
            self.profiler.record_argsort(
                int(input_tensor.numel()),
                time.perf_counter() - t0,
            )
            return result

        torch.argsort = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.argsort = self._orig_argsort
        return False


def instrument_dnl(dnl: TorchDNL, profiler: DNLWorkloadProfiler) -> None:
    """Replace one DNL instance's step method with an instrumented equivalent."""

    def profiled_step(self: TorchDNL):
        profiler.record_snapshot(self)
        step_t0 = time.perf_counter()

        profiler.time_phase("nodes_A", self._process_nodes_A)
        profiler.time_phase("flow_update", self._update_all_flow_accumulation)
        profiler.time_phase("links_B", self._process_links_B)
        profiler.time_phase("demand_C", self._schedule_demand_C)

        self.current_step += 1
        profiler.step_times.append(time.perf_counter() - step_t0)

    dnl.step = types.MethodType(profiled_step, dnl)


def run_benchmark_profile(args: argparse.Namespace) -> DNLWorkloadProfiler:
    print("\n" + "=" * 72)
    print("  Fixed-route benchmark workload profile")
    print("=" * 72)
    dnl = build_fixed_route_dnl(args)
    profiler = DNLWorkloadProfiler("benchmark")
    instrument_dnl(dnl, profiler)

    t0 = time.perf_counter()
    with ArgsortPatch(profiler):
        for _ in range(args.max_steps):
            if args.benchmark_stop_on_done and (dnl.status == 3).all().item():
                break
            dnl.step()
    profiler.wall_time = time.perf_counter() - t0
    profiler.record_final_status(dnl)
    profiler.extra.update(
        {
            "num_agents": dnl.num_agents,
            "num_edges": dnl.num_edges,
            "current_step": dnl.current_step,
            "stop_on_done": args.benchmark_stop_on_done,
        }
    )
    return profiler


def run_training_profile(args: argparse.Namespace) -> DNLWorkloadProfiler:
    print("\n" + "=" * 72)
    print("  Bandit training-loop workload profile")
    print("=" * 72)

    if args.agent != "random":
        raise ValueError(
            "This profiler currently supports --agent random for the RL run. "
            "Use the workload output to compare the simulator behavior first."
        )

    bandit = DTABanditEnv(
        scenario_path=args.scenario,
        population_filter=args.population,
        timestep=args.timestep,
        scale_factor=args.scale_factor,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        stuck_threshold=args.stuck_threshold,
        track_events=args.track_events,
        link_tt_interval=args.link_tt_interval,
    )
    bandit.collect_link_tt = args.collect_link_tt

    if args.formulation == "agent":
        env = AgentLevelWrapper(
            bandit=bandit,
            top_k=args.top_k_paths,
            feedback_type=args.bandit_feedback,
        )
    elif args.formulation == "od_pair":
        env = ODLevelWrapper(
            bandit=bandit,
            top_k=args.top_k_paths,
            feedback_type=args.bandit_feedback,
        )
    elif args.formulation == "centralized":
        env = CentralizedLevelWrapper(
            bandit=bandit,
            top_k=args.top_k_paths,
            feedback_type=args.bandit_feedback,
        )
    else:
        raise ValueError(f"Unknown formulation: {args.formulation}")

    profiler = DNLWorkloadProfiler("training")

    original_bandit_reset = bandit.reset
    original_bandit_step = bandit.step
    original_evaluate = getattr(env.evaluator, "evaluate", None)

    def profiled_bandit_reset(paths):
        t0_reset = time.perf_counter()
        result = original_bandit_reset(paths)
        profiler.record_section_time("bandit.reset", time.perf_counter() - t0_reset)
        instrument_dnl(bandit.dnl, profiler)
        profiler.extra["training_dnl_track_events"] = bool(bandit.dnl.track_events)
        profiler.extra["training_dnl_collect_link_tt"] = bool(bandit.dnl.collect_link_tt)
        return result

    def profiled_bandit_step():
        t0_step = time.perf_counter()
        result = original_bandit_step()
        profiler.record_section_time("bandit.step_total", time.perf_counter() - t0_step)
        return result

    bandit.reset = profiled_bandit_reset
    bandit.step = profiled_bandit_step

    if original_evaluate is not None:

        def profiled_evaluate(*eval_args, **eval_kwargs):
            t0_eval = time.perf_counter()
            result = original_evaluate(*eval_args, **eval_kwargs)
            profiler.record_section_time(
                "time_dependent_evaluator.evaluate",
                time.perf_counter() - t0_eval,
            )
            return result

        env.evaluator.evaluate = profiled_evaluate

    agent = RandomAgent(num_agents=env.num_envs, k=args.top_k_paths, seed=args.seed)

    macro_steps_total = 0
    decisions_total = 0
    t0 = time.perf_counter()
    with ArgsortPatch(profiler):
        for episode_idx in range(args.episodes):
            t0_reset = time.perf_counter()
            obs, infos = env.reset()
            profiler.record_section_time("env.reset", time.perf_counter() - t0_reset)

            if "od_indices" in infos:
                aggregation_indices = torch.from_numpy(infos["od_indices"]).to(args.device)
            else:
                aggregation_indices = torch.arange(obs.shape[0], device=args.device)

            obs_t = torch.from_numpy(obs).to(args.device)
            masks_t = torch.from_numpy(infos["action_mask"]).to(args.device)

            t0_action = time.perf_counter()
            actions_t = agent.get_actions_batched(
                obs_t,
                masks_t,
                aggregation_indices=aggregation_indices,
            )
            profiler.record_section_time(
                "agent.get_actions_batched", time.perf_counter() - t0_action
            )

            actions = actions_t.cpu().numpy()
            t0_env_step = time.perf_counter()
            env.step(actions)
            profiler.record_section_time("env.step_total", time.perf_counter() - t0_env_step)

            macro_steps_total += 1
            decisions_total += int(actions_t.numel())

    profiler.wall_time = time.perf_counter() - t0
    if bandit.dnl is not None:
        profiler.record_final_status(bandit.dnl)
    profiler.extra.update(
        {
            "num_agents": bandit.num_agents,
            "num_edges": bandit.scenario.num_edges,
            "num_nodes": bandit.scenario.num_nodes,
            "current_step": bandit.dnl.current_step if bandit.dnl is not None else None,
            "episodes": args.episodes,
            "macro_steps": macro_steps_total,
            "decisions": decisions_total,
            "formulation": args.formulation,
            "bandit_feedback": args.bandit_feedback,
            "top_k_paths": args.top_k_paths,
            "num_envs": env.num_envs,
            "agent": args.agent,
        }
    )
    env.close()
    return profiler


def _fmt(value: float, width: int = 12, precision: int = 3) -> str:
    return f"{value:{width}.{precision}f}"


def print_run_report(summary: dict[str, Any]) -> None:
    name = summary["name"]
    print("\n" + "-" * 72)
    print(f"  {name.upper()} SUMMARY")
    print("-" * 72)
    step = summary["step_times_s"]
    print(f"  Wall time:        {summary['wall_time_s']:.3f}s")
    print(f"  DNL steps:        {summary['dnl_steps']}")
    print(f"  Mean step time:   {step['mean'] * 1000:.3f} ms")
    print(f"  p95 step time:    {step['p95'] * 1000:.3f} ms")
    print(f"  Argsort total:    {summary['argsort_total_s']:.3f}s")
    print(f"  Argsort calls:    {summary['argsort_calls']}")
    print(f"  Final statuses:   {summary['final_status_counts']}")

    if summary["section_times_s"]:
        print("\n  Outer-loop timings")
        print("  section                         calls      total_s      mean_ms")
        for section, stats in summary["section_times_s"].items():
            print(
                f"  {section:<30s}"
                f"{int(stats['count']):8d}"
                f"{stats['total']:13.3f}"
                f"{stats['mean'] * 1000:13.3f}"
            )

    print("\n  Phase timings")
    print("  phase             calls      total_s      mean_ms      p95_ms")
    for phase in PHASES:
        stats = summary["phase_times_s"][phase]
        print(
            f"  {phase:<14s}"
            f"{int(stats['count']):8d}"
            f"{stats['total']:13.3f}"
            f"{stats['mean'] * 1000:13.3f}"
            f"{stats['p95'] * 1000:12.3f}"
        )

    print("\n  Workload sizes before DNL tick")
    print("  metric                    mean       p95       p99       max")
    for key in WORKLOAD_KEYS:
        stats = summary["workloads"][key]
        print(
            f"  {key:<23s}"
            f"{stats['mean']:10.1f}"
            f"{stats['p95']:10.1f}"
            f"{stats['p99']:10.1f}"
            f"{stats['max']:10.1f}"
        )

    print("\n  torch.argsort by phase")
    print("  phase             calls    total_s    mean_size    p95_size    max_size")
    for phase, stats in sorted(summary["argsort"].items()):
        size_stats = stats["sizes"]
        sec_stats = stats["seconds"]
        print(
            f"  {phase:<14s}"
            f"{int(size_stats['count']):8d}"
            f"{sec_stats['total']:11.3f}"
            f"{size_stats['mean']:13.1f}"
            f"{size_stats['p95']:12.1f}"
            f"{size_stats['max']:12.1f}"
        )


def _ratio(training_value: float, benchmark_value: float) -> str:
    if benchmark_value == 0:
        return "n/a"
    return f"{training_value / benchmark_value:.2f}x"


def print_comparison(benchmark: dict[str, Any], training: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  TRAINING / BENCHMARK COMPARISON")
    print("=" * 72)
    rows = [
        (
            "mean DNL step ms",
            benchmark["step_times_s"]["mean"] * 1000,
            training["step_times_s"]["mean"] * 1000,
        ),
        (
            "p95 DNL step ms",
            benchmark["step_times_s"]["p95"] * 1000,
            training["step_times_s"]["p95"] * 1000,
        ),
        (
            "argsort total s",
            benchmark["argsort_total_s"],
            training["argsort_total_s"],
        ),
        (
            "argsort calls",
            float(benchmark["argsort_calls"]),
            float(training["argsort_calls"]),
        ),
    ]

    for section in ("env.step_total", "bandit.step_total", "time_dependent_evaluator.evaluate"):
        if section in training["section_times_s"]:
            rows.append(
                (
                    f"training {section} s",
                    0.0,
                    training["section_times_s"][section]["total"],
                )
            )

    for phase in ("nodes_A", "links_B", "demand_C"):
        rows.append(
            (
                f"{phase} total s",
                benchmark["phase_times_s"][phase]["total"],
                training["phase_times_s"][phase]["total"],
            )
        )

    for key in (
        "nodes_A_candidates",
        "links_B_candidates",
        "demand_C_waiting",
        "rl_decision_waiters",
    ):
        rows.append(
            (
                f"{key} p95",
                benchmark["workloads"][key]["p95"],
                training["workloads"][key]["p95"],
            )
        )
        rows.append(
            (
                f"{key} max",
                benchmark["workloads"][key]["max"],
                training["workloads"][key]["max"],
            )
        )

    print("  metric                         benchmark      training        ratio")
    for label, bench_value, train_value in rows:
        print(
            f"  {label:<28s}"
            f"{_fmt(bench_value)}"
            f"{_fmt(train_value)}"
            f"{_ratio(train_value, bench_value):>13s}"
        )


def write_tick_csv(
    output_dir: str,
    profilers: list[DNLWorkloadProfiler],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for profiler in profilers:
        path = os.path.join(output_dir, f"{profiler.name}_tick_workloads.csv")
        rows = zip(*(profiler.workloads.get(key, []) for key in WORKLOAD_KEYS))
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("tick_index", *WORKLOAD_KEYS))
            for tick_idx, row in enumerate(rows):
                writer.writerow((tick_idx, *row))
        print(f"  Wrote tick workload CSV: {path}")


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    values: dict[str, Any] = {
        "scenario": None,
        "population": None,
        "max_steps": 86400,
        "timestep": 1.0,
        "scale_factor": 1.0,
        "device": "cpu",
        "seed": None,
        "stuck_threshold": 10,
        "agent": "random",
        "formulation": "agent",
        "bandit_feedback": "full",
        "top_k_paths": 3,
        "episodes": 1,
        "link_tt_interval": 60.0,
    }

    if args.config:
        _load_runtime_deps()
        config_values = load_config(args.config)
        mapping = {
            "scenario_path": "scenario",
            "population_filter": "population",
            "max_steps": "max_steps",
            "stuck_threshold": "stuck_threshold",
            "timestep": "timestep",
            "device": "device",
            "seed": "seed",
            "agent_type": "agent",
            "formulation": "formulation",
            "bandit_feedback": "bandit_feedback",
            "top_k_paths": "top_k_paths",
            "n_episodes": "episodes",
            "link_tt_interval": "link_tt_interval",
        }
        for source_key, target_key in mapping.items():
            if source_key in config_values:
                values[target_key] = config_values[source_key]

    for key in values:
        cli_value = getattr(args, key)
        if cli_value is not None:
            values[key] = cli_value

    for key, value in values.items():
        setattr(args, key, value)

    if args.scenario is None:
        raise SystemExit("Missing --scenario or config scenario.path")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile and compare DNL workload sizes in fixed-route benchmark and RL training loops."
        )
    )
    parser.add_argument("--config", default=None, help="Training-style JSON config")
    parser.add_argument("--scenario", default=None, help="Scenario folder")
    parser.add_argument("--population", default=None, help="Population file filter")
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--scale_factor", "--scale-factor", dest="scale_factor", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stuck_threshold", "--stuck-threshold", dest="stuck_threshold", type=int)

    parser.add_argument("--agent", choices=("random",), default=None)
    parser.add_argument(
        "--formulation",
        choices=("agent", "od_pair", "centralized"),
        default=None,
    )
    parser.add_argument(
        "--bandit_feedback",
        "--bandit-feedback",
        dest="bandit_feedback",
        choices=("full", "semi"),
        default=None,
    )
    parser.add_argument("--top_k_paths", "--top-k-paths", dest="top_k_paths", type=int)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--link_tt_interval",
        "--link-tt-interval",
        dest="link_tt_interval",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--mode",
        choices=("both", "benchmark", "training"),
        default="both",
        help="Which run(s) to execute.",
    )
    parser.add_argument(
        "--benchmark-stop-on-done",
        action="store_true",
        help="Stop fixed-route benchmark once all agents are done.",
    )
    parser.add_argument(
        "--collect-link-tt",
        action="store_true",
        help="Enable dynamic link travel-time collection in DNL.",
    )
    parser.add_argument(
        "--track-events",
        action="store_true",
        help="Enable DNL event tracking while profiling.",
    )
    parser.add_argument("--json-out", default=None, help="Write summary JSON here.")
    parser.add_argument(
        "--csv-out-dir",
        default=None,
        help="Optional directory for per-tick workload CSV files.",
    )
    return parser


def main() -> None:
    args = _apply_config(build_parser().parse_args())
    _load_runtime_deps()

    print("=" * 72)
    print("  DNL workload comparison profiler")
    print("=" * 72)
    print(f"  Scenario:      {args.scenario}")
    print(f"  Population:    {args.population}")
    print(f"  Max steps:     {args.max_steps}")
    print(f"  Timestep:      {args.timestep}")
    print(f"  Scale factor:  {args.scale_factor}")
    print(f"  Device:        {args.device}")
    print(f"  Seed:          {args.seed}")
    print(f"  Formulation:   {args.formulation}")
    print(f"  Feedback:      {args.bandit_feedback}")
    print(f"  Top-k paths:   {args.top_k_paths}")
    print(f"  Mode:          {args.mode}")

    profilers: list[DNLWorkloadProfiler] = []
    if args.mode in ("both", "benchmark"):
        profilers.append(run_benchmark_profile(args))
    if args.mode in ("both", "training"):
        profilers.append(run_training_profile(args))

    summaries = {profiler.name: profiler.summarize() for profiler in profilers}
    for summary in summaries.values():
        print_run_report(summary)

    if "benchmark" in summaries and "training" in summaries:
        print_comparison(summaries["benchmark"], summaries["training"])

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n  Wrote summary JSON: {args.json_out}")

    if args.csv_out_dir:
        write_tick_csv(args.csv_out_dir, profilers)


if __name__ == "__main__":
    main()
