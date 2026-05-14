"""Compare DNL workload profiles for fixed-route benchmark and RL training.

This script answers a question that cProfile alone does not: when
``dnl_matsim.step`` is slower in the RL loop, are the internal batches larger?

It instruments a ``TorchDNLMATSim`` instance at each tick and records:

* per-phase timings for nodes_A, flow_update, links_B, and demand_C;
* workload sizes before each tick, such as spatial candidates and demand queue;
* every ``torch.argsort`` call made inside a DNL phase, including input size.

The fixed-route benchmark run uses the MATSim route strings as ``paths``.
The RL run uses ``DTAMarkovGameEnv`` and the batched random policy by default.
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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from tamarl.core.dnl_matsim import TorchDNLMATSim
from tamarl.envs.dta_markov_game_parallel import DTAMarkovGameEnv
from tamarl.envs.scenario_loader import parse_network
from tamarl.rl.train import load_config
from tamarl.rl_models.random_agent import RandomAgent


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
        raise ValueError(f"Unsupported MATSim time value: {value!r}")
    h, m, s = (int(float(p)) for p in parts)
    return h * 3600 + m * 60 + s


def _locate_scenario_files(
    root_folder: str,
    population_filter: Optional[str],
) -> Tuple[str, str]:
    files = [f for f in os.listdir(root_folder) if f.endswith(".xml")]
    net_candidates = [f for f in files if "network" in f.lower()]
    pop_candidates = [
        f for f in files if "population" in f.lower() or "plans" in f.lower()
    ]

    if population_filter:
        token_matches = []
        for candidate in pop_candidates:
            tokens = candidate.replace("-", "_").replace(".", "_").split("_")
            if population_filter in tokens:
                token_matches.append(candidate)
        pop_candidates = token_matches or [
            p for p in pop_candidates if population_filter in p
        ]

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
    link_id_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Parse MATSim plans into full fixed paths for benchmark-mode DNL."""
    agents: List[Dict[str, Any]] = []
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
        person_legs: List[List[int]] = []
        act_end_times: List[int] = []
        act_durations: List[int] = []

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
                    route_tag.text.strip()
                    if route_tag is not None and route_tag.text
                    else None
                )
                if not route_text:
                    skipped_legs += 1
                    continue

                path_indices: List[int] = []
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
    agents: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_agents = len(agents)
    max_path_len = max(
        sum(len(leg) for leg in agent["legs"]) + len(agent["legs"]) - 1
        for agent in agents
    )
    max_acts = max(len(agent["act_end_times"]) for agent in agents)

    paths = torch.full((num_agents, max_path_len), -1, dtype=torch.int32)
    departure_times = torch.tensor(
        [agent["dep_time"] for agent in agents], dtype=torch.int32
    )
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


def build_fixed_route_dnl(args: argparse.Namespace) -> TorchDNLMATSim:
    network_file, population_file = _locate_scenario_files(
        args.scenario, args.population
    )
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
    paths, departure_times, act_end_times, act_durations, num_legs = (
        _pack_fixed_route_agents(agents)
    )

    return TorchDNLMATSim(
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
    )


def _summary(values: Iterable[float]) -> Dict[str, float]:
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
        self.current_phase: Optional[str] = None
        self.wall_time = 0.0
        self.step_times: List[float] = []
        self.phase_times: Dict[str, List[float]] = defaultdict(list)
        self.workloads: Dict[str, List[int]] = defaultdict(list)
        self.argsort_records: List[ArgsortRecord] = []
        self.final_status_counts: Dict[str, int] = {}
        self.extra: Dict[str, Any] = {}

    def record_snapshot(self, dnl: TorchDNLMATSim) -> None:
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
            "nodes_A_candidates": int(
                (buffer_ready & (dnl.next_edge != -1)).sum().item()
            ),
            "rl_decision_waiters": int(
                (buffer_ready & (dnl.next_edge == -1)).sum().item()
            ),
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

    def record_argsort(self, size: int, seconds: float) -> None:
        self.argsort_records.append(
            ArgsortRecord(self.current_phase or "outside_dnl_phase", size, seconds)
        )

    def record_final_status(self, dnl: TorchDNLMATSim) -> None:
        status = dnl.status
        self.final_status_counts = {
            "waiting_or_activity": int((status == 0).sum().item()),
            "traveling": int((status == 1).sum().item()),
            "buffer": int((status == 2).sum().item()),
            "done": int((status == 3).sum().item()),
            "exiter": int((status == 4).sum().item()),
        }

    def summarize(self) -> Dict[str, Any]:
        argsort_by_phase: Dict[str, Dict[str, Dict[str, float]]] = {}
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
            "phase_times_s": {
                phase: _summary(self.phase_times.get(phase, [])) for phase in PHASES
            },
            "workloads": {
                key: _summary(self.workloads.get(key, [])) for key in WORKLOAD_KEYS
            },
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


def instrument_dnl(dnl: TorchDNLMATSim, profiler: DNLWorkloadProfiler) -> None:
    """Replace one DNL instance's step method with an instrumented equivalent."""

    def profiled_step(self: TorchDNLMATSim):
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


def _get_random_actions(
    agent: RandomAgent,
    obs: torch.Tensor,
    masks: torch.Tensor,
    deciding: torch.Tensor,
) -> torch.Tensor:
    # Current RandomAgent does not accept a deterministic kwarg.
    return agent.get_actions_batched(obs, masks, deciding)


def run_training_profile(args: argparse.Namespace) -> DNLWorkloadProfiler:
    print("\n" + "=" * 72)
    print("  RL training-loop workload profile")
    print("=" * 72)

    formulation = args.formulation
    if formulation not in ("link-based", "path-based"):
        print(
            f"  Warning: formulation={formulation!r} is not implemented by "
            "DTAMarkovGameEnv; using link-based semantics."
        )

    if args.agent != "random":
        raise ValueError(
            "This profiler currently supports --agent random for the RL run. "
            "Use the workload output to compare the simulator behavior first."
        )

    env = DTAMarkovGameEnv(
        scenario_path=args.scenario,
        population_filter=args.population,
        timestep=args.timestep,
        scale_factor=args.scale_factor,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        stuck_threshold=args.stuck_threshold,
        track_events=args.track_events,
        formulation=formulation,
        top_k_paths=args.top_k_paths,
    )
    env.dnl.collect_link_tt = args.collect_link_tt
    if args.collect_link_tt and getattr(env.dnl, "interval_tt_sum", None) is None:
        env.dnl.max_intervals = 100
        env.dnl.interval_tt_sum = torch.zeros(
            (env.dnl.max_intervals, env.dnl.num_edges),
            device=env.dnl.device,
            dtype=torch.float32,
        )
        env.dnl.interval_tt_count = torch.zeros(
            (env.dnl.max_intervals, env.dnl.num_edges),
            device=env.dnl.device,
            dtype=torch.float32,
        )

    profiler = DNLWorkloadProfiler("training")
    instrument_dnl(env.dnl, profiler)
    agent = RandomAgent(seed=args.seed)

    macro_steps_total = 0
    decisions_total = 0
    t0 = time.perf_counter()
    with ArgsortPatch(profiler):
        for episode_idx in range(args.episodes):
            obs_all, masks, deciding, _ = env.reset_batched(
                seed=args.seed if episode_idx == 0 else None
            )
            macro_steps = 0
            decisions = 0

            while env.has_active_agents():
                num_deciding = deciding.numel()
                if num_deciding > 0:
                    if obs_all.shape[0] == env.dnl.num_agents:
                        obs_deciding = obs_all[deciding]
                    else:
                        obs_deciding = env._obs_builder.build_observations_batched(
                            deciding
                        )
                    actions = _get_random_actions(agent, obs_deciding, masks, deciding)
                    decisions += int(num_deciding)
                else:
                    actions = torch.empty(
                        0, device=env.dnl.device, dtype=torch.long
                    )

                obs_all, _, _, _, masks, deciding = env.step_batched(
                    deciding
                    if num_deciding > 0
                    else torch.empty(0, device=env.dnl.device, dtype=torch.long),
                    actions,
                )
                macro_steps += 1

            macro_steps_total += macro_steps
            decisions_total += decisions

    profiler.wall_time = time.perf_counter() - t0
    profiler.record_final_status(env.dnl)
    profiler.extra.update(
        {
            "num_agents": env.dnl.num_agents,
            "num_edges": env.dnl.num_edges,
            "num_nodes": env.dnl.num_nodes,
            "max_out_degree": env.dnl.max_out_degree,
            "current_step": env.dnl.current_step,
            "episodes": args.episodes,
            "macro_steps": macro_steps_total,
            "decisions": decisions_total,
            "formulation": formulation,
            "agent": args.agent,
        }
    )
    env.close()
    return profiler


def _fmt(value: float, width: int = 12, precision: int = 3) -> str:
    return f"{value:{width}.{precision}f}"


def print_run_report(summary: Dict[str, Any]) -> None:
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


def print_comparison(benchmark: Dict[str, Any], training: Dict[str, Any]) -> None:
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
    profilers: List[DNLWorkloadProfiler],
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
    values: Dict[str, Any] = {
        "scenario": None,
        "population": None,
        "max_steps": 86400,
        "timestep": 1.0,
        "scale_factor": 1.0,
        "device": "cpu",
        "seed": None,
        "stuck_threshold": 10,
        "agent": "random",
        "formulation": "link-based",
        "top_k_paths": 3,
        "episodes": 1,
    }

    if args.config:
        config_values = load_config(args.config)
        mapping = {
            "scenario_path": "scenario",
            "population_filter": "population",
            "max_steps": "max_steps",
            "timestep": "timestep",
            "device": "device",
            "seed": "seed",
            "agent_type": "agent",
            "formulation": "formulation",
            "top_k_paths": "top_k_paths",
            "n_episodes": "episodes",
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
            "Profile and compare DNL workload sizes in fixed-route benchmark "
            "and RL training loops."
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
    parser.add_argument("--formulation", default=None)
    parser.add_argument("--top_k_paths", "--top-k-paths", dest="top_k_paths", type=int)
    parser.add_argument("--episodes", type=int, default=None)

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
    print(f"  Mode:          {args.mode}")

    profilers: List[DNLWorkloadProfiler] = []
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
