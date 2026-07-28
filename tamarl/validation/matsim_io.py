"""Parse MATSim reference data (events + fixed/converged route population) into
the exact tensor/tuple formats TorchDNL's own tooling already consumes.

Two independent things are parsed here, both against the *same*
`link_id_to_idx` mapping produced by `tamarl.envs.scenario_loader.parse_network`
for the scenario under comparison, so that link/edge indices line up between
TorchDNL and MATSim:

1. `parse_matsim_events` — MATSim's `output_events.xml(.gz)` into TorchDNL's
   native event-tuple format `(time, evt_type_int, agent_idx, edge_idx)`, the
   same format `TorchDNL.get_events()` returns. This is a drop-in for
   `tamarl.visualisation.plot_histogram._compute_leg_histogram_data`,
   `tamarl.rl.render_helper.write_events_csv`-style export, etc.

2. `parse_fixed_route_population` — a converged/fixed MATSim route population
   (e.g. Siouxfalls_route_population.xml) into literal per-agent CSR
   `paths_flat`/`path_offsets`, for direct replay through TorchDNL
   (`TorchDNL(paths_flat=..., path_offsets=...)`), bypassing route re-selection
   entirely. This is deliberately NOT `scenario_loader.parse_population`,
   which only keeps the first/last link of each leg's `<route>` (it feeds
   TorchDNL's own route-choice wrappers, where the full path is chosen later
   by a top-K candidate search) — here we need the literal MATSim-assigned
   link sequence, unabridged, matching TorchDNL's native paths_flat contract
   where leg boundaries are marked by a single `-2` sentinel and the agent's
   very first entered edge IS the first element of its path (see
   TorchDNL.__init__ docstring / `self.next_edge[:] = paths_flat[path_offsets[:-1]]`).
"""

from __future__ import annotations

import csv
import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Event type constants (mirrored from tamarl/core/torchdnl.py)
EVT_ACTEND = 0
EVT_DEPARTURE = 1
EVT_ENTERS_TRAFFIC = 2
EVT_LEFT_LINK = 3
EVT_ENTERED_LINK = 4
EVT_LEAVES_TRAFFIC = 5
EVT_ARRIVAL = 6
EVT_ACTSTART = 7
EVT_STUCKANDABORT = 8
EVT_ENTERED_BUFFER = 9

# MATSim event "type" attribute string -> TorchDNL integer event code.
# "PersonEntersVehicle"/"PersonLeavesVehicle" have no TorchDNL equivalent
# (vehicle assignment, not link-level movement) and are dropped.
# EVT_ENTERED_BUFFER (queue model's capacity-buffer transition) has no
# standard MATSim public event and is never emitted from the MATSim side.
MATSIM_EVENT_TYPE_MAP: dict[str, int] = {
    "actend": EVT_ACTEND,
    "departure": EVT_DEPARTURE,
    "vehicle enters traffic": EVT_ENTERS_TRAFFIC,
    "left link": EVT_LEFT_LINK,
    "entered link": EVT_ENTERED_LINK,
    "vehicle leaves traffic": EVT_LEAVES_TRAFFIC,
    "arrival": EVT_ARRIVAL,
    "actstart": EVT_ACTSTART,
    "stuckAndAbort": EVT_STUCKANDABORT,
}


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def parse_matsim_events(
    events_path: str,
    link_id_to_idx: dict[str, int],
    person_id_to_idx: dict[str, int] | None = None,
) -> tuple[list[tuple[float, int, int, int]], dict[str, int]]:
    """Parse a MATSim events.xml(.gz) into TorchDNL-native event tuples.

    Args:
        events_path: path to MATSim's output_events.xml or output_events.xml.gz.
        link_id_to_idx: link string ID -> edge index, from `scenario_loader.parse_network`
            run on the *same* network file used for the MATSim run.
        person_id_to_idx: optional pre-built person ID -> agent index map (e.g. from
            `parse_fixed_route_population` on the same population), so agent indices
            line up between the TorchDNL replay and the MATSim reference. If omitted,
            a mapping is built in first-seen order (still stable/deterministic, just
            not guaranteed to match another parse's agent ordering).

    Returns:
        events: list of (time, evt_type_int, agent_idx, edge_idx) tuples, unsorted
            (callers typically only bin/histogram these, but sort by time first if
            order matters, matching TorchDNL.get_events()'s (time, event_type) sort).
        person_id_to_idx: the mapping actually used (either the one passed in, or the
            newly built one), so callers can build a compatible mapping for the
            TorchDNL side.
    """
    person_id_to_idx = {} if person_id_to_idx is None else dict(person_id_to_idx)
    events: list[tuple[float, int, int, int]] = []

    context = ET.iterparse(_open_maybe_gzip(events_path), events=("end",))
    for _, elem in context:
        if elem.tag != "event":
            continue

        type_str = elem.get("type")
        evt_type = MATSIM_EVENT_TYPE_MAP.get(type_str)
        if evt_type is None:
            elem.clear()
            continue

        person_str = elem.get("person") or elem.get("vehicle")
        if person_str is None:
            elem.clear()
            continue
        agent_idx = person_id_to_idx.setdefault(person_str, len(person_id_to_idx))

        link_str = elem.get("link")
        edge_idx = link_id_to_idx.get(link_str, -1) if link_str is not None else -1

        time_val = float(elem.get("time"))
        events.append((time_val, evt_type, agent_idx, edge_idx))
        elem.clear()

    return events, person_id_to_idx


@dataclass
class FixedRouteScenario:
    """Literal per-agent routes extracted from a converged/fixed MATSim population,
    ready to feed directly into `TorchDNL(paths_flat=..., path_offsets=...)`."""

    departure_times: np.ndarray  # [A] int32, seconds
    num_legs: np.ndarray  # [A] int32
    act_end_times: np.ndarray  # [A, MaxActs] int32, -1 if missing
    act_durations: np.ndarray  # [A, MaxActs] int32, -1 if missing
    paths_flat: np.ndarray  # [TotalEdges] int32, -2 marks leg boundaries
    path_offsets: np.ndarray  # [A+1] int64
    person_id_to_idx: dict[str, int] = field(default_factory=dict)


def parse_fixed_route_population(
    pop_file: str,
    link_id_to_idx: dict[str, int],
) -> FixedRouteScenario:
    """Parse a MATSim population XML with pre-assigned `<route>` link sequences
    (e.g. a converged, lastIteration=0-replayed plans file) into literal CSR paths.

    Mirrors `scenario_loader.parse_population`'s person/plan/leg selection logic
    (selected plan only, car-mode legs only) but keeps the *entire* route link
    sequence instead of only the first/last link, since here routes are fixed
    ground truth to replay, not OD endpoints for TorchDNL's own route search.
    """

    def time_to_sec(t_str: str) -> int:
        h, m, s = map(int, t_str.split(":"))
        return h * 3600 + m * 60 + s

    person_id_to_idx: dict[str, int] = {}
    departure_times: list[int] = []
    num_legs: list[int] = []
    act_end_times_per_agent: list[list[int]] = []
    act_durations_per_agent: list[list[int]] = []
    path_segments_per_agent: list[list[np.ndarray]] = []  # per agent, per leg, edge-index array

    context = ET.iterparse(pop_file, events=("end",))
    for _, elem in context:
        if elem.tag != "person":
            continue

        person_id = elem.get("id")
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
        leg_paths: list[np.ndarray] = []
        act_end_times: list[int] = []
        act_durations: list[int] = []

        for el in selected_plan:
            if el.tag in ("act", "activity"):
                end_time_str = el.get("end_time")
                duration_str = el.get("duration")

                act_end = time_to_sec(end_time_str) if end_time_str else -1
                act_dur = time_to_sec(duration_str) if duration_str else -1

                if first_act:
                    first_dep_time = act_end if act_end >= 0 else max(act_dur, 0)
                    first_act = False
                else:
                    act_end_times.append(act_end)
                    act_durations.append(act_dur)

            elif el.tag == "leg":
                if el.get("mode") != "car":
                    continue
                route_tag = el.find("route")
                route_str = route_tag.text.strip() if (route_tag is not None and route_tag.text) else None
                if not route_str:
                    continue

                link_ids = route_str.split(" ")
                try:
                    edge_idxs = np.array([link_id_to_idx[lid] for lid in link_ids], dtype=np.int32)
                except KeyError:
                    # A link in this route isn't in the (car-only) network — skip this leg.
                    continue
                leg_paths.append(edge_idxs)

        elem.clear()

        if not leg_paths:
            continue

        num_boundaries = len(leg_paths) - 1
        act_end_times = act_end_times[:num_boundaries]
        act_durations = act_durations[:num_boundaries]

        person_id_to_idx[person_id] = len(person_id_to_idx)
        departure_times.append(first_dep_time)
        num_legs.append(len(leg_paths))
        act_end_times_per_agent.append(act_end_times)
        act_durations_per_agent.append(act_durations)
        path_segments_per_agent.append(leg_paths)

    num_agents = len(path_segments_per_agent)
    max_acts = max((len(a) for a in act_end_times_per_agent), default=0)

    act_end_times_arr = np.full((num_agents, max_acts), -1, dtype=np.int32)
    act_durations_arr = np.full((num_agents, max_acts), -1, dtype=np.int32)
    for i, (ends, durs) in enumerate(zip(act_end_times_per_agent, act_durations_per_agent)):
        if ends:
            act_end_times_arr[i, : len(ends)] = ends
            act_durations_arr[i, : len(durs)] = durs

    flat_chunks: list[np.ndarray] = []
    path_offsets = np.zeros(num_agents + 1, dtype=np.int64)
    boundary = np.array([-2], dtype=np.int32)
    for i, legs in enumerate(path_segments_per_agent):
        agent_chunks: list[np.ndarray] = []
        for j, leg in enumerate(legs):
            if j > 0:
                agent_chunks.append(boundary)
            agent_chunks.append(leg)
        agent_path = np.concatenate(agent_chunks) if agent_chunks else np.array([], dtype=np.int32)
        flat_chunks.append(agent_path)
        path_offsets[i + 1] = path_offsets[i] + len(agent_path)

    paths_flat = np.concatenate(flat_chunks) if flat_chunks else np.array([], dtype=np.int32)

    return FixedRouteScenario(
        departure_times=np.array(departure_times, dtype=np.int32),
        num_legs=np.array(num_legs, dtype=np.int32),
        act_end_times=act_end_times_arr,
        act_durations=act_durations_arr,
        paths_flat=paths_flat,
        path_offsets=path_offsets,
        person_id_to_idx=person_id_to_idx,
    )


_EVENT_NAME_TO_INT = {
    "actend": EVT_ACTEND,
    "departure": EVT_DEPARTURE,
    "enters_traffic": EVT_ENTERS_TRAFFIC,
    "left_link": EVT_LEFT_LINK,
    "entered_link": EVT_ENTERED_LINK,
    "leaves_traffic": EVT_LEAVES_TRAFFIC,
    "arrival": EVT_ARRIVAL,
    "actstart": EVT_ACTSTART,
    "stuckAndAbort": EVT_STUCKANDABORT,
    "entered_buffer": EVT_ENTERED_BUFFER,
}


def parse_torchdnl_events_csv(
    csv_path: str,
    link_id_to_idx: dict[str, int],
) -> list[tuple[float, int, int, int]]:
    """Read back a TorchDNL events CSV (as written by
    `tamarl.rl.render_helper.write_events_csv` / `scripts/run_torchdnl_fixed_routes.py`)
    into the same `(time, evt_type_int, agent_idx, edge_idx)` tuple format
    `parse_matsim_events` produces, so both sides can be fed identically into
    `plot_histogram._compute_leg_histogram_data` or per-link GEH binning.

    Agent indices are recovered directly from the "agent_{idx}" person string
    (TorchDNL's own convention), so they already match the original TorchDNL run.
    """
    events: list[tuple[float, int, int, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evt_type = _EVENT_NAME_TO_INT.get(row["type"])
            if evt_type is None:
                continue
            agent_idx = int(row["person"].removeprefix("agent_"))
            edge_idx = link_id_to_idx.get(row["link"], -1) if row["link"] else -1
            events.append((float(row["time"]), evt_type, agent_idx, edge_idx))
    return events
