"""Render helper for RL training episodes.

Converts DNL events (integer tuples) to CSV format compatible with
the visualization renderer, then calls render_animation or render_live.
"""

import os
import csv
import tempfile
from typing import Dict, Optional

from tamarl.core.dnl_matsim import TorchDNLMATSim, EVENT_TYPE_NAMES


def write_events_csv(
    dnl: TorchDNLMATSim,
    output_path: str,
    idx_to_link_id: Dict[int, str],
):
    """Write DNL events to a CSV file compatible with the renderer.

    Args:
        dnl: TorchDNLMATSim instance with track_events=True
        output_path: path to write the events CSV
        idx_to_link_id: reverse map from edge index → link string ID
    """
    events = dnl.get_events()
    if not events:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'type', 'person', 'link', 'extra'])

        for evt in events:
            time_val = evt[0]
            evt_type_int = int(evt[1])
            agent_id = int(evt[2])
            edge_id = int(evt[3])

            evt_name = EVENT_TYPE_NAMES.get(evt_type_int, f'unknown_{evt_type_int}')
            link_str = idx_to_link_id.get(edge_id, str(edge_id))
            person_str = f'agent_{agent_id}'

            writer.writerow([time_val, evt_name, person_str, link_str, ''])

    return True


def render_episode(
    scenario_path: str,
    dnl: TorchDNLMATSim,
    idx_to_link_id: Dict[int, str],
    episode: int,
    fmt: str = 'gif',
    output_dir: Optional[str] = None,
    render_fps: int = 5,
    render_hours: Optional[tuple] = None,
    render_speed: int = 1,
    filename: Optional[str] = None,
):
    """Render a completed episode's events to gif/mp4 or show live.

    Args:
        scenario_path: path to the scenario folder (contains network XML)
        dnl: TorchDNLMATSim instance after episode completion
        idx_to_link_id: reverse map edge_idx → link string ID
        episode: episode number (for naming the output file)
        fmt: 'gif', 'mp4', or 'live'
        output_dir: where to store rendered files (default: scenario_path/renders/)
    """
    from tamarl.visualisation.renderer import render_animation, render_live

    if output_dir is None:
        output_dir = os.path.join(scenario_path, 'renders')
    os.makedirs(output_dir, exist_ok=True)

    time_range = None
    if render_hours is not None:
        time_range = (render_hours[0] * 3600.0, render_hours[1] * 3600.0)

    # Write events to a system temp CSV
    fd, events_csv_path = tempfile.mkstemp(suffix='.csv', prefix=f'events_ep{episode}_')
    os.close(fd)
    has_events = write_events_csv(dnl, events_csv_path, idx_to_link_id)

    if not has_events:
        print(f"⚠️  No events for episode {episode}, skipping render.")
        try:
            os.remove(events_csv_path)
        except OSError:
            pass
        return

    if fmt == 'live':
        render_live(scenario_path, output_dir, time_range=time_range, initial_speed=render_speed, events_file=events_csv_path)
    else:
        if filename:
            output_path = os.path.join(output_dir, f"{filename}.{fmt}")
        else:
            output_path = os.path.join(output_dir, f'episode_{episode}.{fmt}')
            
        render_animation(
            scenario_path,      # scenario_folder (positional)
            output_dir,         # output_folder (positional)
            output_path=output_path,
            fmt=fmt,
            fps=render_fps,
            dpi=100,
            time_range=time_range,
            speed=render_speed,
            events_file=events_csv_path,
        )

    # Clean up the events CSV after rendering
    try:
        os.remove(events_csv_path)
    except OSError:
        pass
