"""Core rendering logic for TAMARL_Env simulation visualization.

Parses the network XML and events CSV, reconstructs per-agent states at each
timestep, and renders an animated GIF or MP4.
"""

import os
import glob
import re
import xml.etree.ElementTree as ET
import csv
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from collections import defaultdict
from tqdm import tqdm


# ─── Agent visual states ───────────────────────────────────────────────────────
STATE_WAITING     = 'waiting'
STATE_DEPARTED    = 'departed'      # Shown for exactly 1 frame (ping), then gone
STATE_TRAVELING   = 'traveling'
STATE_BUFFER      = 'buffer'
STATE_ARRIVED     = 'arrived'       # Shown for exactly 1 frame (ping), then gone
STATE_STUCK       = 'stuck'         # Shown for exactly 1 frame (ping), then gone
STATE_GONE        = 'gone'          # Not drawn

STATE_COLORS = {
    STATE_WAITING:   '#F5C542',   # warm yellow
    STATE_DEPARTED:  '#9B59B6',   # purple
    STATE_TRAVELING: '#4A90D9',   # blue
    STATE_BUFFER:    '#E67E22',   # orange
    STATE_ARRIVED:   '#2ECC71',   # green
    STATE_STUCK:     '#E74C3C',   # red
}

# Margin fraction: traveling agents use [MARGIN, 1-MARGIN] of the link
# so they never sit exactly on a node.
LINK_MARGIN = 0.10


# ─── Network parsing ──────────────────────────────────────────────────────────

def parse_network(scenario_folder: str):
    """Parse a MATSim network XML from the scenario folder.

    Returns:
        nodes: dict  node_id -> {'x': float, 'y': float}
        links: dict  link_id -> {'from': node_id, 'to': node_id,
                                  'length': float, 'freespeed': float}
    """
    candidates = glob.glob(os.path.join(scenario_folder, '*network*.xml'))
    if not candidates:
        raise FileNotFoundError(
            f"No *network*.xml found in {scenario_folder}"
        )
    network_file = candidates[0]
    print(f"📄 Parsing network: {os.path.basename(network_file)}")

    nodes = {}
    links = {}

    tree = ET.parse(network_file)
    root = tree.getroot()

    for node_elem in root.iter('node'):
        nid = node_elem.get('id')
        nodes[nid] = {
            'x': float(node_elem.get('x')),
            'y': float(node_elem.get('y')),
        }

    for link_elem in root.iter('link'):
        lid = link_elem.get('id')
        length = float(link_elem.get('length'))
        freespeed = float(link_elem.get('freespeed'))
        capacity_h = float(link_elem.get('capacity'))  # veh/h
        lanes = float(link_elem.get('permlanes', '1'))

        # Derived metrics
        ff_time = length / freespeed if freespeed > 0 else 0  # seconds
        flow_cap_s = capacity_h / 3600.0                      # veh/s
        eff_cell_size = 7.5
        storage_cap = (length * lanes) / eff_cell_size        # veh
        storage_cap = max(storage_cap, flow_cap_s)             # at least flow cap
        storage_cap = max(storage_cap, ff_time * flow_cap_s)   # at least ff flux

        links[lid] = {
            'from': link_elem.get('from'),
            'to': link_elem.get('to'),
            'length': length,
            'freespeed': freespeed,
            'ff_time': ff_time,
            'flow_cap': flow_cap_s,
            'storage_cap': storage_cap,
        }

    print(f"   → {len(nodes)} nodes, {len(links)} links")
    return nodes, links


# ─── Events parsing ───────────────────────────────────────────────────────────

def parse_events(output_folder: str):
    """Parse a MATSim-style events CSV from the output folder.

    Returns:
        events: list of dicts sorted by time
        max_time: float
    """
    candidates = glob.glob(os.path.join(output_folder, '*events*.csv'))
    if not candidates:
        raise FileNotFoundError(
            f"No *events*.csv found in {output_folder}"
        )
    events_file = candidates[0]
    print(f"📄 Parsing events: {os.path.basename(events_file)}")

    events = []
    with open(events_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                'time': float(row['time']),
                'type': row['type'].strip(),
                'person': row['person'].strip(),
                'link': row['link'].strip(),
                'extra': row.get('extra', '').strip(),
            })

    events.sort(key=lambda e: e['time'])
    max_time = max(e['time'] for e in events) if events else 0
    print(f"   → {len(events)} events, max_time={max_time}")
    return events, max_time


# ─── Agent state reconstruction ───────────────────────────────────────────────

def build_agent_timelines(events, max_time):
    """Reconstruct per-agent states at each integer timestep.

    Arrived and stuck agents are visible for exactly 1 frame (ping effect),
    then become STATE_GONE.

    Returns:
        agent_ids: sorted list of agent ids
        timelines: dict  agent_id -> list of state dicts per timestep
        total_steps: int
        per_step_stats: list of dicts per timestep with cumulative/delta counts
    """
    total_steps = int(max_time) + 2  # +1 for the arrival frame, +1 buffer

    agent_ids = sorted(set(e['person'] for e in events))

    # Group events by integer time
    events_by_time = defaultdict(list)
    for e in events:
        events_by_time[int(e['time'])].append(e)

    # Current state tracker per agent
    current = {}
    for aid in agent_ids:
        current[aid] = {
            'state': STATE_GONE,  # Not yet departed
            'link': None,
            'enter_time': 0.0,
        }

    # Cumulative counters
    cumulative_arrived = 0
    cumulative_stuck = 0
    cumulative_departed = 0

    timelines = {aid: [] for aid in agent_ids}
    per_step_stats = []

    for t in range(total_steps):
        # First: 1-frame ping states → GONE
        for aid in agent_ids:
            if current[aid]['state'] in (STATE_ARRIVED, STATE_STUCK):
                current[aid]['state'] = STATE_GONE
            elif current[aid]['state'] == STATE_DEPARTED:
                # After departure ping, agent waits until enters_traffic
                current[aid]['state'] = STATE_WAITING

        # Track new events at this timestep
        new_arrived = 0
        new_stuck = 0
        new_departed = 0

        # Process events at this timestep
        for e in events_by_time.get(t, []):
            aid = e['person']
            etype = e['type']

            if etype == 'actend':
                current[aid]['state'] = STATE_WAITING
                current[aid]['link'] = e['link']

            elif etype == 'departure':
                # Ping: show departure for 1 frame
                current[aid]['state'] = STATE_DEPARTED
                current[aid]['link'] = e['link']
                new_departed += 1
                cumulative_departed += 1

            elif etype == 'enters_traffic':
                # Agent enters capacity buffer of first link
                current[aid]['state'] = STATE_BUFFER
                current[aid]['link'] = e['link']
                current[aid]['enter_time'] = t

            elif etype == 'entered_link':
                # Agent starts traveling on a new link
                current[aid]['state'] = STATE_TRAVELING
                current[aid]['link'] = e['link']
                current[aid]['enter_time'] = t

            elif etype == 'left_link':
                # Agent leaves a link → momentarily in buffer at end
                current[aid]['state'] = STATE_BUFFER
                current[aid]['link'] = e['link']

            elif etype == 'entered_buffer':
                # Agent moved from traveling to capacity buffer
                current[aid]['state'] = STATE_BUFFER
                current[aid]['link'] = e['link']

            elif etype in ('leaves_traffic', 'arrival', 'actstart'):
                # Only count arrival once per agent
                if current[aid]['state'] != STATE_ARRIVED:
                    current[aid]['state'] = STATE_ARRIVED
                    current[aid]['link'] = e['link']
                    new_arrived += 1
                    cumulative_arrived += 1

            elif etype == 'stuckAndAbort':
                if current[aid]['state'] != STATE_STUCK:
                    current[aid]['state'] = STATE_STUCK
                    current[aid]['link'] = e['link']
                    new_stuck += 1
                    cumulative_stuck += 1

        # Count active agents by state
        active_counts = defaultdict(int)
        for aid in agent_ids:
            st = current[aid]['state']
            if st != STATE_GONE:
                active_counts[st] += 1

        per_step_stats.append({
            'waiting': active_counts[STATE_WAITING],
            'traveling': active_counts[STATE_TRAVELING],
            'buffer': active_counts[STATE_BUFFER],
            'departed_total': cumulative_departed,
            'arrived_total': cumulative_arrived,
            'stuck_total': cumulative_stuck,
            'new_departed': new_departed,
            'new_arrived': new_arrived,
            'new_stuck': new_stuck,
        })

        # Snapshot current state
        for aid in agent_ids:
            st = current[aid]
            timelines[aid].append({
                'state': st['state'],
                'link': st['link'],
                'enter_time': st['enter_time'],
            })

    return agent_ids, timelines, total_steps, per_step_stats


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _link_geometry(nodes, link):
    """Get start/end coords for a link."""
    n_from = nodes[link['from']]
    n_to = nodes[link['to']]
    return n_from['x'], n_from['y'], n_to['x'], n_to['y']


def _interpolate_on_link(x0, y0, x1, y1, progress):
    """Interpolate position along a link. progress in [0, 1]."""
    p = min(max(progress, 0.0), 1.0)
    return x0 + p * (x1 - x0), y0 + p * (y1 - y0)


def _perpendicular_offset(x0, y0, x1, y1, offset):
    """Compute a perpendicular offset vector from the link direction."""
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return 0.0, 0.0
    nx = -dy / length
    ny = dx / length
    return nx * offset, ny * offset


# ─── Rendering ─────────────────────────────────────────────────────────────────

def _compute_agent_positions(agent_ids, timelines, timestep, nodes, links):
    """Compute (x, y, color, state) for each visible agent at a timestep.

    Handles overlap by fanning out agents at the same position.
    """
    raw_positions = []

    for aid in agent_ids:
        if timestep >= len(timelines[aid]):
            continue
        st = timelines[aid][timestep]

        if st['state'] in (STATE_GONE,):
            continue

        link_id = st['link']
        if link_id is None or link_id not in links:
            continue

        link = links[link_id]
        x0, y0, x1, y1 = _link_geometry(nodes, link)

        if st['state'] == STATE_WAITING:
            # Waiting at the origin node of their link
            px, py = x0, y0

        elif st['state'] == STATE_DEPARTED:
            # Ping at origin node of their link
            px, py = x0, y0

        elif st['state'] == STATE_TRAVELING:
            # Interpolate along [MARGIN, 1-MARGIN] of the link so
            # traveling agents never sit exactly on a node.
            ff_time = link['length'] / link['freespeed'] if link['freespeed'] > 0 else 1.0
            elapsed = timestep - st['enter_time']
            raw_progress = elapsed / ff_time if ff_time > 0 else 1.0
            raw_progress = min(max(raw_progress, 0.0), 1.0)
            progress = LINK_MARGIN + raw_progress * (1.0 - 2 * LINK_MARGIN)
            px, py = _interpolate_on_link(x0, y0, x1, y1, progress)

        elif st['state'] == STATE_BUFFER:
            # At the end of the link (destination node)
            px, py = x1, y1

        elif st['state'] in (STATE_ARRIVED, STATE_STUCK):
            # Ping frame: show at end of link
            px, py = x1, y1

        else:
            continue

        color = STATE_COLORS.get(st['state'], '#888888')

        raw_positions.append({
            'aid': aid,
            'x': px, 'y': py,
            'color': color,
            'state': st['state'],
            'link': link_id,
        })

    # Fan out overlapping agents perpendicular to the link
    pos_groups = defaultdict(list)
    for p in raw_positions:
        key = (round(p['x'], 4), round(p['y'], 4))
        pos_groups[key].append(p)

    result = []
    avg_link_len = np.mean([l['length'] for l in links.values()]) if links else 10
    spacing = avg_link_len * 0.06

    for (gx, gy), group in pos_groups.items():
        n = len(group)
        if n == 1:
            result.append(group[0])
        else:
            link_id = group[0]['link']
            link = links[link_id]
            x0, y0, x1, y1 = _link_geometry(nodes, link)

            for i, p in enumerate(group):
                offset = (i - (n - 1) / 2.0) * spacing
                ox, oy = _perpendicular_offset(x0, y0, x1, y1, offset)
                p['x'] += ox
                p['y'] += oy
                result.append(p)

    return result


def _draw_frame(ax, nodes, links, agent_ids, timelines, timestep,
                per_step_stats, node_size, agent_size, scale_factor=1.0,
                text_scale_factor=1.0, show_labels=True):
    """Draw a single frame of the visualization."""
    ax.clear()
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal')

    # Compute bounds — use the max range for both axes so that 1D layouts
    # (e.g. all nodes on a horizontal line) get enough padding on both axes.
    all_x = [n['x'] for n in nodes.values()]
    all_y = [n['y'] for n in nodes.values()]
    range_x = max(all_x) - min(all_x) if len(all_x) > 1 else 10
    range_y = max(all_y) - min(all_y) if len(all_y) > 1 else 10
    span = max(range_x, range_y, 10)
    margin = span * 0.10 + 2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw links (directed arrows)
    link_lw = 1.5 * scale_factor
    arrow_scale = 12 * scale_factor
    link_fontsize = 5.5 * scale_factor * text_scale_factor
    for lid, link in links.items():
        x0, y0, x1, y1 = _link_geometry(nodes, link)
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='-|>',
                color='#4a4a6a',
                lw=link_lw,
                mutation_scale=arrow_scale,
            ),
        )
        # Link info labels at midpoint with small perpendicular offset
        if show_labels:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            label_offset = 0.3 * scale_factor  # Small offset in data units
            ox, oy = _perpendicular_offset(x0, y0, x1, y1, label_offset)

            d_val = link.get('flow_cap', 0)
            s_val = link.get('storage_cap', 0)
            ff_val = link.get('ff_time', 0)
            top_label = f"{d_val:.1f} v/s | {s_val:.0f} veh"
            bot_label = f"{ff_val:.0f}s"
            
            ax.text(mx + ox, my + oy, top_label, fontsize=link_fontsize, color='#7a7a9a',
                    ha='center', va='bottom', fontweight='light')
            ax.text(mx - ox, my - oy, bot_label, fontsize=link_fontsize, color='#7a7a9a',
                    ha='center', va='top', fontweight='light')

    # Draw nodes
    node_fontsize = 7 * scale_factor
    node_edge_w = 1.2 * scale_factor
    for nid, node in nodes.items():
        ax.plot(node['x'], node['y'], 'o',
                color='#e0e0e0', markersize=node_size,
                markeredgecolor='#8888aa', markeredgewidth=node_edge_w,
                zorder=5)
        if show_labels:
            ax.text(node['x'], node['y'], nid,
                    fontsize=node_fontsize, color='#1a1a2e', ha='center', va='center',
                    fontweight='bold', zorder=6)

    # Draw agents
    positions = _compute_agent_positions(agent_ids, timelines, timestep, nodes, links)

    for p in positions:
        # Ping effect for 1-frame states: glow ring
        if p['state'] in (STATE_ARRIVED, STATE_STUCK, STATE_DEPARTED):
            ax.plot(p['x'], p['y'], 'o',
                    color=p['color'], markersize=agent_size * 2.0,
                    alpha=0.3, zorder=9)
            ax.plot(p['x'], p['y'], 'o',
                    color=p['color'], markersize=agent_size * 1.4,
                    alpha=0.6, zorder=9)

        ax.plot(p['x'], p['y'], 'o',
                color=p['color'], markersize=agent_size,
                alpha=1.0,
                markeredgecolor='white', markeredgewidth=0.5 * scale_factor,
                zorder=10)

        # Draw agent ID inside dot
        if show_labels:
            aid_str = p['aid']
            m = re.search(r'(\d+)', aid_str)
            label = str(int(m.group(1))) if m else aid_str
            agent_fontsize = agent_size * 0.35
            ax.text(p['x'], p['y'], label,
                    fontsize=agent_fontsize, color='white',
                    ha='center', va='center', fontweight='bold',
                    zorder=11)

    # ─── HUD (multi-color, rendered with fig.text) ──────────────────────────
    stats = per_step_stats[timestep] if timestep < len(per_step_stats) else {}

    waiting    = stats.get('waiting', 0)
    traveling  = stats.get('traveling', 0)
    buffer_n   = stats.get('buffer', 0)
    departed_t = stats.get('departed_total', 0)
    arrived_t  = stats.get('arrived_total', 0)
    stuck_t    = stats.get('stuck_total', 0)
    new_dep    = stats.get('new_departed', 0)
    new_arr    = stats.get('new_arrived', 0)
    new_stk    = stats.get('new_stuck', 0)

    # Helper to add a HUD section with optional (+N) indicator
    def _hud_cumulative(segments, state, label, total, new_count):
        segments.append(('● ', STATE_COLORS[state]))
        txt = f'{label} {total}'
        segments.append((txt, '#e0e0e0'))
        if new_count > 0:
            segments.append((f'  (+{new_count})', STATE_COLORS[state]))

    def _hud_dynamic(segments, state, label, count):
        segments.append(('● ', STATE_COLORS[state]))
        segments.append((f'{label} {count}', '#e0e0e0'))

    SEP = ('   |   ', '#666680')

    hud_segments = [
        (f't={timestep}', '#e0e0e0'),
        SEP,
    ]
    _hud_dynamic(hud_segments, STATE_WAITING, 'Waiting', waiting)
    hud_segments.append(SEP)
    _hud_cumulative(hud_segments, STATE_DEPARTED, 'Departed', departed_t, new_dep)
    hud_segments.append(SEP)
    _hud_dynamic(hud_segments, STATE_TRAVELING, 'Traveling', traveling)
    hud_segments.append(SEP)
    _hud_dynamic(hud_segments, STATE_BUFFER, 'Buffer', buffer_n)
    hud_segments.append(SEP)
    _hud_cumulative(hud_segments, STATE_ARRIVED, 'Arrived', arrived_t, new_arr)
    hud_segments.append(SEP)
    _hud_cumulative(hud_segments, STATE_STUCK, 'Stuck', stuck_t, new_stk)

    # Render as concatenated fig.text segments
    fig = ax.get_figure()
    for txt in list(fig.texts):
        txt.remove()

    fontsize = 11
    approx_char_width = 0.007
    full_text = ''.join(seg[0] for seg in hud_segments)
    total_width = len(full_text) * approx_char_width
    x_cursor = 0.5 - total_width / 2

    for text, color in hud_segments:
        fig.text(x_cursor, 0.96, text, fontsize=fontsize, color=color,
                 fontfamily='monospace', va='top', ha='left',
                 transform=fig.transFigure)
        x_cursor += len(text) * approx_char_width

    # Clean axes
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ─── Public API ────────────────────────────────────────────────────────────────

def _filter_events_by_time(events, max_time, time_range):
    """Filter events to a time range and rebase timestamps to start from 0."""
    if time_range is None:
        return events, max_time

    t_start, t_end = time_range
    filtered = [e for e in events if t_start <= e['time'] < t_end]
    # Rebase timestamps
    for e in filtered:
        e['time'] = e['time'] - t_start
    new_max = min(max_time, t_end) - t_start
    print(f"⏱️  Time filter: {t_start/3600:.0f}h–{t_end/3600:.0f}h → {len(filtered)} events, {new_max:.0f}s")
    return filtered, new_max


def render_animation(scenario_folder: str, output_folder: str,
                     output_path: str = None, fmt: str = 'gif', scale_factor: float = 1.0,
                     text_scale_factor: float = 1.0, show_labels: bool = True,
                     fps: int = 5, dpi: int = 150, fade_steps: int = 5,
                     time_range: tuple = None):
    """Generate a GIF or MP4 animation of a simulation.

    Args:
        scenario_folder: Path containing *network*.xml
        output_folder: Path containing *events*.csv
        output_path: Output file path; defaults to output_folder/simulation.<fmt>
        fmt: 'gif' or 'mp4'
        fps: Frames per second
        dpi: Resolution
        fade_steps: (unused, kept for CLI compat)
        time_range: Optional (start_sec, end_sec) tuple to filter events
    """
    matplotlib.use('Agg')  # Non-interactive backend for file rendering

    if output_path is None:
        output_path = os.path.join(output_folder, f"simulation.{fmt}")

    print(f"🎬 TAMARL_Env Visualization")
    print(f"   Scenario: {scenario_folder}")
    print(f"   Output:   {output_path}")
    print()

    # 1. Parse
    nodes, links = parse_network(scenario_folder)
    events, max_time = parse_events(output_folder)

    if not events:
        print("⚠️  No events found. Nothing to render.")
        return

    # Filter by time range
    events, max_time = _filter_events_by_time(events, max_time, time_range)
    if not events:
        print("⚠️  No events in the specified time range.")
        return

    # 2. Build agent timelines
    agent_ids, timelines, total_steps, per_step_stats = build_agent_timelines(
        events, max_time
    )
    print(f"👥 {len(agent_ids)} agents, {total_steps} timesteps")

    # Trim trailing frames where all agents are gone
    last_visible = 0
    for t in range(total_steps):
        for aid in agent_ids:
            if t < len(timelines[aid]) and timelines[aid][t]['state'] != STATE_GONE:
                last_visible = t
                break
    total_steps = last_visible + 1
    print(f"   Trimmed to {total_steps} visible timesteps")

    # 3. Adaptive sizing (doubled defaults) + scale_factor
    num_nodes = len(nodes)
    if num_nodes <= 10:
        base_node, base_agent, base_fig = 28, 22, (10, 7)
    elif num_nodes <= 50:
        base_node, base_agent, base_fig = 18, 14, (12, 9)
    else:
        base_node, base_agent, base_fig = 10, 8, (16, 12)

    node_size = base_node * scale_factor
    agent_size = base_agent * scale_factor
    # Scale figure dimensions so larger elements have room
    fig_w = base_fig[0] * max(1.0, scale_factor ** 0.5)
    fig_h = base_fig[1] * max(1.0, scale_factor ** 0.5)

    # 4. Create animation
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#1a1a2e')

    pbar = tqdm(total=total_steps, desc='🎞️  Rendering', unit='frame',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def animate(frame):
        _draw_frame(ax, nodes, links, agent_ids, timelines, frame,
                    per_step_stats, node_size, agent_size, scale_factor,
                    text_scale_factor, show_labels)
        pbar.update(1)

    print(f"\n🎞️  Rendering {total_steps} frames at {fps} FPS...")
    anim = FuncAnimation(fig, animate, frames=total_steps, interval=1000 // fps)

    if fmt == 'gif':
        print("💾 Saving as GIF...")
        writer = PillowWriter(fps=fps)
    elif fmt == 'mp4':
        print("💾 Saving as MP4...")
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi)
    pbar.close()
    plt.close(fig)

    file_size = os.path.getsize(output_path) / 1024
    print(f"\n✅ Saved to {output_path} ({file_size:.1f} KB)")


def render_live(scenario_folder: str, output_folder: str,
                scale_factor: float = 1.0, text_scale_factor: float = 1.0,
                show_labels: bool = True, initial_speed: int = 1,
                time_range: tuple = None):
    """Open an interactive matplotlib window for live simulation playback.

    Args:
        scenario_folder: Path containing *network*.xml
        output_folder: Path containing *events*.csv
        scale_factor: Scale factor for graph elements
        text_scale_factor: Scale factor for text labels
        show_labels: Whether to show text labels
        initial_speed: Initial timesteps per frame tick
        time_range: Optional (start_sec, end_sec) tuple to filter events
    """
    matplotlib.use('TkAgg')  # Interactive backend
    from matplotlib.widgets import Button, Slider

    print(f"🎬 TAMARL_Env Live Viewer")
    print(f"   Scenario: {scenario_folder}")
    print()

    # 1. Parse
    nodes, links = parse_network(scenario_folder)
    events, max_time = parse_events(output_folder)

    if not events:
        print("⚠️  No events found. Nothing to render.")
        return

    # Filter by time range
    events, max_time = _filter_events_by_time(events, max_time, time_range)
    if not events:
        print("⚠️  No events in the specified time range.")
        return

    # 2. Build agent timelines
    agent_ids, timelines, total_steps, per_step_stats = build_agent_timelines(
        events, max_time
    )
    print(f"👥 {len(agent_ids)} agents, {total_steps} timesteps")

    # Trim trailing frames
    last_visible = 0
    for t in range(total_steps):
        for aid in agent_ids:
            if t < len(timelines[aid]) and timelines[aid][t]['state'] != STATE_GONE:
                last_visible = t
                break
    total_steps = last_visible + 1
    print(f"   Trimmed to {total_steps} visible timesteps")

    # 3. Adaptive sizing
    num_nodes = len(nodes)
    if num_nodes <= 10:
        base_node, base_agent, base_fig = 28, 22, (10, 7)
    elif num_nodes <= 50:
        base_node, base_agent, base_fig = 18, 14, (12, 9)
    else:
        base_node, base_agent, base_fig = 10, 8, (16, 12)

    node_size = base_node * scale_factor
    agent_size = base_agent * scale_factor
    fig_w = base_fig[0] * max(1.0, scale_factor ** 0.5)
    fig_h = base_fig[1] * max(1.0, scale_factor ** 0.5)

    # 4. Create figure with space for controls at the bottom
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 1.5), facecolor='#1a1a2e')
    fig.subplots_adjust(bottom=0.18)

    # State
    state = {'playing': False, 'current_frame': 0, 'speed': initial_speed, 'timer': None}

    def draw_current():
        _draw_frame(ax, nodes, links, agent_ids, timelines,
                    state['current_frame'], per_step_stats,
                    node_size, agent_size, scale_factor,
                    text_scale_factor, show_labels)
        fig.canvas.draw_idle()

    # ─── Widgets ────────────────────────────────────────────────────────────

    # Time scrubber slider
    ax_time = fig.add_axes([0.15, 0.08, 0.55, 0.03], facecolor='#2a2a4a')
    slider_time = Slider(ax_time, 't', 0, total_steps - 1,
                         valinit=0, valstep=1, color='#6c63ff')

    # Speed slider
    max_speed = max(10, total_steps // 100)
    ax_speed = fig.add_axes([0.15, 0.03, 0.35, 0.03], facecolor='#2a2a4a')
    slider_speed = Slider(ax_speed, 'Speed', 1, max_speed,
                          valinit=initial_speed, valstep=1, color='#ff6b6b')

    # Play/Pause button
    ax_btn = fig.add_axes([0.78, 0.03, 0.12, 0.05])
    btn_play = Button(ax_btn, '▶ Play', color='#2a2a4a', hovercolor='#4a4a6a')
    btn_play.label.set_color('white')

    def on_time_change(val):
        state['current_frame'] = int(val)
        draw_current()

    def on_speed_change(val):
        state['speed'] = int(val)

    def tick():
        """Advance one tick while playing."""
        if not state['playing']:
            return
        new_frame = state['current_frame'] + state['speed']
        if new_frame >= total_steps:
            new_frame = total_steps - 1
            state['playing'] = False
            btn_play.label.set_text('▶ Play')
        state['current_frame'] = new_frame
        slider_time.set_val(new_frame)
        # Schedule next tick (~30ms target, enough for smooth UI)
        if state['playing']:
            state['timer'] = fig.canvas.manager.window.after(30, tick)

    def on_play_pause(event):
        state['playing'] = not state['playing']
        if state['playing']:
            btn_play.label.set_text('⏸ Pause')
            # Reset to start if at end
            if state['current_frame'] >= total_steps - 1:
                state['current_frame'] = 0
                slider_time.set_val(0)
            tick()
        else:
            btn_play.label.set_text('▶ Play')
            if state['timer'] is not None:
                fig.canvas.manager.window.after_cancel(state['timer'])
                state['timer'] = None

    slider_time.on_changed(on_time_change)
    slider_speed.on_changed(on_speed_change)
    btn_play.on_clicked(on_play_pause)

    # Initial draw
    draw_current()
    print("🖥️  Live viewer ready. Close the window to exit.")
    plt.show()
