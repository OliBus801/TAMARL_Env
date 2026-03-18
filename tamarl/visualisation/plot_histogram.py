import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Event type constants (mirrored from dnl_matsim.py)
EVT_DEPARTURE = 1
EVT_ARRIVAL = 6
EVT_STUCKANDABORT = 8


def _sec_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))


def _compute_leg_histogram_data(events, max_steps, dt=1.0, bucket_size_sec=300):
    """
    Compute histogram data from events.
    
    Returns:
        bins (np.array): Bin edges in seconds.
        dep_counts, arr_counts, stuck_counts, en_route_counts (np.array): Counts per bin.
    """
    max_time_sec = max_steps * dt

    departure_times = np.array([t * dt for (t, evt, a, e) in events if evt == EVT_DEPARTURE])
    arrival_times = np.array([t * dt for (t, evt, a, e) in events if evt == EVT_ARRIVAL])
    stuck_times = np.array([t * dt for (t, evt, a, e) in events if evt == EVT_STUCKANDABORT])

    bins = np.arange(0, max_time_sec + bucket_size_sec, bucket_size_sec)

    dep_counts, _ = np.histogram(departure_times, bins=bins)
    arr_counts, _ = np.histogram(arrival_times, bins=bins)
    stuck_counts, _ = np.histogram(stuck_times, bins=bins) if len(stuck_times) > 0 else (np.zeros(len(bins) - 1, dtype=int), None)

    cum_dep = np.cumsum(dep_counts)
    cum_arr = np.cumsum(arr_counts)
    cum_stuck = np.cumsum(stuck_counts)
    en_route_counts = cum_dep - cum_arr - cum_stuck

    return bins, dep_counts, arr_counts, stuck_counts, en_route_counts


def plot_leg_histogram(events, max_steps, dt=1.0, bucket_size_sec=300, output_file='leg_histogram.png'):
    """
    Plots the number of departures, arrivals, and en-route agents over time,
    using simulation events as the data source.

    Args:
        events (list): List of event tuples (time_step, event_type, agent_id, edge_id).
        max_steps (int): Total duration of simulation in steps.
        dt (float): Simulation timestep in seconds (to convert steps to real time).
        bucket_size_sec (int): Size of the time bucket in seconds for histogram.
        output_file (str): Path to save the plot.
    """
    bins, dep_counts, arr_counts, stuck_counts, en_route_counts = _compute_leg_histogram_data(
        events, max_steps, dt, bucket_size_sec)

    max_time_sec = max_steps * dt
    x_axis = bins[:-1] / 3600.0
    max_hour = int(max_time_sec // 3600) + 1

    plt.figure(figsize=(12, 6))
    plt.step(x_axis, dep_counts, where='post', label='Departures', color='red', linewidth=2, alpha=0.7)
    plt.step(x_axis, arr_counts, where='post', label='Arrivals', color='blue', linewidth=2, alpha=0.7)
    plt.step(x_axis, en_route_counts, where='post', label='En Route', color='green', linewidth=2, alpha=0.7)
    plt.xlabel('Time (Hours)')
    plt.ylabel(f'Agents per {bucket_size_sec}s')
    plt.xticks(np.arange(0, max_hour, 1))
    plt.title('Agent Status Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Plot saved to {output_file}")


def export_leg_histogram_csv(events, max_steps, dt=1.0, bucket_size_sec=300, output_file='leg_histogram.csv'):
    """
    Export leg histogram data to CSV.

    Args:
        events (list): List of event tuples (time_step, event_type, agent_id, edge_id).
        max_steps (int): Total duration of simulation in steps.
        dt (float): Simulation timestep in seconds.
        bucket_size_sec (int): Size of the time bucket in seconds.
        output_file (str): Path to save the CSV.
    """
    bins, dep_counts, arr_counts, stuck_counts, en_route_counts = _compute_leg_histogram_data(
        events, max_steps, dt, bucket_size_sec)

    time_seconds = bins[:-1].astype(int)
    time_hms = [_sec_to_hms(s) for s in time_seconds]

    df = pd.DataFrame({
        'time': time_hms,
        'time_s': time_seconds,
        'departures': dep_counts,
        'arrivals': arr_counts,
        'stuck': stuck_counts,
        'en-route': en_route_counts,
    })

    df.to_csv(output_file, index=False, sep=';')
    print(f"Leg histogram CSV saved to {output_file}")


def plot_leg_histogram_from_arrays(start_steps, arrival_steps, max_steps, bucket_size_sec=300, output_file='leg_histogram.png'):
    """
    Legacy wrapper: builds fake events from arrays and calls the main function.
    
    Args:
        start_steps (np.array): Array of departure time steps for each agent.
        arrival_steps (np.array): Array of arrival time steps for each agent.
                                  Values <= 0 typically imply not arrived.
        max_steps (int): Total duration of simulation in steps (seconds).
        bucket_size_sec (int): Size of the time bucket in seconds for histogram.
        output_file (str): Path to save the plot.
    """
    events = []
    
    # Build departure events
    for i, t in enumerate(start_steps):
        events.append((int(t), EVT_DEPARTURE, i, -1))
    
    # Build arrival events (only valid ones)
    for i, t in enumerate(arrival_steps):
        if t > 0 and t <= max_steps:
            events.append((int(t), EVT_ARRIVAL, i, -1))
    
    plot_leg_histogram(events, max_steps, dt=1.0, bucket_size_sec=bucket_size_sec, output_file=output_file)
