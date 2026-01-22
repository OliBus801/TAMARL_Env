import matplotlib.pyplot as plt
import numpy as np

def plot_agent_status(start_steps, arrival_steps, max_steps, bucket_size_sec=300, output_file='agent_status.png'):
    """
    Plots the number of departures, arrivals, and en-route agents over time.

    Args:
        start_steps (np.array): Array of departure time steps for each agent.
        arrival_steps (np.array): Array of arrival time steps for each agent. 
                                  Values <= 0 or > max_steps typically imply not arrived, 
                                  but logic here filters strictly positive valid arrivals.
        max_steps (int): Total duration of simulation in steps (seconds).
        bucket_size_sec (int): Size of the time bucket in seconds for histogram.
        output_file (str): Path to save the plot.
    """
    
    # 1. Define bins
    # Bins from 0 to max_steps
    bins = np.arange(0, max_steps + bucket_size_sec, bucket_size_sec)
    
    # 2. Departures Histogram
    # Count how many agents depart in each bin
    dep_counts, _ = np.histogram(start_steps, bins=bins)
    
    # 3. Arrivals Histogram
    # Filter valid arrivals (assuming arrival_steps > 0 means arrived)
    # Also filter if arrival > max_steps (arrived after simulation end)
    valid_arrivals = arrival_steps[(arrival_steps > 0) & (arrival_steps <= max_steps)]
    arr_counts, _ = np.histogram(valid_arrivals, bins=bins)
    
    # 4. En Route Calculation
    # En Route count at time T = Total Departed (<= T) - Total Arrived (<= T)
    # We want the count at the END of each bucket.
    
    # Cumulative counts up to the right edge of each bin
    # np.histogram returns count in [bin[i], bin[i+1])
    # So cumsum gives total count < bin[i+1]
    
    cum_dep = np.cumsum(dep_counts)
    cum_arr = np.cumsum(arr_counts)
    
    en_route_counts = cum_dep - cum_arr
    
    # 5. Plotting
    # Center the time points for plotting (mid-point of bucket) or end-point?
    # User asked for "interval specified". 
    # Convention: counts are for the interval. Plot at center or start.
    # Time axis in Hours
    x_axis = bins[:-1] / 3600.0  # Start of bucket in hours
    width_h = bucket_size_sec / 3600.0
    
    plt.figure(figsize=(12, 6))
    # Use step plots to represent counts over intervals
    plt.step(x_axis, dep_counts, where='post', label='Departures', color='red', linewidth=2, alpha=0.7)
    plt.step(x_axis, arr_counts, where='post', label='Arrivals', color='blue', linewidth=2, alpha=0.7)
    plt.step(x_axis, en_route_counts, where='post', label='En Route', color='green', linewidth=2, alpha=0.7)
    plt.xlabel('Time (Hours)')
    plt.ylabel(f'Agents per {bucket_size_sec}s')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    plt.title('Agent Status Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"Plot saved to {output_file}")
