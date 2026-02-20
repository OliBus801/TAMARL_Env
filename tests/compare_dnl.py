import os
import argparse
import subprocess
import pandas as pd
import sys

def run_benchmark(script, root_folder, population, output_folder, scale_factor=1.0):
    cmd = [
        sys.executable, script,
        root_folder,
        "--population", population,
        "--output_folder", output_folder,
        "--scale_factor", str(scale_factor),
        "--n_hours", "24",
        "--save_pickle"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    print(f"Running {script}...")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        return False
    return True

def compare_results(matsim_out, gemsim_out, population):
    matsim_file = os.path.join(matsim_out, f"{population}_average_metrics.csv")
    gemsim_file = os.path.join(gemsim_out, f"{population}_average_metrics.csv")
    
    if not os.path.exists(matsim_file) or not os.path.exists(gemsim_file):
        print("Error: Results files not found for comparison.")
        return

    df_m = pd.read_csv(matsim_file)
    df_g = pd.read_csv(gemsim_file)
    
    print("\n--- 📊 Comparison Results (MATSim vs GEMSim) ---")
    
    metrics = ['avg_trav_dist', 'avg_trav_time', 'avg_trav_speed', 'compute_time', 'peak_memory', 'peak_vram']
    
    for m in metrics:
        val_m = df_m[m].iloc[0]
        val_g = df_g[m].iloc[0]
        diff = val_g - val_m
        diff_pct = (diff / val_m) * 100 if val_m != 0 else 0.0
        
        print(f"{m:<15}: MATSim={val_m:.2f} | GEMSim={val_g:.2f} | Diff={diff:+.2f} ({diff_pct:+.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", help="Root folder containing network.xml and population.xml")
    parser.add_argument("--population", help="Population filter (e.g. '10000')", required=True)
    parser.add_argument("--scale_factor", help="Scale factor", default=1.0)
    
    args = parser.parse_args()
    
    root = args.root_folder
    pop = args.population
    
    out_m = os.path.join(root, "output_matsim_bench")
    out_g = os.path.join(root, "output_gemsim_bench")
    
    # Run MATSim Benchmark
    # Using existing benchmark_matsim_dnl.py
    # Assuming it is in tests/benchmark_matsim_dnl.py relative to execution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_m = os.path.join(base_dir, "benchmark_matsim_dnl.py")
    script_g = os.path.join(base_dir, "benchmark_gemsim_dnl.py")
    
    if run_benchmark(script_m, root, pop, "output_matsim_bench", args.scale_factor):
        if run_benchmark(script_g, root, pop, "output_gemsim_bench", args.scale_factor):
            compare_results(out_m, out_g, pop)
