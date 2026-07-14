#!/bin/bash
set -e

echo "=== TrafficGym Demo GIF Generator ==="
echo "Running a 3x3 grid world scenario and exporting events..."

# Ensure we're in the repository root
cd "$(dirname "$0")/.."

# 1. Run benchmark to export events
PYTHONPATH=. python tests/benchmark_torchdnl.py tamarl/data/scenarios/grid_world/3x3 --population 100 --save_events

# 2. Render GIF
echo "Generating GIF visualization..."
python -m tamarl.visualisation --format gif --fps 15 --speed 5 tamarl/data/scenarios/grid_world/3x3 output

echo "Done! The GIF is saved at: tamarl/data/scenarios/grid_world/3x3/output/simulation.gif"
