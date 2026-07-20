#!/bin/bash
set -e

echo "=== TrafficGym Demo GIF Generator ==="
echo "Running a 3x3 grid world scenario and exporting events..."

# Ensure we're in the repository root
cd "$(dirname "$0")/.."

# 1. Run benchmark to export events
PYTHONPATH=. python tests/benchmark_torchdnl.py "tamarl/data/scenarios/shockwave/simple" --population 2000 --save_events

# 2. Render GIF
echo "Generating GIF visualization..."
python -m tamarl.visualisation --format gif --fps 60 --speed 20 --hours 7.5 8.5 "tamarl/data/scenarios/shockwave/simple" output

echo "Done! The GIF is saved at: tamarl/data/scenarios/shockwave/simple/output/simulation.gif"
