# TrafficGym 🚦

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange)

![TrafficGym at scale](tamarl/data/scenarios/berlin/output/1pct_8to12/simulation-ezgif.com-crop.gif)

**TrafficGym** is a GPU-accelerated Dynamic Traffic Assignment (DTA) environment and benchmark for massively multi-agent routing games.

It is built around **TorchDNL**, a vectorized Dynamic Network Loading engine implemented entirely in PyTorch tensor ops (no Python-level per-agent loops in the hot path), paired with an episodic Multi-Agent Multi-Armed Bandit (MAMAB) formulation of route choice. TorchDNL scales to metropolitan-scale networks and populations exceeding $10^6$ concurrently learning agents.

![TrafficGym Demo](tamarl/data/scenarios/shockwave/simple/output/simulation.gif)
*A demonstration of the environment in a toy bottleneck scenario.*

## ✨ Features

- **Blazing Fast Simulation**: TorchDNL computes traffic dynamics directly on the GPU using vectorized tensor operations, bypassing the overhead of traditional CPU-based simulators (MATSim, SUMO).
- **Gymnasium & PettingZoo Compatible**: Wrappers provide seamless integration with standard RL libraries.
- **Multiple Formulations**: Route traffic using different bandit formulations (Agent-Level, OD-Level, Centralized), or a sequential POMDP formulation for step-by-step MARL (IPPO/MAPPO).
- **Extensible & Research-Ready**: Comes with benchmarking scripts, memory profilers, and diverse test scenarios (synthetic grid worlds, Sioux Falls, Berlin, Los Angeles, Braess, and more).
- **Visualization Suite**: Built-in renderer to animate network traffic and evaluate congestion.

## 🚀 Installation

TrafficGym requires **Python 3.10+**.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnonymousAuthors/TrafficGym.git
   cd TrafficGym
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   conda create -n trafficgym python=3.10
   conda activate trafficgym
   ```

3. **Install the package (editable, with dev extras — pytest, ruff):**
   ```bash
   pip install -e .
   ```

## 🏁 Quick Start: Training via the CLI

The experiments reported in the paper are run through `tamarl.rl.train_bandit`, the one-shot bandit DTA training runner. Point it at any scenario folder under `tamarl/data/scenarios/` and pick an agent:

```bash
# Train UCB on a small synthetic 3x3 grid world (fast sanity check)
python -m tamarl.rl.train_bandit --scenario tamarl/data/scenarios/grid_world/3x3 --population 100 --agent ucb
```

Key flags (see `python -m tamarl.rl.train_bandit --help` for the full list):

| Flag | Meaning |
|---|---|
| `--scenario PATH` | Path to a scenario folder (network + population XML) |
| `--population FILTER` | Substring/token filter to select a population file (e.g. `100`, `5pct`, `100pct`) |
| `--agent {random,epsilon_greedy,ucb,aon,frank_wolfe,ts,exp3,msa,evo_swap,rd}` | Routing policy to train |
| `--formulation {agent,od_pair,centralized}` | How learning signal is aggregated across vehicles |
| `--top_k_paths N` | Size of the candidate-route action space per OD pair (default: 3) |
| `--episodes N` / `--max_steps N` | Training horizon (episodes) and per-episode simulation horizon (seconds) |
| `--device {cpu,cuda}` / `--seed N` | Device and random seed |
| `--wandb` / `--wandb_project` | Enable Weights & Biases logging |
| `--render {interval,end}` / `--render_format {gif,mp4,live}` | Render episodes to GIF/MP4/live view |
| `--reload_paths` | Force recomputation of the cached top-k candidate routes |
| `--sanity_checks` | Emit FFTT scatter / V-C histogram / regret violin plots at the end of training |

Every per-algorithm hyperparameter is also overridable from the CLI (e.g. `--ucb_c`, `--epsilon_start`/`--epsilon_end`/`--epsilon_decay`, `--exp3_eta`/`--exp3_gamma`, `--rd_beta`, `--ts_prior_std`/`--ts_env_std`).

### Reproducing paper experiments via JSON configs

The recommended way to reproduce a specific experiment is a JSON config under `tamarl/data/configs/` (see `default_config.json` for the full schema). Resolution order is: `train()` defaults → JSON config → explicit CLI flags (CLI always wins):

```bash
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_SF.json --agent ucb
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_Berlin.json --agent ts
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_LosAngeles.json --agent ts
```

### Sequential POMDP / Deep MARL training

For step-by-step (one decision per departure event) training with IPPO/MAPPO instead of the one-shot bandit formulation, use `scripts/train_marl.py`:

```bash
python scripts/train_marl.py --scenario tamarl/data/scenarios/sioux_falls --agent mappo
```

## 🗺️ Scenarios

All scenarios live under `tamarl/data/scenarios/`:

| Scenario | Description |
|---|---|
| `grid_world/{3x3, 8x8, 16x16, 32x32, 64x64, 128x128, ...}` | Synthetic Manhattan grid networks, for controlled scaling experiments (population and network-size stress tests) |
| `sioux_falls` | Classic small transportation-literature benchmark network |
| `berlin` | Real-world MATSim Berlin scenario (multiple population sample fractions) |
| `los-angeles` | Real-world MATSim Los Angeles scenario |
| `braess` | Braess's paradox toy network |
| `equil`, `equil_biased` | Small equilibrium-analysis toy networks |
| `shockwave` | Toy bottleneck network used for the demo GIF above |
| `toy_corridor`, `toy_corridor_2` | Minimal corridor networks for unit testing |
| `ingolstadt`, `leipzig`, `vulkaneifel`, `saint_arnoult`, `ortuzar_willumsen`, `Exemple_Bench` | Additional real/derived MATSim-style networks |

## 🐍 Using TrafficGym as a Library

For programmatic use (e.g. embedding TrafficGym in another RL codebase), the lower-level `Gymnasium`-style API is available directly:

```python
import torch
from tamarl.envs.dta_bandit_env import DTABanditEnv
from tamarl.envs.agent_level_wrapper import AgentLevelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Instantiate the base bandit environment
bandit = DTABanditEnv(
    scenario_path="tamarl/data/scenarios/grid_world/3x3",
    population_filter="100",
    device=device,
)

# 2. Wrap it for RL (Agent-level formulation)
env = AgentLevelWrapper(bandit=bandit, top_k=3)

# 3. Standard RL loop
obs, info = env.reset()
done = False

while not done:
    # Random actions for demonstration
    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}

    obs, rewards, terminations, truncations, infos = env.step(actions)

    done = all(terminations.values()) or all(truncations.values())

print("Simulation complete! ✅")
```

## 🧪 Tests and Benchmarks

```bash
# Full test suite
pytest

# Benchmark the core DNL engine against MATSim/SUMO
PYTHONPATH=. python tests/benchmark_matsim_dnl.py tamarl/data/scenarios/grid_world/3x3 --population 100
```

## 📊 Generating the Demo GIF

To generate the demonstration GIF yourself, run the included script:
```bash
./scripts/generate_demo_gif.sh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use TrafficGym in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2025trafficgym,
  title={TrafficGym: A GPU-Accelerated Dynamic Traffic Assignment Environment for MARL},
  author={Anonymous Authors},
  booktitle={Under Double-Blind Review},
  year={2025}
}
```
