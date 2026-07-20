# TrafficGym 🚦

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange)

![TrafficGym at scale](tamarl/data/scenarios/berlin/output/1pct_8to12/simulation-ezgif.com-crop.gif)

**TrafficGym** is a high-performance, GPU-accelerated Dynamic Traffic Assignment (DTA) environment designed specifically for Multi-Agent Reinforcement Learning (MARL). 

It features **TorchDNL**, a vectorized Dynamic Network Loading engine implemented entirely in PyTorch, which is capable of scaling to large city networks while retaining micro-level agent resolution.

![TrafficGym Demo](tamarl/data/scenarios/shockwave/simple/output/simulation.gif)
*A demonstration of the environment in a toy bottleneck scenario.*

## ✨ Features

- **Blazing Fast Simulation**: TorchDNL computes traffic dynamics directly on the GPU using sparse tensor operations, bypassing the overhead of traditional CPU-based simulators.
- **Gymnasium & PettingZoo Compatible**: Wrappers provide seamless integration with standard RL libraries.
- **Multiple Formulations**: Route traffic using different bandit formulations (Agent-Level, OD-Level, Centralized).
- **Extensible & Research-Ready**: Comes with benchmarking scripts, memory profilers, and diverse test scenarios (Grid Worlds, Sioux Falls, Los Angeles).
- **Visualization Suite**: Built-in visualizer to render your networks and evaluate traffic conditions.

## 🚀 Installation

TrafficGym is built on modern Python packaging.

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

3. **Install the package:**
   ```bash
   pip install -e .
   ```

## 🏁 Quick Start

TrafficGym acts like any other `Gymnasium` environment. Here's a quick example running a random agent on a 3x3 grid world:

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
