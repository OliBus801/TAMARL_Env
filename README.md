# TAMARL_Env

Modular multi-agent reinforcement learning environment for dynamic traffic assignment. Built on **PyTorch**, it features a fully vectorized, highly optimized mesoscopic traffic simulator (`TorchDNLMATSim`) inspired by MATSim's queue model.

## 🚀 Features

- **GPU-Accelerated Simulation**: Fully tensorized operations for routing, traffic flow, and queueing capable of handling massive agent populations.
- **MATSim Queue Dynamics**: Accurately models downstream spatial buffers, capacity flow limits, and spillback/gridlock resolution (`stuck_threshold`).
- **Reinforcement Learning Mode**: Interactively inject dynamic `next_edge` routing decisions at each timestep without pre-calculated paths.
- **Event Tracking**: Optional output of standard MATSim telemetry (`DEPARTURE`, `ENTERED_LINK`, `ARRIVAL`, etc.) for downstream analytics.
- **PettingZoo & SB3 Ready**: Flexible foundation tailored for Multi-Agent Reinforcement Learning (MARL) research.

## 🏗️ Structure

- `tamarl/core/dnl_matsim.py`: The core `TorchDNLMATSim` PyTorch simulator engine.
- `tamarl/envs/`: Environment wrappers and scenario loading.
- `tamarl/data/scenarios/`: XML network and MATSim population definition files.
- `tamarl/visualisation/`: Tools for rendering networks and plotting simulation histograms.
- `tests/`: Simple Pytest suite and benchmarking tools.

## 💻 Usage

### 1. Basic Simulation (Static Paths)
```python
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim

dnl = TorchDNLMATSim(
    edge_static=edges_tensor,      # [length, free_speed, storage_cap, flow_cap, ff_time]
    paths=agent_paths,             # Pre-calculated routes
    edge_endpoints=endpoints,      # Network connectivity
    device='cuda'
)

for _ in range(num_steps):
    dnl.step()
```

### 2. Reinforcement Learning Mode (Dynamic Routing)
```python
dnl = TorchDNLMATSim(
    edge_static=edges_tensor, 
    paths=None,                    # Explicitly enables RL mode
    first_edges=starting_edges,
    destinations=target_nodes,
    edge_endpoints=endpoints,
    device='cuda'
)

dnl.step()

# Environment provides the next edge dynamically for routing agents
dnl.next_edge[agent_id] = chosen_next_edge 
```

### 3. Benchmarks & Testing

**Run Simulator Unit Tests:**
```bash
PYTHONPATH=. pytest tests/test_dnl_matsim.py
```

**Run Performance Benchmarks:**
```bash
PYTHONPATH=. python tests/benchmark_matsim_dnl.py tamarl/data/scenarios/grid_world/3x3 --population 100
```
