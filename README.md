# TAMARL_Env
Modular multi-agent reinforcement learning environment for traffic assignment using PettingZoo, PyTorch, and torch_geometric. Features vectorized BPR network loading, advanced action masking, and Stable-Baselines3 compatibility. Designed as a flexible foundation for dynamic traffic assignment and MARL research.

## Visualization
- Use `env.render(mode="human")` to open a dark-themed Matplotlib view of the network with animated agent markers and a live mean-reward HUD. `mode="ansi"` keeps the text-only output.
- Matplotlib is an optional dependency; install it with `pip install matplotlib` if you want the graphical renderer.
