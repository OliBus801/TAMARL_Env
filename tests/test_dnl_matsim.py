
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tamarl.core.dnl_matsim import TorchDNLMATSim

def test_basic_movement():
    print("Test 1: Basic Movement")
    # Network: 0 -> 1 -> 2
    # Edges: [Len, Speed, CapStorage, CapFlow, FFTime]
    # FFTime = 2 steps
    # CapFlow = 100/hr (~0.027/s) -> Make it high for this test
    # Let's say dt=1, CapFlow=3600 (1/s)
    
    edge_static = torch.tensor([
        [100, 10, 10, 3600, 2], # Edge 0
        [100, 10, 10, 3600, 2], # Edge 1
        [100, 10, 10, 3600, 2]  # Edge 2
    ], dtype=torch.float32)
    
    # 1 Agent
    # Path: 0 -> 1
    paths = torch.tensor([[0, 1]], dtype=torch.long)
    # Pad with -1 if MaxLen=3
    paths = torch.tensor([[0, 1, -1]], dtype=torch.long)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10)
    
    # Step 1: Agent should enter Edge 0
    dnl.step()
    print(f"Step 1 Status: {dnl.agent_state[0, 0]} (Expected 1)")
    print(f"Step 1 Arrival Time: {dnl.agent_state[0, 4]} (Expected 1+2=3)")
    
    # Step 2: Traveling
    dnl.step()
    print(f"Step 2 Status: {dnl.agent_state[0, 0]} (Expected 1)")
    
    # Step 3: Arrives at end of Edge 0 -> Buffer
    dnl.step()
    print(f"Step 3 Status: {dnl.agent_state[0, 0]} (Expected 2)")
    
    # Step 4: Moves to Edge 1
    dnl.step()
    print(f"Step 4 Status: {dnl.agent_state[0, 0]} (Expected 1)")
    print(f"Step 4 Current Edge: {dnl.agent_state[0, 1]} (Expected 1)")
    
    # Step 5, 6: Traveling Edge 1
    dnl.step()
    dnl.step() # Arrival at step 4+2=6?
    print(f"Step 6 Status: {dnl.agent_state[0, 0]} (Expected 2 - Buffer)")
    
    # Step 7: Moves to Exit (Path done)
    # Next edge -1
    dnl.step()
    print(f"Step 7 Status: {dnl.agent_state[0, 0]} (Expected 3 - Arrived)")
    
    metrics = dnl.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Expected distance: Length of Edge 0 (100) + Length of Edge 1 (100) = 200
    if abs(metrics['avg_travel_dist'] - 200.0) < 1e-5:
         print("Distance Metric Correct (200.0)")
    else:
         print(f"Distance Metric FAILED: {metrics['avg_travel_dist']}")

    print("Test 1 Passed if output matches expectations.\n")

def test_spillback():
    print("Test 2: Spillback")
    # Edge 0 -> Edge 1
    # Edge 1 has Storage Capacity = 1
    edge_static = torch.tensor([
        [100, 10, 10, 3600, 1], # Edge 0
        [100, 10, 1, 3600, 1]   # Edge 1 (Cap 1)
    ], dtype=torch.float32)
    
    # 2 Agents
    paths = torch.tensor([
        [0, 1, -1],
        [0, 1, -1]
    ], dtype=torch.long)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10)
    
    # Fill Edge 1
    # Cheat: set Edge 1 occupancy to 1
    dnl.edge_occupancy[1] = 1
    
    # Agents enter Edge 0
    dnl.step()
    # Travel
    dnl.step() # Arrival at 1+1=2. Status Buffer.
    
    print(f"Step 2 Status: {dnl.agent_state[:, 0]} (Expected [2, 2])")
    
    # Step 3: Try to move to Edge 1.
    # Edge 1 full. Should be blocked.
    dnl.step()
    print(f"Step 3 Status: {dnl.agent_state[:, 0]} (Expected [2, 2] - Still Buffer)")
    print(f"Step 3 Current Edge: {dnl.agent_state[:, 1]} (Expected [0, 0])")
    
    # Force Stuck? Need > 10 steps.
    # Let's free space.
    dnl.edge_occupancy[1] = 0
    dnl.step()
    print(f"Step 4 Status: {dnl.agent_state[:, 0]} (Expected [1, 1] or [1, 2] depending on capacity sharing)")
    # Both might move if storage cap allows 1? No, logic checks (N+avail).
    # Cap is 1. Avail is 1.
    # Ranks: 0, 1.
    # 0 < 1 -> Agent 0 moves.
    # 1 < 1 -> False. Agent 1 stays.
    
    print("Test 2 Passed if blocked then one moves.\n")

if __name__ == "__main__":
    test_basic_movement()
    test_spillback()
