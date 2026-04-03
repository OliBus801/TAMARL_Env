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
    paths = torch.tensor([[0, 1, -1]], dtype=torch.long)
    
    edge_endpoints = torch.tensor([
        [0, 1],
        [1, 2],
        [2, 3]
    ], dtype=torch.int32)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10, edge_endpoints=edge_endpoints)
    
    # Step 1: Agent should enter Edge 0 -> capacity buffer (Status 2) during schedule_demand
    dnl.step()
    
    status = dnl.status[0].item()
    curr_edge = dnl.current_edge[0].item()
    print(f"Step 1 Status: {status} (Expected 2)")
    print(f"Step 1 Current Edge: {curr_edge} (Expected 0)")
    assert status == 2, f"Failed: status {status} != 2"
    assert curr_edge == 0, f"Failed: current_edge {curr_edge} != 0"
    
    # Step 2: Agent moves to Edge 1 spatial buffer during process_nodes_A
    dnl.step()
    
    status = dnl.status[0].item()
    curr_edge = dnl.current_edge[0].item()
    arr_time = dnl.arrival_time[0].item()
    
    print(f"Step 2 Status: {status} (Expected 1)")
    print(f"Step 2 Current Edge: {curr_edge} (Expected 1)")
    print(f"Step 2 Arrival Time: {arr_time} (Expected 3)")
    assert status == 1, f"Failed: status {status} != 1"
    assert curr_edge == 1, f"Failed: current_edge {curr_edge} != 1"
    assert arr_time == 3, f"Failed: arrival_time {arr_time} != 3"
    
    # Step 3: Traveling
    dnl.step()
    status = dnl.status[0].item()
    print(f"Step 3 Status: {status} (Expected 1)")
    assert status == 1, f"Failed: status {status} != 1"
    
    # Step 4: Arrives at end of Edge 1 -> next step is -1, so it exits (Status 3)
    dnl.step()
    status = dnl.status[0].item()
    print(f"Step 4 Status: {status} (Expected 3)")
    assert status == 3, f"Failed: status {status} != 3"

    metrics = dnl.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Expected distance: Length of Edge 1 (100) = 100.0
    # Note: agents enter the network directly into the capacity buffer of their first link (Edge 0),
    # so they do not traverse its spatial buffer and its distance is not added.
    if abs(metrics['avg_travel_dist'] - 100.0) < 1e-5:
         print("Distance Metric Correct (100.0)")
    else:
         print(f"Distance Metric FAILED: {metrics['avg_travel_dist']}")
         assert False, "Distance Metric FAILED"

    print("Test 1 Passed\n")

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
    
    edge_endpoints = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.int32)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10, edge_endpoints=edge_endpoints)
    
    # Fill Edge 1
    # Cheat: set Edge 1 occupancy to 1
    dnl.edge_occupancy[1] = 1
    
    # Step 1: Schedule Demand -> Agents enter Edge 0 capacity buffer (Status 2)
    dnl.step()
    statuses = dnl.status.tolist()
    print(f"Step 1 Status: {statuses} (Expected [2, 2])")
    assert statuses == [2, 2], f"Failed: statuses {statuses} != [2, 2]"
    
    # Step 2: Try to move to Edge 1 spatial buffer. Blocked since it's full.
    dnl.step()
    statuses = dnl.status.tolist()
    edges = dnl.current_edge.tolist()
    print(f"Step 2 Status: {statuses} (Expected [2, 2] - Still Buffer)")
    print(f"Step 2 Current Edge: {edges} (Expected [0, 0])")
    assert statuses == [2, 2], f"Failed: statuses {statuses} != [2, 2]"
    assert edges == [0, 0], f"Failed: edges {edges} != [0, 0]"
    
    # Step 3: Free space, now one agent moves to Edge 1 spatial buffer
    dnl.edge_occupancy[1] = 0
    dnl.step()
    sorted_statuses = sorted(dnl.status.tolist())
    print(f"Step 3 Status: {sorted_statuses} (Expected [1, 2])")
    assert sorted_statuses == [1, 2], f"Failed: statuses {sorted_statuses} != [1, 2]"
    
    print("Test 2 Passed\n")

def test_capacity_flow():
    print("Test 3: Capacity Flow")
    # 2 Edges, flow cap = 2
    edge_static = torch.tensor([
        [100, 10, 100, 2, 2],
        [100, 10, 100, 2, 2]
    ], dtype=torch.float32)
    
    # 5 Agents
    paths = torch.tensor([[0, 1, -1] for _ in range(5)], dtype=torch.long)
    edge_endpoints = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10, edge_endpoints=edge_endpoints)
    
    # Step 1: 2 agents should enter buffer, 3 wait
    dnl.step()
    statuses = sorted(dnl.status.tolist())
    print(f"Step 1 Statuses: {statuses} (Expected [0, 0, 0, 2, 2])")
    assert statuses == [0, 0, 0, 2, 2], "Flow capacity limit failed at step 1"
    
    # Step 2: 2 more agents should enter
    dnl.step()
    statuses = sorted(dnl.status.tolist())
    # 2 are in spatial (1), 2 in buffer (2), 1 waiting (0)
    print(f"Step 2 Statuses: {statuses} (Expected [0, 1, 1, 2, 2])")
    assert statuses == [0, 1, 1, 2, 2], "Flow capacity limit failed at step 2"
    
    print("Test 3 Passed\n")

def test_force_spillback_entry():
    print("Test 4: Force Spillback Entry")
    edge_static = torch.tensor([
        [100, 10, 10, 3600, 1], # Edge 0
        [100, 10, 0, 3600, 1]   # Edge 1 (Cap 0 -> always full)
    ], dtype=torch.float32)
    paths = torch.tensor([[0, 1, -1]], dtype=torch.long)
    edge_endpoints = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    
    # stuck_threshold = 2
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=2, edge_endpoints=edge_endpoints)
    
    # Cheat: Fill Edge 1 unconditionally
    dnl.edge_occupancy[1] = 10
    
    # Step 1: Enters Edge 0 (status 2)
    dnl.step()
    assert dnl.status[0].item() == 2
    
    # Step 2: Stuck time = 1. Blocked!
    dnl.step()
    assert dnl.status[0].item() == 2
    assert dnl.current_edge[0].item() == 0
    
    # Step 3: Stuck time = 2. Still blocked!
    dnl.step()
    assert dnl.status[0].item() == 2
    
    # Step 4: Stuck time = 3. > threshold (2). Forced entry!
    dnl.step()
    assert dnl.status[0].item() == 1
    assert dnl.current_edge[0].item() == 1
    
    print("Test 4 Passed\n")

def test_disconnected_abort():
    print("Test 5: Disconnected Abort")
    edge_static = torch.tensor([
        [100, 10, 10, 3600, 1],
        [100, 10, 10, 3600, 1],
        [100, 10, 10, 3600, 1]
    ], dtype=torch.float32)
    
    # Path tries to jump from Edge 0 to Edge 2
    paths = torch.tensor([[0, 2, -1]], dtype=torch.long)
    
    # Edge endpoints: 0->1, 1->2, 2->3. Edge 0 -> Edge 2 is physically disconnected!
    # Because 0's to_node=1, 2's from_node=2.
    edge_endpoints = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10, edge_endpoints=edge_endpoints)
    
    dnl.step() # Enter edge 0 buffer
    assert dnl.status[0].item() == 2
    
    # Try to enter edge 2 downstream
    dnl.step() 
    # Must be disconnected and abort (status 3)
    assert dnl.status[0].item() == 3, "Agent did not abort on disconnected path"
    assert dnl.stuck_count == 1
    
    print("Test 5 Passed\n")

def test_event_tracking():
    print("Test 6: Event Tracking")
    # Same as Test 1
    edge_static = torch.tensor([
        [10, 10, 10, 3600, 1],
        [10, 10, 10, 3600, 1]
    ], dtype=torch.float32)
    paths = torch.tensor([[0, 1, -1]], dtype=torch.long)
    edge_endpoints = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    
    dnl = TorchDNLMATSim(edge_static, paths, device='cpu', stuck_threshold=10, edge_endpoints=edge_endpoints, track_events=True)
    
    for _ in range(5):
        dnl.step()
        
    events = dnl.get_events()
    assert len(events) >= 5, "Not enough events fired"
    event_types = [e[1] for e in events]
    assert 2 in event_types, "ENTERS_TRAFFIC missing"
    assert 4 in event_types, "ENTERED_LINK missing"
    assert 6 in event_types, "ARRIVAL missing"
    
    print("Test 6 Passed\n")


def test_rl_mode():
    print("Test 7: RL Mode")
    edge_static = torch.tensor([
        [10, 10, 10, 3600, 1],
        [10, 10, 10, 3600, 1]
    ], dtype=torch.float32)
    
    # RL mode: paths=None
    departure_times = torch.tensor([0], dtype=torch.int32)
    edge_endpoints = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    first_edges = torch.tensor([[0]], dtype=torch.long)
    destinations = torch.tensor([[2]], dtype=torch.long) # Dest node = 2
    
    dnl = TorchDNLMATSim(edge_static, paths=None, device='cpu', 
                         departure_times=departure_times, 
                         edge_endpoints=edge_endpoints,
                         stuck_threshold=10,
                         first_edges=first_edges, 
                         destinations=destinations)
    
    # Agent should enter first_edge (0)
    dnl.step()
    assert dnl.status[0].item() == 2
    assert dnl.current_edge[0].item() == 0
    assert dnl.next_edge[0].item() == -1
    
    # Environment supplies the next edge (1)
    dnl.next_edge[0] = 1
    
    # Agent moves to edge 1 spatial
    dnl.step()
    assert dnl.status[0].item() == 1
    assert dnl.current_edge[0].item() == 1
    
    # Agent reaches edge 1 end. As node is 2 (the destination), agent should exit!
    dnl.step()
    assert dnl.status[0].item() == 3
    
    print("Test 7 Passed\n")

if __name__ == "__main__":
    test_basic_movement()
    test_spillback()
    test_capacity_flow()
    test_force_spillback_entry()
    test_disconnected_abort()
    test_event_tracking()
    test_rl_mode()
