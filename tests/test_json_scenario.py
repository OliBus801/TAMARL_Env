import json
import torch

from torch_geometric.data import Data
from tamarl.core.network import load_scenario_from_json

def test_load_scenario_from_json(tmp_path):
    # Temporary JSON file
    scenario_path = tmp_path / "scenario.json"

    json_data = {
    "network" :
    {
        "nodes": [
            {"id": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "x": 1.0, "y": 0.0},
            {"id": 2, "x": 1.0, "y": 1.0}
        ],
        "edges": [
            {"source": 0, "target": 1, "capacity": 100.0, "freeflow_travel_time": 1.0},
            {"source": 1, "target": 2, "capacity": 80.0, "freeflow_travel_time": 1.2},
            {"source": 0, "target": 2, "capacity": 60.0, "freeflow_travel_time": 1.5}
        ]
    },
    "demand" : 
    {
        "0, 1": 10,
        "0, 2": 5,
        "1, 2": 8
    }
}
    
    with open(scenario_path, "w") as f:
        json.dump(json_data, f)

    network_data, agents = load_scenario_from_json(scenario_path)

    assert type(network_data) == Data
    assert network_data.num_nodes == 3
    assert network_data.num_edges == 3
    assert network_data.edge_index.shape == (2, 3)
    assert network_data.edge_attr.shape == (3, 2)  # freeflow_time and capacity
    assert network_data.edge_attr.shape[0] == network_data.num_edges

    assert agents.shape == torch.Size([2, 23])  # Total demand is 23 agents
    assert agents.dtype == torch.int64
    