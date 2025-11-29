import torch
from torch_geometric.data import Data

from tamarl.core.network import build_out_edges_per_node, map_action_to_edge


def build_sample_graph():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    data = Data(num_nodes=3, edge_index=edge_index)
    return data


def test_out_edges_metadata():
    data = build_sample_graph()
    meta = build_out_edges_per_node(data)
    assert meta.out_edges_per_node == [[0, 1], [2], [3]]
    assert meta.max_out_degree == 2


def test_action_mapping_valid():
    data = build_sample_graph()
    meta = build_out_edges_per_node(data)
    assert map_action_to_edge(0, 0, meta.out_edges_per_node) == 0
    assert map_action_to_edge(0, 1, meta.out_edges_per_node) == 1
    assert map_action_to_edge(1, 0, meta.out_edges_per_node) == 2


def test_action_mapping_invalid():
    data = build_sample_graph()
    meta = build_out_edges_per_node(data)
    try:
        map_action_to_edge(1, 1, meta.out_edges_per_node)
    except ValueError:
        return
    assert False, "Expected ValueError for invalid action"
