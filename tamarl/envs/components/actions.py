"""Action manager: builds action masks and applies agent actions."""

from typing import Dict, Tuple
import numpy as np
import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


class ActionManager:
    """Handles action masking and action application for the DTA environment.
    
    Action space: Discrete(max_out_degree) where each action corresponds to
    a local outgoing edge index from the agent's current node.
    """

    def __init__(self, dnl: TorchDNLMATSim):
        self.dnl = dnl

    def get_action_masks(self, deciding_agent_indices: torch.Tensor) -> Dict[str, np.ndarray]:
        """Build action masks for deciding agents.
        
        Args:
            deciding_agent_indices: tensor of agent indices that need decisions
            
        Returns:
            Dict mapping agent_id string to binary mask ndarray of shape (max_out_degree,)
        """
        masks = {}
        if deciding_agent_indices.numel() == 0:
            return masks

        # Get to_node of current edge for each deciding agent
        curr_edges = self.dnl.current_edge[deciding_agent_indices]
        curr_to_nodes = self.dnl.edge_endpoints[curr_edges, 1].long()

        # Get outgoing edges per node using pre-built lookup
        node_out = self.dnl.node_out_edges  # [N, max_out_degree]
        max_deg = self.dnl.max_out_degree

        for i, agent_idx in enumerate(deciding_agent_indices.tolist()):
            node = curr_to_nodes[i].item()
            out_edges = node_out[node]  # [max_out_degree], padded with -1
            mask = (out_edges != -1).cpu().numpy().astype(np.int8)
            masks[f"agent_{agent_idx}"] = mask

        return masks

    def get_action_mask_for_agent(self, agent_idx: int) -> np.ndarray:
        """Get action mask for a single agent.
        
        Args:
            agent_idx: integer index of the agent
            
        Returns:
            Binary mask ndarray of shape (max_out_degree,)
        """
        curr_edge = self.dnl.current_edge[agent_idx].item()
        node = self.dnl.edge_endpoints[curr_edge, 1].item()
        out_edges = self.dnl.node_out_edges[node]
        return (out_edges != -1).cpu().numpy().astype(np.int8)

    def apply_actions(self, action_dict: Dict[str, int]) -> Dict[str, dict]:
        """Apply agent actions: map local action index → global edge ID → dnl.next_edge.
        
        Args:
            action_dict: mapping of agent_id string → local action index
            
        Returns:
            Dict of per-agent info dicts (e.g., warnings for invalid actions)
        """
        infos = {}
        for agent_id, action in action_dict.items():
            agent_idx = int(agent_id.split("_")[-1])
            info = {}

            curr_edge = self.dnl.current_edge[agent_idx].item()
            node = self.dnl.edge_endpoints[curr_edge, 1].item()
            out_edges = self.dnl.node_out_edges[node]
            n_valid = self.dnl.node_out_degree[node].item()

            if action < 0 or action >= n_valid:
                # Fallback to first valid edge
                info["invalid_action"] = True
                info["requested_action"] = action
                action = 0

            edge_id = out_edges[action].item()
            self.dnl.next_edge[agent_idx] = edge_id
            infos[agent_id] = info

        return infos

    def apply_actions_tensor(self, agent_indices: torch.Tensor, actions: torch.Tensor):
        """Vectorised action application for batched policies.
        
        Args:
            agent_indices: [K] tensor of agent indices
            actions: [K] tensor of local action indices
        """
        curr_edges = self.dnl.current_edge[agent_indices]
        nodes = self.dnl.edge_endpoints[curr_edges, 1].long()
        # Gather chosen edge from node_out_edges lookup
        chosen_edges = self.dnl.node_out_edges[nodes, actions.long()]
        self.dnl.next_edge[agent_indices] = chosen_edges
