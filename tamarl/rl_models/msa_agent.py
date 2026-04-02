import numpy as np
import math
from tamarl.envs.components.time_dependent_dijkstra import build_adjacency_list, compute_td_shortest_paths

class MSAAgent:
    """Stochastic Method of Successive Averages (S-MSA) Agent.
    
    Approximates Dynamic User Equilibrium by updating paths with a decaying 
    probability alpha_k = alpha_min + (alpha_max - alpha_min) * exp(-lambda * k).
    """
    
    def __init__(self, num_agents, num_nodes, num_edges, edge_endpoints, ff_times, dt,
                 alpha_max=1.0, alpha_min=0.05, alpha_decay=0.05, seed=None):
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.k = 0
        
        self.dt = dt
        self.edge_endpoints_np = edge_endpoints.cpu().numpy()
        self.ff_times_np = ff_times.cpu().numpy()
        self.adj = build_adjacency_list(self.num_nodes, self.edge_endpoints_np)
        
        self.rng = np.random.default_rng(seed)
        
        # Mapping from node u to list of (outgoing) edge_ids connecting from u.
        # This matches env.node_to_edges ensuring actions align with environment action space!
        self.node_to_edges = {i: [] for i in range(self.num_nodes)}
        for e in range(self.num_edges):
            u = int(self.edge_endpoints_np[e, 0])
            self.node_to_edges[u].append(e)
            
        # agent_paths[agent_id][leg_idx] = list of edge indices.
        self.agent_paths = [[] for _ in range(self.num_agents)]
        
    def get_actions(self, observations, infos):
        actions = {}
        for agent_str, info in infos.items():
            mask = info.get("action_mask")
            if mask is None or mask.sum() == 0:
                continue
                
            obs = observations.get(agent_str)
            if obs is None:
                continue
                
            agent_id = int(agent_str.split("_")[-1])
            node = int(obs[0])
            leg = info.get("curr_leg", 0)
            
            valid_edges = self.node_to_edges[node]
            
            if len(valid_edges) == 0:
                actions[agent_str] = 0
                continue
                
            path = self.agent_paths[agent_id][leg]
            
            # Find which edge in the path starts at `node`
            chosen_edge = -1
            for e_id in path:
                if self.edge_endpoints_np[e_id, 0] == node:
                    chosen_edge = e_id
                    break
                    
            if chosen_edge != -1:
                # Find its index in valid_edges
                try:
                    action_idx = valid_edges.index(chosen_edge)
                except ValueError:
                    action_idx = 0
            else:
                # Fallback if agent is somehow off-path
                action_idx = 0
                
            actions[agent_str] = action_idx
            
        return actions

    def end_episode(self, dnl, is_init=False):
        """
        Updates the MSA routes. If is_init=True, computes using Free Flow times perfectly.
        Otherwise uses TT matrix measured during the episode.
        """
        num_legs = dnl.num_legs.cpu().numpy()
        all_first_edges = dnl._all_first_edges.cpu().numpy()
        dests = dnl.destinations.cpu().numpy()
        leg_departure_times = dnl.leg_departure_times.cpu().numpy()

        if is_init:
            # TT Matrix is just FF times repeated. Shape [1, num_edges] in seconds.
            tt_matrix = (self.ff_times_np * self.dt)[np.newaxis, :]
            alpha_k = 1.0 # 100% update to initialize paths
        else:
            tt_tensor = dnl.get_dynamic_link_travel_times()
            if tt_tensor is None:
                # Fallback to FF if no collection
                tt_matrix = (self.ff_times_np * self.dt)[np.newaxis, :]
            else:
                tt_matrix = tt_tensor.cpu().numpy()
                
            # Decay alpha
            alpha_k = self.alpha_min + (self.alpha_max - self.alpha_min) * math.exp(-self.alpha_decay * self.k)
            self.k += 1
            
        # Collect all routing queries across all agents and legs
        start_times_list = []
        origins_list = []
        destinations_list = []
        query_map = [] # stores (agent_id, leg_idx) for each query
        
        for agent_id in range(self.num_agents):
            if is_init:
                # Initialize empty legs lists
                self.agent_paths[agent_id] = [[] for _ in range(num_legs[agent_id])]
                
            for leg_idx in range(num_legs[agent_id]):
                first_edge = all_first_edges[agent_id, leg_idx]
                origin_node = int(self.edge_endpoints_np[first_edge, 1])
                dest_node = int(dests[agent_id, leg_idx])
                
                if origin_node == dest_node:
                    continue
                    
                dep_time_sec = float(leg_departure_times[agent_id, leg_idx] * self.dt)
                
                start_times_list.append(dep_time_sec)
                origins_list.append(origin_node)
                destinations_list.append(dest_node)
                query_map.append((agent_id, leg_idx))
                
        if len(query_map) == 0:
            return
            
        _, paths = compute_td_shortest_paths(
            adj=self.adj,
            start_times=np.array(start_times_list, dtype=np.float32),
            origin_nodes=np.array(origins_list, dtype=np.int32),
            destination_nodes=np.array(destinations_list, dtype=np.int32),
            tt_matrix=tt_matrix,
            interval=dnl.link_tt_interval if getattr(dnl, "collect_link_tt", False) else 300.0
        )
        
        # Stochastic Path Replacement
        for q_idx, (agent_id, leg_idx) in enumerate(query_map):
            new_path = paths[q_idx]
            
            if is_init or self.rng.random() < alpha_k:
                self.agent_paths[agent_id][leg_idx] = new_path
