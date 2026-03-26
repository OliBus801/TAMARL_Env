import argparse
import os
import random
import numpy as np
from typing import Tuple, List

def write_network_xml(filename: str, size: int, block_size: float = 1000.0,
                      freespeed: float = 13.9, capacity: float = 2000.0, permlanes: float = 1.0):
    """
    Generates a generic grid network and writes it to a MATSim network XML file.
    
    Args:
        filename: Output path for the network XML.
        size: Number of nodes along one dimension (creating a size x size grid).
        block_size: Distance between adjacent nodes in meters.
        freespeed: Free speed in m/s.
        capacity: Link capacity in veh/h.
        permlanes: Number of permanent lanes.
    """
    print(f"Generating Network: {size}x{size} grid...")
    
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n')
        f.write('<network name="grid_world">\n')
        f.write('\t<nodes>\n')

        # Generate Nodes
        # IDs will be "x_y" assuming 0-indexed integer coordinates
        for x in range(size):
            for y in range(size):
                node_id = f"{x}_{y}"
                # Coordinates: x corresponds to Easting, y to Northing
                coord_x = x * block_size
                coord_y = y * block_size
                f.write(f'\t\t<node id="{node_id}" x="{coord_x}" y="{coord_y}" />\n')
        
        f.write('\t</nodes>\n')
        f.write('\t<links>\n')

        # Generate Links
        # Bidirectional links between (x, y) and neighbors (x+1, y) and (x, y+1)
        # Using Manhattan grid logic
        
        link_count = 0
        def write_link(u, v):
            nonlocal link_count
            link_count += 1
            # link id format: from_to
            f.write(f'\t\t<link id="{u}-{v}" from="{u}" to="{v}" length="{block_size}" freespeed="{freespeed}" capacity="{capacity}" permlanes="{permlanes}" oneway="1" modes="car" />\n')

        for x in range(size):
            for y in range(size):
                u = f"{x}_{y}"
                
                # Right neighbor
                if x + 1 < size:
                    v = f"{x+1}_{y}"
                    write_link(u, v)
                    write_link(v, u)
                
                # Top neighbor
                if y + 1 < size:
                    v = f"{x}_{y+1}"
                    write_link(u, v)
                    write_link(v, u)
                    
        f.write('\t</links>\n')
        f.write('</network>\n')
    
    print(f"Network generated with {size*size} nodes and {link_count} links.")


def write_population_xml(filename: str, num_agents: int, size: int, block_size: float = 1000.0,
                        seed: int = 42,
                        o_mean: Tuple[float, float] = None, o_std: float = None,
                        d_mean: Tuple[float, float] = None, d_std: float = None):
    """
    Generates a population with O/D distributed normally (hotspots) and writes to MATSim population XML.
    
    Args:
        filename: Output path for the population XML.
        num_agents: Total number of agents to generate.
        size: Network grid size (N).
        block_size: Distance between nodes.
        seed: Random seed.
        o_mean: (mean_x, mean_y) for Origins in grid local coords (0 to size). If None, randomize.
        o_std: Standard deviation for Origins in grid local coords.
        d_mean: (mean_x, mean_y) for Destinations in grid local coords.
        d_std: Standard deviation for Destinations.
    """
    print(f"Generating Population: {num_agents} agents...")
    np.random.seed(seed)
    random.seed(seed)
    
    # Defaults if not provided (centering logic)
    # E.g. Origin hotspot bottom-left, Destination hotspot top-right, or user-defined
    if o_mean is None:
        o_mean = (size * 0.25, size * 0.25)
    if d_mean is None:
        d_mean = (size * 0.75, size * 0.75)
    if o_std is None:
        o_std = size * 0.15
    if d_std is None:
        d_std = size * 0.15
        
    # Vectorized generation of coords
    # These are in "grid index" space initially, need to clip and scale
    
    # Origins
    ox = np.random.normal(o_mean[0], o_std, num_agents)
    oy = np.random.normal(o_mean[1], o_std, num_agents)
    
    # Destinations
    dx = np.random.normal(d_mean[0], d_std, num_agents)
    dy = np.random.normal(d_mean[1], d_std, num_agents)
    
    # Clip to grid boundaries (Integers)
    # round() ensures we pick the nearest integer node
    # Then clip to [0, size-1] to stay in valid range
    oxi = np.clip(np.round(ox), 0, size - 1).astype(int)
    oyi = np.clip(np.round(oy), 0, size - 1).astype(int)
    dxi = np.clip(np.round(dx), 0, size - 1).astype(int)
    dyi = np.clip(np.round(dy), 0, size - 1).astype(int)
    
    # Departure Times
    t0_mean = 8.0 * 3600
    t0_std = 3600.0 / 2.0
    t1_mean = 17.0 * 3600
    t1_std = 3600.0 / 2.0
    
    dep_am = np.random.normal(t0_mean, t0_std, num_agents)
    dep_am = np.maximum(0, dep_am)
    
    dep_pm = np.random.normal(t1_mean, t1_std, num_agents)
    dep_pm = np.maximum(dep_pm, dep_am + 4 * 3600) 
    
    # Helper to get valid INCOMING links for a node (x, y)
    def get_incoming_links(nx, ny):
        # Returns list of link IDs ENDING at node (nx, ny)
        links = []
        # Check neighbors that could have links TO nx, ny
        # Neighbor at Left (nx-1, ny) -> Right to me
        if nx - 1 >= 0: links.append(f"{nx-1}_{ny}-{nx}_{ny}")
        # Neighbor at Right (nx+1, ny) -> Left to me
        if nx + 1 < size: links.append(f"{nx+1}_{ny}-{nx}_{ny}")
        # Neighbor Below (nx, ny-1) -> Up to me
        if ny - 1 >= 0: links.append(f"{nx}_{ny-1}-{nx}_{ny}")
        # Neighbor Above (nx, ny+1) -> Down to me
        if ny + 1 < size: links.append(f"{nx}_{ny+1}-{nx}_{ny}")
        return links

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n')
        f.write('<population>\n')
        
        for i in range(num_agents):
            pid = f"agent_{i}"
            
            # Origin Indices (Node where we want to be)
            cx_o, cy_o = oxi[i], oyi[i]
            cx_d, cy_d = dxi[i], dyi[i]
            
            # Select Activity Links (Incoming to the node)
            o_links = get_incoming_links(cx_o, cy_o)
            d_links = get_incoming_links(cx_d, cy_d)
            
            # Fallback
            if not o_links or not d_links:
                continue 
                
            home_link = random.choice(o_links)
            work_link = random.choice(d_links)
            
            # Extract Link Head/Tail
            def parse_link(lid):
                parts = lid.split('-')
                u_str, v_str = parts[0], parts[1]
                ux, uy = map(int, u_str.split('_'))
                vx, vy = map(int, v_str.split('_'))
                return (ux, uy), (vx, vy)
            
            # --- Home->Work Trip ---
            # Start: End of home_link = (cx_o, cy_o)
            # End: Must traverse work_link. So target is StartNode(work_link)
            ((wl_start_x, wl_start_y), _) = parse_link(work_link)
            
            # Generate Link Sequence from (cx_o, cy_o) to (wl_start_x, wl_start_y)
            link_route = []
            
            curr_x, curr_y = cx_o, cy_o
            target_x, target_y = wl_start_x, wl_start_y
            
            # Move X
            step_x = 1 if target_x > curr_x else -1
            if curr_x != target_x:
                for x in range(curr_x, target_x, step_x):
                    link_route.append(f"{x}_{curr_y}-{x+step_x}_{curr_y}")
                curr_x = target_x
            
            # Move Y
            step_y = 1 if target_y > curr_y else -1
            if curr_y != target_y:
                for y in range(curr_y, target_y, step_y):
                    link_route.append(f"{curr_x}_{y}-{curr_x}_{y+step_y}")
            
            full_route_am = [home_link] + link_route + [work_link]
            route_str_am = " ".join(full_route_am)
            
            # --- Work->Home Trip ---
            # Start: End of work_link = (cx_d, cy_d)
            # End: Must traverse home_link. Target is StartNode(home_link)
            ((hl_start_x, hl_start_y), _) = parse_link(home_link)
            
            link_route_pm = []
            curr_x, curr_y = cx_d, cy_d
            target_x, target_y = hl_start_x, hl_start_y
            
            # Move X
            step_x = 1 if target_x > curr_x else -1
            if curr_x != target_x:
                for x in range(curr_x, target_x, step_x):
                    link_route_pm.append(f"{x}_{curr_y}-{x+step_x}_{curr_y}")
                curr_x = target_x
            
            # Move Y
            step_y = 1 if target_y > curr_y else -1
            if curr_y != target_y:
                for y in range(curr_y, target_y, step_y):
                    link_route_pm.append(f"{curr_x}_{y}-{curr_x}_{y+step_y}")
            
            full_route_pm = [work_link] + link_route_pm + [home_link]
            route_str_pm = " ".join(full_route_pm)
            
            # Times
            def fmt_time(s):
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"
            
            start_am = fmt_time(dep_am[i])
            start_pm = fmt_time(dep_pm[i])
            
            # XML
            f.write(f'\t<person id="{pid}">\n')
            f.write('\t\t<plan selected="yes">\n')
            
            # Act Home
            f.write(f'\t\t\t<act type="home" link="{home_link}" x="{cx_o*block_size:.1f}" y="{cy_o*block_size:.1f}" end_time="{start_am}" />\n')
            
            # Leg to Work
            f.write('\t\t\t<leg mode="car">\n')
            f.write(f'\t\t\t\t<route type="links" start_link="{home_link}" end_link="{work_link}">{route_str_am}</route>\n')
            f.write('\t\t\t</leg>\n')
            
            # Act Work
            f.write(f'\t\t\t<act type="work" link="{work_link}" x="{cx_d*block_size:.1f}" y="{cy_d*block_size:.1f}" end_time="{start_pm}" />\n')
            
            # Leg to Home
            f.write('\t\t\t<leg mode="car">\n')
            f.write(f'\t\t\t\t<route type="links" start_link="{work_link}" end_link="{home_link}">{route_str_pm}</route>\n')
            f.write('\t\t\t</leg>\n')
            
            # Act Home
            f.write(f'\t\t\t<act type="home" link="{home_link}" x="{cx_o*block_size:.1f}" y="{cy_o*block_size:.1f}" />\n')
            
            f.write('\t\t</plan>\n')
            f.write('\t</person>\n')
            
        f.write('</population>\n')
    
    print(f"Population generated for {num_agents} agents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MATSim Grid Scenario")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output XMLs")
    parser.add_argument("--size", type=int, default=32, help="Grid size N (size x size nodes)")
    parser.add_argument("--agents", type=int, default=10000, help="Number of agents")
    parser.add_argument("--prefix", type=str, default="grid", help="Filename prefix")
    parser.add_argument("--block_size", type=float, default=500.0, help="Distance between nodes in meters")
    parser.add_argument("--capacity", type=float, default=2000.0, help="Capacity of links")
    parser.add_argument("--freespeed", type=float, default=13.89, help="Free speed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Hotspot control
    parser.add_argument("--std_dev", type=float, default=None, help="Standard deviation as fraction of size (default 0.15 * size)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    net_file = os.path.join(args.output_dir, f"{args.prefix}_{args.size}x{args.size}_network.xml")
    pop_file = os.path.join(args.output_dir, f"{args.prefix}_{args.size}x{args.size}_{args.agents}_population.xml")
    
    write_network_xml(net_file, args.size, block_size=args.block_size, 
                      freespeed=args.freespeed, capacity=args.capacity)
    
    # Calculate means for O/D to be in corners roughly
    # O = bottom-leftish (25% corner)
    # D = top-rightish (75% corner)
    
    s = args.size
    std = args.std_dev if args.std_dev else s * 0.15
    
    write_population_xml(pop_file, args.agents, args.size, block_size=args.block_size, seed=args.seed,
                         o_mean=(s*0.25, s*0.25), o_std=std,
                         d_mean=(s*0.75, s*0.75), d_std=std)
    
    print("Done.")