import argparse
import os
import random
import numpy as np

def write_network_xml(filename: str, size: int, block_size: float = 100.0,
                      freespeed: float = 13.9, capacity: float = 2000.0, permlanes: float = 1.0):
    print(f"Génération du réseau : {size}x{size} avec segments de {block_size}m...")
    
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n')
        f.write('<network name="random_grid_world">\n')
        f.write('\t<nodes>\n')

        for x in range(size):
            for y in range(size):
                f.write(f'\t\t<node id="{x}_{y}" x="{x * block_size}" y="{y * block_size}" />\n')
        
        f.write('\t</nodes>\n')
        f.write('\t<links>\n')

        def write_link(u, v):
            f.write(f'\t\t<link id="{u}-{v}" from="{u}" to="{v}" length="{block_size}" freespeed="{freespeed}" capacity="{capacity}" permlanes="{permlanes}" oneway="1" modes="car" />\n')

        for x in range(size):
            for y in range(size):
                u = f"{x}_{y}"
                if x + 1 < size:
                    v = f"{x+1}_{y}"
                    write_link(u, v)
                    write_link(v, u)
                if y + 1 < size:
                    v = f"{x}_{y+1}"
                    write_link(u, v)
                    write_link(v, u)
                    
        f.write('\t</links>\n')
        f.write('</network>\n')

def write_population_xml(filename: str, num_agents: int, size: int, block_size: float, duration_s: int):
    print(f"Génération de la population : {num_agents} agents...")
    
    oxi = np.random.randint(0, size, num_agents)
    oyi = np.random.randint(0, size, num_agents)
    dxi = np.random.randint(0, size, num_agents)
    dyi = np.random.randint(0, size, num_agents)
    
    dep_times = np.random.uniform(0, duration_s, num_agents)
    
    def get_incoming_links(nx, ny):
        links = []
        if nx - 1 >= 0: links.append(f"{nx-1}_{ny}-{nx}_{ny}")
        if nx + 1 < size: links.append(f"{nx+1}_{ny}-{nx}_{ny}")
        if ny - 1 >= 0: links.append(f"{nx}_{ny-1}-{nx}_{ny}")
        if ny + 1 < size: links.append(f"{nx}_{ny+1}-{nx}_{ny}")
        return links

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n')
        f.write('<population>\n')
        
        valid_agents = 0
        for i in range(num_agents):
            while oxi[i] == dxi[i] and oyi[i] == dyi[i]:
                dxi[i] = random.randint(0, size - 1)
                dyi[i] = random.randint(0, size - 1)

            cx_o, cy_o = oxi[i], oyi[i]
            cx_d, cy_d = dxi[i], dyi[i]
            
            o_links = get_incoming_links(cx_o, cy_o)
            d_links = get_incoming_links(cx_d, cy_d)
            
            if not o_links or not d_links:
                continue 
                
            start_link = random.choice(o_links)
            end_link = random.choice(d_links)
            
            curr_x, curr_y = cx_o, cy_o 
            target_x, target_y = map(int, end_link.split('-')[0].split('_'))
            
            dx = target_x - curr_x
            dy = target_y - curr_y
            
            moves = ['X'] * abs(dx) + ['Y'] * abs(dy)
            random.shuffle(moves)
            
            route_links = []
            tmp_x, tmp_y = curr_x, curr_y
            for move in moves:
                if move == 'X':
                    nx = tmp_x + (1 if dx > 0 else -1)
                    route_links.append(f"{tmp_x}_{tmp_y}-{nx}_{tmp_y}")
                    tmp_x = nx
                else:
                    ny = tmp_y + (1 if dy > 0 else -1)
                    route_links.append(f"{tmp_x}_{tmp_y}-{tmp_x}_{ny}")
                    tmp_y = ny
            
            full_route = [start_link] + route_links + [end_link]
            route_str = " ".join(full_route)
            
            s = dep_times[i]
            h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
            start_time_str = f"{h:02d}:{m:02d}:{sec:02d}"
            
            pid = f"agent_{valid_agents}"
            f.write(f'\t<person id="{pid}">\n')
            f.write('\t\t<plan selected="yes">\n')
            f.write(f'\t\t\t<act type="origin" link="{start_link}" x="{cx_o*block_size:.1f}" y="{cy_o*block_size:.1f}" end_time="{start_time_str}" />\n')
            f.write('\t\t\t<leg mode="car">\n')
            f.write(f'\t\t\t\t<route type="links" start_link="{start_link}" end_link="{end_link}">{route_str}</route>\n')
            f.write('\t\t\t</leg>\n')
            f.write(f'\t\t\t<act type="destination" link="{end_link}" x="{cx_d*block_size:.1f}" y="{cy_d*block_size:.1f}" />\n')
            f.write('\t\t</plan>\n')
            f.write('\t</person>\n')
            valid_agents += 1
            
        f.write('</population>\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--block_size", type=float, default=100.0)
    parser.add_argument("--duration", type=int, default=360, help="Fenêtre d'insertion en secondes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    net_file = os.path.join(args.output_dir, f"network_{args.size}x{args.size}.xml")
    write_network_xml(net_file, args.size, block_size=args.block_size)
    
    # 4 * N * (N-1) liens. L_tot_km = nb_liens * block_size / 1000
    L_tot_km = (4 * args.size * (args.size - 1) * args.block_size) / 1000.0
    duration_h = args.duration / 3600.0
    
    for k in range(60, 601, 60):
        # Calcul de la population basée sur la densité d'insertion (veh / h / km)
        num_agents = int(k * L_tot_km * duration_h)
        pop_file = os.path.join(args.output_dir, f"population_k{k}_{num_agents}agents.xml")
        write_population_xml(pop_file, num_agents, args.size, args.block_size, args.duration)
        
    print("Scénarios générés.")