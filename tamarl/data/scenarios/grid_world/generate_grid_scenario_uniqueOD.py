import argparse
import os
import random

import numpy as np


def write_network_xml(
    filename: str,
    size: int,
    block_size: float = 1000.0,
    freespeed: float = 13.9,
    capacity: float = 2000.0,
    permlanes: float = 1.0,
):
    """
    Generates a generic grid network and writes it to a MATSim network XML file.
    """
    print(f"Generating Network: {size}x{size} grid...")

    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n')
        f.write('<network name="grid_world">\n')
        f.write("\t<nodes>\n")

        for x in range(size):
            for y in range(size):
                node_id = f"{x}_{y}"
                coord_x = x * block_size
                coord_y = y * block_size
                f.write(f'\t\t<node id="{node_id}" x="{coord_x}" y="{coord_y}" />\n')

        f.write("\t</nodes>\n")
        f.write("\t<links>\n")

        link_count = 0

        def write_link(u, v):
            nonlocal link_count
            link_count += 1
            f.write(
                f'\t\t<link id="{u}-{v}" from="{u}" to="{v}" length="{block_size}" freespeed="{freespeed}" capacity="{capacity}" permlanes="{permlanes}" oneway="1" modes="car" />\n'
            )

        for x in range(size):
            for y in range(size):
                u = f"{x}_{y}"
                if x + 1 < size:
                    v = f"{x + 1}_{y}"
                    write_link(u, v)
                    write_link(v, u)
                if y + 1 < size:
                    v = f"{x}_{y + 1}"
                    write_link(u, v)
                    write_link(v, u)

        f.write("\t</links>\n")
        f.write("</network>\n")

    print(f"Network generated with {size * size} nodes and {link_count} links.")


def write_population_xml(
    filename: str, num_agents: int, size: int, block_size: float = 1000.0, seed: int = 42
):
    """
    Generates a benchmark population with exactly TWO OD pairs.
    Morning: Node (0,0) -> Node (size-1, size-1)
    Evening: Node (size-1, size-1) -> Node (0,0)
    """
    print(f"Generating Benchmark Population: {num_agents} agents with exactly 2 OD pairs...")
    np.random.seed(seed)

    # Heures de départ avec une variance pour étaler la congestion
    t0_mean, t0_std = 8.0 * 3600, 3600.0 / 2.0
    t1_mean, t1_std = 17.0 * 3600, 3600.0 / 2.0

    dep_am = np.maximum(0, np.random.normal(t0_mean, t0_std, num_agents))
    dep_pm = np.maximum(dep_am + 4 * 3600, np.random.normal(t1_mean, t1_std, num_agents))

    def get_incoming_links(nx, ny):
        links = []
        if nx - 1 >= 0:
            links.append(f"{nx - 1}_{ny}-{nx}_{ny}")
        if nx + 1 < size:
            links.append(f"{nx + 1}_{ny}-{nx}_{ny}")
        if ny - 1 >= 0:
            links.append(f"{nx}_{ny - 1}-{nx}_{ny}")
        if ny + 1 < size:
            links.append(f"{nx}_{ny + 1}-{nx}_{ny}")
        return links

    # Fixer l'origine et la destination pour TOUT LE MONDE
    cx_o, cy_o = 0, 0
    cx_d, cy_d = size - 1, size - 1

    # On force le choix du même lien entrant pour s'assurer d'avoir 1 seule paire de liens OD
    home_link = get_incoming_links(cx_o + 1, cy_o)[
        0
    ]  # On décale de 1 pour avoir un lien entrant valide
    work_link = get_incoming_links(cx_d, cy_d)[0]

    def parse_link(lid):
        u_str, v_str = lid.split("-")
        ux, uy = map(int, u_str.split("_"))
        vx, vy = map(int, v_str.split("_"))
        return (ux, uy), (vx, vy)

    # Pré-calculer la route géométrique de base (Manhattan) une seule fois
    ((wl_start_x, wl_start_y), _) = parse_link(work_link)
    ((hl_start_x, hl_start_y), _) = parse_link(home_link)

    def build_manhattan_route(start_x, start_y, end_x, end_y):
        route = []
        cx, cy = start_x, start_y
        step_x = 1 if end_x > cx else -1
        if cx != end_x:
            for x in range(cx, end_x, step_x):
                route.append(f"{x}_{cy}-{x + step_x}_{cy}")
            cx = end_x
        step_y = 1 if end_y > cy else -1
        if cy != end_y:
            for y in range(cy, end_y, step_y):
                route.append(f"{cx}_{y}-{cx}_{y + step_y}")
        return route

    # Route Aller
    route_am = (
        [home_link] + build_manhattan_route(cx_o + 1, cy_o, wl_start_x, wl_start_y) + [work_link]
    )
    route_str_am = " ".join(route_am)

    # Route Retour
    route_pm = [work_link] + build_manhattan_route(cx_d, cy_d, hl_start_x, hl_start_y) + [home_link]
    route_str_pm = " ".join(route_pm)

    def fmt_time(s):
        return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"

    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'
        )
        f.write("<population>\n")

        for i in range(num_agents):
            start_am, start_pm = fmt_time(dep_am[i]), fmt_time(dep_pm[i])

            f.write(f'\t<person id="agent_{i}">\n\t\t<plan selected="yes">\n')
            f.write(
                f'\t\t\t<act type="home" link="{home_link}" x="{cx_o * block_size:.1f}" y="{cy_o * block_size:.1f}" end_time="{start_am}" />\n'
            )

            f.write('\t\t\t<leg mode="car">\n')
            f.write(
                f'\t\t\t\t<route type="links" start_link="{home_link}" end_link="{work_link}">{route_str_am}</route>\n'
            )
            f.write("\t\t\t</leg>\n")

            f.write(
                f'\t\t\t<act type="work" link="{work_link}" x="{cx_d * block_size:.1f}" y="{cy_d * block_size:.1f}" end_time="{start_pm}" />\n'
            )

            f.write('\t\t\t<leg mode="car">\n')
            f.write(
                f'\t\t\t\t<route type="links" start_link="{work_link}" end_link="{home_link}">{route_str_pm}</route>\n'
            )
            f.write("\t\t\t</leg>\n")

            f.write(
                f'\t\t\t<act type="home" link="{home_link}" x="{cx_o * block_size:.1f}" y="{cy_o * block_size:.1f}" />\n'
            )
            f.write("\t\t</plan>\n\t</person>\n")

        f.write("</population>\n")

    print(f"Benchmark population generated for {num_agents} agents (2 OD pairs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Benchmark MATSim Grid Scenario (2 OD pairs)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output XMLs"
    )
    parser.add_argument("--size", type=int, default=32, help="Grid size N (size x size nodes)")
    parser.add_argument("--agents", type=int, default=10000, help="Number of agents")
    parser.add_argument("--prefix", type=str, default="benchmark_grid", help="Filename prefix")
    parser.add_argument(
        "--block_size", type=float, default=500.0, help="Distance between nodes in meters"
    )
    parser.add_argument("--capacity", type=float, default=2000.0, help="Capacity of links")
    parser.add_argument("--freespeed", type=float, default=13.89, help="Free speed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    net_file = os.path.join(args.output_dir, f"{args.prefix}_{args.size}x{args.size}_network.xml")
    pop_file = os.path.join(
        args.output_dir, f"{args.prefix}_{args.size}x{args.size}_{args.agents}_population.xml"
    )

    write_network_xml(
        net_file,
        args.size,
        block_size=args.block_size,
        freespeed=args.freespeed,
        capacity=args.capacity,
    )
    write_population_xml(
        pop_file, args.agents, args.size, block_size=args.block_size, seed=args.seed
    )

    print("Done.")
