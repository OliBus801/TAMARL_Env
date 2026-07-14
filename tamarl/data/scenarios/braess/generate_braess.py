#!/usr/bin/env python3
"""Script to generate Braess^K networks of arbitrary dimension K for MATSim.

This script generates:
1. network.xml for both "paradox" (with shortcut links) and "vanilla" (without shortcuts)
2. {N}_population.xml where N is the population size (default is 100 * K)

The generated files are saved in:
- tamarl/data/scenarios/braess/k{K}/paradox/
- tamarl/data/scenarios/braess/k{K}/vanilla/
"""

import argparse
import os
import sys


def write_network_xml(K, output_path, include_shortcut=True):
    # s is 1, t is 2K+2
    s = 1
    t = 2 * K + 2

    # Nodes:
    # Column 0: Buffer start (0) at (-DX, y_center)
    # Column 1: Source node s (1) at (0, y_center)
    # Column 2: Upper intermediate nodes v_i (2 to K+1) at (DX, (i-1)*DY)
    # Column 3: Lower intermediate nodes w_i (K+2 to 2K+1) at (2*DX, i*DY)
    # Column 4: Sink node t (2K+2) at (3*DX, y_center)
    # Column 5: Buffer end (2K+3) at (4*DX, y_center)
    DX = 1000.0
    DY = 1000.0
    y_center = (K * DY) / 2.0

    nodes = []
    # Buffer start node (0)
    nodes.append((0, -DX, y_center))
    # Source node s (1)
    nodes.append((1, 0.0, y_center))

    for i in range(1, K + 1):
        # v_i ID: i + 1
        y_v = (i - 1) * DY
        nodes.append((i + 1, DX, y_v))
        # w_i ID: K + 1 + i
        y_w = i * DY
        nodes.append((K + 1 + i, 2.0 * DX, y_w))

    # Sink node t (2K+2)
    nodes.append((t, 3.0 * DX, y_center))
    # Buffer end node (2K+3)
    nodes.append((2 * K + 3, 4.0 * DX, y_center))

    # Links
    links = []

    # Buffer start link: 0 -> 1
    links.append(
        {
            "id": f"0-{s}",
            "from": 0,
            "to": s,
            "length": "30000.00",
            "capacity": "999999",
            "freespeed": "27.78",
        }
    )

    # Map K indices to node IDs
    v = {i: i + 1 for i in range(1, K + 1)}
    w = {i: K + 1 + i for i in range(1, K + 1)}

    # Profil A: GOULOTS (length="750.0", capacity="360", freespeed="27.78")
    for i in range(1, K + 1):
        # s -> v_i
        links.append(
            {
                "id": f"{s}-{v[i]}",
                "from": s,
                "to": v[i],
                "length": "750.00",
                "capacity": "360",
                "freespeed": "27.78",
            }
        )
        # w_i -> t
        links.append(
            {
                "id": f"{w[i]}-{t}",
                "from": w[i],
                "to": t,
                "length": "750.00",
                "capacity": "360",
                "freespeed": "27.78",
            }
        )

    # Profil C: RACCOURCIS (length="5000.0", capacity="999999", freespeed="20.00")
    if include_shortcut:
        for i in range(1, K + 1):
            # v_i -> w_i
            links.append(
                {
                    "id": f"{v[i]}-{w[i]}",
                    "from": v[i],
                    "to": w[i],
                    "length": "5000.00",
                    "capacity": "999999",
                    "freespeed": "20.00",
                }
            )

    # Profil B: LIBRES (length="10000.0", capacity="999999", freespeed="20.00")
    # s -> w_K
    links.append(
        {
            "id": f"{s}-{w[K]}",
            "from": s,
            "to": w[K],
            "length": "10000.00",
            "capacity": "999999",
            "freespeed": "20.00",
        }
    )
    # v_1 -> t
    links.append(
        {
            "id": f"{v[1]}-{t}",
            "from": v[1],
            "to": t,
            "length": "10000.00",
            "capacity": "999999",
            "freespeed": "20.00",
        }
    )
    # Crossing links: v_i -> w_{i-1} for 2 <= i <= K
    for i in range(2, K + 1):
        links.append(
            {
                "id": f"{v[i]}-{w[i - 1]}",
                "from": v[i],
                "to": w[i - 1],
                "length": "10000.00",
                "capacity": "999999",
                "freespeed": "20.00",
            }
        )

    # Buffer end link: t -> 2K+3
    links.append(
        {
            "id": f"{t}-{2 * K + 3}",
            "from": t,
            "to": 2 * K + 3,
            "length": "30000.00",
            "capacity": "999999",
            "freespeed": "30000.00",
        }
    )

    # Write network XML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n\n')
        f.write('<network name="equil test network">\n')
        f.write("   <nodes>\n")
        for nid, x, y in nodes:
            x_str = f"{x:.0f}" if x == int(x) else f"{x:.2f}"
            y_str = f"{y:.0f}" if y == int(y) else f"{y:.2f}"
            f.write(f'      <node id="{nid}" x="{x_str}" y="{y_str}"/>\n')
        f.write("   </nodes>\n")

        f.write("   <links>\n")
        for link in links:
            f.write(
                f'      <link id="{link["id"]}" from="{link["from"]}" to="{link["to"]}" '
                f'length="{link["length"]}" capacity="{link["capacity"]}" freespeed="{link["freespeed"]}" '
                f'permlanes="1"    modes="car"  />\n'
            )
        f.write("   </links>\n")
        f.write("</network>\n")
    print(f"Generated network at: {output_path}")


def write_population_xml(K, N, output_path):
    s = 1
    t = 2 * K + 2
    v_1 = 2

    route_str = f"0-{s} {s}-{v_1} {v_1}-{t} {t}-{2 * K + 3}"
    start_link = f"0-{s}"
    end_link = f"{t}-{2 * K + 3}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'
        )
        f.write("<population>\n")

        width = max(3, len(str(N - 1)))

        # Write agents in descending order to match historical files (can be ascending too)
        for i in range(N - 1, -1, -1):
            agent_id = f"agent_{i:0{width}d}"
            f.write(f'    <person id="{agent_id}">\n')
            f.write('        <plan selected="yes">\n')
            f.write(f'            <act type="h" link="{start_link}" end_time="00:00:00" />\n')
            f.write('            <leg mode="car">\n')
            f.write(
                f'                <route type="links" start_link="{start_link}" end_link="{end_link}">{route_str}</route>\n'
            )
            f.write("            </leg>\n")
            f.write(f'            <act type="w" link="{end_link}" end_time="00:30:00" />\n\n')
            f.write("        </plan>\n")
            f.write("    </person>\n")

        f.write("</population>\n")
    print(f"Generated population ({N} agents) at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Braess^K scenarios of arbitrary dimension K for MATSim."
    )
    parser.add_argument("K", type=int, help="Dimension of the Braess^K network")
    parser.add_argument(
        "--population",
        "-n",
        type=int,
        default=None,
        help="Number of agents to generate. Defaults to 100 * K.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Base directory for scenarios. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--no-paradox", action="store_true", help="Do not generate the paradox configuration"
    )
    parser.add_argument(
        "--no-vanilla", action="store_true", help="Do not generate the vanilla configuration"
    )

    args = parser.parse_args()

    K = args.K
    if K < 1:
        print("Error: K must be a positive integer (>= 1)", file=sys.stderr)
        sys.exit(1)

    N = args.population if args.population is not None else 100 * K
    if N < 1:
        print("Error: population size must be at least 1", file=sys.stderr)
        sys.exit(1)

    # Determine base directory
    if args.output_dir:
        base_dir = args.output_dir
    else:
        # Default to script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

    target_dir = os.path.join(base_dir, f"k{K}")

    if not args.no_paradox:
        paradox_dir = os.path.join(target_dir, "paradox")
        write_network_xml(K, os.path.join(paradox_dir, "network.xml"), include_shortcut=True)
        write_population_xml(K, N, os.path.join(paradox_dir, f"{N}_population.xml"))
        # Create renders directory
        os.makedirs(os.path.join(paradox_dir, "renders"), exist_ok=True)

    if not args.no_vanilla:
        vanilla_dir = os.path.join(target_dir, "vanilla")
        write_network_xml(K, os.path.join(vanilla_dir, "network.xml"), include_shortcut=False)
        write_population_xml(K, N, os.path.join(vanilla_dir, f"{N}_population.xml"))
        # Create renders directory
        os.makedirs(os.path.join(vanilla_dir, "renders"), exist_ok=True)


if __name__ == "__main__":
    main()
