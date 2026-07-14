"""
Generates a MATSim population XML for the Ortuzar-Willumsen scenario.
1700 agents total with the following OD distribution:
  A --> L : 600 agents  (route: A-C-G-J-I-L)
  A --> M : 400 agents  (route: A-C-D-H-K-M)
  B --> L : 300 agents  (route: B-D-G-J-I-L)
  B --> M : 400 agents  (route: B-E-H-K-M)
"""

import os

# --- Configuration ---
N_AGENTS = 128

# Node coordinates (from network.xml)
NODE_COORDS = {
    "A": (0.0, 0.0),
    "B": (0.0, -50.0),
    "L": (200.0, 0.0),
    "M": (200.0, -100.0),
}

# OD pairs: (origin_node, dest_node, count, node_path)
OD_PAIRS = [
    ("A", "L", 32, ["A", "C", "G", "J", "I", "L"]),
    ("A", "M", 32, ["A", "C", "D", "H", "K", "M"]),
    ("B", "L", 32, ["B", "D", "G", "J", "I", "L"]),
    ("B", "M", 32, ["B", "E", "H", "K", "M"]),
]


def node_path_to_links(node_path):
    """Convert a list of nodes [A, C, G, ...] to link IDs [A-C, C-G, ...]"""
    return [f"{node_path[i]}-{node_path[i + 1]}" for i in range(len(node_path) - 1)]


def generate_population():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"{N_AGENTS}_population.xml")

    agent_id = N_AGENTS - 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'
        )
        f.write("<population>\n")

        for origin, dest, count, node_path in OD_PAIRS:
            links = node_path_to_links(node_path)
            start_link = links[0]
            end_link = links[-1]
            route_str = " ".join(links)

            ox, oy = NODE_COORDS[origin]
            dx, dy = NODE_COORDS[dest]

            for _ in range(count):
                pid = f"agent_{agent_id}"

                f.write(f'\t<person id="{pid}">\n')
                f.write('\t\t<plan selected="yes">\n')
                f.write(
                    f'\t\t\t<act type="h" link="{start_link}" x="{ox}" y="{oy}" end_time="00:00:00" />\n'
                )
                f.write('\t\t\t<leg mode="car">\n')
                f.write(
                    f'\t\t\t\t<route type="links" start_link="{start_link}" end_link="{end_link}">{route_str}</route>\n'
                )
                f.write("\t\t\t</leg>\n")
                f.write(f'\t\t\t<act type="w" link="{end_link}" x="{dx}" y="{dy}" />\n')
                f.write("\t\t</plan>\n")
                f.write("\t</person>\n")

                agent_id -= 1

        f.write("</population>\n")

    print(f"[{N_AGENTS}_population.xml] généré avec succès dans {script_dir}")
    print(f"  Total agents: {N_AGENTS}")
    for origin, dest, count, node_path in OD_PAIRS:
        print(f"  {origin} --> {dest}: {count} agents  (route: {'-'.join(node_path)})")


if __name__ == "__main__":
    generate_population()
