import argparse
import os

import numpy as np


def sec_to_hms(seconds):
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_population(n1, mu1, sigma1, n2, mu2, sigma2, output_path):
    np.random.seed(42)

    # Flow 1 (West to East): L1 -> L2
    deps_1 = np.random.normal(loc=mu1, scale=sigma1, size=n1)
    # Flow 2 (South to North): L3 -> L4
    deps_2 = np.random.normal(loc=mu2, scale=sigma2, size=n2)

    # Combine and sort to interleave agents chronologically
    agents = []
    for d in deps_1:
        agents.append((d, "Flow1"))
    for d in deps_2:
        agents.append((d, "Flow2"))

    agents.sort(key=lambda x: x[0])

    header = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">
<population>"""
    footer = "\n</population>"

    agent_template = """
	<person id="agent_{id}">
		<plan selected="yes">
			<act type="home" link="{start_link}" x="{start_x}" y="{start_y}" end_time="{end_time}" />
			<leg mode="car">
				<route type="links" start_link="{start_link}" end_link="{end_link}">{route_str}</route>
			</leg>
			<act type="work" link="{end_link}" x="{end_x}" y="{end_y}" />
		</plan>
	</person>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, (dep_t, flow_type) in enumerate(agents):
            end_time_str = sec_to_hms(dep_t)

            if flow_type == "Flow1":
                start_link, end_link = "L1", "L2"
                route_str = "L1 L2"
                start_x, start_y = -1000.0, 0.0
                end_x, end_y = 1000.0, 0.0
            else:
                start_link, end_link = "L3", "L4"
                route_str = "L3 L4"
                start_x, start_y = 0.0, -1000.0
                end_x, end_y = 0.0, 1000.0

            f.write(
                agent_template.format(
                    id=i,
                    end_time=end_time_str,
                    start_link=start_link,
                    start_x=start_x,
                    start_y=start_y,
                    end_link=end_link,
                    end_x=end_x,
                    end_y=end_y,
                    route_str=route_str,
                )
            )
        f.write(footer)

    print(f"Generated {n1 + n2} agents in {output_path}")
    print(f"Flow 1 (West->East): N={n1}, mu={mu1}, sigma={sigma1}")
    print(f"Flow 2 (South->North): N={n2}, mu={mu2}, sigma={sigma2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate population for MATSim Intersection shockwave scenario."
    )
    parser.add_argument("-N1", type=int, default=1000, help="Total agents Flow 1 (West->East).")
    parser.add_argument("--mu1", type=float, default=28800.0, help="Peak time Flow 1.")
    parser.add_argument("--sigma1", type=float, default=100.0, help="Spread Flow 1.")
    parser.add_argument("-N2", type=int, default=1000, help="Total agents Flow 2 (South->North).")
    parser.add_argument(
        "--mu2", type=float, default=29000.0, help="Peak time Flow 2 (default offset)."
    )
    parser.add_argument("--sigma2", type=float, default=100.0, help="Spread Flow 2.")
    parser.add_argument("-o", "--output", type=str, default="population.xml", help="Output path.")

    args = parser.parse_args()
    generate_population(args.N1, args.mu1, args.sigma1, args.N2, args.mu2, args.sigma2, args.output)
