import argparse
import os

import numpy as np


def sec_to_hms(seconds):
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_population(n, mu, sigma, output_path):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate departure times
    departure_times = np.random.normal(loc=mu, scale=sigma, size=n)
    # Sort departure times in chronological order
    departure_times = np.sort(departure_times)

    # 90% go to L2, 10% go to L3
    routes = np.random.choice(["L2", "L3"], size=n, p=[0.9, 0.1])

    header = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">
<population>"""

    footer = "\n</population>"

    agent_template = """
	<person id="agent_{id}">
		<plan selected="yes">
			<act type="home" link="L0" x="-500.0" y="0.0" end_time="{end_time}" />
			<leg mode="car">
				<route type="links" start_link="L0" end_link="{end_link}">{route_str}</route>
			</leg>
			<act type="work" link="{end_link}" x="{x}" y="{y}" />
		</plan>
	</person>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, dep_t in enumerate(departure_times):
            end_time_str = sec_to_hms(dep_t)
            end_link = routes[i]
            route_str = f"L0 L1 {end_link}"

            if end_link == "L2":
                x, y = 1000.0, 500.0
            else:
                x, y = 1000.0, -500.0

            f.write(
                agent_template.format(
                    id=i, end_time=end_time_str, end_link=end_link, route_str=route_str, x=x, y=y
                )
            )
        f.write(footer)

    print(f"Generated {n} agents in {output_path} (mu={mu}, sigma={sigma})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate population for MATSim Y-shape shockwave scenario."
    )
    parser.add_argument("-N", type=int, default=1000, help="Total number of agents.")
    parser.add_argument(
        "--mu",
        type=float,
        default=28800.0,
        help="Average peak departure time in seconds (default: 28800 for 8:00).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=100.0,
        help="Standard deviation for departure time spreading in seconds.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="population.xml",
        help="Output path for the population.xml file.",
    )

    args = parser.parse_args()
    generate_population(args.N, args.mu, args.sigma, args.output)
