import argparse

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

    header = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">
<population>"""

    footer = "\n</population>"

    agent_template = """
	<person id="agent_{id}">
		<plan selected="yes">
			<act type="home" link="L1" x="0.0" y="0.0" end_time="{end_time}" />
			<leg mode="car">
				<route type="links" start_link="L1" end_link="L3">L1 L2 L3</route>
			</leg>
			<act type="work" link="L3" x="3000.0" y="0.0" />
		</plan>
	</person>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, dep_t in enumerate(departure_times):
            end_time_str = sec_to_hms(dep_t)
            f.write(agent_template.format(id=i, end_time=end_time_str))
        f.write(footer)

    print(f"Generated {n} agents in {output_path} (mu={mu}, sigma={sigma})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate population for MATSim shockwave test scenario."
    )
    parser.add_argument("-N", type=int, default=100, help="Total number of agents.")
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
