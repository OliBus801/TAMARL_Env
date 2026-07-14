import os
import shutil

import numpy as np
import torch

from tamarl.envs.agent_level_wrapper import AgentLevelWrapper
from tamarl.envs.dta_bandit_env import DTABanditEnv


def generate_toy_scenario(scenario_dir):
    os.makedirs(scenario_dir, exist_ok=True)
    network_xml = os.path.join(scenario_dir, "network.xml")
    population_xml = os.path.join(scenario_dir, "population.xml")

    # Write Network
    with open(network_xml, "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">
<network name="corridor">
    <nodes>
        <node id="0" x="0.0" y="0.0" />
        <node id="1" x="1000.0" y="0.0" />
        <node id="2" x="2000.0" y="0.0" />
        <node id="3" x="3000.0" y="0.0" />
    </nodes>
    <links capperiod="01:00:00">
        <link id="0-1" from="0" to="1" length="600.0" freespeed="10.0" capacity="10000.0" permlanes="1.0" oneway="1" modes="car" />
        <link id="1-2" from="1" to="2" length="6000.0" freespeed="10.0" capacity="10.0" permlanes="1.0" oneway="1" modes="car" />
        <link id="2-3" from="2" to="3" length="600.0" freespeed="10.0" capacity="10000.0" permlanes="1.0" oneway="1" modes="car" />
        <link id="3-0" from="3" to="0" length="600.0" freespeed="10.0" capacity="10000.0" permlanes="1.0" oneway="1" modes="car" />
    </links>
</network>
""")

    # Write Population
    with open(population_xml, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'
        )
        f.write("<population>\n")

        # 100 Test Agents (multi-leg)
        for i in range(100):
            f.write(f'  <person id="test_{i}">\n')
            f.write('    <plan selected="yes">\n')
            f.write('      <act type="home" link="3-0" end_time="00:00:00" />\n')
            f.write('      <leg mode="car"><route type="links">3-0 0-1</route></leg>\n')
            f.write('      <act type="dummy" link="0-1" duration="00:00:00" />\n')
            f.write('      <leg mode="car"><route type="links">0-1 1-2</route></leg>\n')
            f.write('      <act type="dummy" link="1-2" duration="00:00:00" />\n')
            f.write('      <leg mode="car"><route type="links">1-2 2-3</route></leg>\n')
            f.write('      <act type="dummy" link="2-3" duration="00:00:00" />\n')
            f.write('      <leg mode="car"><route type="links">2-3 3-0</route></leg>\n')
            f.write('      <act type="work" link="3-0" />\n')
            f.write("    </plan>\n")
            f.write("  </person>\n")

        # 1000 Background Agents (mono-leg)
        for i in range(1000):
            f.write(f'  <person id="bg_{i}">\n')
            f.write('    <plan selected="yes">\n')
            f.write('      <act type="home" link="0-1" end_time="00:00:30" />\n')
            f.write('      <leg mode="car"><route type="links">0-1 1-2</route></leg>\n')
            f.write('      <act type="work" link="1-2" />\n')
            f.write("    </plan>\n")
            f.write("  </person>\n")

        f.write("</population>\n")


def main():
    scenario_dir = "tamarl/data/scenarios/toy_corridor"
    generate_toy_scenario(scenario_dir)

    # max_steps set to 150 to cut the simulation when agents are on link 1-2
    # Link 0-1 FFTT is 60s, so test agents (start at 0s) enter link 1-2 at 60s.
    # Background agents (start at 30s) enter link 1-2 at 90s, heavily congesting it.
    max_steps = 150

    print(f"Instantiating DTABanditEnv with max_steps={max_steps}...")
    bandit = DTABanditEnv(
        scenario_path=scenario_dir,
        timestep=1.0,
        max_steps=max_steps,
        device="cpu",
        track_events=False,
    )
    env = AgentLevelWrapper(bandit=bandit, top_k=1)

    print("Running simulation with fixed routes (Top-1 path)...")
    actions = np.zeros(env.num_envs, dtype=np.int64)
    env.step(actions)
    dnl = env.bandit.dnl

    print(f"Simulation ended at step {dnl.current_step}")

    agent_status = dnl.status  # shape [A]
    not_arrived_mask = agent_status != 3
    not_arrived_indices = torch.where(not_arrived_mask)[0]

    print(f"\nTotal agents not arrived: {len(not_arrived_indices)}")

    violation_count = 0

    agent_leg_to_od = {}
    for i, (a, l) in enumerate(env.leg_to_agent):
        agent_leg_to_od[(a, l)] = env.od_indices_all_legs[i].item()

    print("\nChecking for FFTT violations on incomplete legs...")
    for agent_idx in not_arrived_indices.tolist():
        # Check only test agents (0 to 99)
        if agent_idx >= 100:
            continue

        num_legs = bandit.scenario.num_legs[agent_idx].item()

        current_leg = dnl.current_leg[agent_idx].item()
        agent_stat = dnl.status[agent_idx].item()

        for leg_idx in range(num_legs):
            if leg_idx < current_leg:
                leg_status_str = "completed"
            elif leg_idx == current_leg:
                leg_status_str = f"current (env_status={agent_stat})"
            else:
                leg_status_str = "not_started"

            realized_tt = dnl.leg_metrics[agent_idx, leg_idx, 1].item()

            # Fetch the actual FFTT computed by the route_utils / wrapper
            leg_od = agent_leg_to_od[(agent_idx, leg_idx)]
            path_fftt = float(env.fftt_matrix[leg_od, 0])

            if agent_idx == 0:
                print(
                    f"Agent {agent_idx} Leg {leg_idx}: status={leg_status_str}, realized_tt={realized_tt:.1f}, FFTT={path_fftt:.1f}"
                )
                metrics = dnl.leg_metrics[agent_idx, leg_idx].tolist()
                print(f"   leg_metrics: [dep_time={metrics[0]:.1f}, tt={metrics[1]:.1f}]")

            if realized_tt > 0 and realized_tt < path_fftt:
                violation_count += 1
                if agent_idx != 0:
                    print(
                        f"Agent {agent_idx} Leg {leg_idx}: status={leg_status_str}, realized_tt={realized_tt:.1f}, FFTT={path_fftt:.1f} (VIOLATION)"
                    )

    print(f"\nTotal violations found for test agents: {violation_count}")


if __name__ == "__main__":
    main()
