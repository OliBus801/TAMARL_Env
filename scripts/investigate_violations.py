import argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Investigate Free-Flow violations in sanity check CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the sanity_01_tt_vs_fftt.csv file")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        sys.exit(1)

    print(f"Loading data from {args.csv_path}...\n")
    df = pd.read_csv(args.csv_path)

    # Compute violation: if realized_tt_sec < fftt_sec
    # We add a small tolerance (1e-4) to avoid floating point inaccuracies
    # We must also ignore 'inf' values which just mean the agent was disconnected/failed routing
    finite_mask = np.isfinite(df['fftt_sec']) & np.isfinite(df['realized_tt_sec'])
    valid_df = df[finite_mask].copy()
    
    valid_df['violation_sec'] = valid_df['fftt_sec'] - valid_df['realized_tt_sec']
    violations_df = valid_df[valid_df['violation_sec'] > 1e-4].copy()
    
    total_violations = len(violations_df)
    if total_violations == 0:
        print("🎉 Great news! No free-flow travel time violations found.")
        sys.exit(0)

    print(f"⚠ Found {total_violations} agents with Free-Flow violations.")
    
    # Find the worst offender
    worst_row = violations_df.loc[violations_df['violation_sec'].idxmax()]
    worst_idx = int(worst_row['agent_idx'])
    worst_fftt = worst_row['fftt_sec']
    worst_realized = worst_row['realized_tt_sec']
    worst_violation = worst_row['violation_sec']

    print(f"\n--- WORST OFFENDER SUMMARY ---")
    print(f"Agent Index (Leg #) : {worst_idx}")
    print(f"Calculated FFTT     : {worst_fftt:.2f} seconds")
    print(f"Realized Travel Time: {worst_realized:.2f} seconds")
    print(f"Violation Amount    : {worst_violation:.2f} seconds (arrived too early)")

    # Trace back to the population file
    target_dir = os.path.dirname(os.path.abspath(args.csv_path))
    parts = target_dir.split(os.sep)
    try:
        scen_idx = parts.index("scenarios")
        scenario_path = os.sep.join(parts[:scen_idx+2])
    except ValueError:
        scenario_path = None

    if not scenario_path or not os.path.exists(scenario_path):
        print(f"\nCould not infer scenario path to trace XML data. Assuming directory is {target_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(scenario_path) if f.endswith('.xml')]
    pop_candidates = [f for f in files if 'population' in f.lower() or 'plans' in f.lower()]
    population_file = None
    if pop_candidates:
        routed_candidates = [p for p in pop_candidates if 'routed' in p.lower()]
        if routed_candidates:
            population_file = os.path.join(scenario_path, routed_candidates[0])
        else:
            population_file = os.path.join(scenario_path, pop_candidates[0])

    if not population_file:
        print(f"\n⚠ Could not find population XML in {scenario_path} to trace agent.")
        sys.exit(1)

    print(f"\nTracing Agent Index {worst_idx} in {population_file}...")
    
    context = ET.iterparse(population_file, events=("end",))
    current_leg_idx = 0
    target_person_id = None
    target_route = None
    target_dep_time = None
    
    for event, elem in context:
        if elem.tag == "person":
            person_id = elem.get('id')
            selected_plan = None
            for child in elem:
                if child.tag == 'plan':
                    if child.get('selected') == 'yes':
                        selected_plan = child
                        break
                    if selected_plan is None:
                        selected_plan = child
            
            if selected_plan is not None:
                elements = list(selected_plan)
                last_act_end = None
                
                for el in elements:
                    if el.tag in ['act', 'activity']:
                        end_time = el.get('end_time')
                        if end_time:
                            last_act_end = end_time
                    elif el.tag == 'leg':
                        mode = el.get('mode')
                        if mode == 'car':
                            route_tag = el.find('route')
                            if route_tag is not None and route_tag.text:
                                if current_leg_idx == worst_idx:
                                    target_person_id = person_id
                                    target_route = route_tag.text.strip()
                                    target_dep_time = last_act_end if last_act_end else "00:00:00"
                                    break
                                current_leg_idx += 1
                                
            elem.clear()
            
            if target_person_id is not None:
                break
                
    if target_person_id:
        print(f"\n--- POPULATION TRACE ---")
        print(f"Person ID           : {target_person_id}")
        print(f"Departure Time      : {target_dep_time}")
        
        links = target_route.split(' ')
        print(f"Path Length         : {len(links)} links")
        print(f"Origin Link         : {links[0]}")
        print(f"Destination Link    : {links[-1]}")
        if len(links) < 10:
            print(f"Full Path           : {target_route}")
        else:
            print(f"Path Preview        : {' '.join(links[:5])} ... {' '.join(links[-3:])}")
    else:
        print(f"\n⚠ Agent Index {worst_idx} could not be found in the population file.")

if __name__ == "__main__":
    main()
