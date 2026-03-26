
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
import os
import argparse
import random

def load_network(network_file):
    print(f"Loading network from {network_file}...")
    tree = ET.parse(network_file)
    root = tree.getroot()

    nodes = {}
    node_coords = []
    node_ids = []

    # Parse Nodes
    for node in root.findall(".//node"):
        nid = node.get("id")
        x = float(node.get("x"))
        y = float(node.get("y"))
        nodes[nid] = (x, y)
        node_coords.append([x, y])
        node_ids.append(nid)

    # Parse Links and Build Graph
    G = nx.DiGraph()
    links = {}
    incoming_links = {nid: [] for nid in nodes}

    for link in root.findall(".//link"):
        lid = link.get("id")
        from_node = link.get("from")
        to_node = link.get("to")
        length = float(link.get("length"))
        
        links[lid] = {
            "from": from_node,
            "to": to_node,
            "length": length
        }
        
        G.add_edge(from_node, to_node, weight=length, id=lid)
        
        if to_node in incoming_links:
            incoming_links[to_node].append(lid)

    print(f"Network loaded: {len(nodes)} nodes, {len(links)} links.")
    return G, nodes, node_coords, node_ids, links, incoming_links

def get_nearest_node(kdtree, node_ids, x, y):
    dist, idx = kdtree.query([x, y])
    return node_ids[idx]

def process_population(pop_file, network_file, output_file):
    # Load Network
    G, nodes, node_coords, node_ids, links, incoming_links = load_network(network_file)
    
    # Build KDTree for spatial query
    print("Building KDTree...")
    tree_coords = KDTree(np.array(node_coords))

    print(f"Processing population from {pop_file}...")
    tree = ET.parse(pop_file)
    root = tree.getroot()
    
    # Iterate through all persons
    processed_count = 0
    total_persons = len(root.findall("person"))
    
    for person in root.findall("person"):
        plan = person.find("plan")
        if plan is None:
            continue
            
        acts = plan.findall("act")
        legs = plan.findall("leg")
        
        # 1. Assign Links to Acts
        act_links = [] # Store assigned link for each act to use in routing
        
        for act in acts:
            x = float(act.get("x"))
            y = float(act.get("y"))
            
            # Find nearest node
            nearest_node_id = get_nearest_node(tree_coords, node_ids, x, y)
            
            # Select an incoming link to this node
            # If no incoming links (isolated node?), this is an edge case. 
            # In Sioux Falls, most nodes should have incoming links.
            candidates = incoming_links.get(nearest_node_id, [])
            
            assigned_link = None
            if candidates:
                # Deterministic choice based on person ID or simple random with seed?
                # Using random.choice for now, can be fixed seed if needed.
                assigned_link = random.choice(candidates)
            else:
                # Fallback: Find links where this node is 'from' (outgoing) if no incoming? 
                # Or just error out? For now let's skip/warn.
                print(f"Warning: No incoming links for node {nearest_node_id} (Agent {person.get('id')})")
                assigned_link = list(G.out_edges(nearest_node_id, data="id"))[0][2] if G.out_degree(nearest_node_id) > 0 else None

            if assigned_link:
                act.set("link", assigned_link)
                # Remove facility attribute if present, as requested
                if "facility" in act.attrib:
                    del act.attrib["facility"]
                act_links.append(assigned_link)
            else:
                act_links.append(None)

        # 2. Route Legislations
        # A plan is Act -> Leg -> Act -> Leg -> Act ...
        # acts[0] -> legs[0] -> acts[1] -> legs[1] -> acts[2] ...
        
        for i, leg in enumerate(legs):
            if leg.get("mode") != "car":
                continue # Skip non-car legs if any? Sioux Falls is mostly car.
                
            start_link_id = act_links[i]
            end_link_id = act_links[i+1]
            
            if not start_link_id or not end_link_id:
                continue

            # Route: 
            # Start Link (u -> v)
            # End Link (x -> y)
            # Path should be v -> ... -> x
            
            start_node = links[start_link_id]["to"]
            end_node = links[end_link_id]["from"]
            
            try:
                # Compute Shortest Path
                if start_node == end_node:
                     # Start and end links connected to same node?
                     # e.g. Link A comes into Node N, Link B goes out of Node N (wait, end_link goes INTO Act location)
                     # Start Link: Incoming to Act 1 location (ToNode = Act1_Node)
                     # End Link: Incoming to Act 2 location (FromNode = Act2_PreNode) -- Wait.
                     
                     # Standard MATSim:
                     # Act 1 at Link A (from u to v). Agent is at 'v' end of Link A? Or anywhere on Link A?
                     # Usually departure is from end of Link A. So start routing from 'v'.
                     # Arrival is at Link B (from x to y). We need to arrive at 'x' to enter Link B? Or just traverse Link B?
                     # MATSim standard: Route is sequence of links. Start Link is the first one. End Link is the last one.
                     # But usually the 'route' element contains the INTERMEDIATE links.
                     pass
                
                path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
                
                # Convert node path to link path
                route_links = []
                route_links.append(start_link_id) # To add start_link to route
                for k in range(len(path_nodes) - 1):
                    u, v = path_nodes[k], path_nodes[k+1]
                    edge_data = G.get_edge_data(u, v)
                    route_links.append(edge_data["id"])
                route_links.append(end_link_id) # To add end_link to route
                    
                # Format route string
                route_str = " ".join(route_links)
                
                # Update Leg
                # Check if route tag exists, else create
                route_tag = leg.find("route")
                if route_tag is None:
                    route_tag = ET.SubElement(leg, "route")
                
                route_tag.text = route_str
                route_tag.set("type", "links")
                route_tag.set("start_link", start_link_id)
                route_tag.set("end_link", end_link_id)
                
            except nx.NetworkXNoPath:
                print(f"No path found for agent {person.get('id')} leg {i}")
            except Exception as e:
                print(e)

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count}/{total_persons} agents...")

    print(f"Writing output to {output_file}...")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="Siouxfalls_network_PT.xml", help="Path to network XML")
    parser.add_argument("--population", default="Siouxfalls_population.xml", help="Path to input population XML")
    parser.add_argument("--output", default="Siouxfalls_route_population.xml", help="Path to output population XML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    process_population(args.population, args.network, args.output)
