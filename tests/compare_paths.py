import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
from shapely.geometry import LineString
from collections import defaultdict
import xml.etree.ElementTree as ET

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tamarl.envs.scenario_loader import parse_network, parse_population

def load_network_geometry(scenario_dir):
    """Load network.xml to get node coordinates and edge geometries."""
    network_file = None
    for f in os.listdir(scenario_dir):
        if 'network' in f.lower() and f.endswith('.xml'):
            network_file = os.path.join(scenario_dir, f)
            break
            
    if not network_file:
        raise FileNotFoundError(f"No network file found in {scenario_dir}")

    nodes_coords = {}
    
    context = ET.iterparse(network_file, events=("end",))
    for event, elem in context:
        if elem.tag == "node":
            nid = elem.get('id')
            x = float(elem.get('x', 0.0))
            y = float(elem.get('y', 0.0))
            nodes_coords[nid] = (x, y)
            elem.clear()
            
    return nodes_coords

def jaccard_index(path1, path2):
    set1 = set(path1)
    set2 = set(path2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0
    return intersection / union

def get_average_jaccard_and_length(paths_dict, edge_lengths, allowed_ods=None):
    od_metrics = {}
    for od, paths in paths_dict.items():
        # If population/allowed_ods filter is provided, skip OD pairs not in population
        if allowed_ods is not None and od not in allowed_ods:
            continue
        if len(paths) <= 1:
            continue
            
        jaccards = []
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                jaccards.append(jaccard_index(paths[i], paths[j]))
        avg_jaccard = np.mean(jaccards) if jaccards else 1.0
        
        path_lengths = []
        for p in paths:
            length = sum([edge_lengths[eid] for eid in p if eid < len(edge_lengths)])
            path_lengths.append(length)
            
        if not path_lengths:
            continue
            
        min_length = min(path_lengths)
        if min_length > 0:
            avg_length_ratio = np.mean(path_lengths) / min_length
        else:
            avg_length_ratio = 1.0
            
        od_metrics[od] = {
            'avg_jaccard': avg_jaccard,
            'avg_length_ratio': avg_length_ratio,
            'dissimilarity': 1.0 - avg_jaccard
        }
    return od_metrics

def main():
    parser = argparse.ArgumentParser(description="Compare different top-K path generation methods.")
    parser.add_argument("--scenario", type=str, required=True, help="Path to the scenario directory")
    parser.add_argument("--files", type=str, nargs='+', required=True, help="List of .pkl files to compare")
    parser.add_argument("--labels", type=str, nargs='+', help="Labels for the methods (optional, defaults to filenames)")
    parser.add_argument("--population", type=str, default=None, help="Population filter (e.g. '100' or '1pct')")
    parser.add_argument("--population_file", type=str, default=None, help="Path to a specific population XML file")
    parser.add_argument("--show", action="store_true", help="Show plots on screen")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the generated plots and maps")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.labels and len(args.labels) != len(args.files):
        print("Warning: Number of labels does not match number of files. Using filenames.")
        args.labels = None
        
    labels = args.labels if args.labels else [os.path.basename(f).replace('.pkl', '') for f in args.files]

    # Find network file
    network_file = None
    for f in os.listdir(args.scenario):
        if 'network' in f.lower() and f.endswith('.xml'):
            network_file = os.path.join(args.scenario, f)
            break
            
    if not network_file:
        raise FileNotFoundError(f"No network file found in {args.scenario}")
        
    print(f"Loading network from: {network_file}")
    node_id_to_idx, edges_data, link_id_to_idx = parse_network(network_file)
    
    num_edges = len(edges_data)
    edge_lengths = np.array([e['attr'][0] for e in edges_data])
    edge_endpoints = np.array([[e['u'], e['v']] for e in edges_data])

    print("Extracting node coordinates...")
    nodes_coords = load_network_geometry(args.scenario)
    
    # Check for population file filter
    population_file = None
    if args.population_file:
        population_file = args.population_file
    elif args.population:
        # Search for population candidate files in scenario dir
        files = [f for f in os.listdir(args.scenario) if f.endswith('.xml')]
        pop_candidates = [f for f in files if 'population' in f.lower() or 'plans' in f.lower()]
        
        filtered = []
        for p in pop_candidates:
            tokens = p.replace('-', '_').replace('.', '_').split('_')
            if args.population in tokens:
                filtered.append(p)
        if not filtered:
            filtered = [p for p in pop_candidates if args.population in p]
        pop_candidates = filtered
        
        if pop_candidates:
            # Prioritize 'routed' files
            routed = [p for p in pop_candidates if 'routed' in p.lower()]
            if routed:
                population_file = os.path.join(args.scenario, routed[0])
            else:
                population_file = os.path.join(args.scenario, pop_candidates[0])

    allowed_ods = None
    if population_file:
        print(f"Loading population from: {population_file}")
        agents = parse_population(population_file, link_id_to_idx, node_id_to_idx)
        print(f"Loaded {len(agents)} agents.")
        
        # Build mapping: link_string_id -> to_node_index
        edge_to_node = {e['id']: e['v'] for e in edges_data}
        
        allowed_ods = set()
        for a in agents:
            for leg in a['legs']:
                fe = leg['first_edge']
                dest_link_id = leg['dest_link_id']
                if dest_link_id in link_id_to_idx:
                    orig_node = edge_endpoints[fe, 1]
                    dest_node = edge_to_node[dest_link_id]
                    allowed_ods.add((int(orig_node), int(dest_node)))
        print(f"Found {len(allowed_ods)} unique OD pairs in population file to filter by.")

    # Map edge index to LineString
    idx_to_link_id = {v: k for k, v in link_id_to_idx.items()}
    idx_to_u = {i: edge_endpoints[i, 0] for i in range(num_edges)}
    idx_to_v = {i: edge_endpoints[i, 1] for i in range(num_edges)}
    idx_to_u_id = {v: k for k, v in node_id_to_idx.items()}
    
    edge_geometries = []
    valid_edge_indices = []
    
    for i in range(num_edges):
        u_idx = idx_to_u.get(i)
        v_idx = idx_to_v.get(i)
        if u_idx is not None and v_idx is not None:
            u_id = idx_to_u_id.get(u_idx)
            v_id = idx_to_u_id.get(v_idx)
            if u_id in nodes_coords and v_id in nodes_coords:
                pt_u = nodes_coords[u_id]
                pt_v = nodes_coords[v_id]
                edge_geometries.append(LineString([pt_u, pt_v]))
                valid_edge_indices.append(i)
            else:
                edge_geometries.append(None)
        else:
            edge_geometries.append(None)
            
    results = {}
    
    for file_idx, file_path in enumerate(args.files):
        label = labels[file_idx]
        print(f"Processing {label} from {file_path}...")
        with open(file_path, "rb") as f:
            paths_dict = pickle.load(f)
            
        metrics = get_average_jaccard_and_length(paths_dict, edge_lengths, allowed_ods)
        
        # Edge load
        load_tensor = torch.zeros(num_edges, dtype=torch.int32)
        for od, paths in paths_dict.items():
            if allowed_ods is not None and od not in allowed_ods:
                continue
            for p in paths:
                for eid in p:
                    if 0 <= eid < num_edges:
                        load_tensor[eid] += 1
                        
        results[label] = {
            'metrics': metrics,
            'load': load_tensor.numpy()
        }

    # 1. Overlap Histogram (Jaccard Index)
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        jaccards = [m['avg_jaccard'] for m in data['metrics'].values()]
        if jaccards:
            plt.hist(jaccards, bins=50, alpha=0.5, label=label, density=True)
    
    plt.title('Distribution de l\'Indice de Jaccard Moyen (Overlapping)')
    plt.xlabel('Indice de Jaccard (0 = Disjoint, 1 = Identique)')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'jaccard_histogram.png'), dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # 2. Scatter Plot: Dissimilarity vs Length Ratio
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        metrics = data['metrics']
        dissim = [m['dissimilarity'] for m in metrics.values()]
        length_ratio = [m['avg_length_ratio'] for m in metrics.values()]
        if dissim:
            plt.scatter(dissim, length_ratio, alpha=0.5, label=label, s=10)
        
    plt.title('Trade-off: Diversité vs Allongement du Trajet')
    plt.xlabel('Dissimilarité (1 - Indice de Jaccard)')
    plt.ylabel('Ratio d\'Allongement (Coût Moyen / Coût Min)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0.0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'dissimilarity_vs_length.png'), dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # 3. Heatmaps (GeoJSON)
    for label, data in results.items():
        load = data['load']
        # Filter edges with load > 0
        active_indices = [i for i in valid_edge_indices if load[i] > 0]
        active_geoms = [edge_geometries[i] for i in active_indices]
        active_loads = [load[i] for i in active_indices]
        
        if active_geoms:
            gdf = gpd.GeoDataFrame({
                'edge_id': active_indices,
                'load': active_loads
            }, geometry=active_geoms)
            
            output_file = os.path.join(args.save_dir, f'edge_load_{label}.geojson')
            gdf.to_file(output_file, driver='GeoJSON')
            print(f"Saved GeoJSON for {label} at {output_file}")
            
            # Save simple static map plot as well
            fig, ax = plt.subplots(figsize=(12, 12))
            gdf.plot(column='load', cmap='hot_r', linewidth=gdf['load']/gdf['load'].max() * 5, ax=ax, legend=True)
            plt.title(f'Heatmap de Densité des Chemins - {label}')
            ax.set_axis_off()
            plt.savefig(os.path.join(args.save_dir, f'heatmap_{label}.png'), dpi=300, bbox_inches='tight')
            if args.show:
                plt.show()
            plt.close()

    print("All done!")

if __name__ == "__main__":
    main()
