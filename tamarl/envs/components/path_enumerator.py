"""Top-k loopless path enumeration using Yen's algorithm.

Provides utilities to enumerate the k-shortest loopless paths between
origin-destination pairs in a directed graph, using free-flow travel
times as edge weights.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Variable globale pour stocker le graphe en mémoire dans chaque worker.
# Ceci évite la sérialisation (pickling) coûteuse du graphe complet
# pour chaque tâche envoyée via ProcessPoolExecutor.
_WORKER_GRAPH = None

def _init_worker(graph: ig.Graph):
    """Initialise le graphe igraph dans la mémoire du worker."""
    global _WORKER_GRAPH
    _WORKER_GRAPH = graph

def _compute_paths_for_chunk(od_chunk: np.ndarray, k: int) -> Dict[Tuple[int, int], List[List[int]]]:
    """Exécute l'algorithme de Yen sur un lot d'OD pairs en utilisant igraph."""
    global _WORKER_GRAPH
    result = {}
    
    for i in range(od_chunk.shape[0]):
        o = int(od_chunk[i, 0])
        d = int(od_chunk[i, 1])
        od_key = (o, d)
        
        if o == d:
            result[od_key] = [[]]
            continue
            
        try:
            # get_k_shortest_paths avec output="epath" (edge path) retourne directement
            # les ID internes des arêtes igraph plutôt que les ID des sommets
            paths_edges = _WORKER_GRAPH.get_k_shortest_paths(
                o, to=d, k=k, weights="weight", output="epath"
            )
            
            paths = []
            for path in paths_edges:
                if len(path) == 0:
                    continue
                # Reconversion des ID internes d'igraph vers les index originaux "edge_endpoints"
                # en lisant l'attribut d'arête "original_id" qu'on a configuré lors de l'ajout
                original_path = [_WORKER_GRAPH.es[edge_idx]["original_id"] for edge_idx in path]
                paths.append(original_path)
            
            result[od_key] = paths
        except Exception:
            # igraph peut lever une exception (ex: InternalError ou ValueError)
            # si les noeuds ne sont pas connectés du tout ou isolés.
            result[od_key] = []
            
    return result


def enumerate_top_k_paths(
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Enumerate top-k loopless paths using igraph and multiprocessing."""
    
    # 1. Filtrer les arêtes parallèles en ne gardant que la plus rapide
    # igraph accepte les arêtes parallèles, mais le filtrage natif ici garantit 
    # la même logique que NetworkX et réduit la taille du graphe.
    edge_dict = {}
    for e in range(edge_endpoints.shape[0]):
        u = int(edge_endpoints[e, 0])
        v = int(edge_endpoints[e, 1])
        w = float(ff_times[e])
        
        if (u, v) not in edge_dict or w < edge_dict[(u, v)][1]:
            edge_dict[(u, v)] = (e, w)
            
    filtered_edges = list(edge_dict.keys())
    original_ids = [edge_dict[uv][0] for uv in filtered_edges]
    weights = [edge_dict[uv][1] for uv in filtered_edges]
    
    # 2. Construire le graphe igraph
    # L'argument n=num_nodes s'assure que les identifiants de noeuds 
    # isolés existent bien pour ne pas briser la séquence d'index.
    G = ig.Graph(n=num_nodes, directed=True)
    
    # add_edges() prend une liste de tuples. C'est l'opération de lot (batch) 
    # au niveau C qui est bien plus performante qu'une boucle for.
    G.add_edges(filtered_edges)
    
    # Assigner les attributs aux arêtes (stockés efficacement en C dans igraph)
    G.es["weight"] = weights
    G.es["original_id"] = original_ids
    
    # 3. Préparer les lots (chunks) pour le multiprocessing
    num_workers = os.cpu_count() or 4
    chunks = np.array_split(od_pairs, max(1, num_workers * 4))
    
    result = {}
    
    # 4. Exécuter en parallèle
    # initargs passe le graphe igraph à _init_worker pour qu'il le stocke en global
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(G,)) as executor:
        futures = [
            executor.submit(_compute_paths_for_chunk, chunk, k)
            for chunk in chunks if len(chunk) > 0
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing top-k paths (OD chunks)"):
            result.update(future.result())
            
    return result


def get_or_compute_top_k_paths(
    scenario_dir: str,
    num_nodes: int,
    edge_endpoints: np.ndarray,
    ff_times: np.ndarray,
    od_pairs: np.ndarray,
    k: int,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """Get top-k paths from cache, or compute and cache them if not found."""
    cache_path = os.path.join(scenario_dir, f"top_k_paths_k{k}.pkl")
    
    paths_dict = {}
    if os.path.exists(cache_path):
        print(f"Loading cached top-{k} paths from {cache_path}")
        with open(cache_path, "rb") as f:
            paths_dict = pickle.load(f)
            
    # Check for missing OD pairs in the loaded cache
    missing_od_pairs = []
    for i in range(od_pairs.shape[0]):
        o = int(od_pairs[i, 0])
        d = int(od_pairs[i, 1])
        if (o, d) not in paths_dict:
            missing_od_pairs.append([o, d])
            
    if missing_od_pairs:
        if len(paths_dict) > 0:
            print(f"Found {len(missing_od_pairs)} missing OD pairs in cache. Computing paths for them...")
        else:
            print(f"Computing top-{k} paths with igraph (this may take a while for large networks)...")
            
        missing_od_pairs_np = np.array(missing_od_pairs, dtype=np.int32)
        new_paths_dict = enumerate_top_k_paths(
            num_nodes=num_nodes,
            edge_endpoints=edge_endpoints,
            ff_times=ff_times,
            od_pairs=missing_od_pairs_np,
            k=k,
        )
        
        paths_dict.update(new_paths_dict)
        
        os.makedirs(scenario_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(paths_dict, f)
        print(f"Saved updated paths to {cache_path}")
        
    return paths_dict