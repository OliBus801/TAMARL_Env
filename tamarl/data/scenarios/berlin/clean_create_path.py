import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree
import psutil
import time
from tqdm import tqdm
import multiprocessing
from functools import partial

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class Network:
    def __init__(self, network_file, mode='car'):
        self.network_file = network_file
        self.mode = mode
        
        self.nodes = {}  # id -> (x, y)
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        
        self.links = {}  # id -> attr
        self.link_id_to_idx = {}
        self.idx_to_link_id = {}
        
        self.graph = None # CSR matrix
        self.link_kdtree = None
        self.link_midpoints = []
        self.link_indices = []

    def parse(self):
        print(f"Parsing Network: {self.network_file}")
        context = ET.iterparse(self.network_file, events=("end",))
        
        valid_links = 0
        
        temp_edges = [] # (u_idx, v_idx, cost)
        
        for event, elem in context:
            if elem.tag == "node":
                nid = elem.get('id')
                x = float(elem.get('x'))
                y = float(elem.get('y'))
                
                idx = len(self.nodes)
                self.nodes[nid] = (x, y)
                self.node_id_to_idx[nid] = idx
                self.idx_to_node_id[idx] = nid
                elem.clear()
                
            elif elem.tag == "link":
                modes = elem.get('modes')
                if self.mode in modes:
                    u_id = elem.get('from')
                    v_id = elem.get('to')
                    link_id = elem.get('id')
                    
                    if u_id in self.node_id_to_idx and v_id in self.node_id_to_idx:
                        u_idx = self.node_id_to_idx[u_id]
                        v_idx = self.node_id_to_idx[v_id]
                        
                        length = float(elem.get('length'))
                        freespeed = float(elem.get('freespeed'))
                        
                        # Cost = Travel Time
                        cost = length / freespeed
                        
                        # Store link
                        l_idx = len(self.links)
                        self.links[link_id] = {
                            'u': u_id, 'v': v_id, 
                            'length': length, 
                            'freespeed': freespeed,
                            'cost': cost,
                            'idx': l_idx
                        }
                        self.link_id_to_idx[link_id] = l_idx
                        self.idx_to_link_id[l_idx] = link_id
                        
                        temp_edges.append((u_idx, v_idx, cost))
                        
                        # Midpoint for KDTree
                        u_pos = self.nodes[u_id]
                        v_pos = self.nodes[v_id]
                        mid_x = (u_pos[0] + v_pos[0]) / 2
                        mid_y = (u_pos[1] + v_pos[1]) / 2
                        
                        self.link_midpoints.append([mid_x, mid_y])
                        self.link_indices.append(l_idx)
                        
                        valid_links += 1
                elem.clear()

        print(f"Parsed {len(self.nodes)} nodes and {valid_links} links.")
        print(f"Memory: {get_memory_usage():.2f} MB")
        
        # Build Graph
        print("Building Graph...")
        num_nodes = len(self.node_id_to_idx)
        rows = [e[0] for e in temp_edges]
        cols = [e[1] for e in temp_edges]
        data = [e[2] for e in temp_edges]
        
        self.graph = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        
        # Build KDTree
        print("Building KDTree...")
        self.link_kdtree = KDTree(np.array(self.link_midpoints))
        
    def find_nearest_link(self, x, y):
        dist, idx = self.link_kdtree.query([x, y], k=1)
        link_idx = self.link_indices[idx]
        return self.idx_to_link_id[link_idx]


class Facilities:
    def __init__(self, facilities_file):
        self.facilities_file = facilities_file
        self.facility_coords = {} # id -> (x, y)

    def parse(self):
        print(f"Parsing Facilities: {self.facilities_file}")
        context = ET.iterparse(self.facilities_file, events=("end",))
        count = 0
        for event, elem in context:
            if elem.tag == "facility":
                fid = elem.get('id')
                x = elem.get('x')
                y = elem.get('y')
                if fid and x and y:
                    self.facility_coords[fid] = (float(x), float(y))
                    count += 1
                elem.clear()
        print(f"Parsed {count} facilities.")
        print(f"Memory: {get_memory_usage():.2f} MB")

    def get_coords(self, facility_id):
        return self.facility_coords.get(facility_id)

class PopulationRouter:
    def __init__(self, network, population_file, output_file, facilities=None):
        self.network = network
        self.population_file = population_file
        self.output_file = output_file
        self.facilities = facilities
        
        # Cache for (from_node_idx, to_node_idx) -> space_separated_node_ids
        # Since standard Dijkstra on graph gives nodes, we need to map node-sequence to link-sequence.
        self.path_cache = {} 
        self.link_lookup = {} # (u_idx, v_idx) -> link_id

        # Pre-process link lookup for fast recovery
        print("Building link lookup...")
        for lid, attr in self.network.links.items():
            u = self.network.node_id_to_idx[attr['u']]
            v = self.network.node_id_to_idx[attr['v']]
            self.link_lookup[(u, v)] = lid

    def _should_keep_person(self, person_elem):
        # 1. Get selected plan
        selected_plan = None
        for plan in person_elem.findall('plan'):
            if plan.get('selected') == 'yes':
                selected_plan = plan
                break
        
        if selected_plan is None:
            plans = person_elem.findall('plan')
            if plans:
                selected_plan = plans[0]
                
        if selected_plan is None:
            return False

        # 2. Check if all legs are 'car'
        for leg in selected_plan.findall('leg'):
            if leg.get('mode') != 'car':
                return False
        
        return True


    def collect_od_pairs(self):
        print(f"Pass 1: Collecting OD pairs from {self.population_file}...")
        
        # Requests: start_link -> set(end_link)
        requests = {} 
        
        context = ET.iterparse(self.population_file, events=("end",))
        context = iter(context)
        event, root = next(context)
        
        scanned_count = 0
        kept_count = 0
        
        for event, elem in context:
            if event == 'end' and elem.tag == 'person':
                scanned_count += 1
                if self._should_keep_person(elem):
                    self._collect_from_person(elem, requests)
                    kept_count += 1
                elem.clear()
                
                if scanned_count % 5000 == 0:
                     print(f"\rScanned {scanned_count} persons (Kept {kept_count})...", end="")
                     root.clear()
                     
        print(f"\nFinished scanning {self.population_file}.")
        return requests

    def _collect_from_person(self, person_elem, requests):
        selected_plan = None
        for plan in person_elem.findall('plan'):
            if plan.get('selected') == 'yes':
                selected_plan = plan
                break
        
        if selected_plan is None:
            plans = person_elem.findall('plan')
            if plans:
                selected_plan = plans[0]
        
        # DEBUG
        # if selected_plan is None:
        #      print(f"No plan found for {person_elem.get('id')}")

        if selected_plan is not None:
            elements = list(selected_plan)
            for i, el in enumerate(elements):
                if el.tag == 'leg' and el.get('mode') == 'car':
                    prev_act = elements[i-1]
                    next_act = elements[i+1]
                    
                    try:
                        start_link = self._get_link_from_act(prev_act)
                        end_link = self._get_link_from_act(next_act)
                        
                        if start_link and end_link and start_link != end_link:
                             if start_link not in requests:
                                 requests[start_link] = set()
                             requests[start_link].add(end_link)
                        else:
                             pass
                    except Exception as e:
                        print(f"  Error: {e}")
                        pass

    def apply_routes(self):
        print(f"Pass 2: Writing routes to {self.output_file}...")
        
        with open(self.output_file, 'w') as f_out:
            f_out.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f_out.write('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n')

            f_out.write('<population>\n')

            f_out.write('<attributes>\n')
            f_out.write('<attribute name="coordinateReferenceSystem" class="java.lang.String">EPSG:25832</attribute>\n')
            f_out.write('</attributes>\n')
            f_out.write('<!-- ====================================================================== -->\n')
            
            context = ET.iterparse(self.population_file, events=("start", "end"))
            context = iter(context)
            event, root = next(context)
            
            scanned_count = 0
            kept_count = 0
            total_trips = 0
            
            for event, elem in context:
                if event == 'end' and elem.tag == 'person':
                    scanned_count += 1
                    
                    should_keep = False
                    if self._should_keep_person(elem):
                        start_time = time.time()
                        success, trips = self.route_person(elem)
                        # Only keep if routing successful
                        if success:
                             should_keep = True
                             total_trips += trips
                    
                    if should_keep:
                        xml_str = ET.tostring(elem, encoding='unicode')
                        f_out.write(xml_str + '\n')
                        kept_count += 1
                    
                    elem.clear()
                    
                    if scanned_count % 1000 == 0:
                        print(f"\rScanned {scanned_count} persons (Kept {kept_count})...", end="")
                        root.clear()

            f_out.write('</population>\n')
            
        print(f"Finished writing {kept_count} persons (scanned {scanned_count}).")
        print(f"Total trips generated: {total_trips}")
        print(f"Average trips per person: {total_trips/kept_count if kept_count > 0 else 0:.2f}")

    # ... route_person and _route_plan remain similar but use pre-computed cache only ...
    # Actually, _get_path needs to strictly use cache or return None if not computed (which shouldn't happen if pass 1 worked)

    def route_person(self, person_elem):
        selected_plan = None
        for plan in person_elem.findall('plan'):
            if plan.get('selected') == 'yes':
                selected_plan = plan
                break
        
        if selected_plan is None:
            plans = person_elem.findall('plan')
            if plans:
                selected_plan = plans[0]
                
        if selected_plan is not None:
            return self._route_plan(selected_plan)
        return False, 0

    def _route_plan(self, plan):
        elements = list(plan)
        trips_count = 0
        success = True
        
        for i, el in enumerate(elements):
            if el.tag == 'leg' and el.get('mode') == 'car':
                prev_act = elements[i-1]
                next_act = elements[i+1]
                
                try:
                    start_link = self._get_link_from_act(prev_act)
                    end_link = self._get_link_from_act(next_act)
                    
                    if start_link and end_link:
                        # Lookup from cache directly
                        start_u = self.network.node_id_to_idx[self.network.links[start_link]['v']]
                        end_v = self.network.node_id_to_idx[self.network.links[end_link]['u']]
                        
                        key = (start_u, end_v)
                        route_str = self.path_cache.get(key)
                        
                        if route_str is not None:
                            route_elem = el.find('route')
                            if route_elem is None:
                                route_elem = ET.SubElement(el, 'route')
                            
                            # Construct full route: start_link + intermediate + end_link
                            # If intermediate is empty, just start + end
                            if route_str.strip():
                                final_route = f"{start_link} {route_str} {end_link}"
                            route_elem.text = final_route
                            trips_count += 1
                        else:
                            # Route not found in cache -> check why? 
                            # Maybe unreachable or filtered by heuristic?
                            success = False
                            break
                    else:
                        success = False
                        break
                except Exception:
                    success = False
                    break
        
        return success, trips_count

    def _get_link_from_act(self, act):
        link_id = act.get('link')
        if link_id:
            return link_id
        
        x_str = act.get('x')
        y_str = act.get('y')
        
        if x_str and y_str:
            x = float(x_str)
            y = float(y_str)
        else:
            # Try facility lookup
            fac_id = act.get('facility')
            if fac_id and self.facilities:
                coords = self.facilities.get_coords(fac_id)
                if coords:
                    x, y = coords
                else:
                    return None
            else:
                return None
        
        nearest_link = self.network.find_nearest_link(x, y)
        if nearest_link:
            act.set('link', nearest_link)
        return nearest_link

    def set_path_cache(self, cache):
        self.path_cache = cache


# Global variables for workers
global_graph = None
global_link_lookup = None
global_nodes = None
global_links = None
global_idx_to_node_id = None

def init_worker(graph, link_lookup, nodes, links, idx_to_node_id):
    global global_graph, global_link_lookup, global_nodes, global_links, global_idx_to_node_id
    global_graph = graph
    global_link_lookup = link_lookup
    global_nodes = nodes
    global_links = links
    global_idx_to_node_id = idx_to_node_id

def process_origin_chunk(tasks):
    """
    tasks: list of (start_link, targets_list, start_u, target_nodes_indices)
    """
    local_cache = {}
    found_count = 0
    total_count = 0
    
    # Heuristic settings
    MAX_SPEED = 60.0 / 3.6  # m/s (~16.7 m/s) - lowered to be safer
    FACTOR = 3.0  # Increased safety factor 
    
    for start_link, targets, start_u, target_nodes in tasks:
        # Calculate max Euclidean distance for heuristic limit
        max_dist = 0
        
        # Start Link 'v' node (end of link) is the start of routing
        s_node_id = global_links[start_link]['v']
        s_pos = global_nodes[s_node_id] 
        
        # Calculate max distance to any target
        for t_idx in target_nodes:
            t_id = global_idx_to_node_id[t_idx]
            t_pos = global_nodes[t_id]
            
            # Euclidean
            dist = np.sqrt((s_pos[0]-t_pos[0])**2 + (s_pos[1]-t_pos[1])**2)
            if dist > max_dist:
                max_dist = dist

        # Convert to time limit: (dist / speed) * factor
        # If max_dist is 0 (start=end), limit is 0? No, care.
        if max_dist == 0:
             limit = np.inf # Or small?
        else:
             limit = (max_dist / MAX_SPEED) * FACTOR
             
        # Minimum limit to avoid issues with very short trips
        if limit < 300: limit = 300 # 5 mins min
        
        dist_matrix, predecessors = sp.csgraph.dijkstra(
            global_graph, 
            indices=start_u, 
            return_predecessors=True,
            limit=limit
        )
        
        for t_v in target_nodes:
             if t_v == start_u:
                 key = (start_u, t_v)
                 local_cache[key] = ""
                 found_count += 1
                 total_count += 1
                 continue
             
             path = []
             curr = t_v
             found = True
             MAX_LEN = 10000
             c = 0
             
             while curr != start_u:
                prev = predecessors[curr]
                if prev == -9999:
                    found = False
                    break 
                
                lid = global_link_lookup.get((prev, curr))
                if lid:
                    path.append(lid)
                else:
                    found = False
                    break
                
                curr = prev
                c += 1
                if c > MAX_LEN:
                    found = False
                    break
             
             if found:
                 path.reverse()
                 key = (start_u, t_v)
                 local_cache[key] = " ".join(path)
                 found_count += 1
             
             total_count += 1
             
    return local_cache, found_count, total_count

# ... Fix: We need link_lookup in Network or helper
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--populations", nargs='+', default=["1pct"], help="Population suffixes (e.g. 1pct 10pct)")
    args = parser.parse_args()
    
    base_dir = "/home/olibuss/Documents/Code/TAMARL_Env/tamarl/data/scenarios/los_angeles"
    net_file = os.path.join(base_dir, "los-angeles-v1.0-network_2019-12-10.xml")
    fac_file = os.path.join(base_dir, "los-angeles-v1.0-facilities.xml")
    
    if not os.path.exists(net_file):
        print(f"Network file not found: {net_file}")
        exit(1)
        
    # 1. Load Network & Facilities
    net = Network(net_file)
    net.parse()
    
    # Build Link Lookup
    link_lookup = {}
    print("Building link lookup...")
    for lid, attr in net.links.items():
        u = net.node_id_to_idx[attr['u']]
        v = net.node_id_to_idx[attr['v']]
        link_lookup[(u, v)] = lid

    facilities = None
    if os.path.exists(fac_file):
        facilities = Facilities(fac_file)
        facilities.parse()
    
    routers = []
    global_requests = {} # start_link -> set(end_link)
    
    # 2. Pass 1: Collect
    for pop_suffix in args.populations:
        pop_file = os.path.join(base_dir, f"los-angeles-v1.0-population-{pop_suffix}.xml")
        out_file = os.path.join(base_dir, f"los-angeles-v1.0-population-{pop_suffix}-routed.xml")
        
        if not os.path.exists(pop_file):
            print(f"Skipping missing file: {pop_file}")
            continue
            
        print(f"--- Processing {pop_suffix} ---")
        router = PopulationRouter(net, pop_file, out_file, facilities)
        routers.append(router)
        
        reqs = router.collect_od_pairs()
        
        # Merge requests
        for s, targets in reqs.items():
            if s not in global_requests:
                global_requests[s] = set()
            global_requests[s].update(targets)
            
    print(f"Total unique origins to route: {len(global_requests)}")
    
    # 3. Compute Paths (Parallel)
    path_cache = {}
    total_found = 0
    total_targets = 0
    
    # Prepare tasks
    # Task format: (start_link, targets, start_u, target_nodes)
    # We pre-calculate indices to save worker time/pickle size? 
    # Actually explicit data is better.
    
    pending_tasks = []
    
    print("Preparing tasks...")
    for start_link, targets in global_requests.items():
        if start_link not in net.links:
            continue
        
        start_u = net.node_id_to_idx[net.links[start_link]['v']]
        
        target_nodes = set()
        for t in targets:
            if t in net.links:
                 t_v = net.node_id_to_idx[net.links[t]['u']]
                 target_nodes.add(t_v)
        
        if not target_nodes:
            continue
            
        pending_tasks.append((start_link, targets, start_u, target_nodes))

    print(f"Prepared {len(pending_tasks)} tasks.")
    
    # Chunking
    CHUNK_SIZE = 50 
    chunks = [pending_tasks[i:i + CHUNK_SIZE] for i in range(0, len(pending_tasks), CHUNK_SIZE)]
    print(f"Created {len(chunks)} chunks (size {CHUNK_SIZE}).")
    
    # Worker Init
    # We pass strict data needed: graph, lookup, nodes(for coords if needed), links(for attr)
    # Be careful with memory!
    
    print(f"Starting Pool with {multiprocessing.cpu_count()} processes...")
    with multiprocessing.Pool(
        processes=min(16, multiprocessing.cpu_count()),
        initializer=init_worker,
        initargs=(net.graph, link_lookup, net.nodes, net.links, net.idx_to_node_id)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_origin_chunk, chunks), 
            total=len(chunks), 
            desc="Computing Paths"
        ))
        
    print("Aggregating results...")
    for res_cache, found, total in results:
        path_cache.update(res_cache)
        total_found += found
        total_targets += total
        
    print(f"Computed {total_found}/{total_targets} paths.")
    
    # 4. Pass 2: Apply
    for router in routers:
        router.set_path_cache(path_cache)
        router.apply_routes()


