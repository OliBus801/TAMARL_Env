"""Repair the Ingolstadt population's `<route>` tags into genuine, connected,
free-flow-shortest-path routes, for use as ground truth in the TorchDNL-vs-MATSim
DNL-fidelity validation appendix.

Why this exists
----------------
`tamarl/data/scenarios/ingolstadt/population.xml` looks superficially like a
converged/fixed MATSim route population (it has `<route type="links">` per
leg), but inspection showed every route is literally just
`"{start_link} {end_link}"` -- i.e. only the OD endpoints, with **no**
intermediate links. In 1034/1035 cases `start_link`'s destination node isn't
even the same node as `end_link`'s origin node, so these are not valid
traversable paths: MATSim's queue simulation requires a route to be a
contiguous chain of adjacent links and would reject/mishandle this, and
replaying it through TorchDNL would silently "teleport" agents between
non-adjacent links.

This script recomputes, for every unique (start_link, end_link) OD pair in
the population, the free-flow-shortest-path chain of links connecting them
(Dijkstra over node-to-node link travel times, mirroring
`tamarl.envs.components.path_enumerator`'s edge-weight convention), and
rewrites the population with the full route substituted in. Both the
TorchDNL and MATSim sides of the validation then replay this exact same
(script-generated, not MATSim-converged) route -- which still satisfies the
methodology's core requirement ("identical routes fed to both simulators, so
only DNL-engine stochasticity differs"), it's just that the "ground truth
routes" are shortest-path rather than MATSim-day-to-day-converged. This
should be flagged explicitly wherever Ingolstadt validation results are
reported.

Usage:
    python scripts/generate_ingolstadt_fixed_routes.py \\
        --network tamarl/data/scenarios/ingolstadt/network.xml \\
        --population tamarl/data/scenarios/ingolstadt/population.xml \\
        --output tamarl/data/scenarios/ingolstadt/population_fixed_routes.xml
"""

from __future__ import annotations

import argparse
import heapq
import xml.etree.ElementTree as ET
from collections import defaultdict

from tamarl.envs.scenario_loader import parse_network


def build_adjacency(edges: list[dict]) -> dict[int, list[tuple[int, str, float]]]:
    """node -> list of (dest_node, link_id, ff_time) outgoing links."""
    adj: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for e in edges:
        ff_time = e["attr"][4]
        adj[e["u"]].append((e["v"], e["id"], ff_time))
    return adj


def dijkstra_link_path(
    adj: dict[int, list[tuple[int, str, float]]],
    start_node: int,
    end_node: int,
) -> list[str] | None:
    """Shortest (free-flow-time) chain of link IDs from start_node to end_node.
    Returns None if unreachable."""
    if start_node == end_node:
        return []
    dist = {start_node: 0.0}
    prev: dict[int, tuple[int, str]] = {}
    visited = set()
    pq = [(0.0, start_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end_node:
            break
        for v, link_id, w in adj.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, link_id)
                heapq.heappush(pq, (nd, v))
    if end_node not in dist:
        return None
    path_links: list[str] = []
    cur = end_node
    while cur != start_node:
        p_node, link_id = prev[cur]
        path_links.append(link_id)
        cur = p_node
    path_links.reverse()
    return path_links


def _insert_doctype(output_path: str, source_population_path: str) -> None:
    """xml.etree drops the DOCTYPE on write; MATSim requires it. Re-insert the
    exact DOCTYPE line found in the source population file, right after the
    XML declaration."""
    doctype_line = None
    with open(source_population_path) as f:
        for line in f:
            if line.strip().startswith("<!DOCTYPE"):
                doctype_line = line.rstrip("\n")
                break
    if doctype_line is None:
        return

    with open(output_path) as f:
        lines = f.readlines()
    if lines and lines[0].startswith("<?xml"):
        lines.insert(1, doctype_line + "\n")
    else:
        lines.insert(0, doctype_line + "\n")
    with open(output_path, "w") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    node_id_to_idx, edges, link_id_to_idx, _ = parse_network(args.network)
    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}
    link_endpoints = {e["id"]: (e["u"], e["v"]) for e in edges}
    adj = build_adjacency(edges)

    tree = ET.parse(args.population)
    root = tree.getroot()

    od_cache: dict[tuple[str, str], list[str] | None] = {}
    n_legs = 0
    n_recomputed = 0
    n_unreachable = 0

    for person in root.findall("person"):
        for plan in person.findall("plan"):
            for leg in plan.findall("leg"):
                if leg.get("mode") != "car":
                    continue
                route = leg.find("route")
                if route is None or not route.text:
                    continue
                link_ids = route.text.split()
                if len(link_ids) < 2:
                    continue
                start_link, end_link = link_ids[0], link_ids[-1]
                n_legs += 1

                key = (start_link, end_link)
                if key not in od_cache:
                    if start_link not in link_endpoints or end_link not in link_endpoints:
                        od_cache[key] = None
                    else:
                        u = link_endpoints[start_link][1]  # "to" node of start_link
                        v = link_endpoints[end_link][0]  # "from" node of end_link
                        mid_links = dijkstra_link_path(adj, u, v)
                        od_cache[key] = mid_links

                mid_links = od_cache[key]
                if mid_links is None:
                    n_unreachable += 1
                    continue

                full_path = [start_link] + mid_links + [end_link]
                if full_path != link_ids:
                    n_recomputed += 1
                route.text = " ".join(full_path)

    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    _insert_doctype(args.output, args.population)

    print(f"Legs processed: {n_legs}")
    print(f"Unique OD pairs: {len(od_cache)}")
    print(f"Legs with route recomputed (were not already the shortest contiguous path): {n_recomputed}")
    print(f"Legs with unreachable OD (left unchanged, still broken): {n_unreachable}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
