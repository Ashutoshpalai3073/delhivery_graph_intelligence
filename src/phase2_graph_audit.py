"""
================================================================================
 PHASE 2: Graph construction & bottleneck audit
================================================================================
 Builds a DIRECTED, WEIGHTED graph of the logistics network from legs.csv:

     nodes  = facilities (source_center / destination_center)
     edges  = corridors  (one edge per source->destination pair)
     weight = median delay ratio (actual/osrm) on that corridor

 Then computes graph metrics to find chokepoint hubs and chronic-delay
 corridors, and ranks them by their contribution to SLA breaches.

 WHY A GRAPH (not point-to-point)
 --------------------------------
 A facility's *position* in the network determines blast radius. A hub on many
 shortest paths (high betweenness) that is also slow propagates delay to every
 route flowing through it. Independent A->B estimates can't see that; centrality
 can. We therefore rank hubs on a composite of structural centrality AND delay
 contribution, not delay alone.

 OUTPUTS (written to ../outputs/)
   network.graphml        - the graph, reusable by later phases
   node_metrics.csv       - every facility with its graph metrics
   corridors.csv          - aggregated edge table
   bottleneck_hubs.csv    - hubs ranked by composite bottleneck score
   delayed_corridors.csv  - corridors ranked by total delay contribution

 USAGE
   python phase2_graph_audit.py --legs ../data/legs.csv --out ../outputs
================================================================================
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ----------------------------------------------------------------------------
# 1. Aggregate legs -> corridor (edge) table
# ----------------------------------------------------------------------------
def build_corridor_table(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per (source_center -> destination_center) corridor."""
    grp = legs.groupby(["source_center", "destination_center"])
    corr = grp.agg(
        volume=("trip_uuid", "size"),                 # legs run on this corridor
        median_factor=("factor", "median"),           # robust delay ratio
        median_actual=("actual_time", "median"),
        median_osrm=("osrm_time", "median"),
        total_excess_delay=("delay_min", "sum"),      # SLA-breach fuel (minutes)
        breach_legs=("is_delayed_20", "sum"),         # legs >20% over OSRM
        median_distance=("osrm_distance", "median"),
        ftl_share=("route_type", lambda s: (s == "FTL").mean()),
    ).reset_index()
    corr["breach_rate"] = corr["breach_legs"] / corr["volume"]
    # source/dest readable names (first occurrence)
    names = (legs.groupby("source_center")["source_name"].first().to_dict())
    dnames = (legs.groupby("destination_center")["destination_name"].first().to_dict())
    corr["source_name"] = corr["source_center"].map(names)
    corr["dest_name"] = corr["destination_center"].map(dnames)
    return corr


# ----------------------------------------------------------------------------
# 2. Build the directed graph
# ----------------------------------------------------------------------------
def build_graph(corr: pd.DataFrame, legs: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    # node names lookup (a facility can be a source and/or a destination)
    name_lookup = {}
    for _, r in legs.groupby("source_center")["source_name"].first().items():
        name_lookup[_] = r
    for c, n in legs.groupby("destination_center")["destination_name"].first().items():
        name_lookup.setdefault(c, n)

    for _, e in corr.iterrows():
        G.add_edge(
            e["source_center"], e["destination_center"],
            weight=float(e["median_factor"]),       # delay ratio as edge weight
            volume=int(e["volume"]),
            excess_delay=float(e["total_excess_delay"]),
            median_osrm=float(e["median_osrm"]),
        )
    nx.set_node_attributes(G, name_lookup, "name")
    return G


# ----------------------------------------------------------------------------
# 3. Node-level graph metrics
# ----------------------------------------------------------------------------
def node_metrics(G: nx.DiGraph, legs: pd.DataFrame) -> pd.DataFrame:
    print("[graph] computing centrality on "
          f"{G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges ...")

    betw = nx.betweenness_centrality(G, normalized=True)       # chokepoint metric
    pr = nx.pagerank(G, weight="volume")                       # volume-weighted importance
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    und = G.to_undirected()
    clust = nx.clustering(und)                                 # local redundancy

    # operational throughput + delay attributed to each facility
    as_src = legs.groupby("source_center").agg(
        out_legs=("trip_uuid", "size"),
        out_delay=("delay_min", "sum"),
    )
    as_dst = legs.groupby("destination_center").agg(
        in_legs=("trip_uuid", "size"),
        in_delay=("delay_min", "sum"),
    )

    rows = []
    for n in G.nodes():
        out_legs = int(as_src["out_legs"].get(n, 0))
        in_legs = int(as_dst["in_legs"].get(n, 0))
        out_delay = float(as_src["out_delay"].get(n, 0.0))
        in_delay = float(as_dst["in_delay"].get(n, 0.0))
        rows.append({
            "facility": n,
            "name": G.nodes[n].get("name", n),
            "betweenness": betw.get(n, 0.0),
            "pagerank": pr.get(n, 0.0),
            "in_degree": indeg.get(n, 0),
            "out_degree": outdeg.get(n, 0),
            "degree": indeg.get(n, 0) + outdeg.get(n, 0),
            "clustering": clust.get(n, 0.0),
            "throughput_legs": in_legs + out_legs,
            "total_excess_delay": in_delay + out_delay,   # minutes of lateness handled
        })
    m = pd.DataFrame(rows)
    return m


# ----------------------------------------------------------------------------
# 4. Composite bottleneck score
# ----------------------------------------------------------------------------
def add_bottleneck_score(m: pd.DataFrame) -> pd.DataFrame:
    """
    A hub is a true network bottleneck when it is BOTH structurally central AND
    a large delay contributor AND high-volume. We percentile-rank each component
    (robust to skew) and combine. Weights favour delay contribution because the
    business goal is SLA-breach reduction, with structural centrality next.
    """
    def pct(s):  # percentile rank in [0,1]
        return s.rank(pct=True)

    m["delay_pct"] = pct(m["total_excess_delay"])
    m["betw_pct"] = pct(m["betweenness"])
    m["thru_pct"] = pct(m["throughput_legs"])
    m["bottleneck_score"] = (
        0.50 * m["delay_pct"] + 0.30 * m["betw_pct"] + 0.20 * m["thru_pct"]
    )
    return m.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="../data/legs.csv")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    legs = pd.read_csv(args.legs)
    corr = build_corridor_table(legs)
    G = build_graph(corr, legs)
    m = node_metrics(G, legs)
    hubs = add_bottleneck_score(m)

    # chronic-delay corridors: meaningfully delayed AND carrying volume
    vol_floor = max(5, int(corr["volume"].quantile(0.50)))
    chronic = corr[(corr["median_factor"] > 1.2) & (corr["volume"] >= vol_floor)].copy()
    chronic = chronic.sort_values("total_excess_delay", ascending=False).reset_index(drop=True)

    # persist
    nx.write_graphml(G, out / "network.graphml")
    m.sort_values("betweenness", ascending=False).to_csv(out / "node_metrics.csv", index=False)
    corr.sort_values("total_excess_delay", ascending=False).to_csv(out / "corridors.csv", index=False)
    hubs.to_csv(out / "bottleneck_hubs.csv", index=False)
    chronic.to_csv(out / "delayed_corridors.csv", index=False)

    # report
    print("\n" + "=" * 78)
    print(" TOP 10 BOTTLENECK HUBS (composite score)")
    print("=" * 78)
    show = hubs.head(10)[["name", "bottleneck_score", "betweenness",
                          "throughput_legs", "total_excess_delay"]]
    show = show.rename(columns={"total_excess_delay": "excess_delay_min"})
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n" + "=" * 78)
    print(f" CHRONIC-DELAY CORRIDORS (factor>1.2, volume>={vol_floor}):"
          f" {len(chronic):,} found")
    print("=" * 78)
    cc = chronic.head(8)[["source_name", "dest_name", "volume",
                          "median_factor", "total_excess_delay"]]
    print(cc.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n[done] Phase 2 complete -> outputs/bottleneck_hubs.csv, delayed_corridors.csv")


if __name__ == "__main__":
    main()
