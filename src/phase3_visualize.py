"""
================================================================================
 PHASE 3: Network & bottleneck visualizations
================================================================================
 Produces the figures for the deliverable:

   fig_network.png        - top-hub subgraph; node size = throughput,
                            node colour = excess delay, edge width = volume
   fig_bottleneck_hubs.png- top 15 hubs by composite bottleneck score
   fig_delay_corridors.png- top chronic-delay corridors
   fig_betw_vs_delay.png  - structural centrality vs delay contribution

 The full graph (1,657 nodes) is a hairball, so the network figure plots the
 top-N facilities by throughput - the operationally relevant core.

 USAGE
   python phase3_visualize.py --out ../outputs
================================================================================
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": False})
TOPN_NODES = 60


def short(name: str, n: int = 18) -> str:
    name = str(name).split(" (")[0]
    return name if len(name) <= n else name[: n - 1] + "\u2026"


def fig_network(legs, hubs, out: Path):
    keep = set(hubs.head(TOPN_NODES)["facility"])
    sub = legs[legs["source_center"].isin(keep) & legs["destination_center"].isin(keep)]
    G = nx.DiGraph()
    for (s, d), g in sub.groupby(["source_center", "destination_center"]):
        G.add_edge(s, d, volume=len(g))
    if G.number_of_nodes() == 0:
        return
    info = hubs.set_index("facility")
    sizes = np.array([info.loc[n, "throughput_legs"] for n in G.nodes()], float)
    delays = np.array([info.loc[n, "total_excess_delay"] for n in G.nodes()], float)
    node_size = 60 + 1400 * (sizes / sizes.max())
    vols = np.array([G[u][v]["volume"] for u, v in G.edges()], float)
    ew = 0.3 + 3.0 * (vols / vols.max())

    pos = nx.spring_layout(G, k=0.9, seed=42, iterations=120)
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, width=ew, edge_color="#B4B2A9",
                           alpha=0.5, arrows=True, arrowsize=7,
                           connectionstyle="arc3,rad=0.06", ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=delays,
                                   cmap="YlOrRd", edgecolors="#444441",
                                   linewidths=0.6, ax=ax)
    top_lbl = {n: short(info.loc[n, "name"]) for n in hubs.head(12)["facility"] if n in G}
    nx.draw_networkx_labels(G, pos, labels=top_lbl, font_size=8,
                            font_weight="bold", ax=ax)
    cb = fig.colorbar(nodes, ax=ax, shrink=0.6)
    cb.set_label("Excess delay handled (minutes)")
    ax.set_title(f"Delhivery core network - top {TOPN_NODES} facilities\n"
                 "node size = throughput, colour = delay, edge width = volume",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "fig_network.png", bbox_inches="tight"); plt.close(fig)


def fig_hubs(hubs, out: Path):
    t = hubs.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([short(x) for x in t["name"]], t["bottleneck_score"], color="#D85A30")
    ax.set_xlabel("Composite bottleneck score (0-1)")
    ax.set_title("Top 15 bottleneck hubs\n(0.5*delay + 0.3*betweenness + 0.2*throughput)")
    for i, (s, b) in enumerate(zip(t["bottleneck_score"], t["betweenness"])):
        ax.text(s + 0.005, i, f"betw={b:.2f}", va="center", fontsize=8, color="#5F5E5A")
    ax.set_xlim(0, 1.12)
    fig.tight_layout(); fig.savefig(out / "fig_bottleneck_hubs.png", bbox_inches="tight"); plt.close(fig)


def fig_corridors(chronic, out: Path):
    t = chronic.head(12).iloc[::-1]
    lbl = [f"{short(s,14)} \u2192 {short(d,14)}" for s, d in zip(t["source_name"], t["dest_name"])]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(lbl, t["total_excess_delay"], color="#185FA5")
    ax.set_xlabel("Total excess delay contributed (minutes)")
    ax.set_title("Top chronic-delay corridors (actual > 1.2x OSRM, with volume)")
    for i, (v, f) in enumerate(zip(t["total_excess_delay"], t["median_factor"])):
        ax.text(v, i, f"  {f:.2f}x", va="center", fontsize=8, color="#0C447C")
    fig.tight_layout(); fig.savefig(out / "fig_delay_corridors.png", bbox_inches="tight"); plt.close(fig)


def fig_scatter(hubs, out: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(hubs["betweenness"], hubs["total_excess_delay"],
                    s=20 + 120 * hubs["thru_pct"], c=hubs["bottleneck_score"],
                    cmap="viridis", alpha=0.75, edgecolors="none")
    for _, r in hubs.head(8).iterrows():
        ax.annotate(short(r["name"], 14), (r["betweenness"], r["total_excess_delay"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Betweenness centrality (structural chokepoint)")
    ax.set_ylabel("Total excess delay handled (minutes)")
    ax.set_title("Where structural risk meets delay\n(top-right = priority bottlenecks)")
    fig.colorbar(sc, ax=ax, shrink=0.7).set_label("Bottleneck score")
    fig.tight_layout(); fig.savefig(out / "fig_betw_vs_delay.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="../data/legs.csv")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"

    legs = pd.read_csv(args.legs)
    hubs = pd.read_csv(out / "bottleneck_hubs.csv")
    chronic = pd.read_csv(out / "delayed_corridors.csv")

    print("[viz] network ..."); fig_network(legs, hubs, figdir)
    print("[viz] hubs ...");    fig_hubs(hubs, figdir)
    print("[viz] corridors ..."); fig_corridors(chronic, figdir)
    print("[viz] scatter ...");  fig_scatter(hubs, figdir)
    print(f"[done] Phase 3 -> 4 figures in {figdir}")


if __name__ == "__main__":
    main()
