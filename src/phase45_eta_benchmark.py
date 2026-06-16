"""
================================================================================
 PHASE 4 + 5: ETA prediction - baseline vs graph-enhanced (the benchmark)
================================================================================
 Deliverable 3: prove the "graph advantage", measured not claimed.

   BASELINE  : gradient-boosted regression on trip/leg features only
               (osrm_time, osrm_distance, actual_distance, route_type,
                time-of-day, leg position, ...). Target = actual_time.

   GRAPH     : same model + graph signals for the source & destination
               facilities: node2vec embeddings + centrality
               (betweenness, pagerank, degree). If node2vec is unavailable
               we fall back to centrality-only graph features, so the
               pipeline still runs and the comparison is still valid.

 Both are evaluated on a TRIP-GROUPED hold-out (no leg from a test trip
 appears in training -> no leakage) using:
     MAE (minutes)  and  within-15% accuracy  (the business metric).

 OUTPUTS
   model_comparison.csv         - the metrics table
   figures/fig_model_compare.png

 USAGE
   python phase45_eta_benchmark.py --legs ../data/legs.csv --out ../outputs
================================================================================
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
EMB_DIM = 16


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
BASE_NUM = ["osrm_time", "osrm_distance", "actual_distance", "scan_time",
            "leg_seq", "hour", "n_segments"]
BASE_CAT = ["route_type", "tod_bucket"]


def base_features(legs: pd.DataFrame) -> pd.DataFrame:
    X = legs[BASE_NUM].copy()
    for c in BASE_CAT:
        d = pd.get_dummies(legs[c], prefix=c, dtype=float)
        X = pd.concat([X, d], axis=1)
    return X


def node2vec_embeddings(G: nx.DiGraph) -> dict | None:
    try:
        from node2vec import Node2Vec
        print("[graph] training node2vec embeddings ...")
        # workers=1 -> deterministic embeddings (results reproduce exactly)
        n2v = Node2Vec(G, dimensions=EMB_DIM, walk_length=12, num_walks=30,
                       workers=1, seed=RANDOM_STATE, quiet=True)
        model = n2v.fit(window=5, min_count=1, batch_words=128,
                        seed=RANDOM_STATE, workers=1)
        return {n: model.wv[str(n)] for n in G.nodes() if str(n) in model.wv}
    except Exception as e:
        print(f"[graph] node2vec unavailable ({e}); using centrality-only graph features")
        return None


def graph_features(legs, node_metrics, emb) -> pd.DataFrame:
    nm = node_metrics.set_index("facility")
    cent_cols = ["betweenness", "pagerank", "degree", "clustering", "throughput_legs"]

    def cent(side):
        col = "source_center" if side == "src" else "destination_center"
        f = legs[col].map(lambda n: nm.loc[n, cent_cols] if n in nm.index else None)
        df = pd.DataFrame(list(f), index=legs.index)
        df.columns = [f"{side}_{c}" for c in cent_cols]
        return df

    parts = [cent("src"), cent("dst")]
    if emb is not None:
        dim = len(next(iter(emb.values())))
        zero = np.zeros(dim)
        se = np.vstack([emb.get(n, zero) for n in legs["source_center"]])
        de = np.vstack([emb.get(n, zero) for n in legs["destination_center"]])
        parts.append(pd.DataFrame(se, index=legs.index, columns=[f"src_emb{i}" for i in range(dim)]))
        parts.append(pd.DataFrame(de, index=legs.index, columns=[f"dst_emb{i}" for i in range(dim)]))
    return pd.concat(parts, axis=1)


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    pct = np.abs(y_pred - y_true) / np.clip(y_true, 1e-6, None)
    return {
        "MAE_min": mean_absolute_error(y_true, y_pred),
        "RMSE_min": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "within_15pct": float((pct <= 0.15).mean() * 100),
    }


def fit_eval(X, y, tr, te) -> dict:
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=8,
        l2_regularization=1.0, random_state=RANDOM_STATE)
    model.fit(X.iloc[tr], y.iloc[tr])
    return evaluate(y.iloc[te].values, model.predict(X.iloc[te]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="../data/legs.csv")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    legs = pd.read_csv(args.legs)
    legs = legs[(legs["actual_time"] > 0) & (legs["osrm_time"] > 0)].reset_index(drop=True)
    node_metrics = pd.read_csv(out / "node_metrics.csv")
    G = nx.read_graphml(out / "network.graphml")

    y = legs["actual_time"]
    groups = legs["trip_uuid"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(legs, y, groups))
    print(f"[split] train legs={len(tr):,}  test legs={len(te):,} (grouped by trip)")

    # baseline
    Xb = base_features(legs)
    base = fit_eval(Xb, y, tr, te)

    # graph-enhanced
    emb = node2vec_embeddings(G)
    Xg = pd.concat([Xb, graph_features(legs, node_metrics, emb)], axis=1).astype(float)
    graph = fit_eval(Xg, y, tr, te)

    # comparison table
    comp = pd.DataFrame([
        {"model": "Baseline (trip features)", **base},
        {"model": "Graph-enhanced (+node2vec+centrality)", **graph},
    ])
    comp["MAE_improvement_%"] = [0.0, 100 * (base["MAE_min"] - graph["MAE_min"]) / base["MAE_min"]]
    comp["within15_gain_pts"] = [0.0, graph["within_15pct"] - base["within_15pct"]]
    comp.to_csv(out / "model_comparison.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n" + "=" * 78)
    print(" MODEL COMPARISON (trip-grouped hold-out)")
    print("=" * 78)
    print(comp.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # figure
    labels = ["Baseline", "Graph-enhanced"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(labels, [base["MAE_min"], graph["MAE_min"]], color=["#888780", "#1D9E75"])
    axes[0].set_ylabel("MAE (minutes)  - lower better"); axes[0].set_title("ETA error")
    for i, v in enumerate([base["MAE_min"], graph["MAE_min"]]):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    axes[1].bar(labels, [base["within_15pct"], graph["within_15pct"]], color=["#888780", "#1D9E75"])
    axes[1].set_ylabel("% within 15% of actual  - higher better"); axes[1].set_title("Business metric")
    for i, v in enumerate([base["within_15pct"], graph["within_15pct"]]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.suptitle("Baseline vs graph-enhanced ETA", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "figures" / "fig_model_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("\n[done] Phase 4+5 -> outputs/model_comparison.csv + fig_model_compare.png")


if __name__ == "__main__":
    main()
