"""
================================================================================
 PHASE 6: FTL vs Carting decision framework
================================================================================
 Deliverable 4: an ML-backed route-type recommender with the time-cost
 trade-off quantified by corridor profile (distance, time-of-day, and the
 SOURCE facility's graph position).

 APPROACH
 --------
 1. Descriptive: how do FTL and Carting actually compare (speed, delay) by
    distance band? Grounds the model in observed behaviour.
 2. Counterfactual time model: one regressor predicts actual_time from
    corridor features INCLUDING route_type and source-centrality. For every
    leg we predict time under BOTH route types -> time saved by choosing FTL.
 3. Transparent cost model (tunable assumptions, stated below): FTL runs a
    dedicated truck (faster, dearer); Carting consolidates (cheaper, slower).
    We recommend FTL when the value of time saved beats the cost premium, and
    report the BREAKEVEN SLA value (Rs/shipment-minute) so ops can tune it.

 COST ASSUMPTIONS (edit freely - they are levers, not facts)
   FTL is taken as the cost baseline; Carting ~38% cheaper per shipment.
   Default SLA value of saved time = a tunable Rs/min. The framework reports
   the breakeven so the decision does not hinge on a guessed number.

 OUTPUTS
   ftl_carting_recommendations.csv  - per-corridor-profile recommendation
   ftl_vs_carting_observed.csv      - descriptive comparison by distance band
   figures/fig_ftl_carting.png

 USAGE
   python phase6_ftl_vs_carting.py --legs ../data/legs.csv --out ../outputs
================================================================================
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
# --- cost levers (relative; FTL = 1.0 baseline) -----------------------------
FTL_COST = 1.0
CARTING_COST = 0.62          # carting ~38% cheaper per shipment (assumption)
COST_PER_SHIPMENT_FTL = 1200.0   # Rs, illustrative absolute anchor for premium
DIST_BANDS = [0, 50, 200, 800, 1e9]
DIST_LABELS = ["short <50km", "mid 50-200", "long 200-800", "ultra >800"]


def add_features(legs, node_metrics):
    nm = node_metrics.set_index("facility")
    legs = legs.copy()
    legs["src_betweenness"] = legs["source_center"].map(nm["betweenness"]).fillna(0)
    legs["src_throughput"] = legs["source_center"].map(nm["throughput_legs"]).fillna(0)
    legs["dist_band"] = pd.cut(legs["osrm_distance"], DIST_BANDS, labels=DIST_LABELS)
    return legs


def observed_table(legs, out):
    """Descriptive FTL vs Carting comparison by distance band."""
    rows = []
    for band in DIST_LABELS:
        b = legs[legs["dist_band"] == band]
        for rt in ["FTL", "Carting"]:
            s = b[b["route_type"] == rt]
            if len(s) == 0:
                continue
            rows.append({
                "dist_band": band, "route_type": rt, "legs": len(s),
                "median_actual_min": s["actual_time"].median(),
                "median_factor": s["factor"].median(),
                "median_speed_kmph": s["actual_speed"].median(),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ftl_vs_carting_observed.csv", index=False)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="../data/legs.csv")
    ap.add_argument("--out", default="../outputs")
    args = ap.parse_args()
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    legs = pd.read_csv(args.legs)
    legs = legs[(legs["actual_time"] > 0) & (legs["osrm_time"] > 0) &
                (legs["osrm_distance"] > 0)].reset_index(drop=True)
    node_metrics = pd.read_csv(out / "node_metrics.csv")
    legs = add_features(legs, node_metrics)

    obs = observed_table(legs, out)
    print("=" * 72); print(" OBSERVED: FTL vs Carting by distance band"); print("=" * 72)
    print(obs.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- counterfactual time model -----------------------------------------
    feat_num = ["osrm_time", "osrm_distance", "actual_distance", "hour",
                "src_betweenness", "src_throughput"]
    Xnum = legs[feat_num].copy()
    tod = pd.get_dummies(legs["tod_bucket"], prefix="tod", dtype=float)
    is_ftl = (legs["route_type"] == "FTL").astype(float).rename("is_ftl")
    X = pd.concat([Xnum, tod, is_ftl], axis=1).astype(float)
    y = legs["actual_time"]

    model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06,
                                          max_depth=8, random_state=RANDOM_STATE)
    model.fit(X, y)

    # counterfactual: predict each leg under FTL and under Carting
    X_ftl = X.copy(); X_ftl["is_ftl"] = 1.0
    X_cart = X.copy(); X_cart["is_ftl"] = 0.0
    t_ftl = model.predict(X_ftl)
    t_cart = model.predict(X_cart)
    legs["pred_ftl_min"] = t_ftl
    legs["pred_carting_min"] = t_cart
    legs["time_saved_ftl_min"] = t_cart - t_ftl          # >0 => FTL faster

    # cost premium of FTL per shipment (absolute, illustrative)
    cost_premium = COST_PER_SHIPMENT_FTL * (FTL_COST - CARTING_COST)  # Rs
    # breakeven SLA value: Rs per saved minute at which FTL pays for itself
    legs["breakeven_sla_per_min"] = cost_premium / legs["time_saved_ftl_min"].clip(lower=1e-6)

    # ---- recommendation by corridor profile (distance x tod) ---------------
    prof = legs.groupby(["dist_band", "tod_bucket"], observed=True).agg(
        legs=("trip_uuid", "size"),
        avg_time_saved_ftl_min=("time_saved_ftl_min", "mean"),
        median_breakeven_sla=("breakeven_sla_per_min", "median"),
        avg_src_betweenness=("src_betweenness", "mean"),
    ).reset_index()
    # recommend FTL where it saves meaningful time at a reasonable SLA value
    SLA_VALUE_PER_MIN = 8.0   # Rs/min assumption (tunable lever)
    prof["recommend"] = np.where(
        prof["avg_time_saved_ftl_min"] * SLA_VALUE_PER_MIN >= cost_premium,
        "FTL", "Carting")
    prof = prof.sort_values(["dist_band", "tod_bucket"]).reset_index(drop=True)
    prof.to_csv(out / "ftl_carting_recommendations.csv", index=False)

    print("\n" + "=" * 72)
    print(f" RECOMMENDATIONS (cost premium of FTL = Rs {cost_premium:,.0f}/shipment,"
          f" SLA value = Rs {SLA_VALUE_PER_MIN:.0f}/min)")
    print("=" * 72)
    print(prof.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    piv = obs.pivot(index="dist_band", columns="route_type", values="median_speed_kmph").reindex(DIST_LABELS)
    piv.plot(kind="bar", ax=axes[0], color={"FTL": "#185FA5", "Carting": "#EF9F27"})
    axes[0].set_title("Observed median speed by route type"); axes[0].set_ylabel("km/h")
    axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=20)

    band_save = legs.groupby("dist_band", observed=True)["time_saved_ftl_min"].mean().reindex(DIST_LABELS)
    axes[1].bar(DIST_LABELS, band_save.values, color="#1D9E75")
    axes[1].axhline(0, color="#444441", lw=0.8)
    axes[1].set_title(f"Avg time FTL saves vs Carting\n(breakeven needs Rs {cost_premium:,.0f}/shipment of value)")
    axes[1].set_ylabel("minutes saved per shipment"); axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "figures" / "fig_ftl_carting.png", bbox_inches="tight")
    plt.close(fig)
    print("\n[done] Phase 6 -> ftl_carting_recommendations.csv + fig_ftl_carting.png")


if __name__ == "__main__":
    main()
