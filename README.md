# Delhivery — Graph-Based Delivery Intelligence

Optimizing delivery ETAs and surfacing bottleneck hubs by modelling Delhivery's
logistics network as a **directed weighted graph** (facilities = nodes, corridors
= edges) instead of a set of independent point-to-point estimates.

Submission for the Summer Analytics graph-analytics challenge. Every number in
the strategy memo is reproduced by the code in `src/`.

**Live dashboard:** https://delhivery-graph-intelligence.streamlit.app

---

## TL;DR results

| What | Result |
|------|--------|
| Network | 14,817 trips → 26,369 corridor legs → 1,657 facilities, 2,783 corridors |
| OSRM bias | actual time is a median **2.0×** OSRM; **94.7%** of legs run >20% over |
| #1 bottleneck | **Gurgaon_Bilaspur_HB** — on **22%** of all shortest paths |
| Delay concentration | **top 3 hubs = 39%** of network excess delay (17.5% of legs) |
| ETA model | graph-enhanced cuts MAE **27.5 → 24.3 min (−11.6%)**, within-15% **51.7 → 55.4%** |
| Route policy | FTL on long-haul (saves 60–77 min/shipment), Carting on short/mid |

See **`STRATEGY_MEMO.md`** for the full consulting write-up.

---

## Repository structure

```
delhivery_graph_intelligence/
├── README.md
├── requirements.txt
├── STRATEGY_MEMO.md            # deliverable 5 — the consulting memo
├── run_all.py                  # runs phases 1–6 end to end
├── data/
│   ├── legs.csv                # one row per OD leg (graph edge list + model base)
│   └── trips.csv               # one row per full trip
├── src/
│   ├── phase1_pipeline.py      # clean + un-cumulate raw scans -> legs/trips
│   ├── phase2_graph_audit.py   # build graph, centrality, rank bottlenecks
│   ├── phase3_visualize.py     # network + bottleneck figures
│   ├── phase45_eta_benchmark.py# baseline vs graph-enhanced ETA (deliverable 3)
│   ├── phase6_ftl_vs_carting.py# route-type decision framework (deliverable 4)
│   └── dashboard_streamlit.py  # optional live dashboard (deliverable 6)
└── outputs/
    ├── bottleneck_hubs.csv     delayed_corridors.csv   corridors.csv
    ├── node_metrics.csv        network.graphml
    ├── model_comparison.csv    ftl_carting_recommendations.csv
    └── figures/*.png
```

## How to run

```bash
pip install -r requirements-dev.txt   # full pipeline (graph + ML phases)
# requirements.txt is the slim set the hosted dashboard needs (streamlit/pandas/numpy/altair)

# from the project root, regenerate everything:
python run_all.py --raw path/to/delivery_data.csv

# or run a single phase (from src/):
cd src
python phase1_pipeline.py     --csv ../../delivery_data.csv --out ../data
python phase2_graph_audit.py  --legs ../data/legs.csv       --out ../outputs
python phase3_visualize.py    --legs ../data/legs.csv       --out ../outputs
python phase45_eta_benchmark.py --legs ../data/legs.csv     --out ../outputs
python phase6_ftl_vs_carting.py --legs ../data/legs.csv     --out ../outputs

# optional dashboard:
streamlit run src/dashboard_streamlit.py
```

`data/legs.csv` and `data/trips.csv` plus all `outputs/` are pre-generated, so the
graph/model phases run without the raw file. To rebuild from scratch, supply the
raw `delivery_data.csv` to Phase 1.

## Method notes (the choices that matter)

- **Grain.** Each raw row is a *scan*. Within a trip there are several OD legs, and
  inside each leg `actual_time`/`osrm_time` are **cumulative and reset per leg**.
  Phase 1 collapses to leg grain with `max` on cumulative fields (validated against
  segment sums to within ~0.9%). Getting this wrong double-counts everything.
- **Edge weight = median factor.** The `factor` column has a heavy right tail
  (max 77×), so the median is used throughout — mean would be dominated by outliers.
- **Bottleneck score = 0.5·delay + 0.3·betweenness + 0.2·throughput** (percentile-
  ranked). A hub must be central *and* slow *and* busy to rank.
- **Graph advantage is measured, not claimed:** trip-grouped hold-out (no leg from a
  test trip leaks into training), reported on MAE and the within-15% business metric.
- **Cost numbers in the FTL/Carting framework and memo are explicit, tunable
  assumptions**, flagged as such — the framework reports breakeven values so
  decisions don't hinge on a guessed constant.
```
