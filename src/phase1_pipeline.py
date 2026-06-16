"""
================================================================================
 Delhivery Graph-Intelligence Project  -  PHASE 1: Clean & Reshape
================================================================================
 Turns the raw segment-level export into two analysis-ready tables:

   1. legs_df  -> ONE ROW PER OD LEG (a single hop: source_center -> dest_center)
                  This is the edge list for the graph (Phase 2) and the base
                  table for ETA modelling (Phase 4/5).

   2. trips_df -> ONE ROW PER trip_uuid (the full multi-hop journey)
                  Used for trip-level ETA prediction and SLA accounting.

 WHY THIS IS THE HARD PART
 -------------------------
 Each raw row is a *scan*, not a trip. Within one trip_uuid there are several
 OD legs, and INSIDE each leg the columns `actual_time`, `osrm_time`, `factor`
 are CUMULATIVE running totals that RESET at every new leg. Naively averaging
 the raw rows double-counts everything. The fix:

   - group to the leg grain   (trip_uuid, source_center, destination_center, od_start_time)
   - for cumulative fields (actual_time, osrm_time, distances) take MAX
     (= the value at the last scan of the leg = the leg total)
   - for per-segment fields (segment_*) take SUM (cross-check only)

 Validated on the real file: leg MAX vs segment SUM agree within ~0.9% median,
 so MAX is a reliable, order-independent leg total.

 REQUIREMENTS
 ------------
   pip install pandas numpy
   # optional, for fast columnar output:
   pip install pyarrow

 USAGE
 -----
   python phase1_pipeline.py --csv path/to/delivery_data.csv --out ./data

 Times are in MINUTES, distances in KM (standard for this dataset).
================================================================================
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Small parsing helpers
# ----------------------------------------------------------------------------
def parse_state(name: str) -> str | float:
    """Facility names look like 'Gurgaon_Bilaspur_HB (Haryana)'. State is in parens."""
    if pd.isna(name):
        return np.nan
    m = re.search(r"\(([^)]+)\)", name)
    return m.group(1).strip() if m else np.nan


def parse_city(name: str) -> str | float:
    """City is the token before the first underscore."""
    if pd.isna(name):
        return np.nan
    return name.split("_")[0].strip()


def time_of_day(hour: pd.Series) -> pd.Series:
    """Bucket an hour-of-day Series into 4 operational windows."""
    bins = [-1, 5, 11, 16, 21, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Night"]
    # pandas.cut needs unique labels, so map afterwards
    raw = pd.cut(hour, bins=bins, labels=["Night", "Morning", "Afternoon", "Evening", "LateNight"])
    return raw.astype("object").replace({"LateNight": "Night"})


# ----------------------------------------------------------------------------
# 2. Load + light cleaning
# ----------------------------------------------------------------------------
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    print(f"[load] reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[load] raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Parse the timestamps we actually use
    for col in ["trip_creation_time", "od_start_time", "od_end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop columns irrelevant to graph/ETA work (kept minimal & documented)
    drop_cols = [
        "route_schedule_uuid",   # internal schedule id, not needed
        "is_cutoff", "cutoff_factor", "cutoff_timestamp",  # SLA cutoff metadata
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Parse facility city/state from the human-readable names
    df["source_city"] = df["source_name"].apply(parse_city)
    df["source_state"] = df["source_name"].apply(parse_state)
    df["dest_city"] = df["destination_name"].apply(parse_city)
    df["dest_state"] = df["destination_name"].apply(parse_state)

    # Null facility *names* exist (~few hundred). The center IDs are always
    # present and are what we use as graph node keys, so we fill names from the
    # center id as a readable fallback rather than dropping rows.
    df["source_name"] = df["source_name"].fillna(df["source_center"])
    df["destination_name"] = df["destination_name"].fillna(df["destination_center"])

    return df


# ----------------------------------------------------------------------------
# 3. THE KEY STEP -> collapse cumulative scans into one row per OD leg
# ----------------------------------------------------------------------------
LEG_KEYS = ["trip_uuid", "source_center", "destination_center", "od_start_time"]


def build_legs(df: pd.DataFrame) -> pd.DataFrame:
    print("[legs] collapsing cumulative scans -> one row per OD leg ...")
    g = df.groupby(LEG_KEYS, sort=False)

    legs = g.agg(
        route_type=("route_type", "first"),
        data_split=("data", "first"),                 # 'training' / 'test'
        od_end_time=("od_end_time", "first"),
        trip_creation_time=("trip_creation_time", "first"),
        source_name=("source_name", "first"),
        destination_name=("destination_name", "first"),
        source_city=("source_city", "first"),
        source_state=("source_state", "first"),
        dest_city=("dest_city", "first"),
        dest_state=("dest_state", "first"),
        # cumulative fields -> MAX = value at last scan = leg total
        actual_time=("actual_time", "max"),
        osrm_time=("osrm_time", "max"),
        osrm_distance=("osrm_distance", "max"),
        actual_distance=("actual_distance_to_destination", "max"),
        scan_time=("start_scan_to_end_scan", "max"),
        # per-segment fields -> SUM (cross-check only)
        seg_actual_sum=("segment_actual_time", "sum"),
        seg_osrm_sum=("segment_osrm_time", "sum"),
        n_segments=("actual_time", "size"),
    ).reset_index()

    # --- derived leg features -------------------------------------------------
    legs["od_start_time"] = pd.to_datetime(legs["od_start_time"], errors="coerce")
    legs["factor"] = legs["actual_time"] / legs["osrm_time"]          # delay ratio
    legs["delay_min"] = legs["actual_time"] - legs["osrm_time"]       # raw over-run
    legs["is_delayed_20"] = legs["factor"] > 1.2                      # >20% over OSRM

    # implied speeds (km/h); times are in minutes -> /60 to hours
    legs["osrm_speed"] = legs["osrm_distance"] / (legs["osrm_time"] / 60).replace(0, np.nan)
    legs["actual_speed"] = legs["actual_distance"] / (legs["actual_time"] / 60).replace(0, np.nan)

    # time-of-day + calendar features off the leg's start
    legs["hour"] = legs["od_start_time"].dt.hour
    legs["dow"] = legs["od_start_time"].dt.day_name()
    legs["date"] = legs["od_start_time"].dt.date
    legs["tod_bucket"] = time_of_day(legs["hour"])

    # corridor id (handy for grouping / graph edges later)
    legs["corridor"] = legs["source_center"] + " -> " + legs["destination_center"]

    # leg order within each trip (by start time)
    legs = legs.sort_values(["trip_uuid", "od_start_time"]).reset_index(drop=True)
    legs["leg_seq"] = legs.groupby("trip_uuid").cumcount() + 1

    print(f"[legs] built {len(legs):,} legs across {legs['trip_uuid'].nunique():,} trips")
    return legs


# ----------------------------------------------------------------------------
# 4. Aggregate legs -> one row per full trip
# ----------------------------------------------------------------------------
def build_trips(legs: pd.DataFrame) -> pd.DataFrame:
    print("[trips] aggregating legs -> one row per trip ...")
    legs = legs.sort_values(["trip_uuid", "leg_seq"])

    first = legs.groupby("trip_uuid").first()
    last = legs.groupby("trip_uuid").last()
    agg = legs.groupby("trip_uuid").agg(
        route_type=("route_type", "first"),
        data_split=("data_split", "first"),
        n_legs=("leg_seq", "max"),
        actual_time=("actual_time", "sum"),
        osrm_time=("osrm_time", "sum"),
        osrm_distance=("osrm_distance", "sum"),
        actual_distance=("actual_distance", "sum"),
        scan_time=("scan_time", "sum"),
        trip_start=("od_start_time", "min"),
        trip_end=("od_end_time", "max"),
    )

    trips = agg.copy()
    trips["source_center"] = first["source_center"]
    trips["source_name"] = first["source_name"]
    trips["source_city"] = first["source_city"]
    trips["source_state"] = first["source_state"]
    trips["dest_center"] = last["destination_center"]
    trips["dest_name"] = last["destination_name"]
    trips["dest_city"] = last["dest_city"]
    trips["dest_state"] = last["dest_state"]

    trips = trips.reset_index()
    trips["factor"] = trips["actual_time"] / trips["osrm_time"]
    trips["delay_min"] = trips["actual_time"] - trips["osrm_time"]
    trips["is_delayed_20"] = trips["factor"] > 1.2
    trips["hour"] = trips["trip_start"].dt.hour
    trips["tod_bucket"] = time_of_day(trips["hour"])

    print(f"[trips] built {len(trips):,} trips")
    return trips


# ----------------------------------------------------------------------------
# 5. Save + print a compact summary
# ----------------------------------------------------------------------------
def save(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  ({len(df):,} rows)")
    try:
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"[save] {out_dir / (name + '.parquet')}")
    except Exception as e:  # pyarrow not installed -> CSV is enough
        print(f"[save] (parquet skipped: {e})")


def summary(legs: pd.DataFrame, trips: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(" PHASE 1 SUMMARY")
    print("=" * 70)
    print(f" OD legs                 : {len(legs):,}")
    print(f" Trips                   : {trips.shape[0]:,}")
    print(f" Facilities (nodes)      : {pd.unique(legs[['source_center','destination_center']].values.ravel()).size:,}")
    print(f" Distinct corridors      : {legs['corridor'].nunique():,}")
    print(f" Date range              : {legs['od_start_time'].min()}  ->  {legs['od_start_time'].max()}")
    print("-" * 70)
    print(f" Leg factor (actual/osrm) median : {legs['factor'].median():.2f}")
    print(f" Legs delayed >20%               : {legs['is_delayed_20'].mean()*100:.1f}%")
    print(f" Trip factor median              : {trips['factor'].median():.2f}")
    print(f" Trips delayed >20%              : {trips['is_delayed_20'].mean()*100:.1f}%")
    print("-" * 70)
    print(" Route type mix (legs):")
    print(legs["route_type"].value_counts().to_string())
    print("\n Time-of-day mix (legs):")
    print(legs["tod_bucket"].value_counts().to_string())
    print("=" * 70)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="delivery_data.csv", help="path to raw delivery_data.csv")
    ap.add_argument("--out", default="./data", help="output directory")
    args = ap.parse_args()

    df = load_and_clean(Path(args.csv))
    legs = build_legs(df)
    trips = build_trips(legs)

    out = Path(args.out)
    save(legs, out, "legs")
    save(trips, out, "trips")
    summary(legs, trips)
    print("\n[done] Phase 1 complete. Feed legs.csv into Phase 2 (graph construction).")


if __name__ == "__main__":
    main()
