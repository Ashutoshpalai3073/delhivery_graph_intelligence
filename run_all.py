"""
Run the full pipeline end to end: Phase 1 -> 6.

    python run_all.py --raw path/to/delivery_data.csv

If --raw is omitted, Phase 1 is skipped and the pre-built data/legs.csv is used.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def run(script, *args):
    cmd = [sys.executable, str(SRC / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="path to raw delivery_data.csv")
    args = ap.parse_args()

    data, out = ROOT / "data", ROOT / "outputs"
    if args.raw:
        run("phase1_pipeline.py", "--csv", args.raw, "--out", str(data))
    else:
        print("[run_all] --raw not given; using existing data/legs.csv")

    legs = str(data / "legs.csv")
    run("phase2_graph_audit.py", "--legs", legs, "--out", str(out))
    run("phase3_visualize.py", "--legs", legs, "--out", str(out))
    run("phase45_eta_benchmark.py", "--legs", legs, "--out", str(out))
    run("phase6_ftl_vs_carting.py", "--legs", legs, "--out", str(out))
    print("\n[run_all] complete. See outputs/ and STRATEGY_MEMO.md")


if __name__ == "__main__":
    main()
