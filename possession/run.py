"""End-to-end (pipeline v5): BAR-ATH tracking -> frame,label CSV.

    MTAH_DATA_DIR=~/mtah-data python -m possession.run --out submissions/<team>.csv

The models are trained on the degraded practice games only (possession/v5/learned.py); every
constant is in possession/v5/pipeline.py with its reason. Nothing here reads any BAR-ATH ground
truth. The previous pipeline is kept as possession/run_v4.py.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from possession.v5 import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    args = ap.parse_args()

    lab, runs = pipeline.run()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, c) for i, c in enumerate(lab))

    import possession_metrics as pm
    codes = pm.label_codes(lab)
    print("wrote", args.out, {k: int((lab == k).sum()) for k in "HADX"})
    print("possession stats:", pm.possession_stats(codes, runs))


if __name__ == "__main__":
    main()
