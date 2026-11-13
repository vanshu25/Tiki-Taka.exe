"""Run the LOGO harness for the current feature VERSION and report spellF1 at frozen params,
plus a small decoder grid. Used to measure each change against the previous one.

    python -m possession.experiment [--tag name]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import features, harness, model  # noqa: E402
import possession_metrics as pm  # noqa: E402,I001

GRID = dict(switch=(3, 5, 7, 10, 14), to_dead=(10, 20, 40), from_dead=(5, 10, 20))


def evaluate(pb, sw, td, fd):
    per = {}
    for key, (probs, labels) in sorted(pb.items()):
        per[key] = harness.score(labels, model.decode(probs, pm.valid_runs(labels), sw, td, fd))
    sp = np.array([v["spell"] for v in per.values()])
    ma = np.array([v["macro"] for v in per.values()])
    ratio = np.mean([v["n_pred"] / v["n_gt"] for v in per.values()])
    return sp.mean(), sp.min(), ma.mean(), ratio, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--feature-version", type=int, default=features.VERSION)
    args = ap.parse_args()
    features.VERSION = args.feature_version
    args.tag = args.tag or "f{}_d{}".format(features.VERSION, harness.degrade.VERSION)
    with open(os.path.join(ROOT, "possession", "params.json")) as fh:
        frozen = json.load(fh)
    pb = harness.logo_probs(harness.bar_ath_ball_runs())
    m, lo, ma, r, per = evaluate(pb, frozen["switch"], frozen["to_dead"], frozen["from_dead"])
    print("[{}] frozen params: spell {:.4f} (min {:.4f}) macro {:.4f} poss ratio {:.2f}".format(
        args.tag, m, lo, ma, r))
    for k, v in per.items():
        print("   game {} seed {}: spell {:.4f} macro {:.4f} poss {}/{}".format(
            k[0], k[1], v["spell"], v["macro"], v["n_pred"], v["n_gt"]))
    rows = []
    for sw, td, fd in itertools.product(*GRID.values()):
        m2, lo2, ma2, r2, _ = evaluate(pb, sw, td, fd)
        rows.append((m2, lo2, ma2, r2, sw, td, fd))
    rows.sort(reverse=True)
    print("[{}] best of grid:".format(args.tag))
    for row in rows[:5]:
        print("   spell {:.4f} (min {:.4f}) macro {:.4f} ratio {:.2f}  switch {} to_dead {} "
              "from_dead {}".format(*row))
    os.makedirs(harness.CACHE, exist_ok=True)
    with open(os.path.join(harness.CACHE, "exp_{}.json".format(args.tag)), "w") as fh:
        json.dump({"frozen": [m, lo, ma, r], "grid": rows}, fh)
