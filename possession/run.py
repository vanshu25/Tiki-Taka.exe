"""End-to-end: BAR-ATH tracking -> frame,label CSV.

    MTAH_DATA_DIR=~/mtah-data python -m possession.run --out submissions/<team>.csv

Every parameter used here is frozen in possession/params.json, chosen on the external harness
(possession/harness.py). Nothing here reads any BAR-ATH ground truth.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, anchor_video_frame, asset_paths, half_video_blocks  # noqa: E402
from possession import atd, features, harness, model  # noqa: E402

PARAMS_PATH = os.path.join(ROOT, "possession", "params.json")
LABELS = np.array(["H", "A", "D", "X"])


def load_match():
    p = asset_paths()
    n = GAMES[GAME]["n_video_frames"]
    d = atd.load(p["atd_meta"], p["atd_raw"], n)
    period = np.zeros(n, np.int8)
    blocks = half_video_blocks(GAME)
    for half, a, b in blocks:
        period[a:b + 1] = half
    d["period"] = period
    return d, blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    ap.add_argument("--save-probs", action="store_true",
                    help="also write the per-frame probabilities (gitignored cache/)")
    args = ap.parse_args()

    with open(PARAMS_PATH) as fh:
        params = json.load(fh)
    pitch = tuple(GAMES[GAME]["pitch"])

    # 1. Train on every degraded external game (never on this match).
    ball_runs = harness.bar_ath_ball_runs()
    train = [harness.game_features(n, s, ball_runs)[:3]
             for n in harness.external.GAMES for s in harness.SEEDS]
    clf = model.train(train)

    # 2. Features and probabilities for BAR-ATH.
    d, blocks = load_match()
    prep = features.prepare(d, pitch)
    X0, _ = features.build(prep, 0)
    X1, _ = features.build(prep, 1)
    probs = model.predict(clf, X0, X1)

    # 3. Decode each half. The second half's graded block opens before its kick-off; the ball
    #    is not in play until the kick-off anchor published in games.py, so those frames are D.
    labels = np.full(len(probs), model.X, np.int8)
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    for half, a, b in blocks:
        start = a
        if half == 2 and a < kickoff2 <= b:
            labels[a:kickoff2] = model.D
            start = kickoff2
        labels[start:b + 1] = model.viterbi(probs[start:b + 1], params["switch"],
                                            params["to_dead"], params["from_dead"])

    # 4. Write.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LABELS[c]) for i, c in enumerate(labels))
    if args.save_probs:
        os.makedirs(harness.CACHE, exist_ok=True)
        np.save(os.path.join(harness.CACHE, "bar_ath_probs.npy"), probs)

    graded = labels != model.X
    counts = {k: int((labels == i).sum()) for i, k in enumerate("HADX")}
    print("wrote", args.out, counts)
    import possession_metrics as pm
    runs = [(a, b + 1) for _, a, b in blocks]
    print("possession stats:", pm.possession_stats(labels, runs))
    print("graded frames labelled:", int(graded.sum()))


if __name__ == "__main__":
    main()
