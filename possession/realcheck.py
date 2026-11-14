"""A check on the real match that needs no ground truth: TV replays as a witness of dead balls.

    python -m possession.realcheck [other_submission.csv ...]

When the broadcast feed has no pitch homography for 2 s or more, the director has cut to a replay
or a close-up, which happens almost only while the ball is dead. So a good labelling should call
most of those frames D -- including the ones where the ball is still tracked on the tactical
camera, which is exactly where a model trained on artificial dead balls goes wrong.

v5 is run here WITHOUT its broadcast evidence, so the witness is independent of it. Any CSV given
on the command line is measured the same way.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, anchor_video_frame, asset_paths, half_video_blocks  # noqa: E402
from possession import atd  # noqa: E402
from possession.v5 import learned, pipeline  # noqa: E402
from possession.v5.broadcast import broadcast_evidence  # noqa: E402


def main():
    n = GAMES[GAME]["n_video_frames"]
    paths, bpaths = asset_paths(), asset_paths(source="broadcast")
    blocks = half_video_blocks(GAME)
    hv = [(a, b + 1) for _, a, b in blocks]
    tactical = learned.to_arrays(atd.load(paths["atd_meta"], paths["atd_raw"], n))
    broadcast = learned.to_arrays(atd.load(bpaths["atd_meta"], bpaths["atd_raw"]))
    bc, _ = broadcast_evidence(tactical, broadcast)
    cutaway = pipeline._long_runs_only(bc, pipeline.CONFIG["bc_min_run"])

    p = learned.prepare(tactical, hv, pipeline.goalkeeper_slots(paths["atd_meta"]))
    k2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    forced = [(a, k2) for half, a, b in blocks if half == 2 and a < k2 <= b]
    labels = {"v5 (broadcast evidence off)": pipeline.label_game(
        p, learned.get_models(), forced_dead=forced)}
    for path in sys.argv[1:]:
        labels[os.path.basename(path)] = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str)[:, 1]

    graded = np.zeros(n, bool)
    for a, b in hv:
        graded[a:b] = True
    ok = graded & np.isfinite(cutaway)
    cut, live, vis = ok & (cutaway > 0.5), ok & (cutaway < 0.5), p["fe"]["ball_vis"]
    print("replays and close-ups cover {:.1f}% of graded frames; the ball is tracked on {:.0f}% of "
          "them".format(100 * cut.sum() / ok.sum(), 100 * vis[cut].mean()))
    print("{:34s} {:>10s} {:>22s} {:>14s}".format("share of frames labelled D", "replays",
                                                  "replays, ball tracked", "live picture"))
    for name, lab in labels.items():
        print("{:34s} {:10.3f} {:22.3f} {:14.3f}".format(
            name, np.mean(lab[cut] == "D"), np.mean(lab[cut & vis] == "D"),
            np.mean(lab[live] == "D")))


if __name__ == "__main__":
    main()
