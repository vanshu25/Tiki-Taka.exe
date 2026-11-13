"""External tuning harness: leave-one-game-out on degraded Metrica sample games.

Every model choice and decoder penalty is picked here, on games that are not BAR-ATH. Nothing in
this file reads the BAR-ATH ground truth; the only thing it takes from BAR-ATH is the ball
visible/missing run-length distribution of its *tracking* (no labels), used by degrade.py.

    python -m possession.harness              # features + LOGO probabilities + decoder grid
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import atd, degrade, external, features, model  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
SEEDS = (0, 1)


def bar_ath_ball_runs():
    """Visible / missing ball run lengths on graded BAR-ATH frames (tracking only, no labels)."""
    from games import asset_paths, half_video_blocks, GAME
    p = asset_paths()
    d = atd.load(p["atd_meta"], p["atd_raw"], 199319)
    ok = np.isfinite(d["ball"][:, 0])
    vis, miss = [], []
    for _, a, b in half_video_blocks(GAME):
        on, off = degrade.run_lengths(ok[a:b + 1])
        vis += list(on[1:-1])  # drop the clipped runs at either end of the half
        miss += list(off[1:-1])
    return np.array(vis), np.array(miss)


def game_features(n, seed, ball_runs, use_cache=True):
    path = os.path.join(CACHE, "ext_g{}_s{}_f{}_d{}{}.npz".format(
        n, seed, features.VERSION, degrade.VERSION,
        "" if external.LABEL_VERSION == 1 else "_l{}".format(external.LABEL_VERSION)))
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X0"], z["X1"], z["labels"], list(z["names"])
    g = external.load_game(n)
    dead = (g["labels"] == model.D) if degrade.VERSION >= 2 else None  # D is label-version-free
    dg = degrade.degrade(g, ball_runs, seed=1000 * n + seed, pitch=external.PITCH, dead=dead)
    prep = features.prepare(dg, external.PITCH)
    X0, names = features.build(prep, 0)
    X1, _ = features.build(prep, 1)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X0=X0, X1=X1, labels=g["labels"], names=np.array(names))
    return X0, X1, g["labels"], names


def logo_probs(ball_runs, seeds=SEEDS):
    """{(game, seed): (probs, labels)} with each game predicted by a model trained on the others."""
    data = {(n, s): game_features(n, s, ball_runs) for n in external.GAMES for s in seeds}
    out = {}
    for held in external.GAMES:
        train = [data[k][:3] for k in data if k[0] != held]
        clf = model.train(train)
        for s in seeds:
            X0, X1, labels, _ = data[(held, s)]
            out[(held, s)] = (model.predict(clf, X0, X1), labels)
    return out


def score(labels, pred):
    """The two headline numbers, via the official scorer's own functions."""
    runs = pm.valid_runs(labels)
    filled = pm.fill_abstentions(pred, runs)
    sp = pm.spell_scores(labels, filled, runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(labels, pred)))
    return {"spell": sp["mean_f1"], "macro": macro, "n_pred": sp["n_pred"], "n_gt": sp["n_gt"]}


def grid(probs_by_game, switches, to_deads, from_deads):
    rows = []
    for sw, td, fd in itertools.product(switches, to_deads, from_deads):
        per = {}
        for key, (probs, labels) in probs_by_game.items():
            runs = pm.valid_runs(labels)
            per[key] = score(labels, model.decode(probs, runs, sw, td, fd))
        spell = np.array([v["spell"] for v in per.values()])
        macro = np.array([v["macro"] for v in per.values()])
        ratio = np.mean([v["n_pred"] / v["n_gt"] for v in per.values()])
        rows.append((sw, td, fd, spell.mean(), spell.min(), macro.mean(), ratio))
        print("switch {:5.1f} to_dead {:5.1f} from_dead {:5.1f}  spell {:.4f} (min {:.4f})  "
              "macro {:.4f}  poss ratio {:.2f}".format(*rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--to-dead", type=float, nargs="+", default=[5, 10, 20])
    ap.add_argument("--from-dead", type=float, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()
    runs = bar_ath_ball_runs()
    print("BAR-ATH ball runs: visible median {} missing median {} (n={}/{})".format(
        int(np.median(runs[0])), int(np.median(runs[1])), len(runs[0]), len(runs[1])))
    pb = logo_probs(runs)
    for key, (probs, labels) in sorted(pb.items()):
        pred = np.where(labels == model.X, model.X, np.argmax(probs, 1)).astype(np.int8)
        s = score(labels, pred)
        print("game {} seed {}  argmax  macro {:.4f}  spell {:.4f}  poss {}/{}".format(
            key[0], key[1], s["macro"], s["spell"], s["n_pred"], s["n_gt"]))
    rows = grid(pb, args.switch, args.to_dead, args.from_dead)
    best = max(rows, key=lambda r: r[3])
    print("best", best)
    with open(os.path.join(CACHE, "grid.json"), "w") as fh:
        json.dump(rows, fh)
