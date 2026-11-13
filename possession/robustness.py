"""Robustness: train on the standard degraded games, test on harsher versions of the held-out one.

If the score holds up when the test tracking is worse than we assumed, the choices made on the
harness are less likely to be artefacts of the degradation model.

    python -m possession.robustness
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import degrade, external, features, harness, model  # noqa: E402
import possession_metrics as pm  # noqa: E402

VARIANTS = {
    "standard":          dict(),
    "ball_more_missing": dict(vis_live=0.55, vis_dead=0.25),
    "ball_random_drop":  dict(state_free=True),
    "players_noisy":     dict(player_missing=0.15, player_jitter=0.5),
    "players_missing15": dict(player_missing=0.15),
    "players_drift50":   dict(player_jitter=0.5),
    "all_harsh":         dict(vis_live=0.55, vis_dead=0.25, player_missing=0.15,
                              player_jitter=0.5, ball_jitter=0.5),
}


def variant_features(n, name, ball_runs, seed=0):
    path = os.path.join(harness.CACHE, "rob_g{}_{}_f{}_d{}_l{}.npz".format(
        n, name, features.VERSION, degrade.VERSION, external.LABEL_VERSION))
    if os.path.exists(path):
        z = np.load(path)
        return z["X0"], z["X1"], z["labels"]
    v = dict(VARIANTS[name])
    g = external.load_game(n)
    old = degrade.VIS_LIVE, degrade.VIS_DEAD
    degrade.VIS_LIVE = v.pop("vis_live", degrade.VIS_LIVE)
    degrade.VIS_DEAD = v.pop("vis_dead", degrade.VIS_DEAD)
    dead = None if v.pop("state_free", False) else (g["labels"] == model.D)
    try:
        dg = degrade.degrade(g, ball_runs, seed=1000 * n + 500 + seed, pitch=external.PITCH,
                             dead=dead, **v)
    finally:
        degrade.VIS_LIVE, degrade.VIS_DEAD = old
    prep = features.prepare(dg, external.PITCH)
    X0, _ = features.build(prep, 0)
    X1, _ = features.build(prep, 1)
    np.savez(path, X0=X0, X1=X1, labels=g["labels"])
    return X0, X1, g["labels"]


if __name__ == "__main__":
    with open(os.path.join(ROOT, "possession", "params.json")) as fh:
        p = json.load(fh)
    runs = harness.bar_ath_ball_runs()
    data = {(n, s): harness.game_features(n, s, runs) for n in external.GAMES for s in harness.SEEDS}
    res = {k: [] for k in VARIANTS}
    for held in external.GAMES:
        clf = model.train([data[k][:3] for k in data if k[0] != held])
        for name in VARIANTS:
            X0, X1, labels = variant_features(held, name, runs)
            probs = model.predict(clf, X0, X1)
            pred = model.decode(probs, pm.valid_runs(labels), p["switch"], p["to_dead"],
                                p["from_dead"])
            s = harness.score(labels, pred)
            res[name].append((s["spell"], s["macro"], s["n_pred"] / s["n_gt"]))
            print(held, name, "spell {:.4f} macro {:.4f} ratio {:.2f}".format(
                *res[name][-1]), flush=True)
    print("\nvariant              spellF1  (worst game)  macroF1  poss ratio")
    for name, v in res.items():
        a = np.array(v)
        print("{:<20} {:.4f}   ({:.4f})      {:.4f}   {:.2f}".format(
            name, a[:, 0].mean(), a[:, 0].min(), a[:, 1].mean(), a[:, 2].mean()))
