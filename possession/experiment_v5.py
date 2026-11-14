"""Practice test for pipeline v5: leave one game out, on the degraded Metrica sample games.

    python -m possession.experiment_v5              # v5 only (~8 min)
    python -m possession.experiment_v5 --with-v4    # v4 on the same labels, side by side

For each held-out game the two models are trained on the other two games, then:
  end-to-end        the whole v5 pipeline labels the held-out game (no Smart Tagging or broadcast
                    there, so those two inputs are simply absent);
  possession only   dead balls are taken from the truth, which isolates how well changes of
                    possession are found.
Nothing here reads any BAR-ATH label.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import external, harness, model  # noqa: E402
from possession.v5 import learned, pipeline  # noqa: E402
from possession.v5.decoder import viterbi_tv  # noqa: E402


def _given_dead(score_h, score_a, truth, hv, switch):
    """Decode H/A with the dead balls given by the truth (codes in, codes out)."""
    n = len(truth)
    S = np.zeros((n, 3))
    S[:, 0], S[:, 1] = score_h, score_a
    S[:, 2] = np.where(truth == external.D, 1e6, -1e6)
    C = np.array([[0, switch, 0], [switch, 0, 0], [0, 0, 0]], float)
    out = np.full(n, external.X, np.int8)
    for a, b in hv:
        out[a:b] = viterbi_tv(S[a:b], C, np.zeros(b - a))
    return out


def run_v5(ball_runs, log=print):
    prep = learned.practice_prep(ball_runs, log)
    rows = {}
    for held in external.GAMES:
        models = learned.train(prep, [g for g in external.GAMES if g != held], log=log)
        for s in harness.SEEDS:
            p = prep[(held, s)]
            truth = pm.label_codes(p["lab"])
            pred = pm.label_codes(pipeline.label_game(p, models))
            lo, _ = learned.apply(models, p)
            w = 0.5 * pipeline.CONFIG["w_ha"]
            only = _given_dead(w * lo, -w * lo, truth, p["hv"], pipeline.CONFIG["P_HA"])
            rows[(held, s)] = dict(harness.score(truth, pred),
                                   spell_only=harness.score(truth, only)["spell"])
    return rows


def run_v4(ball_runs):
    with open(os.path.join(ROOT, "possession", "params.json")) as fh:
        p = json.load(fh)
    rows = {}
    for key, (probs, truth) in sorted(harness.logo_probs(ball_runs).items()):
        runs = pm.valid_runs(truth)
        pred = model.decode(probs, runs, p["switch"], p["to_dead"], p["from_dead"])
        lp = np.log(np.clip(probs, 1e-6, 1.0))
        only = _given_dead(lp[:, 0], lp[:, 1], truth, runs, p["switch"])
        rows[key] = dict(harness.score(truth, pred), spell_only=harness.score(truth, only)["spell"])
    return rows


def report(name, rows):
    for (g, s), r in sorted(rows.items()):
        print("  {} game {} seed {}: spell {:.4f} macro {:.4f} poss {}/{} | possession only {:.4f}"
              .format(name, g, s, r["spell"], r["macro"], r["n_pred"], r["n_gt"], r["spell_only"]))
    mean = lambda k: float(np.mean([r[k] for r in rows.values()]))  # noqa: E731
    ratio = float(np.mean([r["n_pred"] / r["n_gt"] for r in rows.values()]))
    print("[{}] spell {:.4f} (min {:.4f})  macro {:.4f}  poss ratio {:.2f}  possession only {:.4f}"
          .format(name, mean("spell"), min(r["spell"] for r in rows.values()), mean("macro"),
                  ratio, mean("spell_only")), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-v4", action="store_true")
    args = ap.parse_args()
    runs = harness.bar_ath_ball_runs()
    if args.with_v4:
        report("v4", run_v4(runs))
    report("v5", run_v5(runs, log=lambda *a: None))
