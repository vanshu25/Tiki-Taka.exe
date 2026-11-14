"""The two models of pipeline v5, trained on the degraded practice games (never on BAR-ATH).

  shape model       players + whether the ball is tracked -> P(home), P(away), P(dead).
                    It never sees where the ball is. In the practice data a dead ball is drawn on
                    a straight line between where play stopped and where it restarts, which is not
                    how a real dead ball moves (it is carried, thrown back, placed), so anything
                    learned from dead-ball geometry does not transfer to the match.
  possession model  ball and players around each frame -> log-odds home vs away. Trained on live
                    frames only, so the artificial dead ball cannot reach it either.

Home and away are swapped as augmentation, so neither model can prefer a side.
"""
import os
import pickle

import numpy as np

from .. import degrade, external, harness
from . import geometry as G
from .geometry import _rolling_nanmean
from .shape import shape_features
from .touches import touch_events

VERSION = 1
LOGIT_CLIP = 4.0
CODE = {"H": 0, "A": 1, "D": 2}
LETTERS = np.array(["H", "A", "D", "X"])


# --- features ------------------------------------------------------------------------------------
def to_arrays(d):
    """Team-repo game dict (players (F,S,2+), ball (F,2+), teams) -> arrays for geometry."""
    P, B = d["players"], d["ball"]
    return dict(px=P[..., 0].astype(np.float32), py=P[..., 1].astype(np.float32),
                bx=B[:, 0].astype(np.float32), by=B[:, 1].astype(np.float32),
                team=np.asarray(d["teams"]),
                # speeds are always differenced from positions, never read from a file
                ps=np.full(P.shape[:2], np.nan, np.float32))


def _ball_visibility(vis):
    """Is the ball tracked, for how long has it been gone, how long until it is back."""
    n = len(vis)
    idx = np.where(vis, np.arange(n), -10 ** 9)
    since = np.minimum(np.arange(n) - np.maximum.accumulate(idx), 3000).astype(float)
    r = vis[::-1]
    idx = np.where(r, np.arange(n), -10 ** 9)
    until = np.minimum(np.arange(n) - np.maximum.accumulate(idx), 3000)[::-1].astype(float)
    v = vis.astype(float)
    cols = {"ball_ok": v, "ball_since": since, "ball_until": until, "ball_gap": since + until}
    for w in (25, 75, 250):
        cols["ball_ok_r%d" % w] = _rolling_nanmean(v, w, w)
    return cols


def player_features(fe, halves, gk):
    """Players and ball visibility, per frame. fe must still hold px, py, pspeed."""
    n = fe["n"]
    X, names = shape_features(fe["px"], fe["py"], fe["pspeed"], fe["team"], halves, gk)
    sp = np.where(np.isfinite(fe["px"]), fe["pspeed"], np.nan)
    cols = {}
    with np.errstate(all="ignore"):
        cols["f_still"] = np.nanmean(sp < 0.5, 1)
        cols["f_walk"] = np.nanmean(sp < 2.0, 1)
        cols["f_run"] = np.nanmean(sp > 4.0, 1)
        cols["f_sprint"] = np.nanmean(sp > 6.0, 1)
        cols["sp_std"] = np.nanstd(sp, 1)
        cols["sp_max"] = np.nanmax(np.where(np.isfinite(sp), sp, -1.0), 1)
        cols["sp_med"] = np.nanmedian(sp, 1)
        acc = np.full_like(sp, np.nan)
        acc[12:-12] = sp[24:] - sp[:-24]
        cols["acc_mean"] = np.nanmean(acc, 1)
        cols["f_decel"] = np.nanmean(acc < -0.5, 1)
        cols["f_accel"] = np.nanmean(acc > 0.5, 1)
    ms = _rolling_nanmean(fe["mean_speed"], 6, 6)
    for o in (-125, -75, -50, -25, 25, 50, 75, 125):
        s = np.full(n, np.nan)
        if o < 0:
            s[-o:] = ms[:o]
        else:
            s[:-o] = ms[o:]
        cols["ms%+d" % o] = s
    for k in ("f_still", "f_walk", "f_run"):
        cols[k + "_b"] = _rolling_nanmean(cols[k], 50, 0)
        cols[k + "_f"] = _rolling_nanmean(cols[k], 0, 50)
    cols.update(_ball_visibility(fe["ball_vis"]))
    return np.column_stack([X] + list(cols.values())), names + list(cols)


def swap_teams(X, names):
    Xs = X.copy()
    for i, nm in enumerate(names):
        if nm.startswith("h_"):
            j = names.index("a_" + nm[2:])
            Xs[:, i], Xs[:, j] = X[:, j], X[:, i]
        elif nm.startswith("d_"):
            Xs[:, i] = -X[:, i]
    return Xs


def _roll_sum(x, left, right):
    return np.nan_to_num(_rolling_nanmean(x, left, right)) * (left + right + 1)


def _since(ev, cap=250):
    out = np.empty(len(ev))
    c = cap
    for k in range(len(ev)):
        c = 0 if ev[k] else min(c + 1, cap)
        out[k] = c
    return out


def possession_features(fe, t_home, t_away, shape_lo):
    """Per-frame features for 'home or away?', and the same frame seen with the teams swapped."""
    uh = fe["c_home"] * (1 - fe["c_away"])
    ua = fe["c_away"] * (1 - fe["c_home"])
    dH = np.where(np.isfinite(fe["d_home"]), np.minimum(fe["d_home"], 30), np.nan)
    dA = np.where(np.isfinite(fe["d_away"]), np.minimum(fe["d_away"], 30), np.nan)
    Hh = {"d": dH, "c": fe["c_home"], "u": uh}
    Aa = {"d": dA, "c": fe["c_away"], "u": ua}
    wins = {"p12": (12, 0), "p25": (25, 0), "p50": (50, 0), "p100": (100, 0),
            "f12": (0, 12), "f25": (0, 25), "f50": (0, 50), "f100": (0, 100)}
    for nm, (l, r) in wins.items():
        Hh["t_" + nm] = _roll_sum(t_home, l, r)
        Aa["t_" + nm] = _roll_sum(t_away, l, r)
        if nm in ("p25", "p50", "f25", "f50"):
            Hh["u_" + nm] = _rolling_nanmean(uh, l, r)
            Aa["u_" + nm] = _rolling_nanmean(ua, l, r)
            Hh["dm_" + nm] = _rolling_nanmean(dH, l, r)
            Aa["dm_" + nm] = _rolling_nanmean(dA, l, r)
    Hh["since"] = _since(t_home > 0)
    Aa["since"] = _since(t_away > 0)
    Hh["until"] = _since((t_home > 0)[::-1])[::-1]
    Aa["until"] = _since((t_away > 0)[::-1])[::-1]
    keys = list(Hh)
    vis = fe["ball_vis"].astype(float)
    sm = _rolling_nanmean(shape_lo, 25, 25)
    X = np.column_stack([Hh[k] for k in keys] + [Aa[k] for k in keys]
                        + [vis, fe["bspeed"], shape_lo, sm])
    Xs = np.column_stack([Aa[k] for k in keys] + [Hh[k] for k in keys]
                         + [vis, fe["bspeed"], -shape_lo, -sm])
    return X, Xs


def shape_logodds(P):
    P = np.clip(P, 1e-3, 1)
    return np.clip(np.log(P[:, 0] / P[:, 1]), -LOGIT_CLIP, LOGIT_CLIP)


# --- one game, ready for the models --------------------------------------------------------------
def prepare(arrays, halves, gk, labels=None):
    """Geometry, player features and touches for one game. labels: H/A/D/X letters or None."""
    fe = G.compute_from(arrays)
    Xp, names = player_features(fe, halves, gk)
    for k in ("px", "py", "pspeed"):
        fe.pop(k)
    t_home, t_away, _ = touch_events(fe)
    return dict(fe=fe, Xp=Xp, names=names, th=t_home, ta=t_away, hv=list(halves), lab=labels)


def goalkeepers(arrays, halves):
    """One slot per team: the regular whose mean position in the first half is nearest a goal."""
    a, b = halves[0]
    out = []
    for t in (0, 1):
        best = None
        for s in np.flatnonzero(arrays["team"] == t):
            x = arrays["px"][a:b, s]
            if np.isfinite(x).mean() > 0.8:
                v = abs(np.nanmean(x) - 0.5)
                if best is None or v > best[0]:
                    best = (v, int(s))
        out.append(best[1])
    return out


_GAMES = {}


def practice_game(n, seed, ball_runs):
    """Practice game n, degraded with the harness's own seed, prepared for the models."""
    if n not in _GAMES:
        g = external.load_game(n)
        hv = [(int(a), int(b)) for a, b in harness.pm.valid_runs(g["labels"])]
        _GAMES[n] = (g, hv, goalkeepers(to_arrays(g), hv))
    g, hv, gk = _GAMES[n]
    dg = degrade.degrade(g, ball_runs, seed=1000 * n + seed, pitch=external.PITCH,
                         dead=(g["labels"] == external.D))
    return prepare(to_arrays(dg), hv, gk, LETTERS[g["labels"]])


# --- training ------------------------------------------------------------------------------------
def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def _fit_shape(prep, keys, step=6):
    Xs, ys = [], []
    for k in keys:
        p = prep[k]
        idx = np.flatnonzero(np.isin(p["lab"], ["H", "A", "D"]))[::step]
        y = np.array([CODE[c] for c in p["lab"][idx]])
        Xs += [p["Xp"][idx], swap_teams(p["Xp"][idx], p["names"])]
        ys += [y, np.where(y == 2, 2, 1 - y)]
    return _gbm().fit(np.vstack(Xs), np.concatenate(ys))


def _fit_possession(prep, shape_lo, keys, step=3):
    Xs, ys = [], []
    for k in keys:
        p = prep[k]
        X, Xsw = possession_features(p["fe"], p["th"], p["ta"], shape_lo[k])
        idx = np.flatnonzero(np.isin(p["lab"], ["H", "A"]))[::step]
        y = (p["lab"][idx] == "H").astype(int)
        Xs += [X[idx], Xsw[idx]]
        ys += [y, 1 - y]
    return _gbm().fit(np.vstack(Xs), np.concatenate(ys))


def train(prep, games, seeds=harness.SEEDS, log=lambda *a: None):
    """prep: {(game, seed): prepare(...)}. Trains on `games` only -> dict(shape, possession)."""
    # The possession model reads the shape model's opinion as a feature. For each training game
    # that opinion comes from a shape model that never saw the game, as it will on the match.
    shape_lo = {}
    for g in games:
        m = _fit_shape(prep, [(h, s) for h in games if h != g for s in seeds])
        for s in seeds:
            shape_lo[(g, s)] = shape_logodds(m.predict_proba(prep[(g, s)]["Xp"]))
        log("shape model without game %d" % g)
    keys = [(g, s) for g in games for s in seeds]
    models = dict(shape=_fit_shape(prep, keys))
    models["possession"] = _fit_possession(prep, shape_lo, keys)
    log("models trained on games %s" % (list(games),))
    return models


def practice_prep(ball_runs, log=lambda *a: None):
    prep = {}
    for n in external.GAMES:
        for s in harness.SEEDS:
            prep[(n, s)] = practice_game(n, s, ball_runs)
        log("practice game %d prepared" % n)
    return prep


def get_models(log=lambda *a: None):
    """Models trained on all practice games, cached under cache/ (gitignored)."""
    os.makedirs(harness.CACHE, exist_ok=True)
    path = os.path.join(harness.CACHE, "v5_models_m{}_d{}_l{}.pkl".format(
        VERSION, degrade.VERSION, external.LABEL_VERSION))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    models = train(practice_prep(harness.bar_ath_ball_runs(), log), external.GAMES, log=log)
    with open(path, "wb") as fh:
        pickle.dump(models, fh)
    return models


def apply(models, p):
    """-> (log-odds home vs away, log-odds dead vs alive), per frame."""
    P = models["shape"].predict_proba(p["Xp"])
    X, _ = possession_features(p["fe"], p["th"], p["ta"], shape_logodds(P))
    q = np.clip(models["possession"].predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    d = np.clip(P[:, 2], 1e-3, 1 - 1e-3)
    return np.clip(np.log(q / (1 - q)), -LOGIT_CLIP, LOGIT_CLIP), np.log(d / (1 - d))
