"""Ball and player geometry per frame, in metres and seconds, plus the Smart Tagging prior.

Shared by the competition match and the practice games: both go through compute_from().
No label of any game is read in this module.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, GAME  # noqa: E402

FPS = 25
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1


def _rolling_nanmean(x, left, right):
    """Mean of x over the window [t-left, t+right], ignoring NaN. Same length as x."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    idx = np.arange(n)
    lo = np.clip(idx - left, 0, n)
    hi = np.clip(idx + right + 1, 0, n)
    s = cs[hi] - cs[lo]
    c = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _central_velocity(p, k=2):
    """Central-difference velocity (units/s) over +-k frames, NaN where either end is missing."""
    v = np.full_like(p, np.nan, dtype=np.float64)
    v[k:-k] = (p[2 * k:] - p[:-2 * k]) / (2 * k / FPS)
    return v


def fill_short_gaps(x, max_gap):
    """Linear interpolation across interior NaN runs no longer than max_gap frames."""
    x = x.copy()
    n = len(x)
    miss = ~np.isfinite(x)
    i = 0
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = np.linspace(x[i - 1], x[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return x


def compute_from(d):
    """d: dict of raw arrays (px, py, ps: frames x slots; bx, by: frames; team: slots), 0..1."""
    team = d["team"]
    n = len(d["bx"])

    # --- ball -------------------------------------------------------------------------------
    bx = fill_short_gaps(d["bx"].astype(np.float64) * PITCH_L, 5)
    by = fill_short_gaps(d["by"].astype(np.float64) * PITCH_W, 5)
    ball_vis = np.isfinite(bx)
    bvx, bvy = _central_velocity(bx), _central_velocity(by)
    bspeed = np.hypot(bvx, bvy)

    # --- players ------------------------------------------------------------------------------
    px = d["px"].astype(np.float64) * PITCH_L
    py = d["py"].astype(np.float64) * PITCH_W
    pvx = np.full_like(px, np.nan)
    pvy = np.full_like(py, np.nan)
    pvx[2:-2] = (px[4:] - px[:-4]) / (4 / FPS)
    pvy[2:-2] = (py[4:] - py[:-4]) / (4 / FPS)
    pspeed = np.hypot(pvx, pvy)
    # The provided speed channel is more complete for long-lived tracks; fall back to it.
    pspeed = np.where(np.isfinite(pspeed), pspeed, d["ps"].astype(np.float64))
    pspeed = np.clip(pspeed, 0, 12)

    present = np.isfinite(px)
    n_players = present.sum(1)
    mean_speed = np.nanmean(np.where(present, pspeed, np.nan), axis=1) if True else None

    # --- player-ball geometry -----------------------------------------------------------------
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    # Relative velocity between ball and each player: a controlled ball moves with its player.
    rel = np.hypot(pvx - bvx[:, None], pvy - bvy[:, None])
    rel = np.where(np.isfinite(rel), rel, 3.0)  # unknown -> neutral-ish
    prox = np.clip((2.0 - dist) / 1.0, 0.0, 1.0)          # 1 at <=1 m, 0 at >=2 m
    calm = np.clip((6.0 - rel) / 3.0, 0.0, 1.0)            # 1 at <=3 m/s, 0 at >=6 m/s
    ctrl = prox * calm

    c_home = ctrl[:, team == HOME].max(1)
    c_away = ctrl[:, team == AWAY].max(1)
    d_home = dist[:, team == HOME].min(1)
    d_away = dist[:, team == AWAY].min(1)
    # Track slot of each team's nearest player (identity of the toucher; -1 when no ball).
    hs, as_ = np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)
    near_home = np.where(np.isfinite(d_home), hs[dist[:, team == HOME].argmin(1)], -1)
    near_away = np.where(np.isfinite(d_away), as_[dist[:, team == AWAY].argmin(1)], -1)

    margin = 0.5
    out = ball_vis & ((bx < -margin) | (bx > PITCH_L + margin) |
                      (by < -margin) | (by > PITCH_W + margin))

    with np.errstate(invalid="ignore"):
        mean_speed = np.where(n_players > 0, mean_speed, np.nan)

    feats = dict(
        n=n, team=team, bx=bx, by=by, ball_vis=ball_vis, bspeed=bspeed,
        c_home=c_home, c_away=c_away, d_home=d_home, d_away=d_away, out=out,
        near_home=near_home, near_away=near_away,
        mean_speed=mean_speed, n_players=n_players,
        px=px, py=py, pspeed=pspeed,
    )
    return feats


def smart_tag_priors(n):
    """Per-frame soft team prior from Metrica's automatic Smart Tagging.

    +1 where a phase says the home team has the ball, -1 for away, 0 elsewhere/conflict.
    Only phases that are *about the team in possession* are used. Zone phases (ball in final
    third, ball/players in the box) are skipped: they describe a location, not an owner.
    """
    import load_smart_tagging as lst
    own = {"build up", "progression", "attacking transition", "creating chances", "long ball"}
    opp = {"defending in defensive third", "defending in middle third",
           "defending in attacking third", "defensive transition"}
    home_name = "barcelona"
    acc = np.zeros(n)
    setp = []
    for e in lst.load():
        code = e["code"].strip().lower()
        t = (e["tags"].get("Team") or "").strip().lower()
        s, f = max(e["start_frame"], 0), min(e["end_frame"], n)
        if code == "set pieces":
            setp.append((s, f, (e["tags"].get("Type") or "").lower(), t))
            continue
        if t in ("", "n/a"):
            continue
        sign = 1.0 if t == home_name else -1.0
        if code in own:
            acc[s:f] += sign
        elif code in opp:
            acc[s:f] -= sign
    return np.sign(acc), setp
