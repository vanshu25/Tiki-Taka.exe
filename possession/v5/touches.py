"""Touch events: the unit the possession rules are written in.

The challenge's rules are phrased in touches: "a simple touch of the ball without control" or "a
deflection that does not intercept" does not change possession; controlled possession does, with
"a team completes passes" as a proxy. So a touch is detected as a kink in the ball's velocity (a
sudden change of speed or direction) with a player within reach, or as the start of calm control
(ball within reach and moving with the player) where no kink was seen.

Each touch is attributed to the nearest player's team. Repeated touches by the same player within
one second are one action (tackle-then-clear, block-then-block), not two.
"""
import numpy as np
from scipy.signal import find_peaks

from .geometry import FPS


def ball_kink(bx, by, k=2, h=3):
    """|change of ball velocity| (m/s) between [t-h] and [t+h], from central differences."""
    n = len(bx)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[k:-k] = (bx[2 * k:] - bx[:-2 * k]) / (2 * k / FPS)
    vy[k:-k] = (by[2 * k:] - by[:-2 * k]) / (2 * k / FPS)
    dv = np.full(n, np.nan)
    dv[h:-h] = np.hypot(vx[2 * h:] - vx[:-2 * h], vy[2 * h:] - vy[:-2 * h])
    return np.nan_to_num(dv)


def touch_events(fe, kink_thr=3.0, reach=2.5, contest=0.5, calm_c=0.5, sep=5, merge=FPS):
    """Return (touch_home, touch_away): per-frame touch weights (1 clean, 0.5 contested)."""
    n = fe["n"]
    kink = ball_kink(fe["bx"], fe["by"])
    dH, dA = fe["d_home"], fe["d_away"]
    cH, cA = fe["c_home"], fe["c_away"]
    nh, na = fe["near_home"], fe["near_away"]

    events = []                         # (frame, team, slot, weight)
    kink_frames = set()
    peaks, _ = find_peaks(kink, height=kink_thr, distance=sep)
    for p in peaks:
        if min(dH[p], dA[p]) >= reach:
            continue                    # the ball changed course with nobody near: a bounce
        w = 0.5 if abs(dH[p] - dA[p]) < contest else 1.0
        events.append((p, 0, nh[p], w) if dH[p] < dA[p] else (p, 1, na[p], w))
        kink_frames.add(p)
    for c, tm, slot in ((cH, 0, nh), (cA, 1, na)):
        on = c >= calm_c
        for s in np.flatnonzero(on & ~np.r_[False, on[:-1]]):
            if not any((s + o) in kink_frames for o in range(-sep, sep + 1)):
                events.append((s, tm, slot[s], 1.0))
    events.sort()

    t_home, t_away = np.zeros(n), np.zeros(n)
    last = {}
    for f, tm, slot, w in events:
        if slot in last and f - last[slot] < merge:
            last[slot] = f
            continue                    # the same player again within a second: one action
        last[slot] = f
        (t_home if tm == 0 else t_away)[f] += w
    return t_home, t_away, kink
