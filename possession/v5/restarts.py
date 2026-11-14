"""Stoppage and restart signatures in the ball track.

A dead ball comes back to life only at a restart: the ball is stationary (or held) at a spot,
the players around it are calm, and it is then kicked or thrown. Out-of-play is the converse:
the ball is seen beyond a line, or it vanishes while flying towards one.
"""
import numpy as np
import pandas as pd

from .geometry import FPS, PITCH_L, PITCH_W, _rolling_nanmean


def ball_speed(bx, by, k=3):
    v = np.full(len(bx), np.nan)
    v[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) / (2 * k / FPS)
    return v


def exit_events(bx, by, vis, min_gap=25, look_ahead_s=0.5, margin=0.5):
    """Frames where the ball disappears for >= min_gap frames while heading out of the pitch."""
    n = len(bx)
    ev = np.zeros(n, bool)
    k = 1
    while k < n:
        if not vis[k] and vis[k - 1]:
            j = k
            while j < n and not vis[j]:
                j += 1
            if j - k >= min_gap:
                i0, i1 = k - 1, max(k - 6, 0)
                if vis[i1]:
                    vx = (bx[i0] - bx[i1]) * FPS / (i0 - i1)
                    vy = (by[i0] - by[i1]) * FPS / (i0 - i1)
                    fx, fy = bx[i0] + vx * look_ahead_s, by[i0] + vy * look_ahead_s
                    if fx < -margin or fx > PITCH_L + margin or fy < -margin or fy > PITCH_W + margin:
                        ev[k] = True
            k = j
        else:
            k += 1
    return ev


def restart_candidates(bx, by, vis, mean_speed, slow_v=3.5, kick_v=6.0, window=12,
                       max_disp=4.0, calm=2.0, gap_min=25, return_kicks=False):
    """Boolean per frame: a plausible restart happens here.

    kick:   ball nearly still for a second (median speed < slow_v, moved < max_disp m), players
            calm over the previous 2 s (mean speed < calm), then the ball accelerates past kick_v.
    appear: the ball is picked up again after a long gap (the tracker routinely loses the ball
            while it is out of play and finds it again once it is put back into play).
    """
    n = len(bx)
    v = pd.Series(ball_speed(bx, by))
    pre_med = v.rolling(window, min_periods=window // 2).median().shift(1).values
    post_max = v[::-1].rolling(window, min_periods=3).max()[::-1].values
    inside = (bx > -1.5) & (bx < PITCH_L + 1.5) & (by > -1.5) & (by < PITCH_W + 1.5)
    disp = np.full(n, np.inf)
    disp[FPS:] = np.hypot(bx[FPS:] - bx[:-FPS], by[FPS:] - by[:-FPS])
    disp = np.where(np.isfinite(disp), disp, np.inf)
    pre_act = _rolling_nanmean(mean_speed, 2 * FPS, 0)
    with np.errstate(invalid="ignore"):
        kick = vis & inside & (pre_med < slow_v) & (post_max > kick_v) & (disp < max_disp) & \
            (pre_act < calm)
    kick = kick & ~np.r_[False, kick[:-1]]

    appear = np.zeros(n, bool)
    run = 0
    for t in range(n):
        if not vis[t]:
            run += 1
        else:
            if run >= gap_min:
                appear[t] = True
            run = 0
    if return_kicks:
        return kick | appear, kick
    return kick | appear
