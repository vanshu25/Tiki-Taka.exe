"""Map the broadcast feed onto the tactical timeline and turn its camera state into evidence.

The broadcast is a real-time TV feed: replays and close-ups replace the live picture but do not
shift time. Matching the two automatic ball tracks (both in pitch coordinates) gives a constant
frame offset per half, found by search below -- no manual sync. Where the broadcast has no pitch
homography (replays, close-ups, crowd shots) a TV director has cut away from play, which happens
overwhelmingly while the ball is dead.
"""
import numpy as np
import pandas as pd

from games import asset_paths

from .geometry import PITCH_L, PITCH_W


def _match_count(tx, ty, bx, by, valid, bi, off, tol):
    ti = bi + off
    ok = (ti >= 0) & (ti < len(tx))
    d = np.abs(tx[ti[ok]] - bx[bi[ok]]) + np.abs(ty[ti[ok]] - by[bi[ok]])
    m = np.isfinite(d) & valid[bi[ok]]
    return (d[m] < tol)


def align(t, b, valid, coarse=range(-30000, 40001, 25), win=1500):
    """Return list of (broadcast_first, broadcast_last, offset): tactical = broadcast + offset."""
    tx, ty = t["bx"] * PITCH_L, t["by"] * PITCH_W
    bx, by = b["bx"] * PITCH_L, b["by"] * PITCH_W
    nb = len(bx)
    bi = np.arange(0, nb, 5)
    scores = sorted(((_match_count(tx, ty, bx, by, valid, bi, o, 2.0).sum(), o) for o in coarse),
                    reverse=True)
    cands = []
    for s, o in scores:
        if all(abs(o - c) > 1000 for c in cands):
            cands.append(o)
        if len(cands) == 2:
            break
    # Windowed refinement: best offset (frame precision) near each candidate, per window.
    win_off = []
    for s in range(0, nb, win):
        w = np.arange(s, min(s + win, nb))
        best = (0, None)
        for c in cands:
            for o in range(c - 60, c + 61):
                sc = _match_count(tx, ty, bx, by, valid, w, o, 1.0).sum()
                if sc > best[0]:
                    best = (sc, o)
        if best[0] >= 50:
            win_off.append((s, best[1]))
    # Collapse to one offset per contiguous run of windows (per half) using the median.
    segs, cur = [], []
    for s, o in win_off:
        if cur and abs(o - cur[-1][1]) > 30:
            segs.append(cur)
            cur = []
        cur.append((s, o))
    segs.append(cur)
    out = []
    for sg in segs:
        o = int(np.median([x[1] for x in sg]))
        first, last = sg[0][0], min(sg[-1][0] + win - 1, nb - 1)
        # trim the edges to frames that actually match at this offset
        w = np.arange(max(first - win, 0), min(last + win, nb))
        ok_t = w + o
        ok = (ok_t >= 0) & (ok_t < len(tx))
        d = np.abs(tx[ok_t[ok]] - bx[w[ok]]) + np.abs(ty[ok_t[ok]] - by[w[ok]])
        good = w[ok][np.isfinite(d) & valid[w[ok]] & (d < 1.5)]
        out.append((int(good.min()), int(good.max()), o))
    return out


def broadcast_evidence(t, b):
    """Per tactical frame: 1.0 = broadcast cut away from play, 0.0 = showing play, NaN = no feed.

    t, b: tactical and broadcast tracking as dicts with bx, by (normalised ball position).
    """
    n_tactical = len(t["bx"])
    hb = pd.read_csv(asset_paths(source="broadcast")["homography"], usecols=["frame", "valid"])
    valid = np.zeros(len(b["bx"]), bool)
    valid[hb["frame"].values] = hb["valid"].values == 1
    segs = align(t, b, valid)
    away = np.full(n_tactical, np.nan)
    for first, last, off in segs:
        bf = np.arange(first, last + 1)
        tf = bf + off
        ok = (tf >= 0) & (tf < n_tactical)
        away[tf[ok]] = (~valid[bf[ok]]).astype(float)
    return away, segs
