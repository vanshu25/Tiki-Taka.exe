"""Team-shape features: what the 22 players are doing, without looking at the ball.

When the ball is not tracked (41% of graded frames) the only remaining evidence of possession is
collective: the team in possession spreads out and moves forward, the defending team compacts and
retreats, and during a stoppage everybody slows down. These features describe that, per frame,
smoothed over about a second. They feed the shape model in learned.py.
"""
import numpy as np

from .geometry import FPS, PITCH_L, PITCH_W, HOME, AWAY, _rolling_nanmean


def attack_sign(px, team, halves, gk_slots):
    """+1 if the home team attacks towards increasing x in that half, else -1. Read off the
    position of the home goalkeeper (the slot the metadata tags as Goalkeeper)."""
    out = {}
    home_gk = [s for s in gk_slots if team[s] == HOME]
    for h, (a, b) in enumerate(halves):
        gx = np.nanmean(px[a:b, home_gk[0]]) if home_gk else np.nanmean(px[a:b, team == HOME])
        out[h] = 1.0 if gx < PITCH_L / 2 else -1.0
    return out


def shape_features(px, py, pspeed, team, halves, gk_slots, smooth=12):
    n = px.shape[0]
    sign = np.zeros(n)
    hs = attack_sign(px, team, halves, gk_slots)
    for h, (a, b) in enumerate(halves):
        sign[a:b] = hs[h]
    # Outside graded halves use the nearest half's direction (those frames are never scored).
    sign[: halves[0][0]] = hs[0]
    sign[halves[0][1]: halves[1][0]] = hs[1]
    sign[halves[1][1]:] = hs[1]

    vx = np.full_like(px, np.nan)
    vx[2:-2] = (px[4:] - px[:-4]) / (4 / FPS)
    cols, names = [], []

    def add(name, v):
        names.append(name)
        cols.append(_rolling_nanmean(v, smooth, smooth))

    is_gk = np.zeros(px.shape[1], bool)
    is_gk[list(gk_slots)] = True
    for tcode, tname in ((HOME, "h"), (AWAY, "a")):
        m = (team == tcode) & ~is_gk
        s = sign if tcode == HOME else -sign  # +1: this team attacks +x
        # x in this team's attacking direction, 0 = own goal line, L = opponent goal line
        xa = np.where(s[:, None] > 0, px[:, m], PITCH_L - px[:, m])
        ya = py[:, m]
        with np.errstate(all="ignore"):
            add(tname + "_cx", np.nanmean(xa, 1))
            add(tname + "_sx", np.nanstd(xa, 1))
            add(tname + "_sy", np.nanstd(ya, 1))
            add(tname + "_xmax", np.nanmax(xa, 1))
            add(tname + "_xmin", np.nanmin(xa, 1))
            add(tname + "_spd", np.nanmean(pspeed[:, m], 1))
            add(tname + "_spd90", np.nanpercentile(pspeed[:, m], 90, axis=1))
            add(tname + "_fwd", np.nanmean(vx[:, m] * s[:, None], 1))
            add(tname + "_n", np.isfinite(xa).sum(1).astype(float))
            add(tname + "_inopp", (xa > PITCH_L / 2).sum(1).astype(float))
    X = np.column_stack(cols)
    # Pairwise differences: the classifier should not need to learn them.
    idx = {k: i for i, k in enumerate(names)}
    extra = []
    for k in ("cx", "sx", "sy", "spd", "fwd", "inopp", "xmax"):
        extra.append(X[:, idx["h_" + k]] - X[:, idx["a_" + k]])
        names.append("d_" + k)
    with np.errstate(all="ignore"):
        allspd = np.nanmean(pspeed, 1)
    for w in (25, 75):
        extra.append(_rolling_nanmean(allspd, w, w))
        names.append("spd_w%d" % w)
    extra.append(_rolling_nanmean(allspd, 0, 50)); names.append("spd_fwd50")
    extra.append(_rolling_nanmean(allspd, 50, 0)); names.append("spd_bwd50")
    X = np.column_stack([X] + extra)
    return X, names
