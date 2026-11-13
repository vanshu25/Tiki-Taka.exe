"""Per-frame, team-relative features from tracking alone.

Everything is computed from positions (velocities are differenced here, never read from the file)
so our match and the external games go through exactly the same code. Features are built from
the point of view of one team ("own") against the other ("opp"); calling build() twice with the
teams swapped gives the two perspectives the classifier is trained and evaluated on.
"""
import numpy as np

FPS = 25
MAX_PLAYER_SPEED = 12.0   # m/s; faster than any sprint -> a tracking jump, not motion
MAX_BALL_SPEED = 40.0     # m/s; faster than a struck ball -> a tracking jump
WINDOWS = (25, 75, 250)   # rolling context: 1 s, 3 s, 10 s (centred)
VERSION = 1               # bump whenever a feature changes (invalidates the harness cache)
# VERSION 2 adds explicit stoppage features (below). Measured on the external harness it made no
# difference (spellF1 0.590 vs 0.594 for v1, degradation v2), so it is skipped.
# VERSION 3 = v1 + touch / sub-second features for short possessions (fine_touch_features).
# Measured +0.006 (0.686 vs 0.680, controlled labels) -- inside noise, so v1 stays in use.


def _nanmean(a, axis):
    n = np.isfinite(a).sum(axis)
    s = np.where(np.isfinite(a), a, 0.0).sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _masked_min(a, axis):
    b = np.where(np.isfinite(a), a, np.inf)
    m = b.min(axis)
    return np.where(np.isfinite(m), m, np.nan)


def _masked_max(a, axis):
    b = np.where(np.isfinite(a), a, -np.inf)
    m = b.max(axis)
    return np.where(np.isfinite(m), m, np.nan)


def _second_min(a, axis=1):
    b = np.sort(np.where(np.isfinite(a), a, np.inf), axis=axis)
    m = b[:, 1] if b.shape[1] > 1 else b[:, 0]
    return np.where(np.isfinite(m), m, np.nan)


def rolling_mean(x, w):
    """Centred NaN-aware rolling mean, +/- w frames (O(n) via cumulative sums)."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    hi = np.minimum(len(x), i + w + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _velocity(pos, lag=2, vmax=MAX_PLAYER_SPEED):
    """Central-difference velocity (m/s) along axis 0; NaN where implausible or unknown."""
    v = np.full_like(pos, np.nan)
    v[lag:-lag] = (pos[2 * lag:] - pos[:-2 * lag]) * (FPS / (2.0 * lag))
    speed = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2) if v.ndim == pos.ndim and pos.shape[-1] == 2 \
        else None
    if speed is not None:
        v[speed > vmax] = np.nan
    return v


def frames_since(mask):
    """Frames since mask was last True (0 where True); large where never."""
    idx = np.where(mask, np.arange(len(mask)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(mask)) - last, 10 ** 6).astype(np.float32)


def frames_until(mask):
    return frames_since(mask[::-1])[::-1]


def attack_sign(players, teams, period, length):
    """+1 if the team attacks towards +x in that period, else -1, from mean team x."""
    out = {}
    for per in np.unique(period[period > 0]):
        sel = period == per
        for t in (0, 1):
            mx = np.nanmean(players[sel][:, teams == t, 0])
            out[(int(per), t)] = 1.0 if mx < length / 2 else -1.0
    return out


def prepare(d, pitch):
    """Metric positions, velocities and per-frame attack direction for a loaded game."""
    L, W = pitch
    P = d["players"][..., :2].astype(np.float64) * np.array([L, W])
    B = d["ball"][..., :2].astype(np.float64) * np.array([L, W])
    period = d["period"]
    teams = d["teams"]
    signs = attack_sign(P, teams, period, L)
    sign = np.zeros((len(period), 2))
    for (per, t), s in signs.items():
        sign[period == per, t] = s
    PV = _velocity(P)
    BV = _velocity(B, lag=1, vmax=MAX_BALL_SPEED)
    return {"P": P, "B": B, "PV": PV, "BV": BV, "teams": teams, "sign": sign,
            "period": period, "L": L, "W": W}


def build(g, own):
    """Feature matrix (F, k) and names, from the perspective of team `own` (0 or 1)."""
    P, B, PV, BV = g["P"], g["B"], g["PV"], g["BV"]
    L, W = g["L"], g["W"]
    opp = 1 - own
    teams = g["teams"]
    F = P.shape[0]
    cols = {}

    def ax(x, t):
        """x in the attacking direction of team t (0 = own goal line, L = goal attacked)."""
        s = g["sign"][:, t]
        if x.ndim == 2:
            s = s[:, None]
        return np.where(s > 0, x, L - x)

    ball_ok = np.isfinite(B[:, 0])
    dist = np.sqrt(((P - B[:, None, :]) ** 2).sum(-1))  # (F, S), NaN where either missing
    speed = np.sqrt((PV ** 2).sum(-1))
    ball_speed = np.sqrt((BV ** 2).sum(-1))

    # Global (team-free) stoppage evidence.
    all_speed = _nanmean(speed, 1)
    cols["speed_all"] = all_speed
    cols["frac_slow"] = _nanmean(np.where(np.isnan(speed), np.nan, (speed < 1.5).astype(float)), 1)
    cols["ball_ok"] = ball_ok.astype(float)
    cols["ball_speed"] = ball_speed
    out_x = np.maximum(-B[:, 0], B[:, 0] - L)
    out_y = np.maximum(-B[:, 1], B[:, 1] - W)
    cols["ball_out"] = np.maximum(out_x, out_y)
    cols["ball_touch_d"] = np.minimum(B[:, 1], W - B[:, 1])
    cols["since_ball"] = np.minimum(frames_since(ball_ok), 3000)
    cols["until_ball"] = np.minimum(frames_until(ball_ok), 3000)
    cols["n_players"] = np.isfinite(P[:, :, 0]).sum(1).astype(float)

    # Explicit stoppage / restart evidence (feature VERSION >= 2). All geometric, no constants
    # taken from BAR-ATH: pitch markings are the Laws of the Game, speeds are plain physics.
    if VERSION == 2:
        still = np.where(ball_ok, (ball_speed < 1.0).astype(float), np.nan)
        cols["ball_still_r25"] = rolling_mean(still, 25)
        cols["ball_still_r75"] = rolling_mean(still, 75)
        is_out = cols["ball_out"] > 0.5
        cols["since_out"] = np.minimum(frames_since(is_out), 3000)
        cols["until_out"] = np.minimum(frames_until(is_out), 3000)
        bx, by = B[:, 0], B[:, 1]
        dx_goal = np.minimum(bx, L - bx)
        cols["ball_corner_d"] = np.sqrt(dx_goal ** 2 + np.minimum(by, W - by) ** 2)
        cols["ball_six_yard"] = ((dx_goal < 5.5) & (np.abs(by - W / 2) < 9.16)).astype(float)
        cols["ball_six_yard"][~ball_ok] = np.nan
        cols["ball_centre_d"] = np.sqrt((bx - L / 2) ** 2 + (by - W / 2) ** 2)
        # Throw-in shape: the nearest player to the ball is standing on/over the touchline.
        j = np.argmin(np.where(np.isfinite(dist), dist, np.inf), 1)
        near_y = P[np.arange(F), j, 1]
        cols["holder_touch_d"] = np.where(ball_ok, np.minimum(near_y, W - near_y), np.nan)
        cols["holder_d"] = _masked_min(dist, 1)
        # Kick-off shape: both teams entirely in their own half.
        own_half = []
        for t in (0, 1):
            xa = ax(P[:, teams == t, 0], t)
            own_half.append(_nanmean(np.where(np.isfinite(xa), (xa < L / 2).astype(float),
                                              np.nan), 1))
        cols["kickoff_shape"] = np.minimum(own_half[0], own_half[1])
        cols["kickoff_shape_r25"] = rolling_mean(cols["kickoff_shape"], 25)
        # Fastest player slowing down: in open play someone is nearly always running.
        top = np.sort(np.where(np.isfinite(speed), speed, -1.0), 1)[:, -3:]
        top = np.where(top < 0, np.nan, top)
        cols["top3_speed"] = _nanmean(top, 1)
        for w in (25, 75, 250):
            cols["top3_speed_r{}".format(w)] = rolling_mean(cols["top3_speed"], w)
        cols["n_within10"] = np.where(ball_ok, (dist < 10).sum(1), np.nan).astype(float)

    for tag, t in (("own", own), ("opp", opp)):
        m = teams == t
        Pt, Vt, Dt = P[:, m], PV[:, m], dist[:, m]
        x_att = ax(Pt[..., 0], t)
        s = g["sign"][:, t][:, None]
        v_att = Vt[..., 0] * s
        cols[tag + "_n"] = np.isfinite(Pt[..., 0]).sum(1).astype(float)
        cols[tag + "_cx"] = _nanmean(x_att, 1)
        cols[tag + "_cy_c"] = np.abs(_nanmean(Pt[..., 1], 1) - W / 2)
        cols[tag + "_sx"] = np.sqrt(np.maximum(_nanmean(x_att ** 2, 1) - _nanmean(x_att, 1) ** 2, 0))
        y = Pt[..., 1]
        cols[tag + "_sy"] = np.sqrt(np.maximum(_nanmean(y ** 2, 1) - _nanmean(y, 1) ** 2, 0))
        cols[tag + "_xmax"] = _masked_max(x_att, 1)
        cols[tag + "_xdef"] = _second_min(x_att)          # last outfield defender
        cols[tag + "_vx"] = _nanmean(v_att, 1)            # pushing forward (+) or retreating
        cols[tag + "_speed"] = _nanmean(np.sqrt((Vt ** 2).sum(-1)), 1)
        cols[tag + "_dmin"] = _masked_min(Dt, 1)
        cols[tag + "_d2"] = _second_min(Dt)
        cols[tag + "_n5"] = np.where(ball_ok, (Dt < 5).sum(1), np.nan).astype(float)
        cols[tag + "_ball_ax"] = ax(B[:, 0], t)
        # Nearest player's velocity relative to the ball (a controlled ball moves with its player).
        j = np.argmin(np.where(np.isfinite(Dt), Dt, np.inf), 1)
        vn = Vt[np.arange(F), j]
        rel = np.sqrt(((vn - BV) ** 2).sum(-1))
        cols[tag + "_relv"] = np.where(ball_ok, rel, np.nan)

    cols["dmin_diff"] = cols["opp_dmin"] - cols["own_dmin"]
    cols["cx_diff"] = cols["own_cx"] - cols["opp_cx"]
    cols["sx_diff"] = cols["own_sx"] - cols["opp_sx"]
    cols["sy_diff"] = cols["own_sy"] - cols["opp_sy"]
    cols["vx_diff"] = cols["own_vx"] - cols["opp_vx"]
    # Whose half are we in: own attackers beyond halfway vs opp attackers beyond halfway.
    cols["line_gap"] = cols["own_xdef"] - (L - cols["opp_xdef"])
    # Last-known control: nearest-team margin carried through ball gaps.
    last = np.where(ball_ok, cols["dmin_diff"], np.nan)
    idx = np.maximum.accumulate(np.where(ball_ok, np.arange(F), 0))
    cols["last_dmin_diff"] = np.where(idx > 0, last[idx], np.nan)
    nxt_idx = frames_until(ball_ok).astype(int) + np.arange(F)
    nxt_idx = np.minimum(nxt_idx, F - 1)
    cols["next_dmin_diff"] = last[nxt_idx]

    base = list(cols)
    roll = ["speed_all", "frac_slow", "ball_speed", "ball_out", "own_cx", "opp_cx", "cx_diff",
            "sx_diff", "sy_diff", "vx_diff", "own_vx", "opp_vx", "dmin_diff", "own_dmin",
            "opp_dmin", "own_relv", "opp_relv", "line_gap", "ball_ok", "own_ball_ax"]
    for w in WINDOWS:
        for name in roll:
            cols["{}_r{}".format(name, w)] = rolling_mean(cols[name], w)
    # Change in stoppage evidence (start/end of a stoppage).
    cols["speed_all_trend"] = rolling_mean(cols["speed_all"], 25)
    cols["speed_all_trend"] = np.r_[np.full(50, np.nan),
                                    cols["speed_all_trend"][50:] - cols["speed_all_trend"][:-50]]
    # Fine-grained control evidence for short possessions (feature VERSION >= 3).
    if VERSION >= 3:
        fine_touch_features(cols, P, B, BV, dist, teams, own, ball_ok)
    names = list(cols)
    X = np.column_stack([np.asarray(cols[n], dtype=np.float32) for n in names])
    return X, names


TOUCH_DV = 3.0      # m/s change in ball velocity over ~0.25 s: a kick, trap or deflection
TOUCH_R = 2.5       # m: the touching player must be this close (a stride plus a leg)


def _shift(x, k):
    """x shifted so out[t] = x[t + k] (NaN-padded)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def fine_touch_features(cols, P, B, BV, dist, teams, own, ball_ok):
    """Touch events and sub-second nearest-team context.

    A touch is a sudden change of ball velocity with a player within TOUCH_R; it is credited
    to that player's team. Both thresholds are physical priors, not fitted to any game.
    """
    F = len(B)
    dv = np.sqrt(((_shift(BV[:, 0], 3) - _shift(BV[:, 0], -3)) ** 2
                  + (_shift(BV[:, 1], 3) - _shift(BV[:, 1], -3)) ** 2))
    dmin_t = {}
    for t in (0, 1):
        dmin_t[t] = _masked_min(dist[:, teams == t], 1)
    touch = {}
    for t in (0, 1):
        o = 1 - t
        touch[t] = (dv > TOUCH_DV) & (dmin_t[t] < TOUCH_R) & ~(dmin_t[o] < dmin_t[t])
    # Collapse each burst of touch frames to its first frame.
    for t in (0, 1):
        x = touch[t]
        touch[t] = x & ~np.r_[False, x[:-1]]
    for tag, t in (("own", own), ("opp", 1 - own)):
        cols[tag + "_since_touch"] = np.minimum(frames_since(touch[t]), 2000)
        cols[tag + "_until_touch"] = np.minimum(frames_until(touch[t]), 2000)
        cols[tag + "_touches_r50"] = rolling_mean(touch[t].astype(float), 50) * 101
    # Team of the last touch and of the next touch (+1 own, -1 opp, 0 none yet).
    last = np.zeros(F)
    nxt = np.zeros(F)
    any_touch = touch[own] | touch[1 - own]
    val = np.where(touch[own], 1.0, np.where(touch[1 - own], -1.0, 0.0))
    idx = np.maximum.accumulate(np.where(any_touch, np.arange(F), -1))
    last[idx >= 0] = val[idx[idx >= 0]]
    ridx = np.where(any_touch, np.arange(F), F)
    ridx = np.minimum.accumulate(ridx[::-1])[::-1]
    nxt[ridx < F] = val[ridx[ridx < F]]
    cols["last_touch_team"] = last
    cols["next_touch_team"] = nxt
    cols["ball_dv"] = dv
    # Sub-second nearest-team context.
    near = np.where(ball_ok, np.sign(cols["dmin_diff"]), np.nan)
    for w in (5, 12):
        cols["near_r{}".format(w)] = rolling_mean(near, w)
    for k in (-25, -12, -5, 5, 12, 25):
        cols["dmin_diff_s{:+d}".format(k)] = _shift(cols["dmin_diff"], k)
