"""Make a clean external game look like our automatic, uncorrected tracking.

What is matched, and where the numbers come from:

* Ball dropout in long bursts. Our ATD loses the ball in ~150 gaps per match with a median of
  ~11 s, not in short occlusions. Visible/missing run lengths are resampled from the empirical
  run-length distribution of the BAR-ATH ball track itself -- a property of the *input* data,
  computed without any label -- passed in as `ball_runs`.
* No "missing means dead" shortcut. Metrica's corrected sample tracking removes the ball while it
  is out of play, which our raw ATD does not. Before dropout the ball is linearly interpolated
  through every gap in the clean track, so a model cannot learn that missing ball = dead ball.
* Player dropout (our match averages 20.5 tracked players of 22) and small position jitter.

* Ball dropout depends on the state (VERSION >= 2). On BAR-ATH the ball is visible on 59% of
  graded frames overall but only ~35% in the 3 s windows that the (automatic) Smart Tagging
  marks just before a restart -- evidence from inputs only, no labels. External dead-ball
  frames therefore keep the ball with probability VIS_DEAD, live frames with VIS_LIVE, chosen so
  the overall rate stays near 59% at a typical ~30% dead share.

* Position noise is smooth drift, not per-frame shake (VERSION >= 3). The BAR-ATH player track
  is very smooth (median second difference ~1 mm/frame, same as Metrica's corrected data), so
  white jitter would invent velocity noise the real data does not have. Noise is white noise
  averaged over ~1 s and rescaled to the requested standard deviation.

Deterministic for a given seed.
"""
import numpy as np

VERSION = 3
VIS_DEAD = 0.35
VIS_LIVE = 0.69


def run_lengths(mask):
    """Lengths of the True runs and the False runs of a boolean array."""
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    on = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    e = np.diff(np.r_[0, (~mask).astype(np.int8), 0])
    off = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    return on, off


def _interp_gaps(x):
    """Linear interpolation through NaN gaps (edges held), column-wise."""
    out = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        ok = np.isfinite(x[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(idx, idx[ok], x[ok, c])
    return out


def _burst_mask(rng, F, ball_runs, visible_share):
    """Alternating visible/missing bursts with BAR-ATH run lengths; visible runs rescaled so the
    long-run visible share is `visible_share`."""
    visible_runs, missing_runs = ball_runs
    v, m = float(np.mean(visible_runs)), float(np.mean(missing_runs))
    scale = visible_share * m / ((1 - visible_share) * v)
    keep = np.zeros(F, bool)
    t = int(rng.integers(0, int(np.median(visible_runs))))
    on = bool(rng.integers(0, 2))
    while t < F:
        n = int(rng.choice(visible_runs) * scale) if on else int(rng.choice(missing_runs))
        keep[t:t + max(n, 1)] = on
        t += max(n, 1)
        on = not on
    return keep


def _smooth_noise(rng, shape, std, width=25):
    """Gaussian noise low-passed along axis 0 (moving average over `width` frames), with the
    requested per-sample standard deviation."""
    if std <= 0:
        return np.zeros(shape)
    white = rng.normal(0, 1, shape)
    k = np.ones(width) / width
    flat = white.reshape(shape[0], -1)
    sm = np.stack([np.convolve(flat[:, j], k, mode="same") for j in range(flat.shape[1])], 1)
    sm *= std / max(sm.std(), 1e-12)
    return sm.reshape(shape)


def degrade(d, ball_runs, seed, player_missing=0.07, player_gap_mean=200,
            ball_jitter=0.1, player_jitter=0.05, pitch=(105.0, 68.0), dead=None):
    """Return a degraded copy of game dict d (normalised coordinates).

    `dead` (bool per frame, from the external game's own labels) switches on state-dependent
    ball dropout; without it the dropout is state-independent (VERSION 1 behaviour).
    """
    rng = np.random.default_rng(seed)
    L, W = pitch
    F = d["ball"].shape[0]
    out = dict(d)

    # Ball: fill every gap, then knock out bursts with our match's own run-length distribution.
    ball = _interp_gaps(d["ball"][:, :2].astype(np.float64))
    if dead is None:
        keep = _burst_mask(rng, F, ball_runs, 1 - np.mean(ball_runs[1]) /
                           (np.mean(ball_runs[0]) + np.mean(ball_runs[1])))
    else:
        keep_live = _burst_mask(rng, F, ball_runs, VIS_LIVE)
        keep_dead = _burst_mask(rng, F, ball_runs, VIS_DEAD)
        keep = np.where(dead, keep_dead, keep_live)
    noise = (_smooth_noise(rng, ball.shape, ball_jitter) if VERSION >= 3
             else rng.normal(0, ball_jitter, ball.shape))
    ball = ball + noise / np.array([L, W])
    ball[~keep] = np.nan
    out["ball"] = ball.astype(np.float32)

    # Players: drop each track in random bursts (geometric gaps), then jitter.
    P = d["players"][..., :2].astype(np.float64).copy()
    S = P.shape[1]
    p_start = player_missing / player_gap_mean / (1 - player_missing)
    for s in range(S):
        starts = np.flatnonzero(rng.random(F) < p_start)
        lens = rng.geometric(1.0 / player_gap_mean, len(starts))
        for a, n in zip(starts, lens):
            P[a:a + n, s] = np.nan
    noise = (_smooth_noise(rng, P.shape, player_jitter) if VERSION >= 3
             else rng.normal(0, player_jitter, P.shape))
    P = P + noise / np.array([L, W])
    out["players"] = P.astype(np.float32)
    return out
