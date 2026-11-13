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

Deterministic for a given seed.
"""
import numpy as np


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


def degrade(d, ball_runs, seed, player_missing=0.07, player_gap_mean=200,
            ball_jitter=0.1, player_jitter=0.05, pitch=(105.0, 68.0)):
    """Return a degraded copy of game dict d (normalised coordinates)."""
    rng = np.random.default_rng(seed)
    L, W = pitch
    F = d["ball"].shape[0]
    out = dict(d)

    # Ball: fill every gap, then knock out bursts with our match's own run-length distribution.
    ball = _interp_gaps(d["ball"][:, :2].astype(np.float64))
    visible_runs, missing_runs = ball_runs
    keep = np.zeros(F, bool)
    t = int(rng.integers(0, int(np.median(visible_runs))))
    on = bool(rng.integers(0, 2))
    while t < F:
        n = int(rng.choice(visible_runs if on else missing_runs))
        keep[t:t + n] = on
        t += n
        on = not on
    ball = ball + rng.normal(0, ball_jitter, ball.shape) / np.array([L, W])
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
    P = P + rng.normal(0, player_jitter, P.shape) / np.array([L, W])
    out["players"] = P.astype(np.float32)
    return out
