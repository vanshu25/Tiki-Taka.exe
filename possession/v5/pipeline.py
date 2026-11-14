"""Pipeline v5: tracking + Smart Tagging + broadcast in, frame labels out.

  1. Geometry    ball/player geometry, touches, out-of-play, restarts (geometry, touches, restarts).
  2. Two models  trained on the degraded practice games (learned.py):
                   shape model       players + ball visibility -> P(home), P(away), P(dead)
                   possession model  ball + players around each frame -> log-odds home vs away
  3. Evidence    the models' output plus physical signals that exist only on the real match:
                 ball seen off the pitch, ball vanishing towards a line, the TV feed cutting to a
                 replay (broadcast.py), Smart Tagging phases.
  4. Decode      3-state Viterbi (H/A/D) with switching costs; a dead ball only comes back to
                 life at a detected restart.
  5. Clean-up    sub-3 s "stoppages" inside one team's possession removed; restarts dated to the
                 kick or throw; the wait before the second-half kick-off is D (games.py anchor).

No BAR-ATH label is read anywhere.
"""
import time
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, anchor_video_frame, asset_paths, half_video_blocks

from .. import atd
from . import geometry as G
from . import learned
from .broadcast import broadcast_evidence
from .decoder import viterbi_tv, H, A, D
from .restarts import exit_events, restart_candidates

CONFIG = dict(
    # --- who has the ball ---
    # Weight on the possession model's log-odds (home vs away), per frame, against a switching
    # cost of 10. Chosen on held-out practice games; the score is flat between 0.5 and 1.0.
    w_ha=0.75,
    # --- dead-ball evidence ---
    w_dead=0.6,        # weight on the shape model's log-odds of "dead", clipped at +-dead_clip
    dead_clip=1.5,
    w_out=2.5,         # ball seen beyond a line (by > 0.5 m): strong per-frame evidence of D
    w_exit=2.0,        # ball vanished while flying out: D evidence for the next second
    exit_len=25,
    # --- alive evidence ---
    w_fast=0.3,        # ball travelling > 6 m/s inside the pitch (passes, shots)
    fast_v=6.0,
    w_tag=0.05,        # Smart Tagging phase naming the team in possession (soft prior)
    # --- switching costs (Viterbi) ---
    P_HA=10.0,         # cost of one change of possession (the unit the evidence is scaled to)
    P_D=15.0,          # alive -> dead
    P_rev_lo=8.0,      # dead -> alive near a detected restart (+-1.2 s)
    P_rev_hi=150.0,    # dead -> alive anywhere else (effectively: only at restarts)
    rev_win=30,
    snap_back=50,      # restart boundary may move back up to 2 s, onto the detected kick/throw
    min_dead=75,       # no restart is taken in under 3 s: a shorter "stoppage" inside one team's
                       # possession is not a stoppage
    # --- broadcast feed (aligned automatically, see broadcast.py) ---
    w_bc=0.15,         # shrink on the per-frame log-likelihood ratio (frames are correlated)
    bc_llr_away=2.1,   # log P(cut-away | dead) / P(cut-away | live), measured self-consistently
    bc_llr_live=-0.85, # on this match's own tracking-only labels (no ground truth)
    bc_min_run=50,     # only cut-aways of >= 2 s count (replays/close-ups, not quick shot changes)
)


def build_scores(fe, cfg, aux):
    n = fe["n"]
    S = np.zeros((n, 3))
    lo = aux["ha_logodds"]
    S[:, H] += 0.5 * cfg["w_ha"] * lo
    S[:, A] -= 0.5 * cfg["w_ha"] * lo
    # Players touch and hold the ball in a stoppage too (throw-ins, free kicks): knowing WHICH
    # team has it says nothing about alive-vs-dead, so the dead state gets the winner's share.
    S[:, D] += 0.5 * cfg["w_ha"] * np.abs(lo)

    S[:, D] += cfg["w_dead"] * np.clip(aux["dead_logodds"], -cfg["dead_clip"], cfg["dead_clip"])
    S[:, D] += cfg["w_out"] * fe["out"]
    ex = np.convolve(aux["exit"].astype(float), np.ones(cfg["exit_len"]), mode="full")[:n] > 0
    S[:, D] += cfg["w_exit"] * ex

    with np.errstate(invalid="ignore"):
        fast = fe["ball_vis"] & ~fe["out"] & (fe["bspeed"] > cfg["fast_v"])
    S[:, H] += cfg["w_fast"] * fast
    S[:, A] += cfg["w_fast"] * fast
    tag = aux.get("tag")
    if tag is not None:
        S[:, H] += cfg["w_tag"] * (tag > 0)
        S[:, A] += cfg["w_tag"] * (tag < 0)
    bc = aux.get("bc_away")
    if bc is not None:
        S[:, D] += cfg["w_bc"] * np.where(np.isnan(bc), 0.0,
                                          np.where(bc > 0.5, cfg["bc_llr_away"], cfg["bc_llr_live"]))
    for a, b in aux.get("forced_dead", ()):
        S[a:b, D] += 1e6
    return S


def decode(S, cfg, revive, hv):
    C = np.array([[0, cfg["P_HA"], cfg["P_D"]],
                  [cfg["P_HA"], 0, cfg["P_D"]],
                  [cfg["P_D"], cfg["P_D"], 0]], dtype=float)
    lab = np.full(S.shape[0], "X", dtype="<U1")
    for a, b in hv:
        lab[a:b] = np.array(["H", "A", "D"])[viterbi_tv(S[a:b], C, revive[a:b])]
    return lab


def snap_restarts(lab, kicks, back):
    """Play restarts the moment the ball is kicked or thrown: move each dead -> alive boundary
    back to the latest detected restart kick in the preceding `back` frames, if there is one."""
    lab = lab.copy()
    for s in np.flatnonzero((lab[1:] != lab[:-1]) & (lab[:-1] == "D") & np.isin(lab[1:], ["H", "A"])) + 1:
        k = np.flatnonzero(kicks[max(s - back, 0):s + 1])
        if len(k):
            t = max(s - back, 0) + k[-1]
            if t < s and np.all(lab[t:s] == "D"):
                lab[t:s] = lab[s]
    return lab


def merge_short_dead(lab, min_len):
    """A dead ball shorter than min_len frames with the same team on both sides is not a
    stoppage (no restart is that quick): give it back to that team."""
    lab = lab.copy()
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for i, j in zip(np.r_[0, b], np.r_[b, len(lab)]):
        if lab[i] == "D" and j - i < min_len and i > 0 and j < len(lab) \
                and lab[i - 1] == lab[j] and lab[j] in "HA":
            lab[i:j] = lab[j]
    return lab


def _long_runs_only(away, min_run):
    """Keep cut-away runs of at least min_run frames; shorter ones are treated as live picture."""
    out = away.copy()
    a = np.nan_to_num(away, nan=0.0) > 0.5
    i, n = 0, len(a)
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i < min_run:
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    return out


def label_game(p, models, cfg=CONFIG, tag=None, kickoffs=(), bc=None, forced_dead=()):
    """Label one prepared game (learned.prepare). Returns H/A/D letters, X outside the halves.

    tag: per-frame team prior (+1 home, -1 away) or None; kickoffs: (start, end) frame windows
    known to contain a kick-off; bc: per-frame broadcast cut-away flag (1/0/NaN) or None;
    forced_dead: (start, end) frame ranges known to be dead.
    """
    fe, hv = p["fe"], p["hv"]
    n = fe["n"]
    cand, kicks = restart_candidates(fe["bx"], fe["by"], fe["ball_vis"], fe["mean_speed"],
                                     return_kicks=True)
    for s, f in kickoffs:
        cand[s:f] = True
    near = np.convolve(cand.astype(float), np.ones(2 * cfg["rev_win"] + 1), mode="same") > 0
    revive = np.where(near, cfg["P_rev_lo"], cfg["P_rev_hi"])
    for a, b in forced_dead:
        revive[b] = 0.0          # play starts exactly where the forced dead ball ends

    lo, dead = learned.apply(models, p)
    aux = dict(tag=tag, ha_logodds=lo, dead_logodds=dead, forced_dead=forced_dead,
               exit=exit_events(fe["bx"], fe["by"], fe["ball_vis"]))
    if bc is not None:
        aux["bc_away"] = _long_runs_only(bc, cfg["bc_min_run"])

    lab = decode(build_scores(fe, cfg, aux), cfg, revive, hv)
    lab = merge_short_dead(lab, cfg["min_dead"])
    lab = snap_restarts(lab, kicks, cfg["snap_back"])
    for a, b in forced_dead:
        lab[a:b] = "D"
    graded = np.zeros(n, bool)
    for a, b in hv:
        graded[a:b] = True
    lab[~graded] = "X"           # never scored; abstain explicitly
    return lab


def goalkeeper_slots(meta_path):
    """Track slots the ATD metadata tags with position_type == Goalkeeper."""
    out = []
    for i, p in enumerate(ET.parse(meta_path).getroot().iter("Player")):
        for q in p.iter("ProviderParameter"):
            if q.findtext("Name") == "position_type" and q.findtext("Value") == "Goalkeeper":
                out.append(i)
    return out


def run(cfg=CONFIG, verbose=True):
    """The competition match -> per-frame letters (H/A/D/X) for all video frames."""
    t0 = time.time()
    log = (lambda *a: print("[%6.1fs]" % (time.time() - t0), *a, flush=True)) if verbose \
        else (lambda *a: None)
    n = GAMES[GAME]["n_video_frames"]
    paths = asset_paths()
    blocks = half_video_blocks(GAME)
    hv = [(a, b + 1) for _, a, b in blocks]

    models = learned.get_models(log)

    tactical = learned.to_arrays(atd.load(paths["atd_meta"], paths["atd_raw"], n))
    p = learned.prepare(tactical, hv, goalkeeper_slots(paths["atd_meta"]))
    log("match prepared")

    tag, setp = G.smart_tag_priors(n)
    kickoffs = [(s, f) for s, f, typ, _ in setp if "kick off" in typ]

    bpaths = asset_paths(source="broadcast")
    broadcast = learned.to_arrays(atd.load(bpaths["atd_meta"], bpaths["atd_raw"]))
    bc, segs = broadcast_evidence(tactical, broadcast)
    log("broadcast aligned", segs)

    # The second half's graded block opens before its kick-off; the ball is not in play until the
    # kick-off anchor the organisers publish in games.py, so those frames are D.
    forced = []
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    for half, a, b in blocks:
        if half == 2 and a < kickoff2 <= b:
            forced.append((a, kickoff2))

    lab = label_game(p, models, cfg, tag, kickoffs, bc, forced)
    log("decoded")
    return lab, hv
