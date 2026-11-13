"""Fast, cached loader for Metrica FIFA/EPTS raw ATD files.

Line format: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" (players, then ball), 0-1 normalised,
NaN for missing. frame_id counts from 1; array row i is video frame i (= frame_id - 1).

The parsed arrays are cached as .npz under cache/ (gitignored -- it is derived match data).
"""
import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache")


def slot_teams(meta_path):
    """0 = home (first team in XML), 1 = away, per track slot, in file order."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_slots):
    """Return frame_ids (N,), players (N, n_slots, g), ball (N, g) as float32.

    g (values per object) is 3 for x,y,speed exports and 2 for x,y-only exports; it is inferred.
    """
    with open(raw_path, "rb") as fh:
        text = fh.read()
    lines = text.split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    n = len(lines)
    first = lines[0].replace(b":", b",").replace(b";", b",").split(b",")
    g = (len(first) - 1) // (n_slots + 1)
    width = 1 + g * (n_slots + 1)
    if len(first) != width:
        raise ValueError("cannot infer ATD group width from first line")
    # Uniform separators, then one vectorised parse.
    flat = b",".join(lines).replace(b":", b",").replace(b";", b",")
    vals = np.array(flat.split(b","), dtype=np.float64)
    if vals.size != n * width:
        raise ValueError("unexpected ATD width: {} values for {} lines".format(vals.size, n))
    vals = vals.reshape(n, width)
    frame_ids = vals[:, 0].astype(np.int64)
    players = vals[:, 1:1 + g * n_slots].reshape(n, n_slots, g).astype(np.float32)
    ball = vals[:, 1 + g * n_slots:].astype(np.float32)
    return frame_ids, players, ball


def load(meta_path, raw_path, n_frames=None, use_cache=True):
    """Dict with players (F, S, 3), ball (F, 3), teams (S,), in normalised units.

    Rows are indexed by video frame (frame_id - 1). Frames absent from the file stay NaN.
    """
    key = hashlib.sha1("{}|{}|{}".format(os.path.abspath(raw_path), os.path.getsize(raw_path),
                                         n_frames).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE_DIR, "atd_{}.npz".format(key))
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    teams = slot_teams(meta_path)
    frame_ids, players, ball = parse_raw(raw_path, len(teams))
    n = n_frames or int(frame_ids.max())
    g = players.shape[2]
    P = np.full((n, len(teams), g), np.nan, dtype=np.float32)
    B = np.full((n, g), np.nan, dtype=np.float32)
    idx = frame_ids - 1
    ok = (idx >= 0) & (idx < n)
    P[idx[ok]] = players[ok]
    B[idx[ok]] = ball[ok]
    out = {"players": P, "ball": B, "teams": teams}
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    return out
