"""External games for tuning: Metrica Sports public sample data (3 anonymised matches).

Source: https://github.com/metrica-sports/sample-data (no formal licence; the README asks for
responsible use and acknowledgement of the source). Point MTAH_EXTERNAL_DIR at the clone.

Each game is returned in the same layout as our match (players (F,S,2), ball (F,2), teams (S,),
normalised 0-1 coordinates, row = frame-1) plus a stand-in ground truth derived mechanically from
Metrica's own human-made event data for that game (NOT for BAR-ATH).
"""
import json
import os

import numpy as np
import pandas as pd

from . import atd

EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.expanduser(
    "~/mtah-data/external/sample-data/data")
PITCH = (105.0, 68.0)
H, A, D, X = 0, 1, 2, 3
GAMES = (1, 2, 3)


# -- tracking ----------------------------------------------------------------------------------

def _read_csv_tracking(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    frames = raw["Frame"].values.astype(np.int64)
    periods = raw["Period"].values.astype(np.int64)
    xy = {}
    for i, c in enumerate(cols):
        if c.startswith("Player") or c == "Ball":
            xy[c] = raw.iloc[:, [i, i + 1]].values.astype(np.float32)
    return frames, periods, xy


def _load_csv_game(n):
    folder = os.path.join(EXTERNAL_DIR, "Sample_Game_{}".format(n))
    fh, ph, home = _read_csv_tracking(os.path.join(folder,
                                      "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(n)))
    fa, pa, away = _read_csv_tracking(os.path.join(folder,
                                      "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(n)))
    assert np.array_equal(fh, fa)
    F = int(fh.max())
    hp = [k for k in home if k != "Ball"]
    ap = [k for k in away if k != "Ball"]
    S = len(hp) + len(ap)
    P = np.full((F, S, 2), np.nan, np.float32)
    idx = fh - 1
    for j, k in enumerate(hp + ap):
        P[idx, j] = (home[k] if j < len(hp) else away[k])
    B = np.full((F, 2), np.nan, np.float32)
    B[idx] = home["Ball"]
    period = np.zeros(F, np.int8)
    period[idx] = ph
    teams = np.array([0] * len(hp) + [1] * len(ap), np.int8)
    return {"players": P, "ball": B, "teams": teams, "period": period}


def _load_epts_game3():
    """Game 3 is FIFA/EPTS like our match, but its 22 slots map to different players in each
    period (substitutions), so slots are remapped per DataFormatSpecification to one column
    per player."""
    import xml.etree.ElementTree as ET
    folder = os.path.join(EXTERNAL_DIR, "Sample_Game_3")
    root = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    teams_xml = [t.get("id") for t in root.iter("Team")]
    player_team = {p.get("id"): (0 if p.get("teamId") == teams_xml[0] else 1)
                   for p in root.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    players = sorted(player_team)
    col = {p: i for i, p in enumerate(players)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [chan_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                 if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    frame_ids, P22, ball = atd.parse_raw(os.path.join(folder, "Sample_Game_3_tracking.txt"), 22)
    F = int(frame_ids.max())
    P = np.full((F, len(players), 2), np.nan, np.float32)
    B = np.full((F, 2), np.nan, np.float32)
    period = np.zeros(F, np.int8)
    for k, (s, e, order) in enumerate(specs):
        sel = (frame_ids >= s) & (frame_ids <= e)
        idx = frame_ids[sel] - 1
        for slot, pid in enumerate(order):
            P[idx, col[pid]] = P22[sel, slot, :2]
        B[idx] = ball[sel, :2]
        period[idx] = k + 1
    teams = np.array([player_team[p] for p in players], np.int8)
    return {"players": P, "ball": B, "teams": teams, "period": period}


# -- events ------------------------------------------------------------------------------------

def _events_csv(n):
    ev = pd.read_csv(os.path.join(EXTERNAL_DIR, "Sample_Game_{0}/Sample_Game_{0}_RawEventsData.csv"
                                  .format(n)))
    return pd.DataFrame({
        "team": np.where(ev["Team"].str.lower() == "home", H, A),
        "type": ev["Type"].str.upper(),
        "subtype": ev["Subtype"].fillna("").str.upper(),
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int) - 1,
        "end": ev["End Frame"].astype(int) - 1,
    })


def _events_json3():
    with open(os.path.join(EXTERNAL_DIR, "Sample_Game_3/Sample_Game_3_events.json")) as fh:
        data = json.load(fh)["data"]
    meta = open(os.path.join(EXTERNAL_DIR, "Sample_Game_3/Sample_Game_3_metadata.xml")).read()
    # Home team = first <Team> in the metadata (EPTS convention), matching slot_teams().
    home_id = meta.split("<Team id=\"", 1)[1].split("\"", 1)[0]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = " ".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        rows.append({
            "team": H if e["team"]["id"] == home_id else A,
            "type": e["type"]["name"].upper(),
            "subtype": (sub or "").upper(),
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]) - 1,
            "end": int(e["end"]["frame"] or e["start"]["frame"]) - 1,
        })
    return pd.DataFrame(rows)


LABEL_VERSION = 3          # 2 = require controlled possession for an open-play switch
                           # 3 = dead ball read from the corrected tracking, not from events
CONTROL_MIN_FRAMES = 50    # "keeps the ball for a few seconds": 2 s, the low end
DEAD_MIN_FRAMES = 25       # the corrected ball is absent this long -> out of play, not a blip

# Events that mean "this team has the ball at the event's start frame".
POSSESSION_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}
# Events that are touches of a live ball (anything a stoppage can follow).
PLAY_TYPES = POSSESSION_TYPES | {"CHALLENGE"}


def _is_restart(row):
    return row.type == "SET PIECE" or (row.type == "PASS" and "GOAL KICK" in row.subtype)


def dead_from_tracking(ball, period, min_len=DEAD_MIN_FRAMES):
    """Dead-ball frames as the corrected tracking marks them: the ball has no position.

    The challenge defines D as "where the corrected tracking marks the ball dead". Metrica's
    corrected sample tracking removes the ball while it is out of play: it is absent in the 2 s
    before 93-100% of free kicks, throw-ins, corners and kick-offs. Gaps shorter than `min_len`
    frames are tracking blips in live play, not stoppages.

    Version 2 derived D from events instead (last live event -> next SET PIECE). That missed
    every goal-kick stoppage in games 1-2, where the keeper's RECOVERY shares the goal kick's
    frame (about 22 stoppages and 9% of the frames of game 1 were labelled as live possession),
    and it called the keeper holding the ball "dead".
    """
    gone = ~np.isfinite(ball[:, 0]) & (period > 0)
    e = np.diff(np.r_[0, gone.astype(np.int8), 0])
    dead = np.zeros(len(gone), bool)
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a >= min_len:
            dead[a:b] = True
    return dead


def events_to_labels(ev, n_frames, period, ball=None):
    """Stand-in H/A/D/X labels from event data, mechanically:

    * live ball: the team of the most recent possession event (pass, carry, shot, recovery,
      set piece, ball lost/out). Challenges, fouls and cards do not change possession, so a
      loose ball stays with the last team -- the challenge definition.
    * dead ball: from the end of the last live-ball event before each restart (set piece or
      goal kick) up to the restart frame.
    * X outside each period's first and last event.
    """
    labels = np.full(n_frames, X, np.int8)
    ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    for per in sorted(ev["period"].unique()):
        pe = ev[ev["period"] == per].reset_index(drop=True)
        first, last = int(pe["start"].min()), int(max(pe["end"].max(), pe["start"].max()))
        cur = None
        pos = None
        for row in pe.itertuples():
            if row.type in POSSESSION_TYPES:
                if cur is not None and pos is not None:
                    labels[pos:row.start] = cur
                cur, pos = row.team, row.start
        if cur is not None:
            labels[pos:last + 1] = cur
        # Dead-ball spans before every restart (the opening kick-off is the period start).
        last_play_end = None
        for row in (pe.itertuples() if LABEL_VERSION < 3 or ball is None else ()):
            if _is_restart(row) and last_play_end is not None and last_play_end < row.start:
                labels[last_play_end:row.start] = D
            if row.type in PLAY_TYPES:
                e = row.end if row.end >= row.start else row.start
                last_play_end = e + 1 if row.type != "SET PIECE" else row.start + 1
        labels[:first][period[:first] == per] = X
    labels[period == 0] = X
    if LABEL_VERSION >= 3 and ball is not None:
        labels[dead_from_tracking(ball, period) & (labels != X)] = D
    if LABEL_VERSION >= 2:
        labels = require_control(labels, ev)
    return labels


def require_control(labels, ev, min_hold=CONTROL_MIN_FRAMES):
    """Apply the challenge's definition of a change of possession to event-derived labels.

    CHALLENGE.md: possession changes only when the other team gains *controlled* possession; a
    touch without control, a deflection or a won duel without control does not count, and a
    loose ball stays with the last team. Its suggested proxies are a player keeping the ball
    for a few seconds, or completed passes. So an open-play H<->A switch is kept only if the new
    team, inside its spell, completes a pass (next event is its own), shoots, or keeps
    the ball for `min_hold` frames; otherwise the spell goes back to the previous team.
    Possessions that start at a restart (after D) are always kept: a restart has a clear owner.
    """
    out = labels.copy()
    ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    starts = ev["start"].values
    teams = ev["team"].values
    types = ev["type"].values
    nxt_team = np.r_[teams[1:], -1]
    # CARRY is deliberately not proof of control: only game 3's event format has it, and the
    # 2 s hold already covers a real dribble -- this keeps the rule identical across games.
    shows_control = ((types == "PASS") & (nxt_team == teams)) | (types == "SHOT")
    n = len(out)
    t = 1
    while t < n:
        prev, cur = out[t - 1], out[t]
        if prev in (H, A) and cur in (H, A) and prev != cur:
            e = t
            while e < n and out[e] == cur:
                e += 1
            sel = (starts >= t) & (starts < e) & (teams == cur)
            if not (e - t >= min_hold or shows_control[sel].any()):
                out[t:e] = prev
                continue  # re-examine from t: the merged spell may now meet another switch
            t = e
        else:
            t += 1
    return out


def load_game(n):
    """Tracking + stand-in labels for Metrica sample game n (1, 2 or 3)."""
    d = _load_csv_game(n) if n in (1, 2) else _load_epts_game3()
    ev = _events_csv(n) if n in (1, 2) else _events_json3()
    F = d["ball"].shape[0]
    d["labels"] = events_to_labels(ev, F, d["period"], d["ball"])
    d["events"] = ev
    d["pitch"] = PITCH
    return d
