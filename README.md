# Tiki-Taka.exe: Possession Challenge (BarcelonaHack 2026)

For every frame of FC Barcelona vs Athletic Club, say who has the ball: `H` (Barcelona), `A`
(Athletic) or `D` (dead ball). Ranked on **spellF1** (whole possessions matched by overlap), with
macroF1 (per-frame) as the tie-break. The organisers' original README is in
[ORGANISER_README.md](ORGANISER_README.md).

**Current submission: pipeline v5** (`python -m possession.run`). Pipeline v4 is kept as
`python -m possession.run_v4`.

---

## What changed in v5

v4 scored well on the practice test and poorly on the one stretch of the real match that has
answers. Three findings explain the gap, and v5 is built on them.

**1. The practice labels missed every goal-kick stoppage.** Dead balls were derived from event
data: from the last live event to the next `SET PIECE`. In practice games 1 and 2 a goal kick is
logged as the keeper's `RECOVERY` and the kick in the same frame, so the rule never fired. In
game 1 that is 22 stoppages and 9% of all frames labelled as live possession. v5 reads dead
balls from the corrected tracking instead: the ball has no position for 1 s or more
(`external.dead_from_tracking`, label version 3). That is the challenge's own definition ("where
the corrected tracking marks the ball dead"), and the corrected ball is absent in the 2 s before
93–100% of free kicks, throw-ins, corners and kick-offs.

**2. The practice dead ball does not behave like a real one.** The degradation fills dead-ball
gaps with a straight line from where play stopped to where it restarts. A real dead ball is
carried, thrown back and placed. Measured on inputs only, with TV replays as the stand-in for
"dead" on the real match:

| Ball is tracked and… | Ball speed (median) | Ball nearly still | Nearest player (median) |
|---|---|---|---|
| practice data, truth dead | 0.7 m/s | 76% | 3.5 m |
| BAR-ATH, during a TV replay | 2.4 m/s | 26% | 2.3 m |

v4 learned "slow, straight ball = dead", which holds on the practice test and not on the match.

**3. v4's edge on the practice test comes from exactly those frames.** With true dead balls
given, v4 finds changes of possession worse than v5 (table below). Its end-to-end score is
level with v5 only because it detects the artificial dead ball almost perfectly.

So v5 splits the problem in two, and neither half can see the artificial dead ball:

* a **shape model** decides dead vs alive from the players and from whether the ball is tracked
  at all, never from where the ball is;
* a **possession model** decides home vs away, and is trained on live frames only.

On the real match it adds evidence the practice games do not have: the ball seen off the pitch,
the TV feed cutting to a replay, restarts, and the Smart Tagging phases.

---

## Results

### 1. Practice test

Three public Metrica Sports matches, degraded to look like our tracking. Train on 2, test on the
3rd, rotating, with 2 degradation seeds each (`python -m possession.experiment_v5 --with-v4`).
Both pipelines are trained and scored on the fixed labels (version 3).

| Run | spellF1 | macroF1 | Possessions vs true | spellF1 with true dead balls given |
|---|---|---|---|---|
| v4, retrained on the fixed labels | 0.702 | 0.904 | 0.91× | 0.751 |
| **v5** | **0.695** | 0.882 | 0.87× | **0.798** |

End to end the two are level: 0.007 apart, inside the ~0.04 noise band. v4 is ahead on macroF1,
which is the artificial dead ball again. v5 is ahead by 0.047 at finding changes of possession,
on all six game/seed runs.

### 2. A check on the real match that needs no ground truth

When the broadcast feed has no pitch homography for 2 s or more, the director has cut to a
replay or a close-up, which happens almost only while the ball is dead. A good labelling should
call most of those frames `D`, including the ones where the ball is still tracked
(`python -m possession.realcheck <csv> ...`). v5 is measured with its own broadcast evidence
switched off, so the witness is independent.

| Share of frames labelled `D` | TV replays | TV replays, ball tracked | Live picture |
|---|---|---|---|
| v4 as submitted | 0.825 | 0.404 | 0.227 |
| v4, retrained on the fixed labels | 0.856 | 0.480 | 0.242 |
| **v5** | **0.907** | **0.678** | 0.265 |

Replays cover 31% of graded frames, and the ball is tracked on 23% of them. On those frames v4
calls the ball dead 40% of the time.

### 3. On Barça–Athletic

| Run | Possessions (full match) | Mean length | Dead-ball share | Full-match spellF1 |
|---|---|---|---|---|
| baseline_smooth50 | 669 | 7.6 s | – | **0.2433** (published) |
| v4 | 288 | 12.5 s | 43% | private until results |
| **v5** | **238** | **13.9 s** | 48% | private until results |

Public 5-minute window, used only to check frame alignment and label meaning, **never to choose
between versions or to set a constant**. It holds 17 possessions, so the 90% bootstrap interval
is about ±0.18:

| Run | spellF1 | macroF1 | Possessions (truth 17) |
|---|---|---|---|
| baseline_smooth50 | 0.326 | 0.621 | 26 |
| v4 | 0.352 | 0.745 | 16 |
| v4, retrained on the fixed labels | 0.387 | 0.800 | 14 |
| v5 | 0.632 | 0.920 | 14 |

v4 misses two of the four stoppages in that window outright. That is what prompted the
investigation above; the decision to move to v5 rests on sections 1 and 2.

### 4. History (label version 2, before the dead-ball fix)

These were scored against the old labels, so they are not comparable with the table in
section 1.

| Run | What it is | spellF1 | macroF1 | Possessions vs true |
|---|---|---|---|---|
| baseline_raw | nearest player to the ball | 0.057 | 0.527 | 6.28× |
| baseline_smooth25 | + 1 s majority vote | 0.211 | 0.533 | 2.44× |
| baseline_smooth50 | + 2 s majority vote (official bar) | 0.248 | 0.527 | 1.94× |
| baseline_smooth125 | + 5 s majority vote | 0.243 | 0.487 | 1.27× |
| v1 | team-relative features + classifier + state machine | 0.640 | 0.877 | 1.04× |
| v2 | + ball goes missing at stoppages, as in our tracking | 0.682 | 0.893 | 1.01× |
| v3 | + practice labels follow the controlled-possession rule | 0.665 | 0.891 | 0.95× |
| v4 | + smooth (realistic) tracking noise | 0.675 | 0.897 | 0.96× |

The baselines score almost the same on the practice test as on the real full match, and in the
same order (raw 0.057 / 0.049, smooth25 0.211 / 0.200, smooth50 0.248 / 0.243, smooth125
0.243 / 0.224).

---

## How v5 works

1. **Geometry** (`possession/v5/geometry.py`, `touches.py`, `restarts.py`): ball and player
   positions in metres, each team's nearest player and control of the ball, **touches** (a kink
   in the ball's velocity with a player within 2.5 m; repeats by the same player within 1 s
   count once), the ball off the pitch, the ball vanishing towards a line, and restart
   candidates.
2. **Shape model** (`possession/v5/learned.py`): gradient-boosted trees giving P(home), P(away),
   P(dead) from 62 features. 55 describe the players: each team's centroid and spread in its
   attacking direction, forward velocity, players in the opponent's half, the share standing,
   walking, running and sprinting, collective acceleration, and mean speed at several lags and
   leads. 7 describe ball *visibility*: tracked or not, for how long it has been gone, how long
   until it is back. Nothing about where the ball is.
3. **Possession model**: gradient-boosted trees giving log-odds home vs away from 46 features,
   built symmetrically for the two teams: distance to each team's nearest player, each team's
   control, each team's touches in windows of 0.5 to 4 s before and after the frame, time since
   and until each team's last and next touch, ball speed, and the shape model's opinion.
   Trained on live frames only.
4. **Evidence** (`possession/v5/pipeline.py`), summed per frame:

   | Signal | Goes to | Weight |
   |---|---|---|
   | possession model log-odds (clipped at ±4) | H or A, and D by the same amount | 0.75 per unit |
   | shape model log-odds of dead (clipped at ±1.5) | D | 0.6 |
   | ball seen more than 0.5 m off the pitch | D | 2.5 per frame |
   | ball vanishes for ≥ 1 s while heading off the pitch | D for 1 s | 2.0 per frame |
   | ball travelling > 6 m/s inside the pitch | H and A | 0.3 per frame |
   | TV feed cut away for ≥ 2 s | D | 0.15 × log-ratio (2.1 / −0.85) |
   | Smart Tagging phase naming the team in possession | that team | 0.05 per frame |

   Knowing which team has the ball says nothing about whether it is in play, because players
   hold the ball at throw-ins and free kicks too. So the dead state receives the same amount as
   the winning team from the possession model.
5. **State machine** (`possession/v5/decoder.py`): Viterbi over H / A / D. A change of
   possession costs 10, alive → dead costs 15. **Dead → alive costs 8 within 1.2 s of a detected
   restart and 150 anywhere else**, so a dead ball only comes back to life at a restart. A
   restart is the ball nearly still for a second with calm players and then struck, or the
   tracker picking the ball up again after a gap of 1 s or more, or a Smart Tagging kick-off.
6. **Clean-up**: a "stoppage" under 3 s with the same team on both sides is removed, because no
   restart is taken that quickly. Each restart is moved back onto the detected kick or throw (up
   to 2 s). The second half's graded frames before kick-off are `D` (kick-off anchor from
   `games.py`). `X` is written only on ungraded frames.

Both models see every frame from both teams' sides (home and away swapped as augmentation), so
neither knows anything about "home" or "away".

### The broadcast feed

The broadcast is not synchronised with the tactical video. Both recordings have automatic ball
tracking in pitch coordinates, so matching the two ball tracks over candidate offsets finds the
mapping: tactical frame = broadcast frame + 4014 in the first half and + 26920 in the second,
stable to ±1 frame in every 60 s window (`possession/v5/broadcast.py`). Neither video is decoded.

### What was chosen where

* The possession-model weight (0.75) was chosen on held-out practice games. The score is flat
  between 0.5 and 1.0, and flat in the dead-ball weight and the switching costs too.
* The physical dead-ball signals, the restart rule and the broadcast weights cannot be measured
  on the practice games, which have no off-pitch ball, no real restarts and no broadcast. They
  were set from football reasoning and physical units before any score was seen.

### The external harness (`possession/harness.py`, `possession/degrade.py`)

- **Stand-in labels** from Metrica's data for those matches (`possession/external.py`). Dead
  ball is where the corrected tracking has no ball for 1 s or more. Possession is the team of
  the latest ball event, and a switch only counts if the new team shows **control** (completed
  pass, shot, or 2 s on the ball), following CHALLENGE.md.
- **Degradation** to match our tracking, measured from BAR-ATH inputs only (no labels):
  - the ball goes missing in long bursts, resampled from our ball-gap lengths (median ~11 s);
  - the ball is visible ~35% of the time at stoppages and ~69% in play, from ball visibility
    around Smart Tagging set pieces;
  - ~7% of players are missing;
  - position noise is smooth drift, because our player tracks are as smooth as corrected data.

### Tried and dropped

| Idea | Result |
|---|---|
| Explicit stoppage features in v4 (ball out, restart shapes, walking pace) | −0.004 spellF1 |
| Snapping hand-overs to the first nearest-player change | −0.004 to −0.06 |
| Touch-detection / sub-second features added to v4's single classifier | +0.006 |
| Averaging v4's and v5's home-vs-away opinions | +0.003, not worth two models |
| Filling tactical ball gaps from the broadcast ball track | covers 12% of the gaps; sits > 3 m from the tactical ball on a fifth of shared frames |
| Collective player deceleration as a whistle detector | two thirds of such events are in live play |

---

## Limitations

* **Dead-ball timing is the weakest part.** With true dead balls given, v5 scores 0.80 on the
  practice test; end to end it scores 0.69. And the practice test cannot measure the dead-ball
  evidence that only exists on the real match.
* **Whistles.** After a foul or offside the ball often keeps rolling and the players slow down
  gradually, so the stoppage can be detected a few seconds late.
* **The practice dead ball is still a straight line.** v5 avoids learning from it; a degradation
  that moves the dead ball the way real ones move would let a model use it.
* **Very short possessions** (under about 2 s) are often merged into their neighbours.
* **Team labels of fragmented tracks** are trusted as given.

---

## Rules compliance

- **No BAR-ATH labels used for training, tuning or selection.** The public 5 minutes were used
  only to check alignment and label meaning.
- **No hand labels or hand corrections** on any data.
- From BAR-ATH we use only **inputs**: tactical tracking, Smart Tagging, and the broadcast feed's
  tracking and homography.
- **No match data in this repository.** `.gitignore` excludes the data, caches and generated
  submissions.

## Reproduce

```bash
# 1. Match data (from the organisers), laid out as data/tactical/... and data/broadcast/...:
export MTAH_DATA_DIR=~/mtah-data

# 2. External data: Metrica Sports sample data
git clone https://github.com/metrica-sports/sample-data ~/mtah-data/external/sample-data
export MTAH_EXTERNAL_DIR=~/mtah-data/external/sample-data/data   # this path is the default

# 3. Environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

# 4. Submission (~5 min the first time, ~1 min once the models are cached; identical on every run)
python -m possession.run --out submissions/<team>.csv
python evaluation/validate_submission.py submissions/<team>.csv

# Practice test and checks
python -m possession.experiment_v5 --with-v4   # leave-one-game-out, v4 and v5 side by side
python -m possession.realcheck submissions/<team>.csv   # TV-replay check on the real match
python -m possession.run_v4 --out submissions/<team>_v4.csv   # the previous pipeline
```

v5 needs these files from the bundle: `tactical/BAR-ATH_tactical_atd.{xml,txt}`,
`tactical/BAR-ATH_tactical_smart_tagging.xml`, `broadcast/BAR-ATH_broadcast_atd.{xml,txt}` and
`broadcast/BAR-ATH_broadcast_homography.csv`.

## External data

- **Metrica Sports sample data**: <https://github.com/metrica-sports/sample-data>. Three
  anonymised matches with tracking and events. No formal licence; the README asks for responsible
  use and acknowledgement of the source. Human-made event data about *other* games, as the rules
  allow.

## Repository layout

```
possession/      our pipelines
  v5/            pipeline v5: geometry, touches, restarts, shape, learned, decoder, broadcast, pipeline
  run.py         v5 entry point            experiment_v5.py   practice test for v5 (and v4)
  run_v4.py      v4 entry point            realcheck.py       TV-replay check on the real match
  atd, external, degrade, harness          shared: loaders, practice labels, degradation, scoring
  features, model, experiment, robustness  pipeline v4
evaluation/      the organisers' scorer and validator (unchanged)
starter/         the organisers' starter kit (unchanged)
games.py         the organisers' match constants (unchanged)
```

The organisers' starter code is under [LICENSE-ORGANISER-STARTER](LICENSE-ORGANISER-STARTER).
Our code is under [LICENSE](LICENSE) (MIT).
