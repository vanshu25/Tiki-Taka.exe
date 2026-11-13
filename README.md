# Tiki-Taka.exe: Possession Challenge (BarcelonaHack 2026)

For every frame of FC Barcelona vs Athletic Club, say who has the ball: `H` (Barcelona), `A`
(Athletic) or `D` (dead ball). Ranked on **spellF1** (whole possessions matched by overlap), with
macroF1 (per-frame) as the tie-break. The organisers' original README is in
[ORGANISER_README.md](ORGANISER_README.md).

**Current submission: pipeline v4.**

---

## Results vs the baseline

### 1. Practice test (how we measure every change)

Nothing can be tuned on Barça–Athletic, so every model and setting is chosen on **3 public Metrica
Sports matches**. Their tracking is degraded to look like ours. We train on 2 matches and test on
the 3rd, rotating through all 3, with 2 degradation seeds each. Everyone below is graded on the
same set (degradation v3, controlled-possession labels).

| Run | What it is | spellF1 | macroF1 | Possessions vs true |
|---|---|---|---|---|
| baseline_raw | nearest player to the ball | 0.057 | 0.527 | 6.28× |
| baseline_smooth25 | + 1 s majority vote | 0.211 | 0.533 | 2.44× |
| **baseline_smooth50** | + 2 s majority vote (**official bar**) | **0.248** | 0.527 | 1.94× |
| baseline_smooth125 | + 5 s majority vote | 0.243 | 0.487 | 1.27× |
| v1 | team-relative features + classifier + state machine | 0.640 | 0.877 | 1.04× |
| v2 | + ball goes missing at stoppages, as in our tracking | 0.682 | 0.893 | 1.01× |
| v3 | + practice labels follow the controlled-possession rule | 0.665 | 0.891 | 0.95× |
| **v4** | + smooth (realistic) tracking noise | **0.675** | **0.897** | **0.96×** |

**Is the practice test a fair stand-in?** The baselines score almost the same on it as on the
real full match, and in the same order:

| Baseline | Practice test | Real full match (published) |
|---|---|---|
| raw | 0.057 | 0.0486 |
| smooth25 | 0.211 | 0.1996 |
| smooth50 | 0.248 | 0.2433 |
| smooth125 | 0.243 | 0.2242 |

v2–v4 are within the ~0.04 spellF1 noise band of each other. v4 is the submission because it is
the most realistic (labels follow the challenge's definition, noise matches our tracker) and has
the best macroF1, which is the tie-break.

### 2. On Barça–Athletic

| Run | Possessions (full match) | Mean length | Full-match spellF1 |
|---|---|---|---|
| baseline_smooth50 | 669 | 7.6 s | **0.2433** (published) |
| **v4** | **288** | **12.5 s** | private until results |

On the practice matches the true count is 251–294 possessions with a mean of 13.5–15 s, and v4
predicts 0.96× the true count there. The baseline invents roughly twice as many.

Public 5-minute window, used only to check frame alignment and label meaning, **never to choose
between versions**. It holds 17 possessions, so the 90% bootstrap interval is about ±0.17:

| Run | spellF1 | macroF1 | Possessions (truth 17) |
|---|---|---|---|
| baseline_smooth50 | 0.326 | 0.621 | 26 |
| v4 | 0.352 | 0.745 | 16 |

### 3. Robustness (v4, trained on standard noise, tested on harsher noise)

| Test condition | spellF1 |
|---|---|
| standard | 0.667 |
| ball missing more often | 0.654 |
| ball dropout independent of play | 0.663 |
| 15% of players missing | 0.641 |
| 0.5 m position drift | 0.488 |

The one failure, 0.5 m drift, inflates player speeds far beyond anything in our tracking (median
team speed 2.8 m/s against 1.47 m/s in BAR-ATH), so it does not apply to this match.

---

## How it works

1. **Load and cache** the tactical ATD (`possession/atd.py`), in metres, with a team per track slot.
2. **Features** (`possession/features.py`): ~100 per frame, computed from each team's point of
   view. Nearest-player distances, ball speed and position, team shape (centroid, spread, last
   defender, most advanced player), movement in the attacking direction, overall player speed,
   and 1 s / 3 s / 10 s rolling context. Velocities are differenced from positions, never read
   from the file.
3. **Classifier** (`possession/model.py`): gradient-boosted trees giving P(own team / opponent /
   dead) per frame. Every frame is seen from both teams' sides, so the model is exactly
   symmetric and knows nothing about "home" or "away".
4. **State machine**: Viterbi decoding over H / A / D. Staying in a state is free and switching
   costs a fixed penalty, which stops flicker and keeps the possession count realistic.
   Penalties are frozen in `possession/params.json`.
5. **Fixed blocks**: the second half's graded frames before kick-off are `D` (kick-off anchor from
   `games.py`). `X` is written only on ungraded frames.

### The external harness (`possession/harness.py`, `possession/degrade.py`)

- **Stand-in labels** from Metrica's event data for those matches (`possession/external.py`). A
  switch only counts if the new team shows **control** (completed pass, shot, or 2 s on the ball),
  following CHALLENGE.md. Dead ball runs from the last live touch to the restart.
- **Degradation** to match our tracking, measured from BAR-ATH inputs only (no labels):
  - the ball goes missing in long bursts, resampled from our ball-gap lengths (median ~11 s);
  - the ball is visible ~35% of the time at stoppages and ~69% in play, from ball visibility around
    Smart Tagging set pieces;
  - ~7% of players are missing;
  - position noise is smooth drift, because our player tracks are as smooth as corrected data.

### Tried and dropped (each measured on the harness, none beat the noise band)

| Idea | Change in spellF1 |
|---|---|
| Explicit stoppage features (ball out, restart shapes, walking pace) | −0.004 |
| Snapping hand-overs to the first nearest-player change | −0.004 to −0.06 |
| Touch-detection / sub-second features | +0.006 |

An oracle check showed where the remaining error is: perfect dead-ball information would add
+0.05, while perfect team information would add +0.34.

---

## Rules compliance

- **No BAR-ATH labels used for training, tuning or selection.** The public 5 minutes were used
  only to check alignment and label meaning.
- **No hand labels or hand corrections** on any data.
- From BAR-ATH we use only **inputs**: tracking, plus one Smart Tagging statistic, to make the
  practice data look like it.
- **No match data in this repository.** `.gitignore` excludes the data, caches and generated
  submissions.

## Reproduce

```bash
# 1. Match data (from the organisers), laid out as data/tactical/... or anywhere else:
export MTAH_DATA_DIR=~/mtah-data

# 2. External data: Metrica Sports sample data
git clone https://github.com/metrica-sports/sample-data ~/mtah-data/external/sample-data
export MTAH_EXTERNAL_DIR=~/mtah-data/external/sample-data/data   # this path is the default

# 3. Environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

# 4. Submission (~2 min; byte-identical on every run)
python -m possession.run --out submissions/<team>.csv
python evaluation/validate_submission.py submissions/<team>.csv

# Harness / experiments
python -m possession.experiment      # leave-one-game-out spellF1 at frozen params + decoder grid
python -m possession.robustness      # harsher-noise checks
```

## External data

- **Metrica Sports sample data**: <https://github.com/metrica-sports/sample-data>. Three
  anonymised matches with tracking and events. No formal licence; the README asks for responsible
  use and acknowledgement of the source. Human-made event data about *other* games, as the rules
  allow.

## Repository layout

```
possession/      our pipeline (atd, external, degrade, features, model, harness, run, ...)
evaluation/      the organisers' scorer and validator (unchanged)
starter/         the organisers' starter kit (unchanged)
games.py         the organisers' match constants (unchanged)
```

The organisers' starter code is under [LICENSE-ORGANISER-STARTER](LICENSE-ORGANISER-STARTER).
Our code is under [LICENSE](LICENSE) (MIT).
