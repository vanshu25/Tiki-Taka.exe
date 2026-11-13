"""Per-frame classifier (own / opp / dead) and the possession decoder.

Classifier: gradient-boosted trees on team-relative features. Each frame is seen from both
teams' points of view, so the model cannot learn anything about "home" or "away" and is exactly
symmetric at prediction time.

Decoder: Viterbi over H / A / D with fixed switching penalties. Staying in a state is free and
switching costs, which is what keeps the possession count realistic; the penalties are chosen on
the external harness only.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

H, A, D, X = 0, 1, 2, 3
OWN, OPP, DEAD = 0, 1, 2

MODEL_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, early_stopping=False, random_state=0)


def perspective_labels(labels, own):
    """Map H/A/D/X labels to own/opp/dead (-1 for X) from team `own`'s point of view."""
    y = np.full(len(labels), -1, np.int8)
    y[labels == D] = DEAD
    y[labels == own] = OWN
    y[labels == 1 - own] = OPP
    return y


def train(feature_sets, stride=4, params=None):
    """feature_sets: list of (X_own0, X_own1, labels). Returns a fitted classifier."""
    Xs, ys = [], []
    for X0, X1, labels in feature_sets:
        for own, Xo in ((0, X0), (1, X1)):
            y = perspective_labels(labels, own)
            sel = np.flatnonzero(y >= 0)[::stride]
            Xs.append(Xo[sel])
            ys.append(y[sel])
    clf = HistGradientBoostingClassifier(**(params or MODEL_PARAMS))
    clf.fit(np.vstack(Xs), np.concatenate(ys))
    return clf


def predict(clf, X0, X1):
    """Symmetric (F, 3) probabilities over H, A, D."""
    p0 = clf.predict_proba(X0)   # own=home: [home, away, dead]
    p1 = clf.predict_proba(X1)   # own=away: [away, home, dead]
    probs = np.empty_like(p0)
    probs[:, H] = (p0[:, OWN] + p1[:, OPP]) / 2
    probs[:, A] = (p0[:, OPP] + p1[:, OWN]) / 2
    probs[:, D] = (p0[:, DEAD] + p1[:, DEAD]) / 2
    return probs


def viterbi(probs, switch, to_dead, from_dead, eps=1e-6):
    """Most likely H/A/D path given per-frame probabilities and transition penalties (in nats).

    Plain-float inner loop: with three states this is ~20x faster than per-frame numpy calls.
    """
    logp = np.log(np.clip(probs, eps, 1.0)).tolist()
    n = len(logp)
    back = bytearray(3 * n)
    h, a, d = logp[0]
    for t in range(1, n):
        lh, la, ld = logp[t]
        # into H
        c0, c1, c2 = h, a - switch, d - from_dead
        if c0 >= c1 and c0 >= c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        # into A
        c0, c1, c2 = h - switch, a, d - from_dead
        if c1 >= c0 and c1 >= c2:
            na, ba = c1, 1
        elif c0 >= c2:
            na, ba = c0, 0
        else:
            na, ba = c2, 2
        # into D
        c0, c1, c2 = h - to_dead, a - to_dead, d
        if c2 >= c0 and c2 >= c1:
            nd, bd = c2, 2
        elif c0 >= c1:
            nd, bd = c0, 0
        else:
            nd, bd = c1, 1
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
        h, a, d = nh + lh, na + la, nd + ld
    path = np.empty(n, np.int8)
    s = 0 if (h >= a and h >= d) else (1 if a >= d else 2)
    path[-1] = s
    for t in range(n - 1, 0, -1):
        s = back[3 * t + s]
        path[t - 1] = s
    return path


def decode(probs, runs, switch, to_dead, from_dead):
    """Viterbi inside each graded run; X elsewhere."""
    out = np.full(len(probs), X, np.int8)
    for a, b in runs:
        out[a:b] = viterbi(probs[a:b], switch, to_dead, from_dead)
    return out
