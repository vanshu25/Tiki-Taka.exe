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
    """Most likely H/A/D path given per-frame probabilities and transition penalties (in nats)."""
    logp = np.log(np.clip(probs, eps, 1.0))
    T = np.array([[0.0, switch, to_dead],
                  [switch, 0.0, to_dead],
                  [from_dead, from_dead, 0.0]])
    n = len(probs)
    back = np.empty((n, 3), np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - T              # from i (rows) to j (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(probs, runs, switch, to_dead, from_dead):
    """Viterbi inside each graded run; X elsewhere."""
    out = np.full(len(probs), X, np.int8)
    for a, b in runs:
        out[a:b] = viterbi(probs[a:b], switch, to_dead, from_dead)
    return out
