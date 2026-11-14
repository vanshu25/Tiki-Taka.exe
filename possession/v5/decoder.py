"""Possession decoding: per-frame evidence + a 3-state Viterbi (H / A / D) with switching costs.

Why Viterbi with switching costs
--------------------------------
Possession is sticky: while the ball is loose or contested it stays with the last team that had
it, and a touch without control does not change it. A hidden-state decoder with a fixed cost per
change of state encodes exactly that: a new team must accumulate more evidence of *control* than
the cost of switching before the label changes, and when it does, the switch is placed where that
evidence starts (the reception), not where it was finally confirmed. Short deflections and
flicker never pay for themselves; real possessions do.
"""
import numpy as np

H, A, D = 0, 1, 2


def viterbi_tv(score, base_cost, revive_cost):
    """Viterbi where the D -> alive cost varies per frame.

    base_cost: (3, 3) costs; revive_cost: (T,) cost of D -> H/A at frame t (overrides base).
    A dead ball can only come back to life at a restart, so revive_cost is low where a restart
    signature is detected and high elsewhere.
    """
    T = score.shape[0]
    back = np.zeros((T, 3), dtype=np.int8)
    cur = score[0].copy()
    trans = -base_cost.astype(float)
    cols = np.arange(3)
    for t in range(1, T):
        tr = trans.copy()
        tr[D, H] = tr[D, A] = -revive_cost[t]
        cand = cur[:, None] + tr
        b = cand.argmax(0)
        back[t] = b
        cur = cand[b, cols] + score[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = cur.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
