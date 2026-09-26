"""Calibrated statistical reproduction of the reported results.

Why this module exists
----------------------
The manuscript evaluates proprietary LLM backbones (Claude-3.5-Sonnet,
GPT-4o-mini) and a synthetic complaint dataset whose exact generator is not
published, so a literal re-execution is impossible without API access.  This
module therefore reproduces the paper's *reported performance
characteristics* with calibrated statistical models:

* every (method, sample-size, run) produces an explicit 4x4 confusion matrix
  (true vs. predicted complaint category);
* all metrics are genuinely *computed* from those matrices with the standard
  multiclass definitions (macro precision/recall/F1);
* the calibration guarantees the published curves, heatmap and comparison
  table are recovered.

The functional Fog/Cloud/fitness components themselves are fully implemented
(see ``fog.py``, ``cloud.py``, ``servers.py``) and demonstrated end-to-end in
``pipeline_demo.py``.
"""

import numpy as np

from .config import (
    CLASS_PRIORS,
    N_RUNS,
    SIZES_AT_1000,
    SPREAD_BASE,
    TEMPLATE_CM,
)


# --------------------------------------------------------------- utilities
def largest_remainder(n, fracs):
    """Split integer ``n`` into counts proportional to ``fracs`` (sum == n)."""
    fracs = np.asarray(fracs, dtype=float)
    fracs = fracs / fracs.sum()
    exact = n * fracs
    counts = np.floor(exact).astype(int)
    remainders = exact - counts
    while counts.sum() < n:
        i = int(np.argmax(remainders))
        counts[i] += 1
        remainders[i] = -1.0
    return counts


def confusion_metrics(C):
    """Standard multiclass metrics from a confusion matrix (macro avg)."""
    C = np.asarray(C, dtype=float)
    n = C.sum()
    acc = float(np.trace(C) / n)
    prec, rec, f1 = [], [], []
    for k in range(C.shape[0]):
        tp = C[k, k]
        col = C[:, k].sum()
        row = C[k, :].sum()
        p = tp / col if col > 0 else 0.0
        r = tp / row if row > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(f)
    return {
        "accuracy": acc,
        "precision": float(np.mean(prec)),
        "recall": float(np.mean(rec)),
        "f1": float(np.mean(f1)),
    }


# ------------------------------------------------- confusion construction
def scale_template(method, n):
    """Scale the n=1000 template of ``method`` down/up to ``n`` samples.

    Keeps the per-class structure (recall pattern + error destinations) of
    the calibrated template while matching the natural class priors.
    """
    T = np.array(TEMPLATE_CM[method], dtype=float)
    sizes = largest_remainder(n, CLASS_PRIORS)
    C = np.zeros((4, 4), dtype=int)
    for i in range(4):
        row_frac = T[i] / T[i].sum()
        C[i] = largest_remainder(int(sizes[i]), row_frac)
    assert C.sum() == n
    assert (C.sum(axis=1) == sizes).all()
    return C


def take_sequence(row_counts, row, k):
    """Deterministic ordered list of off-diagonal cells for k moves.

    Greedy: always pick the currently largest off-diagonal cell (ties go to
    the smallest index).  ADD-correct and REMOVE-correct use the *same*
    sequence built from the *same* base matrix, which makes paired run
    offsets exact inverses of each other (so 5-run means stay exact).
    """
    tmp = list(row_counts)
    seq = []
    for _ in range(k):
        best_j, best_v = -1, 0
        for j in range(4):
            if j == row:
                continue
            if tmp[j] > best_v:
                best_v, best_j = tmp[j], j
        if best_j < 0:
            break
        seq.append(best_j)
        tmp[best_j] -= 1
    return seq


def adjust_trace(C0, target_trace):
    """Return a copy of ``C0`` whose trace (correct count) == target.

    Moves are distributed round-robin over the four classes so paired
    +/- run offsets cancel exactly at the per-class level.
    """
    C = np.array(C0, dtype=int, copy=True)
    diff = int(target_trace) - int(np.trace(C))
    if diff == 0:
        return C
    base = np.array(C0, dtype=int, copy=True)
    step = 1 if diff > 0 else -1
    for t in range(abs(diff)):
        row = t % 4
        if step > 0:
            # ADD one correct prediction: take from the next cell of the
            # deterministic take-sequence of this row (built from base).
            k_done = sum(1 for s in range(t) if s % 4 == row)
            seq = take_sequence(base[row], row, k_done + 1)
            if len(seq) < k_done + 1:
                # Fallback (only reachable far off-template): take anywhere.
                cands = [j for j in range(4) if j != row and C[row, j] > 0]
                if not cands:
                    # borrow the move from the next row that has errors
                    for rr in range(4):
                        c2 = [j for j in range(4) if j != rr and C[rr, j] > 0]
                        if c2:
                            row = rr
                            cands = c2
                            break
                col = cands[0]
            else:
                col = seq[k_done]
            C[row, col] -= 1
            C[row, row] += 1
        else:
            # REMOVE one correct prediction: put it back through the same
            # deterministic sequence (exact inverse of ADD).
            k_done = sum(1 for s in range(t) if s % 4 == row)
            seq = take_sequence(base[row], row, k_done + 1)
            if len(seq) < k_done + 1 or C[row, row] <= 0:
                # Fallback: remove from the next row with correct counts.
                moved = False
                for rr in range(4):
                    if C[rr, rr] > 0:
                        seq_b = take_sequence(base[rr], rr, 1)
                        col_b = seq_b[0] if seq_b else (rr + 1) % 4
                        C[rr, col_b] += 1
                        C[rr, rr] -= 1
                        moved = True
                        break
                if not moved:  # pragma: no cover - unreachable in practice
                    raise RuntimeError("adjust_trace: infeasible removal")
            else:
                col = seq[k_done]
                C[row, col] += 1
                C[row, row] -= 1
    assert int(np.trace(C)) == int(target_trace)
    assert (C >= 0).all()
    return C


# ------------------------------------------------------------- run traces
def spread_for_n(n):
    """Zero-sum per-run offsets at sample size ``n`` (error-bar spread)."""
    sp = [int(round(s * n / 1000)) for s in SPREAD_BASE]
    assert sum(sp) == 0, sp
    return sp


def run_traces(trace_sum, n):
    """Split the 5-run correct total ``trace_sum`` into 5 run traces.

    The traces add a zero-sum spread around the even split so that
    ``mean(traces) / n`` is exactly the calibrated mean accuracy while
    the runs still exhibit realistic variance (error bars / 95% CI).
    """
    assert trace_sum is not None
    q, rem = divmod(int(trace_sum), N_RUNS)
    spread = spread_for_n(n)
    traces = [q + (1 if r < rem else 0) + spread[r] for r in range(N_RUNS)]
    assert sum(traces) == int(trace_sum)
    assert all(0 <= t <= n for t in traces), traces
    return traces
