"""Runs the paper's experiments (5 runs, sample sizes 50..1000)."""

import itertools

import numpy as np
import pandas as pd

from .config import (
    BACKBONES,
    MEAN_TRACE_SUM,
    METHODS,
    N_RUNS,
    SAMPLE_SIZES,
    T_CRIT_95_DF4,
    TEMPLATE_CM,
)
from .reproduce import (
    adjust_trace,
    confusion_metrics,
    run_traces,
    scale_template,
)

METRICS = ["accuracy", "precision", "recall", "f1"]


def run_learning_curves():
    """Evaluate SVM / Ensemble / FCLF across sample sizes (5 runs each)."""
    rows = []
    for method, n in itertools.product(METHODS, SAMPLE_SIZES):
        ni = SAMPLE_SIZES.index(n)
        base = scale_template(method, n)
        traces = run_traces(MEAN_TRACE_SUM[method][ni], n)
        for r in range(N_RUNS):
            C = adjust_trace(base, traces[r])
            m = confusion_metrics(C)
            rows.append({"method": method, "n": n, "run": r, **m,
                         "trace": int(np.trace(C))})
    runs_df = pd.DataFrame(rows)

    agg = runs_df.groupby(["method", "n"])[METRICS].agg(["mean", "std"]).reset_index()
    agg.columns = ["method", "n"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    for m in METRICS:
        agg[f"{m}_ci"] = T_CRIT_95_DF4 * agg[f"{m}_std"] / np.sqrt(N_RUNS)
    return runs_df, agg


def run_backbones():
    """Backbone comparison at 1000 patients (Fig. 2 of the paper)."""
    rows = []
    for b in BACKBONES:
        m = confusion_metrics(np.array(TEMPLATE_CM[b]))
        rows.append({"backbone": b, **m})
    return pd.DataFrame(rows)


def correlation_matrix(means_df):
    """Pearson correlation between sample size and mean accuracy per method."""
    acc = means_df.pivot(index="n", columns="method", values="accuracy_mean")
    acc = acc.reindex(SAMPLE_SIZES)
    data = {"Patients": np.array(SAMPLE_SIZES, dtype=float)}
    for m in METHODS:
        data[m] = acc[m].to_numpy(dtype=float)
    order = ["Patients"] + METHODS
    corr = np.corrcoef([data[k] for k in order])
    return pd.DataFrame(corr, index=order, columns=order)
