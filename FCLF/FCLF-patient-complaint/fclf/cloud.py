"""Cloud layer: aggregation + statistics (paper Sec. 3.4, Eq. 7-8)."""

import numpy as np


# ------------------------------------------------------------------- Eq. 7
def eq7_frequency(records, n_categories=4):
    """F_k = n_k / N * 100 : complaint share per category."""
    counts = np.zeros(n_categories, dtype=int)
    for r in records:
        counts[r["label"]] += 1
    total = counts.sum()
    pct = (counts / total * 100.0) if total else np.zeros(n_categories)
    return {k: {"count": int(counts[k]), "percent": float(pct[k])}
            for k in range(n_categories)}


# ------------------------------------------------------------------- Eq. 8
class AlertSystem:
    """3-sigma automated alert system over weekly complaint counts.

    Alert_k(t) = 1 if y_{k,t} > mu_k + 3*sigma_k else 0,
    where mu/sigma come from the historical weeks of category k.
    """

    def __init__(self, n_categories=4, n_sigma=3.0, min_history=4):
        self.n_categories = n_categories
        self.n_sigma = n_sigma
        self.min_history = min_history
        self.history = {k: [] for k in range(n_categories)}

    def update(self, week_counts):
        """Feed one week (dict/cat-array of counts); return list of alerts."""
        alerts = []
        for k in range(self.n_categories):
            y = float(week_counts[k])
            hist = self.history[k]
            if len(hist) >= self.min_history:
                mu = float(np.mean(hist))
                sigma = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
                threshold = mu + self.n_sigma * sigma
                if sigma == 0:
                    fired = bool(y > mu)
                else:
                    fired = bool(y > threshold)
                if fired:
                    alerts.append({"category": k, "count": y,
                                   "mean": mu, "std": sigma,
                                   "threshold": threshold})
            hist.append(y)
        return alerts
