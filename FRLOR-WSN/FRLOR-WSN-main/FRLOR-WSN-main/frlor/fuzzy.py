"""Mamdani fuzzy inference system of FRLOR (Sec. 3.2-3.3), implemented in plain NumPy.

Inputs : number of neighbours, distance to destination [m], link probability
Output : Candidate Selection Metric F_out in [0,1]
AND = min, implication = min (clipping), aggregation = max, defuzzification = centroid (Eq. 1).
"""
from __future__ import annotations
import csv, os
import numpy as np
from .config import FuzzyCfg

_trapz = getattr(np, "trapezoid", None) or np.trapz

PROFILES = {
    # exactly the triangles printed in the text of Sec. 3.3
    "text": {
        "nbr": {"small": (0, 0, 5), "average": (3, 7, 11), "large": (9, 20, 20)},
        "dist": {"low": (0, 0, 450), "average": (350, 750, 1000), "high": (900, 1200, 1200)},
        "link": {"weak": (0, 0, 0.4), "average": (0.3, 0.5, 0.7), "strong": (0.6, 1, 1)},
        "out": {"very_bad": (0, 0, 0.2), "bad": (0.15, 0.3, 0.45), "average": (0.4, 0.5, 0.6),
                "good": (0.55, 0.7, 0.85), "very_good": (0.8, 1, 1)},
    },
    # approximate reading of Fig. 4 (link) and Fig. 5 (output); the printed figures differ from the text
    "figure": {
        "nbr": {"small": (0, 0, 5), "average": (3, 7, 11), "large": (9, 20, 20)},
        "dist": {"low": (0, 0, 450), "average": (350, 750, 1000), "high": (900, 1200, 1200)},
        "link": {"weak": (0, 0.15, 0.3), "average": (0.25, 0.5, 0.75), "strong": (0.7, 0.85, 1.0)},
        "out": {"very_bad": (0, 0.1, 0.2), "bad": (0.15, 0.3, 0.45), "average": (0.4, 0.55, 0.7),
                "good": (0.65, 0.8, 0.9), "very_good": (0.85, 0.95, 1.0)},
    },
}
DEFAULT_RULES = os.path.join(os.path.dirname(__file__), "data", "rule_base.csv")


def trimf(x, a, b, c):
    """Triangular MF; a==b or b==c give shoulders (membership 1 at the shoulder point)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        m = (x >= a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x > b) & (x <= c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return y


def load_rules(path: str):
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rules = list(csv.DictReader(lines))
    return [(r["neighbors"].strip(), r["distance"].strip(), r["link"].strip(),
             r["output"].strip(), float(r.get("weight", 1.0) or 1.0)) for r in rules]


class MamdaniFIS:
    def __init__(self, cfg: FuzzyCfg):
        self.cfg = cfg
        self.mf = PROFILES[cfg.profile]
        self.rules = load_rules(cfg.rule_file or DEFAULT_RULES)
        self._validate()
        self.y = np.linspace(0.0, 1.0, cfg.n_points)
        self.out_mu = {k: trimf(self.y, *abc) for k, abc in self.mf["out"].items()}

    def _validate(self):
        keys = [r[:3] for r in self.rules]
        if len(set(keys)) != len(keys):
            raise ValueError("rule base has duplicate/conflicting antecedents")
        expected = len(self.mf["nbr"]) * len(self.mf["dist"]) * len(self.mf["link"])
        if len(keys) != expected:
            raise ValueError(f"rule base has {len(keys)} rules, expected {expected}")
        for n, d, l, o, _ in self.rules:
            assert n in self.mf["nbr"] and d in self.mf["dist"] and l in self.mf["link"] and o in self.mf["out"]

    def memberships(self, nbr, dist, link):
        c = self.cfg
        nbr = np.clip(np.atleast_1d(np.asarray(nbr, float)), 0.0, c.nn_max)
        dist = np.clip(np.atleast_1d(np.asarray(dist, float)), 0.0, c.dist_max)
        link = np.clip(np.atleast_1d(np.asarray(link, float)), 0.0, 1.0)
        return ({k: trimf(nbr, *v) for k, v in self.mf["nbr"].items()},
                {k: trimf(dist, *v) for k, v in self.mf["dist"].items()},
                {k: trimf(link, *v) for k, v in self.mf["link"].items()})

    def evaluate(self, nbr, dist, link) -> np.ndarray:
        """Vectorised: returns F_out (crisp centroid, Eq. 1) for every input triple."""
        mn, md, ml = self.memberships(nbr, dist, link)
        n = len(next(iter(mn.values())))
        # max over rules sharing a consequent == aggregating clipped consequents (min-max Mamdani)
        strength = {k: np.zeros(n) for k in self.mf["out"]}
        for rn, rd, rl, ro, w in self.rules:
            s = w * np.minimum(np.minimum(mn[rn], md[rd]), ml[rl])
            strength[ro] = np.maximum(strength[ro], s)
        agg = np.zeros((n, len(self.y)))
        for k, s in strength.items():
            agg = np.maximum(agg, np.minimum(s[:, None], self.out_mu[k][None, :]))
        num = _trapz(agg * self.y[None, :], self.y, axis=1)
        den = _trapz(agg, self.y, axis=1)
        return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
