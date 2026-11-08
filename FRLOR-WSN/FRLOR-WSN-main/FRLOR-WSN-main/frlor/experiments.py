"""Parameter sweep mirroring Sec. 4 (N in {200,400}, candidates 2..5, many seeds).

    python -m frlor.experiments --nodes 200 400 --cands 2 3 4 5 --seeds 20 --out results.csv
(the paper's Algorithm 1 uses 100 seeds: --seeds 100)
"""
from __future__ import annotations
import argparse, csv, itertools
import numpy as np
from .config import Config, apply_overrides
from .pipeline import METHODS, run_once

KEYS = ("ent_analytic", "ent", "pdr", "avg_hops", "exec_time_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--cands", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    rows = []
    for n, k, seed in itertools.product(a.nodes, a.cands, range(1, a.seeds + 1)):
        cfg = apply_overrides(Config(seed=seed), a.set)
        cfg.topology.n_nodes, cfg.cand.max_candidates = n, k
        for m, r in run_once(cfg, METHODS).items():
            rows.append(dict(nodes=n, max_cand=k, seed=seed, method=m, **{q: r[q] for q in KEYS}))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'nodes':>5} {'cand':>4} {'method':>10} " + " ".join(f"{k:>14}" for k in KEYS))
    for n, k, m in itertools.product(a.nodes, a.cands, METHODS):
        sel = [r for r in rows if (r["nodes"], r["max_cand"], r["method"]) == (n, k, m)]
        print(f"{n:5d} {k:4d} {m:>10} " + " ".join(
            f"{np.mean([r[q] for r in sel]):8.3f}±{np.std([r[q] for r in sel]):5.3f}" for q in KEYS))


if __name__ == "__main__":
    main()
