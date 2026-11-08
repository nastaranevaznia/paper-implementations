"""End-to-end offline phase:  topology -> fuzzy candidates -> Q-learning -> export (+ python check).

    python -m frlor.pipeline --out runs/n200_c3_s1 --seed 1 --nodes 200 --max-cand 3
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from . import export
from .candidates import build_fuzzy_candidates, build_por_candidates
from .config import Config, apply_overrides
from .environment import RoutingEnv
from .evaluate import analytic_ent, simulate
from .fuzzy import MamdaniFIS
from .qlearning import rank_with_q, train
from .topology import build_topology

METHODS = ("frlor", "fuzzy_only", "por")


def run_once(cfg: Config, methods=METHODS, out_dir=None, verbose=False) -> dict:
    # three independent, deterministic RNG streams derived from the single seed
    s_topo, s_train, s_eval = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(cfg.seed).spawn(3)]
    topo = build_topology(cfg, s_topo)
    fis = MamdaniFIS(cfg.fuzzy)

    t0 = time.perf_counter()
    fz = build_fuzzy_candidates(topo, fis, cfg)
    t_fuzzy = time.perf_counter() - t0

    orders, times, extra = {}, {}, {}
    if "fuzzy_only" in methods:
        orders["fuzzy_only"], times["fuzzy_only"] = {u: c.ids for u, c in fz.items()}, t_fuzzy
    if "frlor" in methods:
        t0 = time.perf_counter()
        env = RoutingEnv(topo, fz, cfg, np.random.default_rng(s_train))
        Q, hist, info = train(env, cfg.rl, np.random.default_rng(s_train + 1))
        orders["frlor"] = rank_with_q(Q, env, fz, topo.dst, cfg.cand.force_destination_first)
        times["frlor"] = t_fuzzy + (time.perf_counter() - t0)
        extra["frlor"] = dict(info, q_states=len(Q), reordered_nodes=int(sum(
            not np.array_equal(orders["frlor"][u], fz[u].ids) for u in fz)))
    if "por" in methods:
        t0 = time.perf_counter()
        por = build_por_candidates(topo, cfg)
        orders["por"], times["por"] = {u: c.ids for u, c in por.items()}, time.perf_counter() - t0

    results = {}
    for m, order in orders.items():
        res = simulate(topo, order, cfg, np.random.default_rng(s_eval))
        res.update(ent_analytic=analytic_ent(topo, order), exec_time_s=times[m], **extra.get(m, {}))
        results[m] = res
        if verbose:
            print(f"{m:11s} ENT_formula={res['ent_analytic']:.3f} ENT_sim={res['ent']:.3f} PDR={res['pdr']:.3f} hops={res['avg_hops']:.2f} "
                  f"t={res['exec_time_s']:.2f}s " + (f"[{extra[m]}]" if m in extra else ""))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        export.write_topology(os.path.join(out_dir, "topology.txt"), topo)
        files = {}
        for m, order in orders.items():
            md = os.path.join(out_dir, m)
            os.makedirs(md, exist_ok=True)
            export.write_candidates(os.path.join(md, "candidates.txt"), order)
            files[m] = export.sha256(os.path.join(md, "candidates.txt"))
        if "frlor" in methods:
            export.write_train_log(os.path.join(out_dir, "frlor", "train_log.csv"), hist)
            export.write_qtable(os.path.join(out_dir, "frlor", "qtable.json"), Q)
        export.dump_json(os.path.join(out_dir, "run.json"), dict(
            config=cfg.to_dict(), rng_seeds=dict(topology=s_topo, train=s_train, eval=s_eval),
            topology=dict(src=topo.src, dst=topo.dst, resample_attempts=topo.attempts,
                          mean_neighbours=float(np.mean([len(x) for x in topo.nbr]))),
            results=results, candidate_file_sha256=files, environment=export.environment_info()))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--max-cand", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="e.g. rl.eta=0.2 fuzzy.profile=figure")
    a = ap.parse_args(argv)
    cfg = Config(seed=a.seed)
    if a.nodes:
        cfg.topology.n_nodes = a.nodes
    if a.max_cand:
        cfg.cand.max_candidates = a.max_cand
    apply_overrides(cfg, a.set)
    run_once(cfg, a.methods, a.out, verbose=True)
    print("written to", a.out)


if __name__ == "__main__":
    main()
