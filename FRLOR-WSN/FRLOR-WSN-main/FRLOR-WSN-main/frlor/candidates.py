"""Initial candidate set construction (Sec. 3.3) and the POR baseline.

For every node u the FIS scores each eligible neighbour c with
    inputs: |N(c)| (number of neighbours of c), d(c, dst), P_link(u, c)  ->  F_out(u, c)
and the Max best-scored neighbours form u's priority-ordered candidate list.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import Config
from .fuzzy import MamdaniFIS
from .topology import Topology


@dataclass
class CandidateSet:
    ids: np.ndarray       # priority order (index 0 = highest priority)
    fout: np.ndarray
    dist: np.ndarray      # distance of each candidate to the destination [m]
    nn: np.ndarray        # number of neighbours of each candidate
    p: np.ndarray         # P_link(u, candidate)
    void: bool = False    # True if no forward-progress neighbour existed (fallback used)


def _empty() -> CandidateSet:
    z = np.zeros(0)
    return CandidateSet(np.zeros(0, int), z, z, z, z)


def _eligible(topo: Topology, u: int, cfg: Config):
    nb = topo.nbr[u]
    if cfg.cand.require_progress:
        prog = nb[topo.d_dst[nb] < topo.d_dst[u]]
        if len(prog):
            return prog, False
        # routing void: fall back to the neighbours nearest to the destination
        return nb[np.argsort(topo.d_dst[nb], kind="stable")][: cfg.cand.max_candidates], True
    return nb, False


def _order(ids, fout, dist, topo, cfg, key_primary):
    order = np.lexsort((ids, dist, key_primary))            # primary key last; ties -> closer, then lower id
    if cfg.cand.force_destination_first and topo.dst in ids[order]:
        order = np.concatenate([[int(np.flatnonzero(ids == topo.dst)[0])], order[ids[order] != topo.dst]])
    return order[: cfg.cand.max_candidates]


def build_fuzzy_candidates(topo: Topology, fis: MamdaniFIS, cfg: Config) -> dict:
    nn_all = np.array([len(x) for x in topo.nbr], float)
    out = {}
    for u in range(len(topo.xy)):
        if u == topo.dst or len(topo.nbr[u]) == 0:
            out[u] = _empty()
            continue
        ids, void = _eligible(topo, u, cfg)
        p = topo.plink[u, ids]
        fout = fis.evaluate(nn_all[ids], topo.d_dst[ids], p)
        o = _order(ids, fout, topo.d_dst[ids], topo, cfg, -fout)
        out[u] = CandidateSet(ids[o], fout[o], topo.d_dst[ids][o], nn_all[ids][o], p[o], void)
    return out


def build_por_candidates(topo: Topology, cfg: Config) -> dict:
    """POR baseline: purely geographic - neighbours with the smallest distance to the destination."""
    nn_all = np.array([len(x) for x in topo.nbr], float)
    out = {}
    for u in range(len(topo.xy)):
        if u == topo.dst or len(topo.nbr[u]) == 0:
            out[u] = _empty()
            continue
        ids, void = _eligible(topo, u, cfg)
        d = topo.d_dst[ids]
        o = _order(ids, d, d, topo, cfg, d)
        out[u] = CandidateSet(ids[o], np.zeros(len(o)), d[o], nn_all[ids][o], topo.plink[u, ids][o], void)
    return out
