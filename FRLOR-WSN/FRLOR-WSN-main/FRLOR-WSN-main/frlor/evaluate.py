"""Fast python-level check of a candidate assignment (NOT a substitute for NS-2).

Each transmission is a broadcast; every candidate receives independently with its link probability;
the highest-priority receiver forwards (perfect coordination). No receiver -> retransmission.
"""
from __future__ import annotations
import numpy as np
from .config import Config
from .topology import Topology


def simulate(topo: Topology, order: dict, cfg: Config, rng: np.random.Generator) -> dict:
    ev = cfg.evaluation
    tx = delivered = hops_sum = 0
    coord_delay = 0.0
    for _ in range(ev.n_packets):
        u, hops, ok, cd = topo.src, 0, True, 0.0
        while u != topo.dst:
            ids = order[u]
            if len(ids) == 0:
                ok = False
                break
            p = topo.plink[u, ids]
            for _try in range(ev.max_retry + 1):
                tx += 1
                rx = rng.random(len(ids)) < p
                if rx.any():
                    k = int(np.argmax(rx))
                    cd += k * ev.t_default                       # Eq. (7): T_i = (i-1)*T_default
                    u = int(ids[k])
                    break
            else:
                ok = False
                break
            hops += 1
            if hops > ev.max_hops:
                ok = False
                break
        if ok:
            delivered += 1
            hops_sum += hops
            coord_delay += cd
    d = max(delivered, 1)
    return dict(pdr=delivered / ev.n_packets, ent=tx / d, tx_total=tx, avg_hops=hops_sum / d,
                coord_delay_s=coord_delay / d)


def analytic_ent(topo: Topology, order: dict, max_iter: int = 500) -> float:
    """Expected Number of Transmissions from the source under opportunistic any-path forwarding
    (Zhong et al., ref. [15] of the paper; same recursion as EAX/ETX-based OR metrics):

        ENT(u) = ( 1 + sum_i ENT(c_i) * p_i * prod_{j<i}(1-p_j) ) / ( 1 - prod_j (1-p_j) ),   ENT(dst) = 0

    with candidates c_1..c_n in priority order and p_i = P_link(u, c_i). Deterministic (no sampling noise).
    """
    n = len(topo.xy)
    ent = np.full(n, np.inf)
    ent[topo.dst] = 0.0
    for _ in range(max_iter):
        changed = False
        for u in range(n):
            ids = order.get(u, [])
            if u == topo.dst or len(ids) == 0:
                continue
            p = topo.plink[u, ids]
            miss_before = np.concatenate([[1.0], np.cumprod(1 - p)[:-1]])
            w = p * miss_before                                  # P(candidate i is the highest-priority receiver)
            e = ent[ids]
            if np.any(np.isinf(e) & (w > 0)):
                continue
            val = (1.0 + np.sum(w * np.where(np.isinf(e), 0.0, e))) / (1.0 - np.prod(1 - p))
            if val < ent[u] - 1e-12 or np.isinf(ent[u]):
                ent[u], changed = val, True
        if not changed:
            break
    return float(ent[topo.src])
