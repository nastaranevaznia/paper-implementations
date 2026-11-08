"""Random topology (Sec. 4): N nodes uniform in a 1200 x 1200 m area.
Node 0 = source, node 1 = destination (same ids are used in NS-2)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .channel import link_probability
from .config import Config


@dataclass
class Topology:
    xy: np.ndarray        # (N,2)
    src: int
    dst: int
    dist: np.ndarray      # (N,N) pairwise distance
    d_dst: np.ndarray     # (N,)  distance to destination
    plink: np.ndarray     # (N,N) link delivery probability, diagonal 0
    nbr: list             # nbr[i] = np.ndarray of neighbour ids
    attempts: int = 1


def _connected(adj: np.ndarray, a: int, b: int) -> bool:
    seen = np.zeros(len(adj), bool)
    seen[a] = True
    stack = [a]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u] & ~seen):
            seen[v] = True
            stack.append(v)
    return bool(seen[b])


def build_topology(cfg: Config, seed: int) -> Topology:
    t = cfg.topology
    rng = np.random.default_rng(seed)
    for attempt in range(1, t.max_resample + 1):
        xy = rng.uniform([0.0, 0.0], [t.width, t.height], size=(t.n_nodes, 2))
        xy[0], xy[1] = t.src_xy, t.dst_xy
        dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        p = link_probability(dist, cfg.channel)
        np.fill_diagonal(p, 0.0)
        adj = p >= t.neighbor_min_prob
        if _connected(adj, 0, 1):
            nbr = [np.flatnonzero(adj[i]) for i in range(t.n_nodes)]
            return Topology(xy, 0, 1, dist, dist[:, 1].copy(), p, nbr, attempt)
    raise RuntimeError("could not generate a connected topology; increase n_nodes or lower neighbor_min_prob")
