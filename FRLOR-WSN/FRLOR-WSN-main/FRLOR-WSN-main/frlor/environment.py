"""MDP of Sec. 3.4 (Steps 0-3).

Episode  = one packet travelling source -> destination, at most `episode_len` transmissions.
State    = discretised Eq. (3) feature vector of the CURRENT node
           s = (F_out, D_g, N_n, L_p, T_e, C_a)   with
           F_out : mean fuzzy score of the node's candidate set
           D_g   : mean distance candidate->destination / D
           N_n   : mean #neighbours of the candidates / N_max
           L_p   : mean link probability node->candidates
           T_e   : transmissions already spent on the current hop (0, 1, >=2)      [ASSUMPTION]
           C_a   : candidate capacity = number of candidates
Action   = index a in {0..n-1} of the candidate (rank in the FIS list) to be used as next hop.
Transition T(s,a,s'): the transmission succeeds with the shadowing link probability; on success the
           packet moves to the candidate, on failure the node stays and T_e increases.
Reward   = Eq. (4) evaluated for the chosen candidate when the hop succeeds (0 on a failed attempt):
           r = w_dist*(1 - D_g/D) + w_nbr*(N_n/N_max) + w_etx*(1 - T_e/T_min),  T_min = 1,
           (1 - T_e/T_min) clipped to [-1, 0]; D_g, N_n are those of the chosen candidate.  [ASSUMPTION]
"""
from __future__ import annotations
import numpy as np
from .candidates import CandidateSet
from .config import Config
from .topology import Topology

_EDGES = dict(f=[0.2, 0.4, 0.6, 0.8], d=[0.25, 0.5, 0.75], n=[1 / 3, 2 / 3], l=[0.4, 0.7])


class RoutingEnv:
    def __init__(self, topo: Topology, cands: dict, cfg: Config, rng: np.random.Generator):
        self.topo, self.cands, self.cfg, self.rng = topo, cands, cfg, rng
        self.D, self.Nmax = cfg.fuzzy.dist_max, cfg.fuzzy.nn_max
        self.max_actions = cfg.cand.max_candidates
        self.base = {u: self._base_state(cs) for u, cs in cands.items() if len(cs.ids)}
        self.reset()

    def _base_state(self, cs: CandidateSet):
        b = lambda v, e: int(np.digitize(v, e))
        return (b(cs.fout.mean(), _EDGES["f"]),
                b(np.clip(cs.dist.mean() / self.D, 0, 1), _EDGES["d"]),
                b(np.clip(cs.nn.mean() / self.Nmax, 0, 1), _EDGES["n"]),
                b(cs.p.mean(), _EDGES["l"]),
                len(cs.ids))                                  # C_a

    def key(self, u: int, hop_tx: int):
        return self.base[u][:4] + (self.base[u][4], min(hop_tx, 2))   # (..., C_a, T_e)

    def reset(self):
        self.u, self.hop_tx, self.t = self.topo.src, 0, 0
        return self.key(self.u, 0)

    def step(self, a: int):
        r_cfg, cs = self.cfg.rl, self.cands[self.u]
        c = int(cs.ids[a])
        self.t += 1
        self.hop_tx += 1
        ok = self.rng.random() < cs.p[a]
        reward, terminal = 0.0, False
        if ok:
            d = np.clip(cs.dist[a] / self.D, 0, 1)
            n = np.clip(cs.nn[a] / self.Nmax, 0, 1)
            e = np.clip(1.0 - self.hop_tx / 1.0, -1.0, 0.0)           # T_min = 1
            reward = r_cfg.w_dist * (1 - d) + r_cfg.w_nbr * n + r_cfg.w_etx * e
            self.u, self.hop_tx = c, 0
            if c == self.topo.dst:
                reward += r_cfg.goal_bonus
                terminal = True
            elif self.u not in self.base:                          # dead end (no candidates)
                reward += r_cfg.dead_end_penalty
                terminal = True
        truncated = (not terminal) and self.t >= r_cfg.episode_len
        nxt = None if terminal else self.key(self.u, self.hop_tx)
        return nxt, float(reward), terminal, truncated, {"success": ok, "node": self.u}
