"""Tabular Q-learning (Eq. 5), epsilon-greedy (Eq. 6), Algorithm 1 / Table 3 of the paper."""
from __future__ import annotations
import numpy as np
from .config import RLCfg


def train(env, cfg: RLCfg, rng: np.random.Generator):
    Q, hist = {}, []
    eps, best_avg, last_improve, stable, stop = cfg.eps, -np.inf, 0, 0, "max_episodes"
    returns = []
    zeros = lambda: np.full(env.max_actions, cfg.q_init, float)
    for ep in range(1, cfg.episodes + 1):
        s, G, max_dq, steps = env.reset(), 0.0, 0.0, 0
        delivered = False
        while True:
            q = Q.setdefault(s, zeros())
            nv = s[4]                                                  # number of valid actions (C_a)
            a = int(rng.integers(nv)) if rng.random() < eps else int(np.argmax(q[:nv]))
            s2, r, term, trunc, info = env.step(a)
            if term:
                target = r
            else:
                q2 = Q.setdefault(s2, zeros())
                target = r + cfg.gamma * q2[: s2[4]].max()             # Eq. (5)
            dq = cfg.eta * (target - q[a])
            q[a] += dq
            max_dq, G, steps = max(max_dq, abs(dq)), G + r, steps + 1
            if term:
                delivered = info["node"] == env.topo.dst
            if term or trunc:
                break
            s = s2
        returns.append(G)
        hist.append(dict(episode=ep, ret=G, steps=steps, delivered=int(delivered), eps=eps,
                         max_dq=max_dq, q_states=len(Q)))
        eps = max(cfg.eps_min, eps * cfg.eps_decay)
        stable = stable + 1 if max_dq < cfg.conv_tol else 0
        if cfg.use_conv_stop and stable >= cfg.conv_window:
            stop = "q_converged"
            break
        if ep >= cfg.early_stop_window:
            avg = float(np.mean(returns[-cfg.early_stop_window:]))
            if avg > best_avg + 1e-9:
                best_avg, last_improve = avg, ep
            elif cfg.use_early_stop and ep - last_improve >= cfg.early_stop_window:
                stop = "no_improvement"
                break
    return Q, hist, {"episodes_run": len(hist), "stop_reason": stop}


def rank_with_q(Q, env, cands, dst: int, force_dst_first: bool):
    """Final priority order per node: candidates sorted by learned Q(s_u, a) (stable w.r.t. fuzzy order)."""
    out = {}
    for u, cs in cands.items():
        ids, n = cs.ids.copy(), len(cs.ids)
        if n > 1 and u in env.base:
            q = Q.get(env.key(u, 0))
            if q is not None:
                ids = ids[np.argsort(-q[:n], kind="stable")]
        if force_dst_first and n and dst in ids:
            ids = np.concatenate([[dst], ids[ids != dst]])
        out[u] = ids
    return out
