"""Files consumed by NS-2 and by the analysis scripts."""
from __future__ import annotations
import hashlib, json, os, platform, sys
import numpy as np


def write_topology(path, topo):
    with open(path, "w") as f:
        f.write(f"# id x y   (node {topo.src} = source, node {topo.dst} = destination)\n")
        for i, (x, y) in enumerate(topo.xy):
            f.write(f"{i} {x:.3f} {y:.3f}\n")


def write_candidates(path, order: dict):
    """One line per node:  <node> <cand1> <cand2> ...   (priority order, highest first)."""
    with open(path, "w") as f:
        f.write("# node  candidates(priority order)\n")
        for u in sorted(order):
            f.write(f"{u} " + " ".join(str(int(c)) for c in order[u]) + "\n")


def write_train_log(path, hist):
    with open(path, "w") as f:
        f.write("episode,return,steps,delivered,epsilon,max_dq,q_states\n")
        for h in hist:
            f.write(f"{h['episode']},{h['ret']:.6f},{h['steps']},{h['delivered']},{h['eps']:.6f},"
                    f"{h['max_dq']:.6f},{h['q_states']}\n")


def write_qtable(path, Q):
    with open(path, "w") as f:
        json.dump({",".join(map(str, k)): [float(x) for x in v] for k, v in sorted(Q.items())}, f, indent=1)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def environment_info():
    return dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform())


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
