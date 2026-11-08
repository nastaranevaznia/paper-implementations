"""Compute ENT, PDR, E2E delay, hops and energy from the log written by Agent/FRLOR.

    python -m frlor.analyze_ns2 runs/x/frlor/ns2.log --initial-energy 100
Log lines:  SRC t node seq | TX t node src seq retry hops | RX_DEST t src seq delay hops |
            DROP t node src seq reason | ENERGY node remaining_J
"""
from __future__ import annotations
import argparse, json


def analyze(path, initial_energy=None):
    src, tx, rx, drops, energy = set(), 0, {}, 0, {}
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "SRC":
            src.add((int(t[2]), int(t[3])))
        elif t[0] == "TX":
            tx += 1
        elif t[0] == "RX_DEST":
            rx.setdefault((int(t[2]), int(t[3])), (float(t[4]), int(t[5])))   # first copy only
        elif t[0] == "DROP":
            drops += 1
        elif t[0] == "ENERGY":
            energy[int(t[1])] = float(t[2])
    n_rx = len(rx)
    out = dict(originated=len(src), delivered=n_rx, pdr=n_rx / max(len(src), 1),
               data_transmissions=tx, ent=tx / max(n_rx, 1),
               avg_e2e_delay_ms=1e3 * sum(d for d, _ in rx.values()) / max(n_rx, 1),
               avg_hops=sum(h for _, h in rx.values()) / max(n_rx, 1), drop_events=drops)
    if energy and initial_energy is not None:
        used = sum(initial_energy - e for e in energy.values() if e >= 0)
        out.update(energy_consumed_J=used, energy_per_delivered_pkt_J=used / max(n_rx, 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--initial-energy", type=float, default=None)
    a = ap.parse_args()
    print(json.dumps(analyze(a.log, a.initial_energy), indent=2))
