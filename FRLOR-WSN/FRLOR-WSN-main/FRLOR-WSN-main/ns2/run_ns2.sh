#!/bin/sh
# Offline phase (python) + NS-2 runs for all three methods + analysis.
#   sh ns2/run_ns2.sh <runs_dir> <seed> <n_nodes> <max_cand> [n_packets] [interval_s]
set -e
OUT="$1"; SEED="$2"; NN="$3"; K="$4"; NP="${5:-200}"; IV="${6:-1.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS="${NS:-ns}"                         # override with NS=/path/to/ns-2.34/ns
cd "$ROOT"
python3 -m frlor.pipeline --out "$OUT" --seed "$SEED" --nodes "$NN" --max-cand "$K"
for M in frlor fuzzy_only por; do
  "$NS" ns2/sim/frlor_sim.tcl "$OUT/topology.txt" "$OUT/$M/candidates.txt" "$OUT/$M/ns2.log" \
        "$NN" 0 1 "$NP" "$IV" "$SEED" 100.0
  python3 -m frlor.analyze_ns2 "$OUT/$M/ns2.log" --initial-energy 100 | tee "$OUT/$M/ns2_metrics.json"
done
