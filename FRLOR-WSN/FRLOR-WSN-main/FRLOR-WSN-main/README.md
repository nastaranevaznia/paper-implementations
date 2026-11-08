# FRLOR – Fuzzy Reinforcement Learning-based Opportunistic Routing

Reproducibility package for the paper:

> **Fuzzy Reinforcement Learning in Opportunistic Routing for Wireless Sensor Networks**  
> Toktam Kamali Yazdi, Nastaran Evaznia, Sobhan Esmaeili, Javad Rezazadeh  
> *Future Research in AI & IoT*, 2026, 2(1), pp. 69–83  
> DOI: [10.22080/fraij.2026.30847.1041](https://doi.org/10.22080/fraij.2026.30847.1041)

## Overview

FRLOR is a hybrid opportunistic routing protocol for Wireless Sensor Networks that combines:

1. **Mamdani Fuzzy Inference System (FIS)** – evaluates candidate nodes using geographical distance, neighbour density and link delivery probability.
2. **Tabular Q-Learning** – refines the candidate priority list by learning from network feedback (Eqs. 3–6, Algorithm 1 of the paper).
3. **NS-2 packet-level simulation** – evaluates the final candidate sets under a realistic MAC and physical layer.

The overall pipeline follows Figure 1 of the paper:

```
Topology + Shadowing Channel
        ↓
Mamdani FIS (Candidate Selection Metric)
        ↓
Tabular Q-Learning (Priority Refinement)
        ↓
Candidate lists → NS-2 simulation → Metrics
```

## Repository Structure

| Path | Description | Paper Reference |
|------|-------------|-----------------|
| `frlor/config.py` | All configuration parameters and hyperparameters | Tables 1 & 3 |
| `frlor/channel.py` | Log-normal shadowing model and link probability | Eq. (2), Table 1 |
| `frlor/topology.py` | Random deployment (1200 × 1200 m) | Section 4 |
| `frlor/fuzzy.py` + `frlor/data/rule_base.csv` | Mamdani FIS (centroid defuzzification) + 27 rules | Sections 3.2–3.3, Table 2, Eq. (1) |
| `frlor/candidates.py` | FIS-based candidate set construction + POR baseline | Section 3.3 |
| `frlor/environment.py` | MDP definition (state, action, reward) | Section 3.4, Eqs. (3)–(4) |
| `frlor/qlearning.py` | Q-update, ε-greedy, convergence criteria | Eqs. (5)–(6), Algorithm 1 |
| `frlor/evaluate.py` | Analytic ENT and Monte-Carlo evaluation | Section 4 |
| `frlor/pipeline.py` | Single-run experiment | Section 4 |
| `frlor/experiments.py` | Parameter sweep (nodes, candidates, seeds) | Section 4 |
| `ns2/` | NS-2.34 agent (C++), Tcl script, installer | Section 4 |
| `frlor/analyze_ns2.py` | Metrics extraction from NS-2 logs | Section 4 |
| `tests/test_frlor.py` | Unit tests for FIS, channel model and reproducibility | – |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- (Optional) NS-2.34 for packet-level simulations

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run unit tests
```bash
python -m unittest discover -s tests -v
```

### 2. Single experiment
```bash
python -m frlor.pipeline \
    --out runs/n200_c3_s1 \
    --seed 1 \
    --nodes 200 \
    --max-cand 3
```

Results are written to `runs/n200_c3_s1/`:
- `topology.txt`
- `frlor/candidates.txt` (and corresponding files for other methods)
- `frlor/qtable.json`
- `frlor/train_log.csv`
- `run.json` (full configuration and metrics)

### 3. Full parameter sweep (as in Algorithm 1)
```bash
python -m frlor.experiments \
    --nodes 200 400 \
    --cands 2 3 4 5 \
    --seeds 100 \
    --out results.csv
```

### 4. Override any configuration parameter
```bash
python -m frlor.pipeline --set rl.eta=0.2 fuzzy.profile='"figure"' cand.max_candidates=4
```

## NS-2 Simulation

1. Install NS-2.34 (or use the provided helper):
   ```bash
   cd ns2
   ./install_ns2.sh          # patches and builds the FRLOR agent
   ```

2. Run a simulation:
   ```bash
   ./run_ns2.sh ../runs/n200_c3_s1/frlor/candidates.txt
   ```

3. Analyse the log:
   ```bash
   python -m frlor.analyze_ns2 ns2.log
   ```

The NS-2 agent implements timer-based candidate coordination according to Eq. (7) of the paper.

## Configuration Highlights

All important constants are centralised in `frlor/config.py`:

- **Fuzzy membership functions** – exactly as defined in Section 3.3 of the paper.
- **Q-Learning hyperparameters** – Table 3 (`η = 0.1`, `γ = 0.9`, `ε = 0.1`, decay = 0.99, 1000 episodes, …).
- **Reward weights** – `α = 0.5`, `β = 0.3`, `ω = 0.2` (Eq. 4).
- **Shadowing parameters** – Table 1 (`β = 2.7`, `σ = 6 dB`, …).

## Implementation Notes

A few details that are necessary for exact reproduction but were not fully specified in the paper (mainly due to space limitations) are clarified below:

- **Fuzzy rule base**: The complete set of 27 rules (3 × 3 × 3) is provided in `frlor/data/rule_base.csv`. The printed Table 2 in the paper contains a subset of the rules.
- **Neighbour definition**: Two nodes are considered neighbours if their link delivery probability is ≥ 0.2.
- **Source and destination**: Node 0 is the source and node 1 is the destination (randomly placed in the 1200 × 1200 m area).
- **Forward-progress filter**: Only neighbours that are geographically closer to the destination than the current node are eligible as candidates (can be disabled via configuration).
- **Destination priority**: If the destination itself is a neighbour, it is always placed first in the candidate list.

All of the above can be changed through the configuration system (`frlor/config.py` or the `--set` command-line flag).

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{kamali2026frlor,
  title   = {Fuzzy Reinforcement Learning in Opportunistic Routing for Wireless Sensor Networks},
  author  = {Kamali Yazdi, Toktam and Evaznia, Nastaran and Esmaeili, Sobhan and Rezazadeh, Javad},
  journal = {Future Research in AI and IoT},
  volume  = {2},
  number  = {1},
  pages   = {69--83},
  year    = {2026},
  doi     = {10.22080/fraij.2026.30847.1041}
}
```

## License

This code is released under the MIT License (see [LICENSE](LICENSE)).  
The original paper is distributed under the Creative Commons Attribution (CC-BY) license.
