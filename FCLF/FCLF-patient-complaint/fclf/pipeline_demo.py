"""End-to-end functional demo of the FCLF pipeline on synthetic data.

1. Generate a synthetic complaint dataset (texts + metadata).
2. Train the fog classifier; each center's FogNode classifies locally and
   forwards only (label + metadata) to the cloud (privacy preserved).
3. Cloud aggregates labels (Eq. 7) and runs the 3-sigma alert demo (Eq. 8).
4. Server-selection demo with the dual-objective fitness (Eq. 6).

NOTE: this demo shows the *mechanics* of the framework on template data.
The paper's reported numbers are reproduced by the calibrated evaluation in
``experiment.py`` (see README for why the two tracks exist).
"""

import numpy as np

from .cloud import AlertSystem, eq7_frequency
from .config import CATEGORIES
from .data import generate_dataset
from .fog import FogClassifier, FogNode
from .servers import Node, Task, select_server


def run(seed=7):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------- 1. data + training
    train_df = generate_dataset(1500, seed=seed, noise=0.15)
    test_df = generate_dataset(600, seed=seed + 1, noise=0.15)
    clf = FogClassifier().train(train_df["text"].tolist(),
                                train_df["true_label"].tolist())
    nodes = {c: FogNode(center_id=c, classifier=clf)
             for c in sorted(test_df["center_id"].unique())}

    # -------------------------------------- 2. fog: local classification
    cloud_records, y_true, y_pred = [], [], []
    for _, row in test_df.iterrows():
        rec, _ = nodes[row["center_id"]].process(
            row["text"], week=int(row["week"]), department=row["department"])
        cloud_records.append(rec)
        y_true.append(int(row["true_label"]))
        y_pred.append(int(rec["label"]))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    demo_acc = float((y_true == y_pred).mean())

    # ------------------------------------------- 3a. cloud: Eq. 7 frequency
    freq = eq7_frequency(cloud_records)

    # ------------------------------------------- 3b. cloud: Eq. 8 alert demo
    alert_sys = AlertSystem()
    weekly = rng.poisson(20, size=(12, 4))           # 12 normal weeks
    for w in range(12):
        alert_sys.update(weekly[w])
    surge = np.array([22, 21, 68, 19])               # sudden Management surge
    alerts = alert_sys.update(surge)

    # ------------------------------------------------- 4. server selection
    nodes_srv = [
        Node("fog-center-1", cpu=16, ram=64, storage=2000, cpu_used=12.0, energy=220),
        Node("fog-center-2", cpu=16, ram=64, storage=2000, cpu_used=4.0, energy=150),
        Node("fog-center-3", cpu=8, ram=32, storage=1000, cpu_used=6.5, energy=180),
        Node("cloud-main", cpu=64, ram=256, storage=20000, cpu_used=20.0, energy=900),
    ]
    task = Task("llm-batch-1000", d_cpu=4.0, d_ram=8.0, d_storage=50.0)
    best, table = select_server(nodes_srv, task, alpha=0.6)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "demo_accuracy": demo_acc,
        "freq": {CATEGORIES[k]: v for k, v in freq.items()},
        "surge_week": surge.tolist(),
        "alerts": [{"category": CATEGORIES[a["category"]],
                    "count": a["count"],
                    "threshold": round(a["threshold"], 2)} for a in alerts],
        "server_table": table,
        "best_server": best.name if best else None,
    }
