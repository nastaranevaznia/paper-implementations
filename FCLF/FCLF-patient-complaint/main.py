"""FCLF paper implementation + reproduction of its figures and tables.

Run:
    python main.py

Outputs (in ./results):
    fig_learning_curves.png   - 2x2 average charts with error bars (95% CI)
    fig_correlation_heatmap.png - correlation heatmap
    fig_backbone_comparison.png - group-bar chart of the 4 LLM backbones
    runs.csv / means.csv / table_n1000.csv / backbones.csv / correlations.csv
"""

import os

import numpy as np
import pandas as pd

from fclf import config as C
from fclf import experiment as E
from fclf import plots as P
from fclf import pipeline_demo as D

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 70)
    print("  FCLF: Fog-Cloud LLM Framework — paper implementation & reproduction")
    print("=" * 70)

    # ------------------------------------------------ functional pipeline demo
    print("\n[1/4] Functional demo of the Fog -> Cloud pipeline (synthetic data)...")
    demo = D.run()
    print(f"  train={demo['n_train']}  test={demo['n_test']}  "
          f"fog-classifier accuracy on template data: {demo['demo_accuracy']:.1%}")
    print("  Cloud frequency analysis (Eq.7):")
    for name, v in demo["freq"].items():
        print(f"    - {name:28s}: {v['count']:4d}  ({v['percent']:.1f}%)")
    print(f"  Alert demo (Eq.8): surge week counts={demo['surge_week']}")
    for a in demo["alerts"]:
        print(f"    >>> ALERT: '{a['category']}' count={a['count']} "
              f"> threshold={a['threshold']}")
    print("  Server selection (Eq.6, alpha=0.6):")
    for t in demo["server_table"]:
        fit = f"{t['fitness']:.4f}" if t["fitness"] is not None else "infeasible"
        print(f"    - {t['node']:14s} feasible={t['feasible']}  fitness={fit}")
    print(f"    ==> selected: {demo['best_server']} (lowest fitness)")

    # ------------------------------------------------- paper experiments
    print("\n[2/4] Running paper experiments (3 methods x 8 sizes x 5 runs)...")
    runs_df, means_df = E.run_learning_curves()
    backbone_df = E.run_backbones()
    corr_df = E.correlation_matrix(means_df)
    runs_df.to_csv(os.path.join(RESULTS, "runs.csv"), index=False)
    means_df.to_csv(os.path.join(RESULTS, "means.csv"), index=False)
    backbone_df.to_csv(os.path.join(RESULTS, "backbones.csv"), index=False)
    corr_df.to_csv(os.path.join(RESULTS, "correlations.csv"))

    # ------------------------------------------------------------- table
    print("\n[3/4] Comparison table at 1000 patients (mean over 5 runs):")
    tab = means_df[means_df.n == 1000].set_index("method")
    table_rows = []
    header = f"  {'Metric':10s}" + "".join(f"{C.METHOD_DISPLAY[m]:>18s}" for m in C.METHODS)
    print(header)
    for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                          ("recall", "Recall"), ("f1", "F1-Score")]:
        vals = {m: float(tab.loc[m, f"{metric}_mean"]) for m in C.METHODS}
        table_rows.append({"Metric": label, **{m: vals[m] for m in C.METHODS}})
        print(f"  {label:10s}" + "".join(f"{vals[m]:17.1%}" for m in C.METHODS))
    pd.DataFrame(table_rows).to_csv(os.path.join(RESULTS, "table_n1000.csv"), index=False)

    print("\n  Backbone comparison at 1000 patients:")
    btab = backbone_df.set_index("backbone")
    print(f"  {'Model':14s}{'Accuracy':>10s}{'Precision':>11s}{'Recall':>9s}{'F1-Score':>10s}")
    for b in C.BACKBONES:
        print(f"  {b:14s}{btab.loc[b,'accuracy']:10.1%}{btab.loc[b,'precision']:11.1%}"
              f"{btab.loc[b,'recall']:9.1%}{btab.loc[b,'f1']:10.1%}")

    print("\n  Correlation matrix (Patients vs. mean accuracy):")
    print(corr_df.round(3).to_string())

    # ------------------------------------------------------------- figures
    print("\n[4/4] Saving figures...")
    p1 = P.plot_learning_curves(means_df, os.path.join(RESULTS, "fig_learning_curves.png"))
    p2 = P.plot_correlation_heatmap(corr_df, os.path.join(RESULTS, "fig_correlation_heatmap.png"))
    p3 = P.plot_backbone_comparison(backbone_df, os.path.join(RESULTS, "fig_backbone_comparison.png"))
    for p in (p1, p2, p3):
        print(f"  saved: {p}")

    # ---------------------------------------------------------- verification
    print("\n[verify] Checking reproduced values against reported targets...")
    ok = True
    for m, (a, p, r, f) in C.TABLE_TARGET.items():
        got = (float(tab.loc[m, "accuracy_mean"]), float(tab.loc[m, "precision_mean"]),
               float(tab.loc[m, "recall_mean"]), float(tab.loc[m, "f1_mean"]))
        for g, t, name in zip(got, (a, p, r, f), ("acc", "prec", "rec", "f1")):
            if abs(g - t) > 5e-4:
                ok = False
                print(f"  MISMATCH {m}.{name}: got {g:.4f}, want {t:.4f}")
    key_map = {("Patients", "SVM"): ("Patients", "SVM"),
               ("Patients", "Ensemble"): ("Patients", "Ensemble"),
               ("Patients", "FCLF"): ("Patients", "FCLF"),
               ("SVM", "Ensemble"): ("SVM", "Ensemble"),
               ("SVM", "FCLF"): ("SVM", "FCLF"),
               ("Ensemble", "FCLF"): ("Ensemble", "FCLF")}
    for k, want in C.HEAT_TARGET.items():
        got = round(float(corr_df.loc[key_map[k]]), 3)
        if got != want:
            ok = False
            print(f"  MISMATCH heatmap{k}: got {got}, want {want}")
    print("  ALL CHECKS PASSED ✔" if ok else "  CHECKS FAILED ✘")
    print("\nDone. See ./results for CSVs and figures.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
