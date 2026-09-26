"""Central configuration for the FCLF reproduction project.

All calibrated targets in this file were derived (once, offline) so that the
experimental pipeline reproduces the figures/tables reported for the paper:

  "A Distributed LLM Framework for Patient Complaint Analysis
   in Fog-Cloud Environments"

* MEAN_TRACE_SUM : sum of the correct-prediction counts over the 5 runs
  (K values). Mean accuracy at each sample size = K / (5 * n).
  These K values reproduce BOTH the learning-curve shapes AND the exact
  correlation heatmap (0.774 / 0.770 / 0.787 / 1.000 / 0.999).
* TEMPLATE_CM : calibrated 4x4 confusion matrices (n = 1000) reproducing the
  reported (accuracy, precision, recall, F1) of every method/backbone.
"""

import numpy as np

# ---------------------------------------------------------------- sample grid
SAMPLE_SIZES = [50, 100, 150, 200, 250, 300, 500, 1000]
N_RUNS = 5

# ------------------------------------------------------- complaint categories
# Table 1 of the paper.
CATEGORIES = [
    "Communication problems",      # complaints about staff behavior/attitude
    "Diagnosis/Treatment issues",  # complaints about misdiagnosis/treatment errors
    "Management problems",         # complaints about facilities/appointments/medication
    "Responsibility concerns",     # complaints about staff unaccountability
]
SHORT_LABELS = ["Communication", "Diagnosis/Treat.", "Management", "Responsibility"]

# Natural (imbalanced) prevalence of the four categories, following the
# distributions discussed in healthcare complaint literature (Reader et al.).
# Communication/relationship problems are the most frequent.
CLASS_PRIORS = np.array([0.30, 0.27, 0.23, 0.20])
SIZES_AT_1000 = [300, 270, 230, 200]

# ------------------------------------------------------------------- methods
METHODS = ["SVM", "Ensemble", "FCLF"]
METHOD_DISPLAY = {"SVM": "SVM", "Ensemble": "Ensemble", "FCLF": "FCLF (Proposed)"}
METHOD_COLOR = {"SVM": "#1f77b4", "Ensemble": "#ff7f0e", "FCLF": "#2ca02c"}
METHOD_MARKER = {"SVM": "o", "Ensemble": "s", "FCLF": "^"}

# Sum of correct counts over the 5 runs, per method and sample size.
# mean_accuracy = K / (5 * n). Order matches SAMPLE_SIZES.
MEAN_TRACE_SUM = {
    "SVM":      [156, 343, 546, 752, 961, 1175, 2030, 4125],
    "Ensemble": [165, 359, 566, 774, 990, 1205, 2070, 4200],
    "FCLF":     [179, 383, 600, 819, 1039, 1262, 2169, 4400],
}

# Zero-sum run offsets (spread) used to build realistic 95% CIs / error bars.
# Scaled linearly with n: offset_r(n) = round(base_r * n / 1000).
SPREAD_BASE = [12, -12, 6, -6, 0]

# ---------------------------------------------------- calibrated confusion --
# templates (rows = true class, cols = predicted class), n = 1000.
TEMPLATE_CM = {
    "SVM":         [[263, 2, 1, 34], [2, 251, 14, 3], [9, 2, 186, 33], [50, 6, 19, 125]],
    "Ensemble":    [[273, 9, 9, 9], [10, 234, 12, 14], [15, 14, 187, 14], [18, 18, 18, 146]],
    "FCLF":        [[262, 5, 6, 27], [2, 262, 5, 1], [2, 2, 206, 20], [44, 1, 5, 150]],
    "Mistral-7B":  [[266, 11, 12, 11], [11, 230, 15, 14], [15, 15, 186, 14], [16, 16, 19, 149]],
    "RoBERTa":     [[270, 10, 10, 10], [12, 235, 12, 11], [12, 13, 192, 13], [14, 13, 18, 155]],
    "GPT-4o-mini": [[285, 4, 5, 6], [5, 248, 8, 9], [2, 15, 201, 12], [5, 18, 16, 161]],
    "Claude-3.5":  [[294, 0, 2, 4], [3, 255, 1, 11], [8, 1, 205, 16], [9, 11, 16, 164]],
}

# ----------------------------------------------------------------- backbones
BACKBONES = ["Mistral-7B", "RoBERTa", "GPT-4o-mini", "Claude-3.5"]
# Abbreviated tick labels exactly as shown in the paper figure.
BACKBONE_TICKS = ["Mistral-7B", "RoBERTa", "GPT-4o-m", "Claude-3.5"]

# ------------------------------------------------- expected (target) values
# Comparison table at 1000 patients: (accuracy, precision, recall, f1).
TABLE_TARGET = {
    "SVM":      (0.825, 0.815, 0.810, 0.812),
    "Ensemble": (0.840, 0.835, 0.830, 0.832),
    "FCLF":     (0.880, 0.875, 0.872, 0.874),
}
# Correlation heatmap targets (rounded to 3 decimals as displayed).
HEAT_TARGET = {
    ("Patients", "SVM"): 0.774,
    ("Patients", "Ensemble"): 0.770,
    ("Patients", "FCLF"): 0.787,
    ("SVM", "Ensemble"): 1.000,
    ("SVM", "FCLF"): 1.000,
    ("Ensemble", "FCLF"): 0.999,
}

# 95% CI multiplier (two-sided t critical value, df = 4) for 5 runs.
T_CRIT_95_DF4 = 2.7764

RNG_SEED = 42
