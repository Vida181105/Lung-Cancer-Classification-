"""Central configuration for the v3 MiniConvNet replication.

Everything a notebook might want to tweak (seeds, image size, learning rates,
epoch budgets, the anti-collapse settings, output paths) lives here, so no
notebook hardcodes a value another notebook contradicts.

Import style used by the notebooks:

    from src.config import *

Two things drive every choice in this file:

* **CPU only.** There is no GPU for this project at any point. Epoch budgets,
  fold counts and the baseline set are all scoped so the work can actually
  finish; see ``TIME_*`` below and the notebook time-estimate cells.
* **Partial collapse is the bug to beat** (v2's unresolved failure: the model
  predicted only 2 of 4 classes). The five countermeasures from lesson 2 -
  LeakyReLU, He init, a wide bottleneck, ReduceLROnPlateau and label smoothing
  - are all defaults here, not opt-in flags.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Data geometry
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)          # (height, width)
IMG_SHAPE = IMG_SIZE + (3,)    # models always take 3-channel input
BATCH_SIZE = 32

# Canonical class order. Every label index in this project is
# CLASS_NAMES.index(<class>), so this order must never change.
CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]
NUM_CLASSES = len(CLASS_NAMES)

NORMAL_CLASS = "normal"
TUMOR_CLASSES = [c for c in CLASS_NAMES if c != NORMAL_CLASS]

# The Kaggle dataset is inconsistent: train/ and valid/ use long staging folder
# names ("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib") while test/ uses short
# ones ("adenocarcinoma"). data_utils.normalize_class_name() maps any folder
# name onto CLASS_NAMES by longest-prefix match against these keys.
CLASS_FOLDER_PREFIXES = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "normal": "normal",
}

# --------------------------------------------------------------------------
# Split variants (LESSON 6)
# --------------------------------------------------------------------------
# "faithful" -> the dataset's own train/valid/test folders, duplicates left in
#               place. This is the paper-comparable split.
# "clean"    -> content-hash deduplicated, leakage-free, group-aware
#               stratified re-split. NOT a replication of the paper's number;
#               it is a robustness / generalisation experiment.
SPLIT_VARIANTS = ["faithful", "clean"]
CLEAN_SPLIT_FRACTIONS = (0.70, 0.10, 0.20)   # train / val / test

# --------------------------------------------------------------------------
# MiniConvNet architecture (LESSON 3 - one architecture, built properly)
# --------------------------------------------------------------------------
# v3 commits to the ~0.5M-parameter Flatten reading. The paper's stated layer
# list (GAP -> Dense(64) -> Dense(4)) only reaches ~106K params and contradicts
# the paper's own "~0.5M". v2 built both and split its compute two ways; this
# round builds one and builds it well.
#
# 4 conv blocks with filters below, each Conv2D(3x3, same) + LeakyReLU + Pool,
# then ONE extra pool before the flatten:
#     224 -> 112 -> 56 -> 28 -> 14 -> (extra pool) -> 7
# so the flatten width is 7 x 7 x 128 = 6272, feeding a single WIDE Dense(64)
# bottleneck. Exact parameter count: 97,440 (backbone) + 401,732 (head)
# = 499,172, i.e. the paper's ~0.5M.
MINICONVNET_FILTERS = (16, 32, 64, 128)
EXTRA_POOL_BEFORE_FLATTEN = True   # 14x14 -> 7x7; this is what buys the wide head
DENSE_UNITS = 64                   # LESSON 2: wide bottleneck, NOT v2's Dense(16) x3
DROPOUT_RATE = 0.5

# LESSON 2, cause A: dying ReLU. A ReLU unit whose pre-activation is negative
# for every input has exactly zero gradient forever; enough dead units and whole
# classes become unreachable. LeakyReLU keeps a gradient on the negative side.
# This is the v3 DEFAULT, not a fallback tried after failure. Costs 0 params.
ACTIVATION = "leaky_relu"          # 'leaky_relu' | 'relu' (relu only as a control)
LEAKY_RELU_ALPHA = 0.1

# LESSON 2, cause C: initialisation. He is the correct scaling for ReLU-family
# activations; do not silently inherit Keras's Glorot default.
KERNEL_INITIALIZER = "he_normal"

# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------
# LESSON 1: MiniConvNet collapses (dead ReLUs) with Adam's default 1e-3.
# 1e-4 + clipnorm=1.0 is the DEFAULT here, not an opt-in flag.
LR_MINICONVNET = 1e-4
CLIPNORM = 1.0
LR_BASELINE = 1e-4

# LESSON 2, cause E: label smoothing. Discourages the early, overconfident
# commitment to a subset of classes that precedes a partial collapse.
LABEL_SMOOTHING = 0.05

# LESSON 2, cause D: a learning-rate schedule, so training can self-correct
# instead of sitting at one rate for the whole run. Monitors val_loss (a
# smoother signal than val_accuracy on ~600 training images).
REDUCE_LR_MONITOR = "val_loss"
REDUCE_LR_MODE = "min"
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# Early stopping still watches val_accuracy - it is the quantity being reported.
EARLY_STOPPING_MONITOR = "val_accuracy"
EARLY_STOPPING_MODE = "max"
EARLY_STOPPING_PATIENCE = 12

# CPU-scoped epoch budgets. Early stopping usually ends a run well short of
# these; they are ceilings, not targets.
EPOCHS_MINICONVNET = 60
EPOCHS_ABLATION = 60       # both ablation arms must get the SAME budget
EPOCHS_CV = 40
EPOCHS_BASELINE = 25

# Class weighting: default ON for the (imbalanced) clean variant only.
USE_CLASS_WEIGHTS_BY_SPLIT = {"faithful": False, "clean": True}

# --------------------------------------------------------------------------
# Cross-validation (LESSON 10)
# --------------------------------------------------------------------------
# 3 folds, not 5: a disclosed CPU-budget decision. 3 folds still gives a real
# distributional estimate (mean +/- std over independent test partitions),
# which is the point of lesson 10, at ~40% less training than 5 folds.
CV_FOLDS = 3

# --------------------------------------------------------------------------
# Collapse detection (LESSON 4) - both modes, from the first run
# --------------------------------------------------------------------------
COLLAPSE_WINDOW = 10        # trailing epochs inspected for flatness
COLLAPSE_STD_TOL = 1e-4     # std below this over the window == "flat"
COLLAPSE_KAPPA_TOL = 0.02   # |kappa| or |MCC| below this == degenerate
COLLAPSE_TAG = "INVALID_collapsed"
# The check v2 was missing: a model predicting only some classes, at a kappa
# clearly above zero, passed every other test and was logged as "ok".
PARTIAL_COLLAPSE_TAG = "INVALID_partial_collapse"
MIN_PREDICTED_CLASSES = NUM_CLASSES   # every class must be predicted >= once
VALID_TAG = "ok"
INVALID_TAGS = (COLLAPSE_TAG, PARTIAL_COLLAPSE_TAG)

# --------------------------------------------------------------------------
# CPU time budgeting
# --------------------------------------------------------------------------
# Any run whose estimate exceeds this must be reported and agreed BEFORE it
# starts. Notebooks time 1-2 real epochs first, then extrapolate - they never
# guess from a table.
TIME_ASK_THRESHOLD_MINUTES = 20
TIME_PROBE_EPOCHS = 1        # epochs to time before extrapolating

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IS_KAGGLE = os.path.exists("/kaggle/input")

# Keep every artefact inside the repo when the repo is writable (true both
# locally and for a repo cloned into /kaggle/working). Fall back to
# /kaggle/working only if the checkout itself is read-only.
if os.access(PROJECT_ROOT, os.W_OK):
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    MODELS_DIR = PROJECT_ROOT / "models"
else:  # pragma: no cover - only hit on a read-only Kaggle input mount
    OUTPUT_ROOT = Path("/kaggle/working/outputs")
    MODELS_DIR = Path("/kaggle/working/models")

FIGURES_DIR = OUTPUT_ROOT / "figures"
HISTORY_DIR = OUTPUT_ROOT / "history"
REPORTS_DIR = OUTPUT_ROOT / "reports"
SPLITS_DIR = OUTPUT_ROOT / "splits"
# LESSON 5: raw per-run predictions, so a diagnostic invented later can be
# applied to an old run without retraining it.
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

RESULTS_TABLE_CSV = OUTPUT_ROOT / "results_table.csv"
EXPERIMENTS_LOG_CSV = OUTPUT_ROOT / "experiments_log.csv"
ABLATION_CSV = OUTPUT_ROOT / "ablation_dropout.csv"

LOCAL_DATA_DIR = PROJECT_ROOT / "Data"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dirs():
    """Create every output directory. Safe to call repeatedly."""
    for d in (OUTPUT_ROOT, FIGURES_DIR, HISTORY_DIR, REPORTS_DIR, SPLITS_DIR,
              PREDICTIONS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "FIGURES_DIR": str(FIGURES_DIR),
        "HISTORY_DIR": str(HISTORY_DIR),
        "REPORTS_DIR": str(REPORTS_DIR),
        "SPLITS_DIR": str(SPLITS_DIR),
        "PREDICTIONS_DIR": str(PREDICTIONS_DIR),
        "MODELS_DIR": str(MODELS_DIR),
    }


def describe_environment():
    """Small human-readable dict printed at the top of every notebook."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "IS_KAGGLE": IS_KAGGLE,
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "MODELS_DIR": str(MODELS_DIR),
        "SEED": SEED,
        "IMG_SIZE": IMG_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "CLASS_NAMES": CLASS_NAMES,
    }


def describe_anti_collapse_settings():
    """The lesson-2 countermeasures, printed before every training run.

    All five are on by default. Printing them means a run can never be reported
    without a record of which of them were actually active.
    """
    return {
        "activation": ACTIVATION,
        "leaky_relu_alpha": LEAKY_RELU_ALPHA,
        "kernel_initializer": KERNEL_INITIALIZER,
        "dense_units (wide bottleneck)": DENSE_UNITS,
        "learning_rate": LR_MINICONVNET,
        "clipnorm": CLIPNORM,
        "label_smoothing": LABEL_SMOOTHING,
        "reduce_lr": f"{REDUCE_LR_MONITOR} patience={REDUCE_LR_PATIENCE} "
                     f"factor={REDUCE_LR_FACTOR}",
    }
