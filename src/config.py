"""Central configuration for the MiniConvNet lung-cancer replication.

Everything that a notebook might want to tweak (seeds, image size, learning
rates, epoch budgets, output paths) lives here so that no notebook hardcodes
a value that another notebook contradicts.

Import style used by the notebooks:

    from src.config import *
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

# The Kaggle dataset is inconsistent: train/ and valid/ use long staging
# folder names ("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib") while test/ uses
# short ones ("adenocarcinoma"). data_utils.normalize_class_name() maps any
# folder name onto CLASS_NAMES by longest-prefix match against these keys.
CLASS_FOLDER_PREFIXES = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "normal": "normal",
}

# --------------------------------------------------------------------------
# Split variants
# --------------------------------------------------------------------------
# "faithful" -> the dataset's own train/valid/test folders, duplicates left in
#               place. This is the paper-comparable split.
# "clean"    -> content-hash deduplicated, leakage-free, group-aware
#               stratified re-split. NOT a replication of the paper's number;
#               it is a robustness / generalisation experiment.
SPLIT_VARIANTS = ["faithful", "clean"]
CLEAN_SPLIT_FRACTIONS = (0.70, 0.10, 0.20)   # train / val / test

# --------------------------------------------------------------------------
# Architecture variants (see README "Architecture ambiguity")
# --------------------------------------------------------------------------
ARCH_VARIANTS = ["gap", "flatten"]
MINICONVNET_FILTERS = (16, 32, 64, 128)   # 4 conv blocks -> 14x14x128 = 25088
GAP_DENSE_UNITS = 64                      # GAP head  -> ~106K total params
FLATTEN_DENSE_UNITS = 16                  # Flatten head -> ~0.5M total params
DROPOUT_RATE = 0.5

# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------
# LESSON 1: MiniConvNet collapses (dead ReLUs) with Adam's default 1e-3.
# 1e-4 + clipnorm=1.0 is the DEFAULT here, not an opt-in flag.
LR_MINICONVNET = 1e-4
CLIPNORM = 1.0
LR_BASELINE = 1e-4

EPOCHS_MINICONVNET = 100
EPOCHS_ABLATION = 100      # ablation must use the full budget, not a short run
EPOCHS_CV = 60
EPOCHS_BASELINE = 30

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# Class weighting: default ON for the (imbalanced) clean variant only.
USE_CLASS_WEIGHTS_BY_SPLIT = {"faithful": False, "clean": True}

# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Collapse detection (LESSON 3)
# --------------------------------------------------------------------------
COLLAPSE_WINDOW = 10        # trailing epochs inspected for flatness
COLLAPSE_STD_TOL = 1e-4     # std below this over the window == "flat"
COLLAPSE_KAPPA_TOL = 0.02   # |kappa| or |MCC| below this == degenerate
COLLAPSE_TAG = "INVALID_collapsed"
VALID_TAG = "ok"

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

RESULTS_TABLE_CSV = OUTPUT_ROOT / "results_table.csv"
EXPERIMENTS_LOG_CSV = OUTPUT_ROOT / "experiments_log.csv"
ABLATION_CSV = OUTPUT_ROOT / "ablation_dropout.csv"

LOCAL_DATA_DIR = PROJECT_ROOT / "Data"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dirs():
    """Create every output directory. Safe to call repeatedly."""
    for d in (OUTPUT_ROOT, FIGURES_DIR, HISTORY_DIR, REPORTS_DIR, SPLITS_DIR,
              MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "FIGURES_DIR": str(FIGURES_DIR),
        "HISTORY_DIR": str(HISTORY_DIR),
        "REPORTS_DIR": str(REPORTS_DIR),
        "SPLITS_DIR": str(SPLITS_DIR),
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
