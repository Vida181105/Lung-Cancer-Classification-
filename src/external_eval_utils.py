"""Phase 5 - cross-dataset generalisation: evaluate the existing MiniConvNet
and VGG16 checkpoints on a genuinely external CT dataset.

Entirely additive. Nothing here is imported by notebooks 00-12, and every
artefact goes to ``results/external_eval/``. No existing result, prediction,
or checkpoint is read for writing or modified.

Dataset selection - researched, not assumed (see notebook 13's markdown for
the full write-up with sources)
------------------------------------------------------------------------------
Two candidates were considered, per the Phase 5 brief:

* **IQ-OTH/NCCD** (Iraq-Oncology Teaching Hospital / National Center for
  Cancer Diseases) - SELECTED. Confirmed via web research: genuine CT
  modality (Siemens SOMATOM scanner, originally DICOM), collected at a
  different institution than our primary dataset (which is a web-scraped
  compilation, per its own Kaggle description), 3 whole-slice-image class
  labels (Normal / Benign / Malignant), CC BY 4.0 licence, publicly available
  on Kaggle and Mendeley Data with no special credentialing. Its labels are
  genuinely image-level and genuinely different from - but meaningfully
  reducible against - our 4-class scheme.
* **LIDC-IDRI** - REJECTED. Confirmed via web research: its labels are
  nodule-level malignancy SCORES (1-5, ordinal, not enforced to consensus
  across 4 radiologists) tied to 3D nodule ROIs via XML annotation, not
  whole-slice discrete class labels. Using it would require nodule
  localisation (pylidc/DICOM tooling not in requirements.txt), a
  score-to-label thresholding decision, and reconciling a fundamentally
  different unit of analysis (nodule crop vs. whole CT slice) with what our
  checkpoints were trained to classify. That is closer to *inventing* a task
  than *reducing* one - the "do not force incompatible labels" rule in the
  addendum rules it out for this project's scope.

Why the task must change (label compatibility)
------------------------------------------------------------------------------
Our checkpoints output 4-class softmax probabilities over
``adenocarcinoma / large.cell.carcinoma / normal / squamous.cell.carcinoma``.
IQ-OTH/NCCD provides 3 whole-slice labels: ``benign / malignant / normal``.
There is no direct 4-class comparison possible - IQ-OTH/NCCD carries no NSCLC
subtype information at all. The reduced task implemented here collapses our
model's three tumour-subtype probabilities into one "tumour" probability
(summed, since the three subtype classes and ``normal`` fully partition the
softmax output - no renormalisation is needed) and compares against
IQ-OTH/NCCD's ground truth under **three separate, explicitly labelled
framings** (see :data:`EXTERNAL_TASK_FRAMINGS`), because "benign" has no
counterpart anywhere in our model's training data and forcing it into either
bucket silently would be exactly the kind of unjustified label-forcing the
addendum prohibits. All three framings, plus a purely descriptive
(non-scored) report of what the model does with benign inputs, are computed
from a SINGLE forward pass - see :func:`run_external_forward_pass`.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    NORMAL_CLASS,
    PROJECT_ROOT,
    RESULTS_TABLE_CSV,
    SEED,
)
from src.finetune_utils import FAIR_METRICS_CSV

# --------------------------------------------------------------------------
# Paths - all new
# --------------------------------------------------------------------------

EXTERNAL_DATA_DIR = PROJECT_ROOT / "Data_External_IQOTHNCCD"

EXTERNAL_EVAL_DIR = PROJECT_ROOT / "results" / "external_eval"
EXTERNAL_FIGURES_DIR = EXTERNAL_EVAL_DIR / "figures"
EXTERNAL_PREDICTIONS_DIR = EXTERNAL_EVAL_DIR / "predictions"

EXTERNAL_METRICS_CSV = EXTERNAL_EVAL_DIR / "external_eval_metrics.csv"
EXTERNAL_METRICS_JSON = EXTERNAL_EVAL_DIR / "external_eval_metrics.json"
BENIGN_REPORT_JSON = EXTERNAL_EVAL_DIR / "benign_subset_report.json"


def ensure_external_eval_dirs() -> dict:
    for d in (EXTERNAL_EVAL_DIR, EXTERNAL_FIGURES_DIR, EXTERNAL_PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {"external_eval_dir": str(EXTERNAL_EVAL_DIR), "figures": str(EXTERNAL_FIGURES_DIR),
            "predictions": str(EXTERNAL_PREDICTIONS_DIR)}


# --------------------------------------------------------------------------
# Dataset definition - the EXTERNAL label space, independent of CLASS_NAMES
# --------------------------------------------------------------------------

# IQ-OTH/NCCD's own three classes. Deliberately NOT the same list as, or
# index-aligned with, this project's CLASS_NAMES - conflating the two would
# be exactly the label-forcing this phase must avoid.
EXTERNAL_CLASS_NAMES = ["benign", "malignant", "normal"]
EXTERNAL_NUM_CLASSES = len(EXTERNAL_CLASS_NAMES)

# Folder-name resolution is defensive, not hardcoded to one spelling: the
# commonly-cited public mirror of this dataset spells the benign folder
# "Bengin case" (a known typo carried through multiple re-uploads), but this
# was not independently confirmed byte-for-byte during research (Kaggle's
# file listing is JS-rendered and not visible to a plain page fetch), so
# every folder name actually found is matched case-insensitively against
# BOTH spellings and printed for the runner to verify, exactly as
# lc25000_utils.find_class_dirs() does for LC25000.
EXTERNAL_FOLDER_ALIASES = {
    "benign": ("bengin", "benign"),
    "malignant": ("malignant",),
    "normal": ("normal",),
}

DICOM_EXTENSIONS = (".dcm", ".dicom")


def resolve_external_data_root(explicit=None) -> Path:
    """Find the directory containing the three IQ-OTH/NCCD case-class folders.

    Same contract as ``lc25000_utils.resolve_lc25000_root()``: Kaggle first
    (auto-detected by name pattern), then the local ``Data_External_IQOTHNCCD/``
    folder, with an error message listing what was actually found rather than
    a bare failure.
    """
    import os

    if explicit is not None:
        root = Path(explicit)
        if not root.exists():
            raise FileNotFoundError(f"Explicit external-dataset path does not exist: {root}")
        return root

    if os.path.exists("/kaggle/input"):
        candidates = sorted(os.listdir("/kaggle/input"))
        for c in candidates:
            low = c.lower()
            if "iq-oth" in low or "iqothnccd" in low or "nccd" in low:
                return Path("/kaggle/input") / c
        raise FileNotFoundError(
            "Running on Kaggle but couldn't auto-detect the IQ-OTH/NCCD dataset folder. "
            f"Found: {candidates}. Attach the dataset, or pass an explicit path.")

    if not EXTERNAL_DATA_DIR.exists():
        raise FileNotFoundError(
            f"External dataset not found at {EXTERNAL_DATA_DIR}. Download the IQ-OTH/NCCD lung "
            "cancer dataset (CC BY 4.0) from Kaggle "
            "(kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) or Mendeley Data "
            "(data.mendeley.com/datasets/bhmdr45bh2) into that folder, or pass an explicit path.")
    return EXTERNAL_DATA_DIR


def find_external_class_dirs(root) -> dict:
    """Map canonical external class -> list of directories holding its images.

    Searches at any depth and tries every known spelling variant per class
    (see :data:`EXTERNAL_FOLDER_ALIASES`). Raises with a printed listing of
    what folder names WERE found if any expected class is missing, rather
    than silently proceeding with 2 of 3 classes.
    """
    root = Path(root)
    found = {c: [] for c in EXTERNAL_CLASS_NAMES}
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        low = d.name.strip().lower()
        for cls, aliases in EXTERNAL_FOLDER_ALIASES.items():
            if any(alias in low for alias in aliases):
                found[cls].append(d)
                break

    missing = [c for c, dirs in found.items() if not dirs]
    if missing:
        seen = sorted({p.name for p in root.rglob("*") if p.is_dir()})[:30]
        raise FileNotFoundError(
            f"No folders found for {missing} under {root}. Directory names actually seen "
            f"(first 30): {seen}")
    return found


def scan_for_dicom(root) -> dict:
    """Whether the downloaded copy is raw DICOM (needs pydicom, not in
    requirements.txt) or already-converted PNG/JPG (usable as-is).

    Public redistributions of this dataset are near-universally pre-converted
    to PNG/JPG for accessibility, but this is checked rather than assumed -
    if DICOM files are found, this is reported plainly as a blocker instead
    of silently attempting (and failing) to decode them with PIL.
    """
    root = Path(root)
    dicom_files = [str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in DICOM_EXTENSIONS]
    return {
        "n_dicom_files": len(dicom_files),
        "sample": dicom_files[:10],
        "is_dicom": bool(dicom_files),
        "note": ("Raw DICOM files were found. This module only decodes standard image formats "
                "(PNG/JPG/etc, via PIL) - pydicom is not in requirements.txt and no DICOM "
                "handling is implemented here. Locate a PNG/JPG mirror of this dataset instead "
                "of proceeding with these files."
                if dicom_files else
                "No DICOM files found - the downloaded copy is standard image format(s), "
                "usable directly by this project's existing image pipeline."),
    }


def index_external_dataset(root=None) -> pd.DataFrame:
    """One row per external image: ``filepath, filename, external_class,
    external_label, folder``.

    ``external_label`` indexes into :data:`EXTERNAL_CLASS_NAMES` - this is a
    SEPARATE label space from this project's ``CLASS_NAMES`` and must never
    be used as if it were a 4-class label.
    """
    root = resolve_external_data_root(root)
    class_dirs = find_external_class_dirs(root)

    rows = []
    for cls, dirs in class_dirs.items():
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append({
                    "filepath": str(f), "filename": f.name, "external_class": cls,
                    "external_label": EXTERNAL_CLASS_NAMES.index(cls), "folder": d.name,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root} - check the dataset layout and that "
                           "scan_for_dicom() did not report raw DICOM files.")
    return df.reset_index(drop=True)


def external_class_counts(df: pd.DataFrame) -> pd.Series:
    return df["external_class"].value_counts().reindex(EXTERNAL_CLASS_NAMES, fill_value=0)


def sample_image_properties(df: pd.DataFrame, n=5) -> pd.DataFrame:
    """Format/mode/size of a few actual files - the concrete "confirm it's
    really CT, not X-ray or something else" check, run on whatever was
    downloaded rather than trusted from documentation alone."""
    from PIL import Image

    rows = []
    for _, r in df.sample(n=min(n, len(df)), random_state=SEED).iterrows():
        with Image.open(r["filepath"]) as im:
            rows.append({"filepath": r["filepath"], "external_class": r["external_class"],
                        "format": im.format, "mode": im.mode, "size": im.size})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tf.data pipeline (external images -> our model's expected input)
# --------------------------------------------------------------------------


def _decode(path, img_size):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # raw [0, 255]; the checkpoint's own in-model
    img.set_shape((img_size[0], img_size[1], 3))   # Rescaling/preprocess_input handles scaling
    return img


def make_external_dataset(df: pd.DataFrame, img_size=IMG_SIZE, batch_size=32):
    """``(image[0..255], external_label)`` dataset, unshuffled so predictions
    stay aligned with ``df`` row order - same discipline as every other
    dataset builder in this project."""
    import tensorflow as tf

    if df.empty:
        raise ValueError("make_external_dataset received an empty dataframe.")
    paths = df["filepath"].astype(str).tolist()
    labels = df["external_label"].astype(int).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# STEP 3 - external evaluation: ONE forward pass, no fitting, no tuning
# --------------------------------------------------------------------------


def run_external_forward_pass(checkpoint_path, df, batch_size=32, verbose=True):
    """Load an EXISTING checkpoint and run ONE forward pass over the external
    dataset - the only inference cost this phase incurs per model.

    Returns ``(external_label, y_prob_4class)`` where ``y_prob_4class`` is
    this project's own 4-class softmax output (columns in ``CLASS_NAMES``
    order) - every reduced-task framing and the calibration analysis is
    derived from this single array afterwards, at zero extra inference cost.

    **No `.fit()` call, no gradient computation, no threshold search against
    this data anywhere in this function.**
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(str(checkpoint_path))
    ds = make_external_dataset(df, batch_size=batch_size)

    probs, labels = [], []
    for x, y in ds:
        probs.append(model(x, training=False).numpy())
        labels.append(y.numpy())
    y_prob = np.concatenate(probs, axis=0)
    external_label = np.concatenate(labels, axis=0)

    if verbose:
        print(f"  forward pass complete: {len(external_label)} images, "
             f"output shape {y_prob.shape}")
    return external_label, y_prob, model


def print_external_eval_time_estimate(seconds_for_first_batch, n_batches_total, label=""):
    """Informational only - a single forward pass is cheap (Phase 4's MC
    Dropout needed 15 of these; this phase needs 1), so no stop-and-ask gate
    is used, matching the addendum's own reasoning. Still worth printing,
    especially for VGG16."""
    projected = seconds_for_first_batch * n_batches_total
    tag = f"[{label}] " if label else ""
    print(f"{tag}first batch took {seconds_for_first_batch:.2f}s -> "
         f"~{projected:.1f}s projected for all {n_batches_total} batches (single pass)")
    return {"seconds_per_batch": round(seconds_for_first_batch, 3),
           "projected_total_seconds": round(projected, 1)}


# --------------------------------------------------------------------------
# STEP 2 - the reduced task: collapse our 4-class output, define framings
# --------------------------------------------------------------------------


def reduce_to_binary_tumor_probs(y_prob_4class, class_names=CLASS_NAMES,
                                 normal_class=NORMAL_CLASS) -> np.ndarray:
    """4-class softmax -> 2-column ``[p_normal, p_tumor]``.

    The three tumour-subtype classes and ``normal`` fully partition the
    softmax output, so ``p_tumor = sum of the three subtype probabilities``
    and ``p_normal + p_tumor == 1`` exactly - no renormalisation needed or
    performed.
    """
    y_prob_4class = np.asarray(y_prob_4class, dtype=float)
    normal_idx = class_names.index(normal_class)
    tumor_idx = [i for i in range(len(class_names)) if i != normal_idx]
    p_normal = y_prob_4class[:, normal_idx]
    p_tumor = y_prob_4class[:, tumor_idx].sum(axis=1)
    return np.stack([p_normal, p_tumor], axis=1)


# Three explicitly separate, explicitly labelled reduced-task framings.
# "benign" has no counterpart in our training data, so - rather than picking
# one silent grouping - all three are computed and reported; the notebook
# presents the first as primary and the other two as secondary/exploratory,
# per the addendum's "state this task change explicitly and prominently"
# instruction.
EXTERNAL_TASK_FRAMINGS = {
    "malignant_vs_normal_excl_benign": {
        "label": "Malignant vs Normal (benign EXCLUDED from scoring)",
        "primary": True,
        "description": (
            "Benign cases are excluded entirely - our model was never trained on a benign "
            "class, so there is no ground-truth-correct answer for it in our label space. "
            "This is the most scientifically defensible reduced task: it only scores the two "
            "categories (malignant, normal) that have a genuine counterpart in our training "
            "data."),
        "include": lambda ext_label: np.isin(
            ext_label, [EXTERNAL_CLASS_NAMES.index("malignant"), EXTERNAL_CLASS_NAMES.index("normal")]),
        "binary_true": lambda ext_label: (ext_label == EXTERNAL_CLASS_NAMES.index("malignant")).astype(int),
    },
    "malignant_vs_not_malignant_full": {
        "label": "Malignant vs Not-Malignant (benign grouped with normal)",
        "primary": False,
        "description": (
            "Secondary/exploratory framing: benign folded into the 'not malignant' side, full "
            "external test set scored. Treats benign as closer to normal than to malignant - "
            "arguably the more clinically apt framing given our model was trained to recognise "
            "NSCLC malignant subtypes specifically, not 'any abnormality'."),
        "include": lambda ext_label: np.ones_like(ext_label, dtype=bool),
        "binary_true": lambda ext_label: (ext_label == EXTERNAL_CLASS_NAMES.index("malignant")).astype(int),
    },
    "abnormal_vs_normal_full": {
        "label": "Abnormal (benign OR malignant) vs Normal",
        "primary": False,
        "description": (
            "Secondary/exploratory framing: any non-normal finding (benign or malignant) "
            "counted as a correct 'tumour' call, full external test set scored. The most "
            "lenient framing - most favourable to the model, least specific to what it was "
            "actually trained to detect."),
        "include": lambda ext_label: np.ones_like(ext_label, dtype=bool),
        "binary_true": lambda ext_label: (ext_label != EXTERNAL_CLASS_NAMES.index("normal")).astype(int),
    },
}


def binary_task_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / macro P-R-F1 / per-class P-R-F1 / confusion matrix / mean
    confidence / AUC for a 2-class task.

    Self-contained rather than reusing ``evaluate_utils.compute_metrics()``,
    which is written around a fixed ``NUM_CLASSES == 4`` (this project's
    primary task) - reusing it for a 2-class, differently-labelled problem
    would risk silently misapplying that contract. This mirrors the same
    "each genuinely different task gets its own small metrics helper"
    decision already made in ``lc25000_utils.py`` and ``calibration_utils.py``.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0, "note": "no samples in this framing"}

    accuracy = float(np.mean(y_true == y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)

    out = {
        "n_samples": int(n),
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_normal": float(precision[0]), "recall_normal": float(recall[0]),
        "f1_normal": float(f1[0]), "support_normal": int(support[0]),
        "precision_tumor": float(precision[1]), "recall_tumor": float(recall[1]),
        "f1_tumor": float(f1[1]), "support_tumor": int(support[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        out["mean_confidence"] = float(y_prob.max(axis=1).mean())
        if len(set(y_true.tolist())) > 1:
            try:
                out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except ValueError:
                out["auc"] = None
        else:
            out["auc"] = None
    return out


def evaluate_external_framing(framing_key, external_label, binary_prob) -> dict:
    """Score one of :data:`EXTERNAL_TASK_FRAMINGS` against the external
    ground truth and our reduced binary probabilities."""
    framing = EXTERNAL_TASK_FRAMINGS[framing_key]
    include = framing["include"](external_label)
    y_true = framing["binary_true"](external_label)[include]
    y_prob = binary_prob[include]
    y_pred = y_prob.argmax(axis=1)

    metrics = binary_task_metrics(y_true, y_pred, y_prob)
    metrics.update({
        "framing": framing_key, "label": framing["label"], "primary": framing["primary"],
        "description": framing["description"], "n_excluded": int((~include).sum()),
    })
    return metrics


def describe_benign_subset(external_label, binary_prob) -> dict:
    """Purely descriptive report of what the model does with benign inputs -
    NOT scored, because our model's training data defines no correct answer
    for a class it was never trained on."""
    mask = external_label == EXTERNAL_CLASS_NAMES.index("benign")
    n = int(mask.sum())
    if n == 0:
        return {"n_benign": 0, "note": "no benign cases in this external dataset copy"}
    pred = binary_prob[mask].argmax(axis=1)
    frac_tumor = float((pred == 1).mean())
    return {
        "n_benign": n,
        "fraction_predicted_tumor": frac_tumor,
        "fraction_predicted_normal": round(1.0 - frac_tumor, 6),
        "mean_confidence": float(binary_prob[mask].max(axis=1).mean()),
        "note": ("Descriptive only, not accuracy-scored - our model was never trained on a "
                "benign class, so there is no ground-truth-correct label to score against. "
                "This reports the model's behaviour on out-of-distribution benign inputs, not "
                "whether that behaviour is 'right'."),
    }


# --------------------------------------------------------------------------
# STEP 4 - domain-shift comparison against the already-known in-domain result
# --------------------------------------------------------------------------


def load_in_domain_binary_reference(secondary_model_name=None, secondary_run_name=None) -> dict:
    """The in-domain tumour-vs-normal binary accuracy already on record for
    each model - read live from the existing results files, never
    hardcoded, so a stale number can never be quoted here silently.

    MiniConvNet's figure comes from ``outputs/results_table.csv`` (the
    ``binary_tumor_acc`` column of its 3-fold-CV row) - already exactly the
    same reduced binary task this phase's PRIMARY framing computes, just
    measured in-domain instead of externally, making this the cleanest
    possible before/after comparison.
    """
    from src.evaluate_utils import load_results

    ref = {}
    canon = load_results("canonical")
    if len(canon):
        row = canon[canon["model"].astype(str).str.startswith("MiniConvNet")]
        if len(row):
            r = row.iloc[0]
            ref["MiniConvNet"] = {
                "binary_tumor_accuracy": float(r["binary_tumor_acc"]),
                "source": f"{RESULTS_TABLE_CSV} ('{r['model']}' row, binary_tumor_acc column)",
            }

    if secondary_model_name and secondary_run_name and Path(FAIR_METRICS_CSV).exists():
        fair = pd.read_csv(FAIR_METRICS_CSV)
        row = fair[fair["run_name"] == secondary_run_name]
        if len(row):
            r = row.iloc[0]
            ref[secondary_model_name] = {
                "binary_tumor_accuracy": float(r["tumour_detection_accuracy"]),
                "source": f"{FAIR_METRICS_CSV} ('{secondary_run_name}' row, "
                         "tumour_detection_accuracy column)",
            }
    return ref


def domain_shift_summary(in_domain_accuracy, external_accuracy) -> dict:
    drop = in_domain_accuracy - external_accuracy
    return {
        "in_domain_accuracy": float(in_domain_accuracy),
        "external_accuracy": float(external_accuracy),
        "absolute_drop": round(float(drop), 6),
        "relative_drop_pct": (round(100 * drop / in_domain_accuracy, 2)
                              if in_domain_accuracy else None),
        "direction": ("external is WORSE (expected direction for a genuine domain shift)"
                     if drop > 0 else
                     "external is BETTER OR EQUAL (unusual - inspect the framing/data before "
                     "trusting this)"),
    }


# --------------------------------------------------------------------------
# STEP 6 - visualisation
# --------------------------------------------------------------------------


def plot_external_confusion(metrics: dict, model_name, save=True, show=True):
    """Confusion matrix for one framing's result."""
    import matplotlib.pyplot as plt

    if metrics.get("n_samples", 0) == 0:
        return None
    cm = np.array(metrics["confusion_matrix"])
    labels = ["normal", "tumor"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (external ground truth)")
    ax.set_title(f"{model_name} - {metrics['label']}\naccuracy={metrics['accuracy']:.4f}",
                fontsize=9)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    path = None
    if save:
        ensure_external_eval_dirs()
        fname = f"{model_name}_{metrics['framing']}_confusion.png"
        path = EXTERNAL_FIGURES_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_domain_shift_bar(model_name, in_domain_acc, external_acc, save=True, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["in-domain", "external"], [in_domain_acc, external_acc],
          color=["#4c78a8", "#e45756"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("tumour-vs-normal accuracy")
    ax.set_title(f"{model_name} - domain shift")
    for i, v in enumerate([in_domain_acc, external_acc]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        ensure_external_eval_dirs()
        path = EXTERNAL_FIGURES_DIR / f"{model_name}_domain_shift.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


# --------------------------------------------------------------------------
# Artefact writing
# --------------------------------------------------------------------------

EXTERNAL_RECORD_COLUMNS = [
    "model", "framing", "label", "primary", "n_samples", "n_excluded",
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_tumor", "recall_tumor", "f1_tumor", "auc", "mean_confidence",
    "in_domain_accuracy", "absolute_drop", "notes", "timestamp",
]


def record_external_result(row: dict) -> Path:
    unknown = set(row) - set(EXTERNAL_RECORD_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column(s): {sorted(unknown)}. Allowed: {EXTERNAL_RECORD_COLUMNS}")
    if not row.get("model") or not row.get("framing"):
        raise ValueError("external-eval rows require both 'model' and 'framing'.")

    ensure_external_eval_dirs()
    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if EXTERNAL_METRICS_CSV.exists():
        table = pd.read_csv(EXTERNAL_METRICS_CSV)
        for c in EXTERNAL_RECORD_COLUMNS:
            if c not in table.columns:
                table[c] = pd.NA
        table = table[EXTERNAL_RECORD_COLUMNS]
        table = table[~((table["model"] == row["model"]) & (table["framing"] == row["framing"]))]
    else:
        table = pd.DataFrame(columns=EXTERNAL_RECORD_COLUMNS)

    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[EXTERNAL_RECORD_COLUMNS]
    table.to_csv(EXTERNAL_METRICS_CSV, index=False)
    table.to_json(EXTERNAL_METRICS_JSON, orient="records", indent=2)
    return EXTERNAL_METRICS_CSV


def load_external_results() -> pd.DataFrame:
    if not EXTERNAL_METRICS_CSV.exists():
        return pd.DataFrame(columns=EXTERNAL_RECORD_COLUMNS)
    return pd.read_csv(EXTERNAL_METRICS_CSV)


def save_external_predictions(model_name, df, external_label, y_prob_4class, binary_prob) -> Path:
    """Raw per-image predictions on the external set - 4-class softmax,
    reduced binary probabilities, and external ground truth, all together so
    this phase's numbers can be re-derived without re-running inference."""
    ensure_external_eval_dirs()
    frame = {
        "filepath": df["filepath"].values, "external_class": df["external_class"].values,
        "external_label": external_label,
    }
    for i, c in enumerate(CLASS_NAMES):
        frame[f"prob_{c}"] = y_prob_4class[:, i]
    frame["prob_reduced_normal"] = binary_prob[:, 0]
    frame["prob_reduced_tumor"] = binary_prob[:, 1]

    path = EXTERNAL_PREDICTIONS_DIR / f"{model_name}_external_predictions.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def save_benign_report(all_reports: dict) -> Path:
    ensure_external_eval_dirs()
    with open(BENIGN_REPORT_JSON, "w") as fh:
        json.dump(all_reports, fh, indent=2, default=str)
    return BENIGN_REPORT_JSON
