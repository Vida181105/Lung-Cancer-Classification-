"""LC25000 (lung subset) support - the project's SECOND dataset.

Everything specific to the second-dataset experiment lives here so that the
primary CT-scan pipeline is untouched. The CT experiment is finalised and
verified; nothing in this module is imported by any of notebooks 00-06.

Why a separate module rather than parameters everywhere
-------------------------------------------------------
``src/evaluate_utils.py`` is written around the CT dataset's **4** classes -
``CLASS_NAMES``, ``NUM_CLASSES`` and the tumour-vs-``normal`` split are baked
into its metrics, confusion matrices and reports. LC25000's lung subset has
**3** classes with different names, so reusing those functions directly would
silently mislabel columns and invent a fourth, always-empty class. Instead:

* **Reused unchanged** (class-count agnostic): ``compute_metrics`` for
  everything except AUC, ``detect_collapse`` for the flat-curve and
  kappa/MCC rules, ``detect_partial_collapse`` for the dead-class rule, plus
  all of ``train_utils`` (compile, callbacks, timing, history) and both model
  builders, which already take ``num_classes``.
* **Written here**: the 3-class-aware wrappers for metrics, confusion matrix,
  per-class report, prediction saving and results routing.

The collapse *thresholds and status tags* are imported from ``src.config``, so
this experiment cannot drift away from the primary one's definition of a valid
run. Only ONE existing function was touched to make this work:
``detect_partial_collapse`` gained an optional ``class_names`` argument whose
default preserves its previous behaviour exactly.

Expected dataset layout (either published form works)
-----------------------------------------------------
``resolve_lc25000_root()`` searches for directories named ``lung_aca``,
``lung_n`` and ``lung_scc`` at any depth, so both of these work as-is::

    Data_LC25000/lung_colon_image_set/lung_image_sets/lung_aca/*.jpeg
                                                     /lung_n/*.jpeg
                                                     /lung_scc/*.jpeg
    Data_LC25000/<any 10-fold layout>/.../lung_aca/*.jpeg

Colon folders (``colon_aca``, ``colon_n``) are ignored by construction - only
the three lung folder names are ever indexed.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BATCH_SIZE,
    COLLAPSE_TAG,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    OUTPUT_ROOT,
    PARTIAL_COLLAPSE_TAG,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    SEED,
    VALID_TAG,
)
from src.evaluate_utils import compute_metrics, detect_collapse, detect_partial_collapse

# --------------------------------------------------------------------------
# Dataset definition
# --------------------------------------------------------------------------

# Canonical class order for THIS dataset. Independent of the CT dataset's
# CLASS_NAMES - never mix the two.
LC25000_CLASS_NAMES = [
    "lung_adenocarcinoma",
    "lung_benign",
    "lung_squamous_cell_carcinoma",
]
LC25000_NUM_CLASSES = len(LC25000_CLASS_NAMES)
LC25000_BENIGN_CLASS = "lung_benign"

# Folder name (lowercased) -> canonical class. Lung only; colon is excluded by
# simply never appearing here.
LC25000_FOLDER_MAP = {
    "lung_aca": "lung_adenocarcinoma",
    "lung_n": "lung_benign",
    "lung_scc": "lung_squamous_cell_carcinoma",
}

LOCAL_LC25000_DIR = PROJECT_ROOT / "Data_LC25000"

# Split fractions for the single train/val/test partition (no CV here - see the
# disclosed scope reduction in notebook 07).
LC25000_SPLIT_FRACTIONS = (0.70, 0.10, 0.20)

# Disclosed sampling decision: LC25000 ships 5,000 images per lung class
# (15,000 total), which is ~15x the CT dataset on CPU. The notebook's timing
# gate decides whether to use this stratified subsample or the full set.
LC25000_IMAGES_PER_CLASS = 1200

# New, separate results table - the primary one is never touched.
LC25000_RESULTS_CSV = OUTPUT_ROOT / "results_table_lc25000.csv"
LC25000_PREFIX = "lc25000_"

LC25000_RESULTS_COLUMNS = [
    "model", "arch_variant", "dataset", "params", "accuracy", "precision_macro",
    "recall_macro", "f1_macro", "cohen_kappa", "mcc", "auc_macro",
    "benign_vs_malignant_acc", "subtype_acc", "n_train", "n_test",
    "epochs_trained", "n_runs", "status", "notes", "timestamp",
]


# --------------------------------------------------------------------------
# Discovery and indexing
# --------------------------------------------------------------------------


def resolve_lc25000_root(explicit=None) -> Path:
    """Find the directory containing the three lung class folders.

    Mirrors ``data_utils.resolve_data_root()``'s contract: Kaggle first, then
    the local ``Data_LC25000/`` folder, with an error message that lists what
    was actually found rather than just failing.
    """
    if explicit is not None:
        root = Path(explicit)
        if not root.exists():
            raise FileNotFoundError(f"Explicit LC25000 path does not exist: {root}")
        return root

    if os.path.exists("/kaggle/input"):
        candidates = sorted(os.listdir("/kaggle/input"))
        for c in candidates:
            low = c.lower()
            if "lc25000" in low or "lung" in low or "histopath" in low:
                return Path("/kaggle/input") / c
        raise FileNotFoundError(
            "Running on Kaggle but couldn't auto-detect the LC25000 dataset folder. "
            f"Found: {candidates}. Attach the dataset, or pass an explicit path."
        )

    if not LOCAL_LC25000_DIR.exists():
        raise FileNotFoundError(
            f"LC25000 not found at {LOCAL_LC25000_DIR}. Download the Kaggle "
            "'Lung and Colon Cancer Histopathological Images' dataset (or the "
            "lung-only LC25000 split) into that folder, or pass an explicit path. "
            "Only the three lung_* folders are used; colon folders are ignored."
        )
    return LOCAL_LC25000_DIR


def find_class_dirs(root) -> dict:
    """Map canonical class -> list of directories holding its images.

    Searches at any depth, so both the combined lung+colon layout and a
    pre-split lung-only layout work without configuration. A class appearing in
    several directories (e.g. a 10-fold layout) has all of them collected.
    """
    root = Path(root)
    found = {c: [] for c in LC25000_CLASS_NAMES}
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        cls = LC25000_FOLDER_MAP.get(d.name.strip().lower())
        if cls is not None:
            found[cls].append(d)

    missing = [c for c, dirs in found.items() if not dirs]
    if missing:
        seen = sorted({p.name for p in root.rglob("*") if p.is_dir()})[:25]
        raise FileNotFoundError(
            f"No image folders found for {missing} under {root}. Expected folders named "
            f"{sorted(LC25000_FOLDER_MAP)}. Directory names seen (first 25): {seen}"
        )
    return found


def index_lc25000(root=None) -> pd.DataFrame:
    """One row per lung image: ``filepath, filename, class, label, folder``.

    Colon images are never indexed - only the three lung folder names are
    looked up, so the colon half of the dataset is excluded by construction
    rather than filtered out afterwards.
    """
    root = resolve_lc25000_root(root)
    class_dirs = find_class_dirs(root)

    rows = []
    for cls, dirs in class_dirs.items():
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append({
                    "filepath": str(f),
                    "filename": f.name,
                    "class": cls,
                    "label": LC25000_CLASS_NAMES.index(cls),
                    "folder": d.name,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root} - check the dataset layout.")
    return df.reset_index(drop=True)


def lc25000_class_counts(df: pd.DataFrame) -> pd.Series:
    return df["class"].value_counts().reindex(LC25000_CLASS_NAMES, fill_value=0)


def check_duplicate_filenames(df: pd.DataFrame) -> dict:
    """Cheap sanity check for repeated filenames across class folders.

    LC25000 is augmentation-generated from 250 source images per class, so
    near-duplicates exist *by construction*. This does not attempt a content
    hash over 15,000 images (expensive, and not what the second-dataset
    experiment is for) - it only flags exact filename collisions, and the
    caveat is stated in the notebook instead of being silently ignored.
    """
    counts = df["filename"].value_counts()
    repeated = counts[counts > 1]
    return {
        "n_files": int(len(df)),
        "n_unique_filenames": int(df["filename"].nunique()),
        "n_repeated_filenames": int(len(repeated)),
        "note": ("LC25000 is generated by augmenting 250 original images per class, so "
                 "near-duplicate IMAGES exist by design. This checks filenames only; the "
                 "train/test split below is random within class, so that augmentation "
                 "relationship can straddle the split. State it whenever these numbers "
                 "are quoted."),
    }


# --------------------------------------------------------------------------
# Sampling and splitting
# --------------------------------------------------------------------------


def stratified_subsample(df: pd.DataFrame, per_class=LC25000_IMAGES_PER_CLASS,
                         seed=SEED) -> pd.DataFrame:
    """Take ``per_class`` images from each class (all of them if fewer exist).

    A disclosed CPU-budget decision, not a silent truncation: the notebook
    prints how many were dropped and records the sample size in the results
    row's ``n_train`` / ``notes``.
    """
    if per_class is None:
        return df.reset_index(drop=True)
    parts = []
    for cls in LC25000_CLASS_NAMES:
        sub = df[df["class"] == cls]
        if len(sub) > per_class:
            sub = sub.sample(n=per_class, random_state=seed)
        parts.append(sub)
    return (pd.concat(parts, ignore_index=True)
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))


def build_lc25000_split(df: pd.DataFrame, fractions=LC25000_SPLIT_FRACTIONS,
                        seed=SEED) -> pd.DataFrame:
    """Stratified train/val/test split, adding a ``split`` column."""
    from sklearn.model_selection import train_test_split

    train_frac, val_frac, test_frac = fractions
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1.0"

    train_df, rest = train_test_split(
        df, train_size=train_frac, stratify=df["class"], random_state=seed, shuffle=True)
    rel_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest, train_size=rel_val, stratify=rest["class"], random_state=seed, shuffle=True)

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out["dataset"] = "lc25000_lung"
    return out


def lc25000_split_counts(split_df: pd.DataFrame) -> pd.DataFrame:
    table = (split_df.pivot_table(index="class", columns="split", values="filepath",
                                  aggfunc="count", fill_value=0)
                     .reindex(LC25000_CLASS_NAMES, fill_value=0))
    ordered = [c for c in ("train", "val", "test") if c in table.columns]
    table = table[ordered]
    table["total"] = table.sum(axis=1)
    return table


def save_lc25000_split(split_df: pd.DataFrame, name="lc25000") -> Path:
    from src.config import SPLITS_DIR

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}_split.csv"
    split_df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# tf.data pipeline (3-class)
# --------------------------------------------------------------------------


def _decode(path, img_size):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # raw [0, 255]; the model rescales
    img.set_shape((img_size[0], img_size[1], 3))
    return img


def make_lc25000_dataset(df: pd.DataFrame, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                         shuffle=False, augment=False, seed=SEED, one_hot=False):
    """``(image[0..255], label)`` dataset. Same contract as the CT pipeline.

    ``one_hot=True`` produces **3**-wide targets - required whenever the model
    is compiled with label smoothing. This is why the CT ``make_dataset`` is not
    reused: it one-hots to 4 columns.
    """
    import tensorflow as tf

    if df.empty:
        raise ValueError("make_lc25000_dataset received an empty dataframe.")

    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    if one_hot:
        labels = tf.keras.utils.to_categorical(labels, num_classes=LC25000_NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode(p, img_size), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        from src.data_utils import get_augmenter     # class-count agnostic

        augmenter = get_augmenter(seed=seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def make_lc25000_datasets(split_df: pd.DataFrame, one_hot=True, augment_train=True,
                          img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED):
    """``(train_ds, val_ds, test_ds, frames)`` for a built split."""
    frames = {s: split_df[split_df["split"] == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    for s, f in frames.items():
        if f.empty:
            raise ValueError(f"Split '{s}' is empty - check the split definition.")
    train_ds = make_lc25000_dataset(frames["train"], img_size, batch_size, shuffle=True,
                                    augment=augment_train, seed=seed, one_hot=one_hot)
    val_ds = make_lc25000_dataset(frames["val"], img_size, batch_size, one_hot=one_hot)
    test_ds = make_lc25000_dataset(frames["test"], img_size, batch_size, one_hot=one_hot)
    return train_ds, val_ds, test_ds, frames


# --------------------------------------------------------------------------
# Metrics, collapse checks and reports (3-class)
# --------------------------------------------------------------------------


def lc25000_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / macro P-R-F1 / kappa / MCC from the shared implementation,
    plus a correctly 3-class macro AUC.

    ``compute_metrics`` is reused for everything class-count agnostic; only its
    AUC step hardcodes the CT dataset's 4 labels, so AUC is computed here.
    """
    metrics = compute_metrics(y_true, y_pred, y_prob=None)
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score

        try:
            metrics["auc_macro"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=list(range(LC25000_NUM_CLASSES))))
        except ValueError:
            metrics["auc_macro"] = float("nan")
    return metrics


def lc25000_detect_collapse(history=None, kappa=None, mcc=None, y_pred=None) -> dict:
    """Both collapse checks, with LC25000's 3 classes and names.

    Reuses the shared detector rather than reimplementing it:

    * ``detect_collapse(..., y_pred=None)`` supplies the flat-curve and
      kappa/MCC rules and the project's thresholds;
    * ``detect_partial_collapse(..., num_classes=3, class_names=...)`` supplies
      the dead-class rule;
    * the single-class rule is added here because it needs this dataset's names.

    Status values are the project's own ``COLLAPSE_TAG`` /
    ``PARTIAL_COLLAPSE_TAG`` / ``VALID_TAG``, so "valid run" means the same
    thing on both datasets.
    """
    base = detect_collapse(history=history, kappa=kappa, mcc=mcc, y_pred=None)
    reasons = list(base["reasons"])
    details = dict(base["details"])

    partial = {"partial": False, "reasons": [], "details": {}}
    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
        uniq = np.unique(y_pred)
        if len(uniq) == 1:
            reasons.append(
                f"model predicts a single class for every sample "
                f"(class {int(uniq[0])} = {LC25000_CLASS_NAMES[int(uniq[0])]})")
        partial = detect_partial_collapse(
            y_pred, num_classes=LC25000_NUM_CLASSES,
            min_predicted_classes=LC25000_NUM_CLASSES,
            class_names=LC25000_CLASS_NAMES)
        details.update(partial["details"])

    collapsed = len(reasons) > 0
    partial_only = partial["partial"] and not collapsed
    if partial_only:
        reasons = list(partial["reasons"])

    status = COLLAPSE_TAG if collapsed else (PARTIAL_COLLAPSE_TAG if partial_only
                                             else VALID_TAG)
    return {
        "collapsed": collapsed or partial_only,
        "fully_collapsed": collapsed,
        "partial_collapse": partial_only,
        "reasons": reasons,
        "details": details,
        "status": status,
    }


def lc25000_confusion(y_true, y_pred) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred,
                            labels=list(range(LC25000_NUM_CLASSES)))


def lc25000_per_class_report(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import classification_report

    rep = classification_report(y_true, y_pred,
                                labels=list(range(LC25000_NUM_CLASSES)),
                                target_names=LC25000_CLASS_NAMES,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def plot_lc25000_confusion(y_true, y_pred, run_name, normalize=False, save=True,
                           show=True, cmap="Blues"):
    """Annotated 3-class confusion matrix; flags all-zero columns like the
    primary pipeline's version does."""
    import matplotlib.pyplot as plt

    from src.config import FIGURES_DIR

    cm = lc25000_confusion(y_true, y_pred)
    dead = [LC25000_CLASS_NAMES[j] for j in range(LC25000_NUM_CLASSES)
            if cm[:, j].sum() == 0]

    display = cm.astype(float)
    if normalize:
        rows = display.sum(axis=1, keepdims=True)
        display = np.divide(display, np.maximum(rows, 1))

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(display, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(LC25000_NUM_CLASSES))
    ax.set_yticks(range(LC25000_NUM_CLASSES))
    ax.set_xticklabels(LC25000_CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LC25000_CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    title = f"{run_name} - confusion matrix"
    if normalize:
        title += " (row-normalised)"
    if dead:
        title += f"\nNEVER PREDICTED: {', '.join(dead)}"
    ax.set_title(title, color="crimson" if dead else "black")

    thresh = display.max() / 2 if display.max() else 0.5
    for i in range(LC25000_NUM_CLASSES):
        for j in range(LC25000_NUM_CLASSES):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black")

    fig.tight_layout()
    path = None
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_cm_norm" if normalize else "_cm"
        path = FIGURES_DIR / f"{LC25000_PREFIX}{run_name}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cm, path


def benign_vs_subtype_breakdown(y_true, y_pred) -> dict:
    """The LESSON 11 diagnostic, translated to this dataset.

    On the CT dataset every model detected tumour far better than it told the
    subtypes apart. The analogous split here is benign-vs-malignant detection
    versus adenocarcinoma-vs-squamous discrimination, so the two datasets can
    be compared on the thing that actually distinguished them.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    benign_idx = LC25000_CLASS_NAMES.index(LC25000_BENIGN_CLASS)

    true_mal = y_true != benign_idx
    pred_mal = y_pred != benign_idx
    binary_acc = float((true_mal == pred_mal).mean())

    subtype_acc = (float((y_true[true_mal] == y_pred[true_mal]).mean())
                   if true_mal.any() else float("nan"))
    both = true_mal & pred_mal
    subtype_given = (float((y_true[both] == y_pred[both]).mean())
                     if both.any() else float("nan"))

    pred_counts = pd.Series(y_pred).value_counts().reindex(
        range(LC25000_NUM_CLASSES), fill_value=0)
    true_counts = pd.Series(y_true).value_counts().reindex(
        range(LC25000_NUM_CLASSES), fill_value=0)
    over = pred_counts - true_counts
    over.index = LC25000_CLASS_NAMES

    return {
        "benign_vs_malignant_accuracy": binary_acc,
        "subtype_accuracy_all_malignant": subtype_acc,
        "subtype_accuracy_given_detected_malignant": subtype_given,
        "n_malignant_samples": int(true_mal.sum()),
        "over_prediction_by_class": over.to_dict(),
        "most_over_predicted_class": str(over.idxmax()),
    }


def interpret_lc25000_breakdown(breakdown: dict) -> str:
    b = breakdown["benign_vs_malignant_accuracy"]
    s = breakdown["subtype_accuracy_all_malignant"]
    worst = breakdown["most_over_predicted_class"]
    gap = b - s
    lines = [
        f"Benign-vs-malignant accuracy: {b:.4f}.",
        f"Subtype accuracy among true malignant samples: {s:.4f}.",
        f"Gap between the two tasks: {gap:.4f}.",
    ]
    if gap > 0.05:
        lines.append(
            f"As on the CT dataset, detection is the easier task and subtype "
            f"discrimination carries the error; '{worst}' is the most "
            f"over-predicted class.")
    else:
        lines.append(
            "Detection and subtype discrimination are of comparable quality here - "
            "unlike the CT dataset, where detection was far easier than subtyping.")
    return " ".join(lines)


# --------------------------------------------------------------------------
# Persistence - all files clearly prefixed, none of them shared with the CT run
# --------------------------------------------------------------------------


def save_lc25000_predictions(run_name, y_true, y_pred, y_prob=None, meta=None) -> Path:
    """Raw per-sample predictions to ``outputs/predictions/lc25000_<run>_...``.

    Written here rather than through ``evaluate_utils.save_predictions`` because
    that function names its probability columns from the CT dataset's class
    list. The ``lc25000_`` prefix guarantees no existing prediction file can be
    overwritten.
    """
    import json

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    frame = {"y_true": y_true, "y_pred": y_pred}
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        for i in range(min(y_prob.shape[1], LC25000_NUM_CLASSES)):
            frame[f"prob_{LC25000_CLASS_NAMES[i]}"] = y_prob[:, i]

    stem = f"{LC25000_PREFIX}{run_name}"
    path = PREDICTIONS_DIR / f"{stem}_predictions.csv"
    pd.DataFrame(frame).to_csv(path, index=False)

    sidecar = dict(meta or {})
    sidecar.update({
        "run_name": stem,
        "dataset": "LC25000 (lung subset, 3 classes)",
        "n_samples": int(len(y_true)),
        "class_names": LC25000_CLASS_NAMES,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    })
    with open(PREDICTIONS_DIR / f"{stem}_predictions_meta.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)
    return path


def load_lc25000_predictions(run_name):
    """``(y_true, y_pred, y_prob or None)`` for a saved LC25000 run."""
    stem = run_name if run_name.startswith(LC25000_PREFIX) else LC25000_PREFIX + run_name
    path = PREDICTIONS_DIR / f"{stem}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - was this run trained?")
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    y_prob = df[prob_cols].to_numpy(dtype=float) if prob_cols else None
    return df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int), y_prob


def init_lc25000_results(overwrite=False) -> Path:
    """Create ``outputs/results_table_lc25000.csv`` with headers only."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if overwrite or not LC25000_RESULTS_CSV.exists():
        pd.DataFrame(columns=LC25000_RESULTS_COLUMNS).to_csv(
            LC25000_RESULTS_CSV, index=False)
    return LC25000_RESULTS_CSV


def record_lc25000_result(row: dict) -> Path:
    """One row per model in the LC25000 table; re-running a model replaces it.

    Deliberately a separate file from ``results_table.csv`` - the CT results are
    final and this experiment must not be able to touch them. Unknown column
    names raise, same as the primary router.
    """
    unknown = set(row) - set(LC25000_RESULTS_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column(s): {sorted(unknown)}. "
                       f"Allowed: {LC25000_RESULTS_COLUMNS}")
    if not row.get("model"):
        raise ValueError("LC25000 result rows require a 'model' name.")

    row = dict(row)
    row.setdefault("dataset", "lc25000_lung")
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("status", VALID_TAG)

    init_lc25000_results()
    table = pd.read_csv(LC25000_RESULTS_CSV)
    for c in LC25000_RESULTS_COLUMNS:
        if c not in table.columns:
            table[c] = pd.NA
    table = table[LC25000_RESULTS_COLUMNS]
    if len(table):
        table = table[table["model"].astype(str) != str(row["model"])]
    table = pd.concat([table, pd.DataFrame([row])],
                      ignore_index=True)[LC25000_RESULTS_COLUMNS]
    table.to_csv(LC25000_RESULTS_CSV, index=False)
    return LC25000_RESULTS_CSV


def load_lc25000_results() -> pd.DataFrame:
    if not LC25000_RESULTS_CSV.exists():
        return pd.DataFrame(columns=LC25000_RESULTS_COLUMNS)
    return pd.read_csv(LC25000_RESULTS_CSV)


def lc25000_result_row(model_name, metrics, collapse, breakdown=None, params=None,
                       arch_variant=None, n_train=None, n_test=None,
                       epochs_trained=None, notes="") -> dict:
    """Assemble a results row; ``status`` comes from the collapse check."""
    def _r(v):
        return round(v, 6) if v is not None else None

    return {
        "model": model_name,
        "arch_variant": arch_variant,
        "dataset": "lc25000_lung",
        "params": params,
        "accuracy": _r(metrics.get("accuracy")),
        "precision_macro": _r(metrics.get("precision_macro")),
        "recall_macro": _r(metrics.get("recall_macro")),
        "f1_macro": _r(metrics.get("f1_macro")),
        "cohen_kappa": _r(metrics.get("cohen_kappa")),
        "mcc": _r(metrics.get("mcc")),
        "auc_macro": _r(metrics.get("auc_macro")),
        "benign_vs_malignant_acc": _r(breakdown.get("benign_vs_malignant_accuracy")
                                      if breakdown else None),
        "subtype_acc": _r(breakdown.get("subtype_accuracy_all_malignant")
                          if breakdown else None),
        "n_train": n_train,
        "n_test": n_test,
        "epochs_trained": epochs_trained,
        "n_runs": 1,
        "status": collapse["status"] if collapse else VALID_TAG,
        "notes": notes,
    }
