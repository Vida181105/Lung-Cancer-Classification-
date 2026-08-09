"""Evaluation: collapse detection, metrics, confusion matrices, results routing.

Two things in this file exist because of specific failures in the earlier
attempt:

* :func:`detect_collapse` (LESSON 3) - a dead network with a frozen output
  looks *stable* (zero variance across epochs) and can be mistaken for a good,
  consistent result. Every training run in this project is checked, and a
  collapsed run is tagged ``INVALID_collapsed`` instead of being reported as a
  real number.
* :func:`record_result` (LESSON 7) - routes a result row to the canonical
  table, the experiment log, or the ablation table, so nobody has to remember
  which CSV a run belongs in.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ABLATION_CSV,
    CLASS_NAMES,
    COLLAPSE_KAPPA_TOL,
    COLLAPSE_STD_TOL,
    COLLAPSE_TAG,
    COLLAPSE_WINDOW,
    EXPERIMENTS_LOG_CSV,
    FIGURES_DIR,
    NORMAL_CLASS,
    NUM_CLASSES,
    OUTPUT_ROOT,
    RESULTS_TABLE_CSV,
    VALID_TAG,
)

# --------------------------------------------------------------------------
# Collapse detection (LESSON 3)
# --------------------------------------------------------------------------


def detect_collapse(history=None, kappa=None, mcc=None, y_pred=None,
                    window=COLLAPSE_WINDOW, std_tol=COLLAPSE_STD_TOL,
                    kappa_tol=COLLAPSE_KAPPA_TOL) -> dict:
    """Flag a run whose network died instead of learning.

    A run is collapsed if ANY of the following holds:

    1. training loss is flat (std < ``std_tol``) over the trailing ``window``
       epochs, or training accuracy is flat over the same window;
    2. Cohen's kappa or MCC is ~0.0 on evaluation (chance-level agreement);
    3. the model predicts a single class for every test sample.

    Returns ``{"collapsed": bool, "reasons": [...], "details": {...},
    "status": "ok" | "INVALID_collapsed"}``.
    """
    reasons, details = [], {}

    if history is not None:
        hist = getattr(history, "history", history)
        for key in ("loss", "accuracy"):
            values = hist.get(key)
            if not values:
                continue
            tail = np.asarray(values[-window:], dtype=float)
            if len(tail) < min(window, 3):
                continue
            std = float(tail.std())
            details[f"{key}_tail_std"] = std
            details[f"{key}_tail_n"] = int(len(tail))
            if std < std_tol:
                reasons.append(
                    f"train {key} flat over last {len(tail)} epochs "
                    f"(std={std:.2e} < {std_tol:.1e})"
                )

        val_acc = hist.get("val_accuracy")
        if val_acc:
            tail = np.asarray(val_acc[-window:], dtype=float)
            details["val_accuracy_tail_std"] = float(tail.std())

    if kappa is not None:
        details["cohen_kappa"] = float(kappa)
        if abs(float(kappa)) < kappa_tol:
            reasons.append(f"Cohen's kappa ~ 0 ({float(kappa):.4f})")

    if mcc is not None:
        details["mcc"] = float(mcc)
        if abs(float(mcc)) < kappa_tol:
            reasons.append(f"MCC ~ 0 ({float(mcc):.4f})")

    if y_pred is not None:
        uniq = np.unique(np.asarray(y_pred))
        details["n_predicted_classes"] = int(len(uniq))
        if len(uniq) == 1:
            reasons.append(
                f"model predicts a single class for every sample "
                f"(class {int(uniq[0])} = {CLASS_NAMES[int(uniq[0])]})"
            )

    collapsed = len(reasons) > 0
    return {
        "collapsed": collapsed,
        "reasons": reasons,
        "details": details,
        "status": COLLAPSE_TAG if collapsed else VALID_TAG,
    }


def print_collapse_report(result: dict, run_name=""):
    """Loud, unmissable stdout summary - call after every training run."""
    header = f"COLLAPSE CHECK{' [' + run_name + ']' if run_name else ''}"
    print(header)
    print("-" * len(header))
    if result["collapsed"]:
        print(f"  !!! COLLAPSED -> status = {result['status']}")
        for r in result["reasons"]:
            print(f"      - {r}")
        print("  This run's metrics are NOT a valid result. Do not report them.")
    else:
        print("  PASS - no collapse detected.")
    for k, v in result["details"].items():
        print(f"  {k}: {v}")
    return result


# --------------------------------------------------------------------------
# Prediction + metrics
# --------------------------------------------------------------------------


def predict(model, dataset):
    """Return ``(y_true, y_pred, y_prob)`` for an unshuffled dataset."""
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.concatenate([np.asarray(y) for _, y in dataset.as_numpy_iterator()])
    if y_true.ndim > 1:                      # one-hot
        y_true = np.argmax(y_true, axis=1)
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Label/prediction length mismatch ({len(y_true)} vs {len(y_pred)}). "
            "The evaluation dataset must not be shuffled or drop_remainder'd."
        )
    return y_true.astype(int), y_pred.astype(int), y_prob


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy, macro/weighted P-R-F1, Cohen's kappa, MCC and macro AUC."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            metrics["auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                              labels=list(range(NUM_CLASSES)))
            )
        except ValueError:
            metrics["auc_macro"] = float("nan")   # a class missing from y_true
    return metrics


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import classification_report

    rep = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    return pd.DataFrame(rep).T


def confusion(y_true, y_pred) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))


def plot_confusion_matrix(y_true, y_pred, run_name, normalize=False,
                          save=True, show=True, cmap="Blues"):
    """Annotated confusion matrix saved to ``outputs/figures``."""
    import matplotlib.pyplot as plt

    cm = confusion(y_true, y_pred)
    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, np.maximum(row_sums, 1))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(display, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    title = f"{run_name} - confusion matrix"
    ax.set_title(title + (" (row-normalised)" if normalize else ""))

    thresh = display.max() / 2 if display.max() else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black")

    fig.tight_layout()
    path = None
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_cm_norm" if normalize else "_cm"
        path = FIGURES_DIR / f"{run_name}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cm, path


# --------------------------------------------------------------------------
# Tumour-vs-healthy / subtype breakdown (LESSON 9)
# --------------------------------------------------------------------------


def tumor_vs_subtype_breakdown(y_true, y_pred) -> dict:
    """Separate the two very different jobs the model is doing.

    * ``binary_accuracy``  - tumour vs healthy, collapsing all three tumour
      classes into one. In the earlier attempt this was ~97%.
    * ``subtype_accuracy`` - among samples that are genuinely tumour AND were
      correctly recognised as tumour, how often the subtype is right. This is
      where the gap versus the paper actually lives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    normal_idx = CLASS_NAMES.index(NORMAL_CLASS)

    true_tumor = y_true != normal_idx
    pred_tumor = y_pred != normal_idx

    binary_acc = float((true_tumor == pred_tumor).mean())

    tumor_mask = true_tumor
    subtype_acc = (float((y_true[tumor_mask] == y_pred[tumor_mask]).mean())
                   if tumor_mask.any() else float("nan"))

    both_tumor = tumor_mask & pred_tumor
    subtype_acc_given_tumor = (float((y_true[both_tumor] == y_pred[both_tumor]).mean())
                               if both_tumor.any() else float("nan"))

    # Which class absorbs the confusions (earlier attempt: adenocarcinoma).
    pred_counts = pd.Series(y_pred).value_counts().reindex(
        range(NUM_CLASSES), fill_value=0)
    true_counts = pd.Series(y_true).value_counts().reindex(
        range(NUM_CLASSES), fill_value=0)
    over_pred = (pred_counts - true_counts)
    over_pred.index = CLASS_NAMES

    return {
        "binary_tumor_vs_healthy_accuracy": binary_acc,
        "subtype_accuracy_all_tumors": subtype_acc,
        "subtype_accuracy_given_detected_tumor": subtype_acc_given_tumor,
        "n_tumor_samples": int(tumor_mask.sum()),
        "over_prediction_by_class": over_pred.to_dict(),
        "most_over_predicted_class": str(over_pred.idxmax()),
    }


def interpret_breakdown(breakdown: dict) -> str:
    """One-paragraph written interpretation, saved into the report."""
    b = breakdown["binary_tumor_vs_healthy_accuracy"]
    s = breakdown["subtype_accuracy_all_tumors"]
    worst = breakdown["most_over_predicted_class"]
    gap = b - s
    lines = [
        f"Tumour-vs-healthy accuracy: {b:.4f}.",
        f"Subtype accuracy among true tumour samples: {s:.4f}.",
        f"Gap between the two tasks: {gap:.4f}.",
    ]
    if gap > 0.05:
        lines.append(
            f"The model separates tumour from healthy far better than it "
            f"separates the three tumour subtypes, so the headline accuracy "
            f"gap versus the paper is a subtype-discrimination problem, not a "
            f"detection problem. '{worst}' is the most over-predicted class "
            f"and absorbs most of the subtype confusions."
        )
    else:
        lines.append(
            "Detection and subtype discrimination are of comparable quality, "
            "so the error is spread across both tasks rather than being a "
            "subtype-only problem."
        )
    return " ".join(lines)


# --------------------------------------------------------------------------
# Cross-validation aggregation (LESSON 8)
# --------------------------------------------------------------------------


def summarize_cv(fold_metrics, keys=("accuracy", "f1_macro", "cohen_kappa",
                                     "mcc", "precision_macro", "recall_macro")):
    """mean / std across folds. Collapsed folds must be excluded by the caller
    (or passed through so the summary is visibly invalid)."""
    df = pd.DataFrame(list(fold_metrics))
    out = {"n_folds": int(len(df))}
    for k in keys:
        if k in df.columns:
            out[f"{k}_mean"] = float(df[k].mean())
            out[f"{k}_std"] = float(df[k].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def format_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


# --------------------------------------------------------------------------
# Results routing (LESSON 7)
# --------------------------------------------------------------------------

RESULTS_COLUMNS = [
    "model", "arch_variant", "split_variant", "params", "accuracy",
    "accuracy_std", "precision_macro", "recall_macro", "f1_macro",
    "cohen_kappa", "mcc", "auc_macro", "epochs_trained", "n_runs", "status",
    "notes", "timestamp",
]

EXPERIMENTS_COLUMNS = RESULTS_COLUMNS + ["config_note"]

ABLATION_COLUMNS = [
    "run_name", "arch_variant", "split_variant", "dropout_enabled",
    "dropout_rate", "accuracy", "f1_macro", "cohen_kappa", "mcc",
    "epochs_trained", "status", "notes", "timestamp",
]

_KIND_SPEC = {
    "canonical": (RESULTS_TABLE_CSV, RESULTS_COLUMNS, "model"),
    "experiment": (EXPERIMENTS_LOG_CSV, EXPERIMENTS_COLUMNS, None),
    "ablation": (ABLATION_CSV, ABLATION_COLUMNS, "run_name"),
}


def init_results_files(overwrite=False):
    """Create the three CSVs with headers only. Never writes numbers."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    created = {}
    for kind, (path, columns, _) in _KIND_SPEC.items():
        if overwrite or not Path(path).exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            created[kind] = str(path)
    return created


def record_result(row: dict, kind="experiment", replace_existing=True) -> Path:
    """Write one result row to the correct CSV.

    ``kind='canonical'``  -> ``outputs/results_table.csv``   (one row per model;
                             an existing row for the same ``model`` is replaced)
    ``kind='experiment'`` -> ``outputs/experiments_log.csv`` (append-only;
                             ``config_note`` is required)
    ``kind='ablation'``   -> ``outputs/ablation_dropout.csv`` (one row per
                             ``run_name``)

    Unknown keys raise, so a typo can never silently vanish into an
    unwritten column.
    """
    if kind not in _KIND_SPEC:
        raise ValueError(f"kind must be one of {list(_KIND_SPEC)}, got '{kind}'.")
    path, columns, key = _KIND_SPEC[kind]

    unknown = set(row) - set(columns)
    if unknown:
        raise KeyError(f"Unknown column(s) for kind='{kind}': {sorted(unknown)}. "
                       f"Allowed: {columns}")
    if kind == "experiment" and not row.get("config_note"):
        raise ValueError("experiment rows require a non-empty 'config_note'.")
    if kind == "canonical" and not row.get("model"):
        raise ValueError("canonical rows require a 'model' name.")

    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("status", VALID_TAG)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if Path(path).exists():
        table = pd.read_csv(path)
        for c in columns:                      # tolerate an older header
            if c not in table.columns:
                table[c] = pd.NA
        table = table[columns]
    else:
        table = pd.DataFrame(columns=columns)

    if key and replace_existing and key in row and len(table):
        table = table[table[key].astype(str) != str(row[key])]

    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[columns]
    table.to_csv(path, index=False)
    return Path(path)


def record_canonical(row: dict) -> Path:
    """Final, reportable result for one model - goes in the main table."""
    return record_result(row, kind="canonical")


def record_experiment(row: dict) -> Path:
    """Any debugging / exploratory / non-headline run."""
    return record_result(row, kind="experiment")


def record_ablation(row: dict) -> Path:
    """A row of the dropout-on vs dropout-off comparison."""
    return record_result(row, kind="ablation")


def result_row_from_metrics(model_name, metrics, collapse, arch_variant=None,
                            split_variant=None, params=None, epochs_trained=None,
                            n_runs=1, notes="", accuracy_std=None,
                            config_note=None) -> dict:
    """Assemble a CSV row from a metrics dict + a collapse-check result.

    ``status`` is taken from the collapse check, so a collapsed run is written
    as ``INVALID_collapsed`` and can never be mistaken for a real number.
    """
    row = {
        "model": model_name,
        "arch_variant": arch_variant,
        "split_variant": split_variant,
        "params": params,
        "accuracy": round(metrics.get("accuracy", float("nan")), 6),
        "accuracy_std": (round(accuracy_std, 6) if accuracy_std is not None else None),
        "precision_macro": round(metrics.get("precision_macro", float("nan")), 6),
        "recall_macro": round(metrics.get("recall_macro", float("nan")), 6),
        "f1_macro": round(metrics.get("f1_macro", float("nan")), 6),
        "cohen_kappa": round(metrics.get("cohen_kappa", float("nan")), 6),
        "mcc": round(metrics.get("mcc", float("nan")), 6),
        "auc_macro": (round(metrics["auc_macro"], 6)
                      if metrics.get("auc_macro") is not None else None),
        "epochs_trained": epochs_trained,
        "n_runs": n_runs,
        "status": collapse["status"] if collapse else VALID_TAG,
        "notes": notes,
    }
    if config_note is not None:
        row["config_note"] = config_note
    return row


def load_results(kind="canonical") -> pd.DataFrame:
    path, columns, _ = _KIND_SPEC[kind]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def valid_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows tagged ``INVALID_collapsed`` before ranking or plotting."""
    if "status" not in df.columns or df.empty:
        return df
    return df[df["status"].astype(str) != COLLAPSE_TAG]
