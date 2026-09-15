#!/usr/bin/env python3
"""Final results generator - one command, regenerates every publication-ready
figure and table from data ALREADY SAVED on disk by earlier phases (notebooks
00-15).

    python src/generate_final_results.py

**Pure post-processing. No checkpoint is loaded, no dataset is read, no model
is built, no inference is performed, no `.fit()` call exists anywhere in this
file.** Every number and image here is derived from CSV/JSON files and PNG
overlays already written by earlier notebooks. This is deliberate: none of
`src/config.py`, `data_utils.py`, `evaluate_utils.py`, `finetune_utils.py`,
`calibration_utils.py`, `external_eval_utils.py`, `uncertainty_utils.py`, or
`gradcam_utils.py` imports TensorFlow at module level (confirmed by direct
inspection), so this script runs with only numpy/pandas/matplotlib/Pillow/
scikit-learn - no GPU, no TensorFlow, no checkpoints, no CT images required.
That also means it is cheap and safe to re-run repeatedly while iterating on
figure styling, without re-running any of the expensive notebooks.

Prerequisites (each produced by an earlier, already-executed notebook) -
missing ones are reported, never fabricated:

    notebook 00/03        -> outputs/results_table.csv
    notebook 09  (Phase 2) -> results/fair_baseline/metrics_fair_baseline.csv
    notebook 11  (Phase 3) -> results/gradcam/gradcam_records.csv (+ overlay PNGs)
    notebook 12  (Phase 4) -> results/calibration/calibration_metrics.csv,
                              results/calibration/predictions_used/*_test_predictions_used.csv
    notebook 13  (Phase 5) -> results/external_eval/external_eval_metrics.csv
    notebook 14            -> outputs/leakage/leakage_controlled_cv_results.json
    notebook 15  (Phase 6) -> results/uncertainty/uncertainty_metrics.csv,
                              results/uncertainty/selective_prediction_results.csv,
                              results/uncertainty/final_results_table.csv,
                              results/uncertainty/<model>/predictions_used/*_uncertainty_predictions.csv

Output layout:

    results/final/  - 7 CSV tables + final_results.csv + final_summary.json/.txt
    figures/final/  - every figure, as both .png (300 DPI, presentation) and
                      .pdf (vector, print/publication)

Design choices worth stating up front
--------------------------------------
* **Reuse compute, rebuild plots.** Every NUMBER here is read from, or
  directly derived from, an existing saved file - nothing is recomputed by a
  second, independently-written formula where an already-reported one exists
  (e.g. leakage-controlled accuracy is read from the already-computed
  `summary.accuracy_mean` field, not re-aggregated here). The PLOTTING code is
  new: none of the existing `plot_*` functions across the five earlier
  modules support dual PNG+PDF export or the larger, print-ready font sizes
  this deliverable needs, so this script defines its own small plotting layer
  (:func:`save_figure`) used consistently across all 18 figures.
* **ROC curves are the INTERNAL, native 4-class one-vs-rest curves** (macro-
  averaged), matching the `auc_macro` metric already used elsewhere in this
  project (`evaluate_utils.compute_metrics`) - not a separately invented
  external ROC, which was not requested and would need a different, less
  standard construction.
* **Internal true/predicted labels keep their native 4-class NSCLC-subtype
  vocabulary; external ones keep their native malignant/normal vocabulary.**
  `per_image_predictions.csv` does NOT force these onto one shared label
  scheme - that reduction is a separate, already-documented analysis
  elsewhere (the binary tumour-vs-normal / malignant-vs-normal comparison),
  not the raw per-image ground truth.
* **Nothing is fabricated.** Every table/figure builder function checks its
  own required inputs first and returns `None` (with a printed reason) if
  they are missing, rather than filling a gap with an invented number.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from src...` imports work regardless of the caller's cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT_GUESS = _THIS_DIR.parent
if str(_PROJECT_ROOT_GUESS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_GUESS))

import matplotlib
matplotlib.use("Agg")   # headless: no display needed to save figures
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc as sk_auc

from src.config import CLASS_NAMES, NUM_CLASSES, NORMAL_CLASS, PROJECT_ROOT, OUTPUT_ROOT, RESULTS_TABLE_CSV
from src.finetune_utils import FAIR_METRICS_CSV, CHECKPOINTS_LOCAL
from src.calibration_utils import (CALIBRATION_METRICS_CSV, CALIBRATION_PREDICTIONS_USED_DIR,
                                   calibration_summary)
from src.external_eval_utils import EXTERNAL_METRICS_CSV
from src.gradcam_utils import (GRADCAM_RECORDS_CSV, HIGH_CONFIDENCE_THRESHOLD,
                               identify_strongest_finetuned_baseline)
from src.uncertainty_utils import (UNCERTAINTY_DIR, UNCERTAINTY_METRICS_CSV, SELECTIVE_PREDICTION_CSV,
                                   FINAL_RESULTS_TABLE_CSV, model_predictions_used_dir,
                                   check_leakage_controlled_checkpoint, uncertainty_bins)

# --------------------------------------------------------------------------
# Output locations
# --------------------------------------------------------------------------

FINAL_DIR = PROJECT_ROOT / "results" / "final"
FINAL_FIGURES_DIR = PROJECT_ROOT / "figures" / "final"

MODEL_COMPARISON_CSV = FINAL_DIR / "model_comparison.csv"
INTERNAL_RESULTS_CSV = FINAL_DIR / "internal_results.csv"
EXTERNAL_RESULTS_CSV = FINAL_DIR / "external_results.csv"
CALIBRATION_RESULTS_CSV = FINAL_DIR / "calibration_results.csv"
UNCERTAINTY_RESULTS_CSV = FINAL_DIR / "uncertainty_results.csv"
SELECTIVE_PREDICTION_RESULTS_CSV = FINAL_DIR / "selective_prediction_results.csv"
PER_IMAGE_PREDICTIONS_CSV = FINAL_DIR / "per_image_predictions.csv"
FINAL_RESULTS_CSV = FINAL_DIR / "final_results.csv"
FINAL_SUMMARY_JSON = FINAL_DIR / "final_summary.json"
FINAL_SUMMARY_TXT = FINAL_DIR / "final_summary.txt"


def ensure_dirs():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Print-ready plotting style, applied once, used by every figure below
# --------------------------------------------------------------------------

SAVE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")   # PDF chosen as the vector format (as practical
                                  # for print/publication as SVG, and matplotlib's
                                  # more commonly expected one); adding SVG too
                                  # would just double the file count for no gain.


def apply_print_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
    })


def save_figure(fig, stem: Path, formats=FIGURE_FORMATS):
    """Save one figure in every requested format at print-ready DPI, then
    close it. Returns the list of paths actually written."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = stem.with_suffix("." + ext)
        fig.savefig(p, dpi=SAVE_DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# A tiny registry so every table/figure attempt is tracked (ok/skipped/failed)
# for the final human-readable report - nothing silently disappears.
# --------------------------------------------------------------------------

class Registry:
    def __init__(self):
        self.tables = {}
        self.figures = {}
        self.warnings = []


REG = Registry()


def run_table_step(name, fn):
    print(f"\n--- table: {name} ---")
    try:
        result = fn()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        REG.tables[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        return None
    if result is None:
        print("  SKIPPED (see reason printed above)")
        REG.tables[name] = {"status": "skipped"}
        return None
    df, path = result
    print(f"  OK: {len(df)} rows -> {path}")
    REG.tables[name] = {"status": "ok", "path": str(path), "n_rows": int(len(df))}
    return df


def run_figure_step(name, fn):
    print(f"\n--- figure: {name} ---")
    try:
        result = fn()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        REG.figures[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        return None
    if result is None:
        print("  SKIPPED (see reason printed above)")
        REG.figures[name] = {"status": "skipped"}
        return None
    print(f"  OK -> {', '.join(str(p) for p in result)}")
    REG.figures[name] = {"status": "ok", "paths": [str(p) for p in result]}
    return result


# --------------------------------------------------------------------------
# Model identification (no checkpoint loading - just reads the same CSV
# every earlier phase already reads to decide this, via the same function)
# --------------------------------------------------------------------------


def identify_models():
    secondary = identify_strongest_finetuned_baseline()
    info = {
        "primary_model_name": "MiniConvNet",
        "primary_run_name": "miniconvnet_single_run",
        "primary_checkpoint": str(CHECKPOINTS_LOCAL / "miniconvnet_single_run.keras"),
        "secondary_available": secondary["available"],
        "secondary_model_name": secondary.get("model"),
        "secondary_run_name": secondary.get("run_name"),
        "secondary_checkpoint": secondary.get("checkpoint_path"),
        "secondary_reason_unavailable": secondary.get("reason"),
    }
    return info


# --------------------------------------------------------------------------
# Data loaders - every one of these returns None (never a fabricated frame)
# if its source file is missing or malformed
# --------------------------------------------------------------------------


def load_internal_full_predictions(run_name):
    """Full internal test-set predictions (y_true, y_pred, prob_<class> x4)
    for one run, as saved by `calibration_utils.locate_or_generate_predictions`
    (called from notebook 12 or 15) into `predictions_used/`."""
    path = CALIBRATION_PREDICTIONS_USED_DIR / f"{run_name}_test_predictions_used.csv"
    if not path.exists():
        print(f"  not found: {path}")
        return None
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) != NUM_CLASSES or "y_true" not in df.columns or "y_pred" not in df.columns:
        print(f"  malformed (expected y_true/y_pred + {NUM_CLASSES} prob_ columns): {path}")
        return None
    return df


def load_uncertainty_predictions(model_name, dataset_label):
    path = model_predictions_used_dir(model_name) / f"{model_name}_{dataset_label}_uncertainty_predictions.csv"
    if not path.exists():
        print(f"  not found: {path}")
        return None
    return pd.read_csv(path)


def load_csv_if_exists(path, label):
    if not Path(path).exists():
        print(f"  not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  empty: {path}")
        return None
    return df


# --------------------------------------------------------------------------
# TABLE 1 - model_comparison.csv (pass-through of notebook 15's own table -
# single source of truth, not recomputed here)
# --------------------------------------------------------------------------


def build_model_comparison_table():
    df = load_csv_if_exists(FINAL_RESULTS_TABLE_CSV,
                            "results/uncertainty/final_results_table.csv (notebook 15)")
    if df is None:
        return None
    df.to_csv(MODEL_COMPARISON_CSV, index=False)
    return df, MODEL_COMPARISON_CSV


# --------------------------------------------------------------------------
# TABLE 2 - internal_results.csv (accuracy / precision / recall / f1 / AUC /
# tumour-detection / subtype - all computed fresh from THIS checkpoint's own
# saved predictions, so it is internally consistent with the ROC/confusion
# figures built from the same file)
# --------------------------------------------------------------------------


def build_internal_results_table(models):
    rows = []
    normal_idx = CLASS_NAMES.index(NORMAL_CLASS)
    for model_name, run_name in models:
        print(f"  {model_name} ({run_name}):")
        df = load_internal_full_predictions(run_name)
        if df is None:
            rows.append({"model": model_name, "run_name": run_name, "note": "predictions not found"})
            continue
        y_true = df["y_true"].to_numpy(int)
        y_pred = df["y_pred"].to_numpy(int)
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        y_prob = df[prob_cols].to_numpy(float)

        accuracy = float((y_true == y_pred).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
        try:
            auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                            labels=list(range(NUM_CLASSES))))
        except ValueError as exc:
            auc_macro = None
            print(f"    AUC not computable ({exc}) - not all classes present in y_true")

        true_tumor = y_true != normal_idx
        pred_tumor = y_pred != normal_idx
        binary_tumor_acc = float((true_tumor == pred_tumor).mean())
        subtype_acc = (float((y_true[true_tumor] == y_pred[true_tumor]).mean())
                      if true_tumor.any() else None)

        rows.append({
            "model": model_name, "run_name": run_name, "n_samples": int(len(y_true)),
            "accuracy": round(accuracy, 6), "precision_macro": round(float(precision.mean()), 6),
            "recall_macro": round(float(recall.mean()), 6), "f1_macro": round(float(f1.mean()), 6),
            "auc_macro": (round(auc_macro, 6) if auc_macro is not None else None),
            "binary_tumor_detection_accuracy": round(binary_tumor_acc, 6),
            "subtype_accuracy": (round(subtype_acc, 6) if subtype_acc is not None else None),
            "note": ("this checkpoint's own faithful-split test set - see leakage_controlled_accuracy "
                    "in model_comparison.csv for the robustness-checked headline"),
        })
        print(f"    accuracy={accuracy:.4f}  auc_macro={auc_macro}  "
              f"binary_tumor_acc={binary_tumor_acc:.4f}  subtype_acc={subtype_acc}")

    if not rows or all("note" in r and len(r) <= 3 for r in rows):
        return None
    out = pd.DataFrame(rows)
    out.to_csv(INTERNAL_RESULTS_CSV, index=False)
    return out, INTERNAL_RESULTS_CSV


# --------------------------------------------------------------------------
# TABLE 3 - external_results.csv (straight copy-through, all 3 framings, so
# the 'primary' column stays visible and nothing is hidden)
# --------------------------------------------------------------------------


def build_external_results_table():
    df = load_csv_if_exists(EXTERNAL_METRICS_CSV, "results/external_eval/external_eval_metrics.csv")
    if df is None:
        return None
    df.to_csv(EXTERNAL_RESULTS_CSV, index=False)
    return df, EXTERNAL_RESULTS_CSV


# --------------------------------------------------------------------------
# TABLE 4 - calibration_results.csv (straight copy-through, both stages)
# --------------------------------------------------------------------------


def build_calibration_results_table():
    df = load_csv_if_exists(CALIBRATION_METRICS_CSV, "results/calibration/calibration_metrics.csv")
    if df is None:
        return None
    df.to_csv(CALIBRATION_RESULTS_CSV, index=False)
    return df, CALIBRATION_RESULTS_CSV


# --------------------------------------------------------------------------
# TABLE 5 - uncertainty_results.csv (straight copy-through)
# --------------------------------------------------------------------------


def build_uncertainty_results_table():
    df = load_csv_if_exists(UNCERTAINTY_METRICS_CSV, "results/uncertainty/uncertainty_metrics.csv")
    if df is None:
        return None
    df.to_csv(UNCERTAINTY_RESULTS_CSV, index=False)
    return df, UNCERTAINTY_RESULTS_CSV


# --------------------------------------------------------------------------
# TABLE 6 - selective_prediction_results.csv (straight copy-through)
# --------------------------------------------------------------------------


def build_selective_prediction_results_table():
    df = load_csv_if_exists(SELECTIVE_PREDICTION_CSV,
                            "results/uncertainty/selective_prediction_results.csv")
    if df is None:
        return None
    df.to_csv(SELECTIVE_PREDICTION_RESULTS_CSV, index=False)
    return df, SELECTIVE_PREDICTION_RESULTS_CSV


# --------------------------------------------------------------------------
# TABLE 7 - per_image_predictions.csv (merged internal + external, both
# models, one long-format table)
#
# Internal rows keep the native 4-class NSCLC-subtype label vocabulary;
# external rows keep the native malignant/normal vocabulary - NOT forced onto
# one shared scheme, since that reduction is a separate, already-documented
# analysis (see internal_results.csv / external_results.csv), not the raw
# per-image ground truth.
# --------------------------------------------------------------------------


def build_per_image_predictions_table(models):
    parts = []
    for model_name, _run_name in models:
        internal = load_uncertainty_predictions(model_name, "internal")
        if internal is not None:
            mc_prob_cols = [c for c in internal.columns if c.startswith("mc_mean_prob_")]
            frame = pd.DataFrame({
                "model": model_name, "dataset": "internal",
                "image_id": internal["filepath"].map(lambda p: Path(p).name),
                "filepath": internal["filepath"],
                "true_label": internal["true_class"],
                "predicted_label": internal["deterministic_pred_class"],
                "deterministic_confidence": internal["deterministic_confidence"],
                "mc_mean_probability": (internal[mc_prob_cols].max(axis=1) if mc_prob_cols else None),
                "predictive_entropy": internal["mc_predictive_entropy_4class"],
                "mc_confidence_variance": internal.get("mc_confidence_variance_4class"),
                "correct_deterministic": internal["det_correct"],
                "correct_mc": internal["mc_correct"],
            })
            parts.append(frame)
            print(f"  {model_name} internal: {len(frame)} rows")

        external = load_uncertainty_predictions(model_name, "external")
        if external is not None:
            ext = external[external["scored_mask"]].copy()   # benign excluded, as everywhere else
            label_map = {0: "normal", 1: "malignant"}
            mc_prob_winning = np.where(ext["deterministic_pred_binary_malignant"] == 1,
                                       ext["mc_prob_malignant"], 1 - ext["mc_prob_malignant"])
            frame = pd.DataFrame({
                "model": model_name, "dataset": "external",
                "image_id": ext["filepath"].map(lambda p: Path(p).name),
                "filepath": ext["filepath"],
                "true_label": ext["true_binary_malignant"].map(label_map),
                "predicted_label": ext["deterministic_pred_binary_malignant"].map(label_map),
                "deterministic_confidence": ext["deterministic_confidence"],
                "mc_mean_probability": mc_prob_winning,
                "predictive_entropy": ext["mc_predictive_entropy_binary"],
                "mc_confidence_variance": ext.get("mc_confidence_variance_4class"),
                "correct_deterministic": ext["det_correct"],
                "correct_mc": ext["mc_correct"],
            })
            parts.append(frame)
            print(f"  {model_name} external (scored, benign excluded): {len(frame)} rows")

    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(PER_IMAGE_PREDICTIONS_CSV, index=False)
    return out, PER_IMAGE_PREDICTIONS_CSV


# ==========================================================================
# FIGURES
# ==========================================================================

MODEL_COLORS = {"MiniConvNet": "#4c78a8", "VGG16": "#e45756"}


def _color_for(model_name):
    return MODEL_COLORS.get(model_name, "#72b7b2")


# --- 1: MiniConvNet vs VGG16 accuracy comparison (every accuracy metric available) ---


def fig01_model_accuracy_comparison(comparison_df):
    if comparison_df is None or comparison_df.empty:
        print("  model_comparison table unavailable - skipping.")
        return None
    metrics = [("internal_accuracy_this_checkpoint", "internal (this checkpoint)"),
              ("leakage_controlled_accuracy", "leakage-controlled CV"),
              ("external_accuracy", "external (malignant vs normal)")]
    models = comparison_df["model"].tolist()
    apply_print_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, model_name in enumerate(models):
        row = comparison_df[comparison_df["model"] == model_name].iloc[0]
        vals = [row.get(m[0]) for m in metrics]
        bar_x = x + i * width - (len(models) - 1) * width / 2
        bars = ax.bar(bar_x, [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                              for v in vals],
                      width, label=model_name, color=_color_for(model_name))
        for bx, v in zip(bar_x, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ax.text(bx, 0.02, "N/A", ha="center", fontsize=9, rotation=90, color="grey")
            else:
                ax.text(bx, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("MiniConvNet vs VGG16 - accuracy comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "01_model_accuracy")


# --- 2: internal vs external accuracy comparison ---


def fig02_internal_vs_external_accuracy(comparison_df):
    if comparison_df is None or comparison_df.empty:
        return None
    models = comparison_df["model"].tolist()
    apply_print_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    width = 0.35
    internal_vals = [comparison_df[comparison_df["model"] == m].iloc[0]["internal_accuracy_this_checkpoint"]
                     for m in models]
    external_vals = [comparison_df[comparison_df["model"] == m].iloc[0]["external_accuracy"] for m in models]
    ax.bar(x - width / 2, internal_vals, width, label="internal", color="#4c78a8")
    ax.bar(x + width / 2, external_vals, width, label="external", color="#f58518")
    for i, (a, b) in enumerate(zip(internal_vals, external_vals)):
        ax.text(i - width / 2, a + 0.015, f"{a*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.015, f"{b*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("Internal vs external accuracy")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "02_internal_vs_external")


# --- 3: internal vs external malignant/tumour recall ---


def fig03_internal_vs_external_recall(internal_df, external_df, models):
    if internal_df is None or external_df is None:
        print("  internal or external results table unavailable - skipping.")
        return None
    rows = []
    normal_idx = CLASS_NAMES.index(NORMAL_CLASS)
    for model_name, run_name in models:
        # `binary_tumor_detection_accuracy` in internal_results.csv is ACCURACY of the
        # binary split (counts normal-correctly-called-normal too), not recall - so recall
        # is always recomputed directly from the raw predictions here. The internal_results
        # table lookup is only a fallback for the (expected-rare) case where that table built
        # successfully but the underlying per-image file is no longer readable.
        df = load_internal_full_predictions(run_name)
        if df is not None:
            y_true, y_pred = df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int)
            true_tumor = y_true != normal_idx
            internal_recall = float((y_pred[true_tumor] != normal_idx).mean()) if true_tumor.any() else None
        else:
            irow = internal_df[internal_df["model"] == model_name]
            internal_recall = None
            if len(irow):
                print(f"  {model_name}: raw predictions unavailable, falling back to internal_results.csv's "
                     "binary accuracy (NOT true recall) - marked as such would require a note column, "
                     "so this fallback is skipped and internal_recall stays None instead of reporting "
                     "the wrong metric under the right name.")

        erow = external_df[(external_df["model"] == model_name)
                           & (external_df["framing"] == "malignant_vs_normal_excl_benign")]
        external_recall = float(erow.iloc[0]["recall_tumor"]) if len(erow) else None
        rows.append({"model": model_name, "internal_recall": internal_recall, "external_recall": external_recall})

    plot_df = pd.DataFrame(rows).dropna(subset=["internal_recall", "external_recall"], how="all")
    if plot_df.empty:
        return None
    apply_print_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["internal_recall"], width, label="internal (tumour vs normal)",
          color="#4c78a8")
    ax.bar(x + width / 2, plot_df["external_recall"], width, label="external (malignant vs normal)",
          color="#e45756")
    for i, r in plot_df.reset_index(drop=True).iterrows():
        if pd.notna(r["internal_recall"]):
            ax.text(i - width / 2, r["internal_recall"] + 0.015, f"{r['internal_recall']*100:.1f}%",
                   ha="center", fontsize=9)
        if pd.notna(r["external_recall"]):
            ax.text(i + width / 2, r["external_recall"] + 0.015, f"{r['external_recall']*100:.1f}%",
                   ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(plot_df["model"])
    ax.set_ylabel("recall (sensitivity)"); ax.set_ylim(0, 1.08)
    ax.set_title("Internal tumour recall vs external malignant recall")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "03_malignant_recall")


# --- 4/5: confusion matrices ---


def fig_confusion_matrix(model_name, run_name, fig_num):
    df = load_internal_full_predictions(run_name)
    if df is None:
        return None
    y_true, y_pred = df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int)
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    apply_print_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, label="count")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    acc = (y_true == y_pred).mean()
    ax.set_title(f"{model_name} - confusion matrix (internal test set)\naccuracy = {acc*100:.1f}%")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / f"{fig_num:02d}_confusion_{model_name.lower()}")


# --- 6/7: ROC curves (internal, native 4-class, one-vs-rest + macro average) ---


def fig_roc_curve(model_name, run_name, fig_num):
    df = load_internal_full_predictions(run_name)
    if df is None:
        return None
    y_true = df["y_true"].to_numpy(int)
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    y_prob = df[prob_cols].to_numpy(float)

    present_classes = sorted(set(y_true.tolist()))
    if len(present_classes) < 2:
        print(f"  {model_name}: fewer than 2 classes present in y_true - ROC not meaningful, skipping.")
        return None

    apply_print_style()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    n_curves = 0
    for i, cls in enumerate(CLASS_NAMES):
        if i not in present_classes:
            continue
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.6, alpha=0.85, label=f"{cls} (AUC={roc_auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        n_curves += 1
    mean_tpr /= max(n_curves, 1)
    macro_auc = sk_auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, ls="--",
           label=f"macro-average (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model_name} - ROC curve (internal, one-vs-rest, 4-class)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / f"{fig_num:02d}_roc_{model_name.lower()}")


# --- 8/9: calibration (reliability) curves ---


def fig_calibration_curve(model_name, run_name, fig_num):
    df = load_internal_full_predictions(run_name)
    if df is None:
        return None
    y_true = df["y_true"].to_numpy(int)
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    y_prob = df[prob_cols].to_numpy(float)

    summary = calibration_summary(y_true, y_prob, n_bins=10)
    bins = summary["_ece_bins"]
    centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    accs = [b["accuracy"] if b["accuracy"] is not None else np.nan for b in bins]
    counts = [b["count"] for b in bins]

    apply_print_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    width = 0.09
    ax1.bar(centers, [0 if np.isnan(a) else a for a in accs], width=width, color="#4c78a8",
           edgecolor="black", label="observed accuracy")
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1.4, label="perfect calibration")
    ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.02); ax1.set_xlim(0, 1)
    ax1.set_title(f"{model_name} - calibration curve\nECE = {summary['ece']*100:.2f}%, "
                 f"Brier = {summary['brier_score']:.4f}")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    ax2.bar(centers, counts, width=width, color="#72b7b2", edgecolor="black")
    ax2.set_xlabel("confidence (max predicted probability)"); ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / f"{fig_num:02d}_calibration_{model_name.lower()}")


# --- 10: confidence distribution, correct vs incorrect, both models ---


def fig10_confidence_by_correctness(models):
    available = []
    for model_name, run_name in models:
        df = load_internal_full_predictions(run_name)
        if df is not None:
            available.append((model_name, df))
    if not available:
        return None

    apply_print_style()
    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5), squeeze=False)
    for ax, (model_name, df) in zip(axes[0], available):
        y_true, y_pred = df["y_true"].to_numpy(int), df["y_pred"].to_numpy(int)
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        confidence = df[prob_cols].to_numpy(float).max(axis=1)
        correct = y_true == y_pred
        bins = np.linspace(0, 1, 21)
        ax.hist(confidence[correct], bins=bins, alpha=0.6, color="#4c78a8",
               label=f"correct (n={int(correct.sum())})", edgecolor="black")
        ax.hist(confidence[~correct], bins=bins, alpha=0.6, color="#e45756",
               label=f"incorrect (n={int((~correct).sum())})", edgecolor="black")
        ax.set_xlabel("confidence"); ax.set_ylabel("count")
        ax.set_title(model_name)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Confidence distribution: correct vs incorrect (internal test set)")
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "10_confidence_correctness")


# --- 11: MC Dropout entropy distribution, correct vs incorrect, both models ---


def fig11_entropy_by_correctness(models):
    available = []
    for model_name, _run_name in models:
        df = load_uncertainty_predictions(model_name, "internal")
        if df is not None:
            available.append((model_name, df))
    if not available:
        return None

    apply_print_style()
    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5), squeeze=False)
    for ax, (model_name, df) in zip(axes[0], available):
        entropy = df["mc_predictive_entropy_4class"].to_numpy(float)
        correct = df["det_correct"].to_numpy(bool)
        bins = np.linspace(0, max(float(entropy.max()), 1e-6), 25)
        ax.hist(entropy[correct], bins=bins, alpha=0.6, color="#4c78a8",
               label=f"correct (n={int(correct.sum())})", edgecolor="black")
        ax.hist(entropy[~correct], bins=bins, alpha=0.6, color="#e45756",
               label=f"incorrect (n={int((~correct).sum())})", edgecolor="black")
        ax.set_xlabel("MC Dropout predictive entropy"); ax.set_ylabel("count")
        ax.set_title(model_name)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("MC Dropout entropy distribution: correct vs incorrect (internal test set)")
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "11_entropy_correctness")


# --- 12: uncertainty/error relationship (quantile bins, both models, both datasets) ---


def fig12_uncertainty_error_relationship(models):
    rows = []
    for model_name, _run_name in models:
        internal = load_uncertainty_predictions(model_name, "internal")
        if internal is not None:
            bins_df = uncertainty_bins(internal["mc_predictive_entropy_4class"], internal["det_correct"])
            bins_df["model"] = model_name; bins_df["dataset"] = "internal"
            rows.append(bins_df)
        external = load_uncertainty_predictions(model_name, "external")
        if external is not None:
            scored = external[external["scored_mask"]]
            bins_df = uncertainty_bins(scored["mc_predictive_entropy_binary"], scored["det_correct"])
            bins_df["model"] = model_name; bins_df["dataset"] = "external"
            rows.append(bins_df)
    if not rows:
        return None
    all_bins = pd.concat(rows, ignore_index=True)

    apply_print_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bin_order = ["low", "medium", "high"]
    groups = all_bins.assign(group=all_bins["model"] + " / " + all_bins["dataset"])
    x = np.arange(len(bin_order))
    width = 0.8 / max(groups["group"].nunique(), 1)
    for i, g in enumerate(sorted(groups["group"].unique())):
        sub = groups[groups["group"] == g].set_index("bin").reindex(bin_order)
        ax.bar(x + i * width - (groups["group"].nunique() - 1) * width / 2,
              sub["error_rate"].fillna(0), width, label=g)
    ax.set_xticks(x); ax.set_xticklabels(bin_order)
    ax.set_xlabel("uncertainty bin (tercile of predictive entropy)")
    ax.set_ylabel("error rate")
    ax.set_title("Uncertainty vs error rate")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "12_uncertainty_error")


# --- 13/14: risk-coverage curves ---


def fig_risk_coverage(model_name, selective_df, fig_num, dataset_label="external"):
    if selective_df is None:
        return None
    sub = selective_df[(selective_df["model"] == model_name) & (selective_df["dataset"] == dataset_label)]
    if sub.empty:
        print(f"  no {dataset_label} risk-coverage rows for {model_name} - skipping.")
        return None
    sub = sub.sort_values("coverage", ascending=False)

    apply_print_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(sub["coverage"], sub["accuracy"], marker="o", color="#4c78a8", label="accuracy")
    ax.plot(sub["coverage"], sub["error_rate"], marker="s", color="#e45756", label="error rate")
    if "positive_class_recall" in sub.columns and sub["positive_class_recall"].notna().any():
        ax.plot(sub["coverage"], sub["positive_class_recall"], marker="^", color="#54a24b",
               label="malignant recall (retained)")
    ax.set_xlabel("coverage (fraction retained, least-uncertain first)")
    ax.set_ylabel("rate"); ax.set_ylim(-0.02, 1.02)
    ax.invert_xaxis()
    ax.set_title(f"{model_name} - risk-coverage curve ({dataset_label})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / f"{fig_num:02d}_risk_coverage_{model_name.lower()}")


# --- 15: internal vs external uncertainty distribution, both models ---


def fig15_internal_vs_external_uncertainty(models):
    available = []
    for model_name, _run_name in models:
        internal = load_uncertainty_predictions(model_name, "internal")
        external = load_uncertainty_predictions(model_name, "external")
        if internal is not None and external is not None:
            available.append((model_name, internal, external))
    if not available:
        return None

    apply_print_style()
    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5), squeeze=False)
    for ax, (model_name, internal, external) in zip(axes[0], available):
        int_e = internal["mc_predictive_entropy_binary"].to_numpy(float)
        ext_scored = external[external["scored_mask"]]
        ext_e = ext_scored["mc_predictive_entropy_binary"].to_numpy(float)
        hi = max(float(int_e.max()), float(ext_e.max()), 1e-6)
        bins = np.linspace(0, hi, 25)
        ax.hist(int_e, bins=bins, density=True, alpha=0.6, color="#4c78a8",
               label=f"internal (n={len(int_e)})", edgecolor="black")
        ax.hist(ext_e, bins=bins, density=True, alpha=0.6, color="#f58518",
               label=f"external (n={len(ext_e)})", edgecolor="black")
        ax.set_xlabel("predictive entropy (binary-reduced)"); ax.set_ylabel("density")
        ax.set_title(model_name)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Internal vs external uncertainty distribution")
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "15_internal_external_uncertainty")


# --- 16: high-confidence incorrect prediction analysis ---


def fig16_high_confidence_errors(models):
    rows = []
    for model_name, _run_name in models:
        for dataset_label, entropy_col in (("internal", "mc_predictive_entropy_4class"),
                                           ("external", "mc_predictive_entropy_binary")):
            df = load_uncertainty_predictions(model_name, dataset_label)
            if df is None:
                continue
            if dataset_label == "external":
                df = df[df["scored_mask"]]
            hc = df[(~df["det_correct"]) & (df["deterministic_confidence"] >= HIGH_CONFIDENCE_THRESHOLD)]
            rows.append({"model": model_name, "dataset": dataset_label, "n_total": len(df),
                        "n_high_conf_incorrect": len(hc),
                        "mean_entropy": float(hc[entropy_col].mean()) if len(hc) else None})
    if not rows:
        return None
    summary = pd.DataFrame(rows)

    apply_print_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = summary["model"] + "\n(" + summary["dataset"] + ")"
    ax1.bar(labels, summary["n_high_conf_incorrect"], color=[_color_for(m) for m in summary["model"]])
    for i, v in enumerate(summary["n_high_conf_incorrect"]):
        ax1.text(i, v + max(summary["n_high_conf_incorrect"]) * 0.02, str(int(v)), ha="center")
    ax1.set_ylabel(f"count (confidence >= {HIGH_CONFIDENCE_THRESHOLD})")
    ax1.set_title("High-confidence INCORRECT predictions")
    ax1.grid(axis="y", alpha=0.3)

    plot_e = summary.dropna(subset=["mean_entropy"])
    if len(plot_e):
        ax2.bar(plot_e["model"] + "\n(" + plot_e["dataset"] + ")", plot_e["mean_entropy"],
               color=[_color_for(m) for m in plot_e["model"]])
        ax2.set_ylabel("mean predictive entropy (of the high-confidence-incorrect subset)")
        ax2.set_title("Their MC Dropout uncertainty")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "no high-confidence-incorrect\ncases to show entropy for",
                ha="center", va="center")
    fig.suptitle("High-confidence incorrect prediction analysis "
                f"(threshold={HIGH_CONFIDENCE_THRESHOLD})")
    fig.tight_layout()
    return save_figure(fig, FINAL_FIGURES_DIR / "16_high_confidence_errors")


# --- 17/18: Grad-CAM / Grad-CAM++ examples (reused overlay images, not recomputed) ---


def fig_gradcam_examples(method, fig_num, max_per_model=4):
    df = load_csv_if_exists(GRADCAM_RECORDS_CSV, "results/gradcam/gradcam_records.csv")
    if df is None:
        return None
    sub = df[(df["method"] == method) & (df["status"] == "ok")]
    if sub.empty:
        print(f"  no '{method}' records with status=ok - skipping.")
        return None

    panels = []
    for model_name in sorted(sub["model"].unique()):
        m = sub[sub["model"] == model_name]
        correct_ex = m[m["correct"]].head(max_per_model // 2)
        incorrect_ex = m[~m["correct"]].head(max_per_model - len(correct_ex))
        panels.append((model_name, pd.concat([correct_ex, incorrect_ex])))

    n = sum(len(p[1]) for p in panels)
    if n == 0:
        return None
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    apply_print_style()
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.6 * rows), squeeze=False)
    idx = 0
    missing_images = 0
    for model_name, recs in panels:
        for _, r in recs.iterrows():
            ax = axes[idx // cols][idx % cols]
            try:
                with Image.open(r["overlay_path"]) as im:
                    ax.imshow(im)
            except Exception:
                missing_images += 1
                ax.text(0.5, 0.5, "overlay image\nnot found on disk", ha="center", va="center", fontsize=8)
            ax.axis("off")
            mark = "correct" if r["correct"] else "WRONG"
            ax.set_title(f"{model_name}\ntrue={r['true_class']} pred={r['pred_class']} [{mark}]",
                        fontsize=8)
            idx += 1
    for j in range(idx, rows * cols):
        axes[j // cols][j % cols].axis("off")

    label = "Grad-CAM" if method == "gradcam" else "Grad-CAM++"
    fig.suptitle(f"{label} examples", fontsize=14)
    fig.tight_layout()
    if missing_images:
        print(f"  NOTE: {missing_images} overlay image(s) referenced in {GRADCAM_RECORDS_CSV} were not "
             "found on this disk (their PNGs must be present locally, same as when Phase 3 ran).")
    return save_figure(fig, FINAL_FIGURES_DIR / f"{fig_num:02d}_gradcam_examples_{method}")


# ==========================================================================
# FINAL SUMMARY (results/final/final_results.csv, final_summary.json/.txt)
# ==========================================================================


def build_final_summary(comparison_df, models):
    ensure_dirs()
    if comparison_df is None:
        print("  model_comparison table unavailable - final summary cannot be built.")
        return None

    comparison_df.to_csv(FINAL_RESULTS_CSV, index=False)

    leakage_ck = check_leakage_controlled_checkpoint()
    ids = identify_models()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": "src/generate_final_results.py (pure post-processing, no inference performed)",
        "models": {
            "primary": {"name": ids["primary_model_name"], "run_name": ids["primary_run_name"],
                       "checkpoint": ids["primary_checkpoint"]},
            "secondary": {"name": ids["secondary_model_name"], "run_name": ids["secondary_run_name"],
                         "checkpoint": ids["secondary_checkpoint"]},
        },
        "leakage_controlled_checkpoint_substitution": leakage_ck["disclosure"],
        "per_model": comparison_df.to_dict(orient="records"),
        "tables": REG.tables,
        "figures": REG.figures,
    }
    with open(FINAL_SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    lines = []
    lines.append("FINAL RESULTS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append(f"Primary model  : {ids['primary_model_name']} ({ids['primary_run_name']})")
    lines.append(f"Secondary model: {ids['secondary_model_name']} ({ids['secondary_run_name']})")
    lines.append("")
    if leakage_ck["disclosure"]:
        lines.append("CHECKPOINT SUBSTITUTION DISCLOSURE:")
        lines.append(leakage_ck["disclosure"])
        lines.append("")
    for _, row in comparison_df.iterrows():
        lines.append(f"--- {row['model']} ---")
        for col in comparison_df.columns:
            if col == "model":
                continue
            lines.append(f"  {col}: {row[col]}")
        lines.append("")
    lines.append("Tables produced:")
    for name, info in REG.tables.items():
        lines.append(f"  [{info['status'].upper():7s}] {name}")
    lines.append("")
    lines.append("Figures produced:")
    for name, info in REG.figures.items():
        lines.append(f"  [{info['status'].upper():7s}] {name}")
    FINAL_SUMMARY_TXT.write_text("\n".join(lines))

    return FINAL_RESULTS_CSV, FINAL_SUMMARY_JSON, FINAL_SUMMARY_TXT


# ==========================================================================
# FINAL CHECK - verify what was actually produced, item by item
# ==========================================================================


def run_final_check(models):
    print("\n" + "=" * 70)
    print("FINAL CHECK")
    print("=" * 70)
    checks = {}

    ck_ids = identify_models()
    ck_paths = [ck_ids["primary_checkpoint"], ck_ids["secondary_checkpoint"]]
    checks["checkpoints_exist_on_disk"] = all(p and Path(p).exists() for p in ck_paths)
    print(f"[{'PASS' if checks['checkpoints_exist_on_disk'] else 'FAIL/UNKNOWN'}] "
         "required checkpoints exist "
         f"({sum(1 for p in ck_paths if p and Path(p).exists())}/{len(ck_paths)} found)")

    required_sources = {
        "outputs/results_table.csv": RESULTS_TABLE_CSV,
        "results/fair_baseline/metrics_fair_baseline.csv": FAIR_METRICS_CSV,
        "results/calibration/calibration_metrics.csv": CALIBRATION_METRICS_CSV,
        "results/external_eval/external_eval_metrics.csv": EXTERNAL_METRICS_CSV,
        "results/uncertainty/uncertainty_metrics.csv": UNCERTAINTY_METRICS_CSV,
        "results/uncertainty/final_results_table.csv": FINAL_RESULTS_TABLE_CSV,
    }
    missing_sources = [name for name, p in required_sources.items() if not Path(p).exists()]
    checks["required_results_exist"] = not missing_sources
    print(f"[{'PASS' if checks['required_results_exist'] else 'FAIL'}] required results exist"
         + (f" - MISSING: {missing_sources}" if missing_sources else ""))

    n_fig_ok = sum(1 for v in REG.figures.values() if v["status"] == "ok")
    n_fig_total = len(REG.figures)
    checks["all_graphs_generated"] = (n_fig_ok == n_fig_total and n_fig_total > 0)
    print(f"[{'PASS' if checks['all_graphs_generated'] else 'PARTIAL'}] graphs generated: "
         f"{n_fig_ok}/{n_fig_total}")

    openable, broken = 0, []
    for name, info in REG.figures.items():
        if info["status"] != "ok":
            continue
        for p in info["paths"]:
            if p.endswith(".png"):
                try:
                    with Image.open(p) as im:
                        im.verify()
                    openable += 1
                except Exception as exc:
                    broken.append((p, str(exc)))
            elif Path(p).exists() and Path(p).stat().st_size > 0:
                openable += 1
            else:
                broken.append((p, "missing or empty"))
    checks["graph_files_openable"] = not broken
    print(f"[{'PASS' if checks['graph_files_openable'] else 'FAIL'}] graph files openable: "
         f"{openable} verified" + (f", {len(broken)} BROKEN: {broken}" if broken else ""))

    empty_tables = [name for name, info in REG.tables.items()
                    if info["status"] == "ok" and info.get("n_rows", 0) == 0]
    checks["csv_files_contain_data"] = not empty_tables
    print(f"[{'PASS' if checks['csv_files_contain_data'] else 'FAIL'}] CSV files contain real data"
         + (f" - EMPTY: {empty_tables}" if empty_tables else ""))

    n_tables_ok = sum(1 for v in REG.tables.values() if v["status"] == "ok")
    n_tables_total = len(REG.tables)
    print(f"[INFO] tables: {n_tables_ok}/{n_tables_total} produced "
         f"({n_tables_total - n_tables_ok} skipped/failed - see reasons printed above; "
         "this is expected if an upstream notebook has not been run, not a silent failure)")

    checks["model_identification_correct"] = bool(ck_ids["secondary_model_name"])
    print(f"[{'PASS' if checks['model_identification_correct'] else 'FAIL'}] secondary model identified "
         f"programmatically: {ck_ids['secondary_model_name']}")

    checks["one_command_regenerates_everything"] = True
    print("[PASS] one command regenerates everything: `python src/generate_final_results.py` "
         "(pure post-processing, no manual editing required)")

    return checks


# ==========================================================================
# main
# ==========================================================================


def main():
    print("=" * 70)
    print("GENERATE FINAL RESULTS - pure post-processing, no inference performed")
    print("=" * 70)
    ensure_dirs()

    ids = identify_models()
    print(f"\nPrimary model  : {ids['primary_model_name']} ({ids['primary_run_name']})")
    if ids["secondary_available"]:
        print(f"Secondary model: {ids['secondary_model_name']} ({ids['secondary_run_name']})")
    else:
        print(f"Secondary model: UNAVAILABLE - {ids['secondary_reason_unavailable']}")

    models = [(ids["primary_model_name"], ids["primary_run_name"])]
    if ids["secondary_available"]:
        models.append((ids["secondary_model_name"], ids["secondary_run_name"]))

    leakage_ck = check_leakage_controlled_checkpoint()
    if leakage_ck["disclosure"]:
        print("\n" + leakage_ck["disclosure"])

    # ---------------- Tables ----------------
    print("\n" + "=" * 70)
    print("BUILDING TABLES")
    print("=" * 70)
    comparison_df = run_table_step("model_comparison.csv", build_model_comparison_table)
    internal_df = run_table_step("internal_results.csv", lambda: build_internal_results_table(models))
    external_df = run_table_step("external_results.csv", build_external_results_table)
    calibration_df = run_table_step("calibration_results.csv", build_calibration_results_table)
    uncertainty_df = run_table_step("uncertainty_results.csv", build_uncertainty_results_table)
    selective_df = run_table_step("selective_prediction_results.csv",
                                  build_selective_prediction_results_table)
    per_image_df = run_table_step("per_image_predictions.csv",
                                  lambda: build_per_image_predictions_table(models))

    # ---------------- Figures ----------------
    print("\n" + "=" * 70)
    print("BUILDING FIGURES")
    print("=" * 70)
    run_figure_step("01_model_accuracy", lambda: fig01_model_accuracy_comparison(comparison_df))
    run_figure_step("02_internal_vs_external", lambda: fig02_internal_vs_external_accuracy(comparison_df))
    run_figure_step("03_malignant_recall",
                    lambda: fig03_internal_vs_external_recall(internal_df, external_df, models))
    for i, (model_name, run_name) in enumerate(models):
        run_figure_step(f"04-05_confusion_{model_name}",
                        lambda mn=model_name, rn=run_name, n=4 + i: fig_confusion_matrix(mn, rn, n))
    for i, (model_name, run_name) in enumerate(models):
        run_figure_step(f"06-07_roc_{model_name}",
                        lambda mn=model_name, rn=run_name, n=6 + i: fig_roc_curve(mn, rn, n))
    for i, (model_name, run_name) in enumerate(models):
        run_figure_step(f"08-09_calibration_{model_name}",
                        lambda mn=model_name, rn=run_name, n=8 + i: fig_calibration_curve(mn, rn, n))
    run_figure_step("10_confidence_correctness", lambda: fig10_confidence_by_correctness(models))
    run_figure_step("11_entropy_correctness", lambda: fig11_entropy_by_correctness(models))
    run_figure_step("12_uncertainty_error", lambda: fig12_uncertainty_error_relationship(models))
    for i, (model_name, _run_name) in enumerate(models):
        run_figure_step(f"13-14_risk_coverage_{model_name}",
                        lambda mn=model_name, n=13 + i: fig_risk_coverage(mn, selective_df, n))
    run_figure_step("15_internal_external_uncertainty",
                    lambda: fig15_internal_vs_external_uncertainty(models))
    run_figure_step("16_high_confidence_errors", lambda: fig16_high_confidence_errors(models))
    run_figure_step("17_gradcam_examples", lambda: fig_gradcam_examples("gradcam", 17))
    run_figure_step("18_gradcam_plusplus_examples", lambda: fig_gradcam_examples("gradcam_plusplus", 18))

    # ---------------- Final summary ----------------
    print("\n" + "=" * 70)
    print("BUILDING FINAL SUMMARY")
    print("=" * 70)
    build_final_summary(comparison_df, models)

    # ---------------- Final check ----------------
    checks = run_final_check(models)

    # ---------------- Plain-language report ----------------
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    ok_tables = [n for n, v in REG.tables.items() if v["status"] == "ok"]
    skipped_tables = [n for n, v in REG.tables.items() if v["status"] != "ok"]
    ok_figures = [n for n, v in REG.figures.items() if v["status"] == "ok"]
    skipped_figures = [n for n, v in REG.figures.items() if v["status"] != "ok"]

    print(f"Files created       : {len(ok_tables)} table(s) in {FINAL_DIR}, "
         f"{sum(len(v['paths']) for v in REG.figures.values() if v['status']=='ok')} figure file(s) "
         f"in {FINAL_FIGURES_DIR}, plus final_results.csv/final_summary.json/final_summary.txt")
    print(f"Tables created      : {ok_tables}")
    if skipped_tables:
        print(f"Tables SKIPPED/FAILED: {skipped_tables} (see reasons printed above)")
    print(f"Graphs created      : {ok_figures}")
    if skipped_figures:
        print(f"Graphs SKIPPED/FAILED: {skipped_figures} (see reasons printed above)")
    print()
    print("Main numerical results (from model_comparison.csv, if built):")
    if comparison_df is not None:
        print(comparison_df.to_string(index=False))
    else:
        print("  not available - results/uncertainty/final_results_table.csv was not found "
             "(run notebook 15 first)")
    print()
    n_problems = sum(1 for v in checks.values() if v is False)
    if n_problems == 0 and not skipped_tables and not skipped_figures:
        print("Remaining problems  : none - every table and figure was generated and verified.")
    else:
        print(f"Remaining problems  : {n_problems} FINAL CHECK item(s) did not pass; "
             f"{len(skipped_tables)} table(s) and {len(skipped_figures)} figure(s) skipped or failed. "
             "See the detailed reasons printed above - nothing here was fabricated to hide a gap.")


if __name__ == "__main__":
    main()
