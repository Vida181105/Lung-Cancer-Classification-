"""Phase 6 - uncertainty-aware / selective prediction.

Connects three already-completed pieces of work into one final experiment:
Phase 4's MC Dropout + calibration, Phase 5's external IQ-OTH/NCCD evaluation,
and Phase 2/3's checkpoint-identification conventions. The question this phase
answers: **can uncertainty identify predictions that are likely to be wrong,
particularly under external domain shift?**

Entirely additive. Nothing here is imported by notebooks 00-14, and every
artefact goes to ``results/uncertainty/``. No existing result, prediction, or
checkpoint is read for writing or modified.

Hard constraints this module was written under
------------------------------------------------
* **CPU only. No retraining, ever.** Every function either loads an existing
  ``.keras`` checkpoint and runs it forward (deterministic inference, or MC
  Dropout's stochastic-but-still-forward-only passes) or operates on
  already-computed probabilities/predictions. There is no ``.fit()`` call
  anywhere in this file.
* **Never silently substitute a different checkpoint instance.** The one
  documented exception is decided in advance and must be disclosed loudly,
  never silently: :func:`check_leakage_controlled_checkpoint` confirms (by
  direct inspection of ``src/leakage_cv_utils.py`` and notebook 14 - neither
  contains a single ``.save()`` call) that the leakage-controlled 3-fold CV
  never produced a standalone saved model, and returns the exact disclosure
  text every downstream MiniConvNet result in this phase must carry.
* **The external label mapping is not changed.** This module reuses
  ``external_eval_utils.EXTERNAL_TASK_FRAMINGS["malignant_vs_normal_excl_benign"]``
  - the same primary framing Phase 5 already reported (benign excluded from
  scoring) - and nothing else. See :func:`mc_dropout_external`.
* **The MC Dropout mechanism is not reinvented.** :func:`mc_dropout_external`
  calls ``calibration_utils.mc_dropout_raw_passes`` - the exact same
  stochastic-forward-pass core Phase 4 already validated on both models - and
  only adds the binary-task reduction/scoring on top, because the external
  3-class label space is not index-comparable to a 4-class ``argmax`` (see
  that function's docstring for why a naive reuse of
  ``calibration_utils.mc_dropout_predict`` would silently compute meaningless
  "correctness" values here).
* **Thresholds and bins are not tuned to flatter a result.** The high-
  confidence threshold reuses Phase 3's already-established
  ``gradcam_utils.HIGH_CONFIDENCE_THRESHOLD`` (0.75) rather than picking a new
  one now; uncertainty bins (:func:`uncertainty_bins`) are quantile-based
  (terciles of the entropy distribution itself, independent of correctness);
  risk-coverage levels (:data:`DEFAULT_COVERAGE_LEVELS`) are the standard
  100%-down-to-10% decile grid named in the brief, not searched over.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, NORMAL_CLASS, NUM_CLASSES, PROJECT_ROOT, SEED
from src.finetune_utils import CHECKPOINTS_LOCAL, FAIR_METRICS_CSV
from src.gradcam_utils import HIGH_CONFIDENCE_THRESHOLD, identify_strongest_finetuned_baseline

# --------------------------------------------------------------------------
# Paths - all new
# --------------------------------------------------------------------------

UNCERTAINTY_DIR = PROJECT_ROOT / "results" / "uncertainty"
UNCERTAINTY_METRICS_CSV = UNCERTAINTY_DIR / "uncertainty_metrics.csv"
UNCERTAINTY_METRICS_JSON = UNCERTAINTY_DIR / "uncertainty_metrics.json"
SELECTIVE_PREDICTION_CSV = UNCERTAINTY_DIR / "selective_prediction_results.csv"
FINAL_RESULTS_TABLE_CSV = UNCERTAINTY_DIR / "final_results_table.csv"
FINAL_RESULTS_TABLE_JSON = UNCERTAINTY_DIR / "final_results_table.json"


def model_dir(model_name: str) -> Path:
    """Per-model subfolder, e.g. ``results/uncertainty/miniconvnet/``."""
    return UNCERTAINTY_DIR / model_name.lower()


def model_figures_dir(model_name: str) -> Path:
    return model_dir(model_name) / "figures"


def model_predictions_used_dir(model_name: str) -> Path:
    return model_dir(model_name) / "predictions_used"


def ensure_uncertainty_dirs(model_names) -> dict:
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    out = {"uncertainty_dir": str(UNCERTAINTY_DIR)}
    for name in model_names:
        model_figures_dir(name).mkdir(parents=True, exist_ok=True)
        model_predictions_used_dir(name).mkdir(parents=True, exist_ok=True)
        out[name] = {"figures": str(model_figures_dir(name)),
                    "predictions_used": str(model_predictions_used_dir(name))}
    return out


# --------------------------------------------------------------------------
# STEP 1 (checkpoints) - identify models, verify checkpoints, disclose the
# one pre-resolved substitution
# --------------------------------------------------------------------------

# Confirmed by direct inspection (Phase 1 of this task) - neither file
# contains a single `.save(` / `CHECKPOINTS_LOCAL` reference, so Option B
# never produced a standalone model file. Patterns are still searched for at
# runtime rather than hardcoding "it doesn't exist", in case that ever
# changes.
_LEAKAGE_CV_CHECKPOINT_PATTERNS = (
    "*leakage_controlled*.keras", "*option_b*.keras", "*leakage_cv*.keras",
)


def check_leakage_controlled_checkpoint(checkpoints_dir=None) -> dict:
    """Look for a checkpoint saved specifically from the leakage-controlled
    (Option B) 3-fold CV. Confirmed absent by inspecting
    ``src/leakage_cv_utils.py`` and ``notebooks/14_leakage_controlled_cv.ipynb``
    - Option B was built to report aggregate per-fold metrics only, never a
    standalone saved model.

    If none is found (the expected, confirmed case), returns the exact
    disclosure text to print/attach to every MiniConvNet result in this
    phase - this substitution must never be silent, per the addendum.
    """
    d = Path(checkpoints_dir or CHECKPOINTS_LOCAL)
    hits = []
    if d.exists():
        for pat in _LEAKAGE_CV_CHECKPOINT_PATTERNS:
            hits.extend(sorted(str(p) for p in d.glob(pat)))
    found = bool(hits)

    disclosure = None
    if not found:
        disclosure = (
            "MiniConvNet results in this analysis use the SINGLE-RUN checkpoint "
            "(checkpoints_local/miniconvnet_single_run.keras, internal faithful-split test "
            "accuracy 49.52% - see results/calibration/calibration_metrics.csv), the SAME "
            "instance already analysed in Phases 3-4 (Grad-CAM, calibration) - NOT a checkpoint "
            "specifically derived from the leakage-controlled 3-fold CV (74.10% +/- 4.24%, "
            "outputs/leakage/leakage_controlled_cv_results.json), which never produced a "
            "standalone saved model (confirmed: no .save() call exists anywhere in "
            "src/leakage_cv_utils.py or notebook 14 - it was built to report aggregate per-fold "
            "metrics only). This is a disclosed, deliberate substitution: the research question "
            "here (does uncertainty flag likely-wrong predictions) does not depend on which "
            "specific MiniConvNet training run is used, as long as it is stated clearly which "
            "one was.")

    return {"found": found, "candidates": hits, "substitution_required": not found,
           "disclosure": disclosure}


def identify_models_for_this_phase() -> dict:
    """Primary (MiniConvNet, always) + secondary (strongest Phase 2 fine-tuned
    baseline, read live - never hardcoded, same pattern as Phases 3/4/5)."""
    secondary = identify_strongest_finetuned_baseline()
    primary_run_name = "miniconvnet_single_run"
    return {
        "primary_model_name": "MiniConvNet",
        "primary_run_name": primary_run_name,
        "primary_checkpoint": str(CHECKPOINTS_LOCAL / f"{primary_run_name}.keras"),
        "secondary_available": secondary["available"],
        "secondary_model_name": secondary.get("model"),
        "secondary_run_name": secondary.get("run_name"),
        "secondary_checkpoint": secondary.get("checkpoint_path"),
        "secondary_reason_unavailable": secondary.get("reason"),
    }


# --------------------------------------------------------------------------
# STEP 2/3 - MC Dropout on the EXTERNAL dataset (reuses the Phase 4 core)
# --------------------------------------------------------------------------


def deterministic_external_predictions(checkpoint_path, external_df, batch_size=32, verbose=True):
    """One deterministic forward pass over the external set - a thin,
    documented call-through to ``external_eval_utils.run_external_forward_pass``
    (Phase 5's own function, unchanged), so the "deterministic prediction /
    deterministic confidence" columns Step 3 asks for come from the exact same
    code path Phase 5's headline external-accuracy numbers already used.
    """
    from src.external_eval_utils import run_external_forward_pass

    external_label, y_prob_4class, model = run_external_forward_pass(
        checkpoint_path, external_df, batch_size=batch_size, verbose=verbose)
    return external_label, y_prob_4class, model


def mc_dropout_external(model, external_df, n_passes=15, seed=SEED, batch_size=32,
                        verbose=True) -> dict:
    """MC Dropout over the external IQ-OTH/NCCD set, using the SAME
    stochastic-pass mechanism as ``calibration_utils.mc_dropout_predict`` (via
    :func:`calibration_utils.mc_dropout_raw_passes`) - not a different
    methodology.

    Correctness cannot be computed by comparing ``argmax(mean_probs)``
    (4-class) to the external label (3-class) directly - see this module's
    header. Instead this reduces the MC-mean 4-class softmax to the binary
    ``[p_normal, p_tumor]`` task via
    ``external_eval_utils.reduce_to_binary_tumor_probs()`` (the exact
    reduction Phase 5 already established and reported) and scores against
    the PRIMARY, already-reported framing only -
    ``malignant_vs_normal_excl_benign`` (benign excluded from scoring, exactly
    as Phase 5 did; the label mapping is NOT changed here).

    Returns a dict whose ``correct``/``binary_pred``/``binary_true`` entries
    are only MEANINGFUL where ``scored_mask`` is True (i.e. not benign) -
    callers must apply that mask before computing any accuracy/recall number,
    exactly as :func:`external_eval_utils.evaluate_external_framing` already
    does for the deterministic pass.
    """
    from src.calibration_utils import mc_dropout_raw_passes
    from src.external_eval_utils import (EXTERNAL_TASK_FRAMINGS, make_external_dataset,
                                         reduce_to_binary_tumor_probs)

    ds = make_external_dataset(external_df, batch_size=batch_size)
    raw = mc_dropout_raw_passes(model, ds, n_passes=n_passes, seed=seed, verbose=verbose)

    external_label = raw["y_labels"]                      # IQ-OTH/NCCD 3-class label
    mean_probs_4class = raw["mean_probs"]
    binary_mean_prob = reduce_to_binary_tumor_probs(mean_probs_4class)

    eps = 1e-12
    entropy_binary = -np.sum(binary_mean_prob * np.log(binary_mean_prob + eps), axis=1)

    framing = EXTERNAL_TASK_FRAMINGS["malignant_vs_normal_excl_benign"]
    scored_mask = framing["include"](external_label)      # True where NOT benign
    binary_true = framing["binary_true"](external_label)  # only meaningful where scored_mask
    binary_pred = binary_mean_prob.argmax(axis=1)
    correct = (binary_pred == binary_true)                # only meaningful where scored_mask

    return {
        "n_passes": raw["n_passes"], "n_samples": raw["n_samples"],
        "external_label": external_label,
        "mean_probs_4class": mean_probs_4class,
        "predictive_entropy_4class": raw["predictive_entropy"],
        "confidence_variance_4class": raw["confidence_variance"],
        "binary_mean_prob": binary_mean_prob,
        "predictive_entropy_binary": entropy_binary,
        "binary_pred": binary_pred,
        "binary_true": binary_true,
        "scored_mask": scored_mask,
        "correct": correct,
        "framing_used": "malignant_vs_normal_excl_benign",
        "framing_label": framing["label"],
    }


def internal_tumor_recall(y_true_4class, y_pred_4class, class_names=CLASS_NAMES,
                          normal_class=NORMAL_CLASS) -> float:
    """Internal analogue of external 'malignant recall': fraction of true
    tumour samples (any of the three subtypes) correctly predicted as ANY
    tumour class, mirroring (not duplicating) the same normal-vs-tumour
    grouping convention already used by ``evaluate_utils.tumor_vs_subtype_breakdown``.

    Not the same vocabulary as external 'malignant' (a different dataset's own
    label), but the same STRUCTURAL comparison: sensitivity for detecting any
    abnormal case vs the healthy/normal class. Kept distinct and clearly
    labelled wherever plotted (Figure 6) rather than conflated.
    """
    normal_idx = class_names.index(normal_class)
    y_true_4class = np.asarray(y_true_4class).astype(int)
    y_pred_4class = np.asarray(y_pred_4class).astype(int)
    true_tumor = y_true_4class != normal_idx
    pred_tumor = y_pred_4class != normal_idx
    if not true_tumor.any():
        return float("nan")
    return float(pred_tumor[true_tumor].mean())


# --------------------------------------------------------------------------
# STEP 4 - uncertainty vs error
# --------------------------------------------------------------------------


def correct_vs_incorrect_stats(entropy, correct) -> dict:
    """Mean/median/std entropy split by correctness - the direct answer to
    'is uncertainty higher when the model is wrong?'."""
    entropy = np.asarray(entropy, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    def _stats(mask):
        if not mask.any():
            return {"n": 0, "mean_entropy": None, "median_entropy": None, "std_entropy": None}
        e = entropy[mask]
        return {"n": int(mask.sum()), "mean_entropy": float(e.mean()),
                "median_entropy": float(np.median(e)), "std_entropy": float(e.std())}

    correct_stats = _stats(correct)
    incorrect_stats = _stats(~correct)
    delta = (None if correct_stats["mean_entropy"] is None or incorrect_stats["mean_entropy"] is None
            else round(incorrect_stats["mean_entropy"] - correct_stats["mean_entropy"], 6))
    return {"correct": correct_stats, "incorrect": incorrect_stats,
           "mean_entropy_delta_incorrect_minus_correct": delta,
           "higher_for_incorrect": (None if delta is None else bool(delta > 0))}


# Standard tercile quantile split (0/33/66/100th percentiles of the entropy
# distribution ITSELF) - NOT tuned against correctness or chosen after seeing
# the error-rate pattern. Document any deviation from this default explicitly
# if one is ever used.
DEFAULT_UNCERTAINTY_BIN_LABELS = ("low", "medium", "high")


def uncertainty_bins(entropy, correct, n_bins=3,
                     labels=DEFAULT_UNCERTAINTY_BIN_LABELS) -> pd.DataFrame:
    """Quantile bins over the entropy distribution, reporting count and error
    rate per bin.

    Bin edges are percentiles of ``entropy`` alone (independent of
    ``correct``) - a standard, non-arbitrary quantile binning. This is
    computed AFTER the fact for reporting, never used to pick a threshold that
    happens to produce a favourable-looking error-rate gradient.
    """
    entropy = np.asarray(entropy, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.quantile(entropy, np.linspace(0, 1, n_bins + 1))
    edges = edges.astype(float)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(entropy, edges[1:-1], right=True)

    rows = []
    for i, lab in enumerate(labels[:n_bins]):
        mask = bin_idx == i
        rows.append({
            "bin": lab, "entropy_lo": round(float(edges[i]), 6), "entropy_hi": round(float(edges[i + 1]), 6),
            "n": int(mask.sum()),
            "accuracy": (float(correct[mask].mean()) if mask.any() else None),
            "error_rate": (float(1 - correct[mask].mean()) if mask.any() else None),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# STEP 5 - selective prediction / risk-coverage
# --------------------------------------------------------------------------

# Standard decile grid named in the brief - reported at every level
# regardless of shape, never searched over to find a flattering coverage
# point.
DEFAULT_COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def risk_coverage_curve(uncertainty, correct, y_true_binary=None, y_pred_binary=None,
                        positive_label=1, coverage_levels=DEFAULT_COVERAGE_LEVELS) -> pd.DataFrame:
    """At each coverage level, retain the LEAST uncertain fraction of
    predictions (ascending sort by ``uncertainty``) and report accuracy, error
    rate, and (if binary true/pred labels are supplied) the recall of
    ``positive_label`` among the RETAINED predictions - the malignant-recall
    number Step 5 specifically asks for, since overall accuracy alone can hide
    a recall collapse under selective prediction.
    """
    uncertainty = np.asarray(uncertainty, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(uncertainty)
    order = np.argsort(uncertainty, kind="stable")   # most confident (lowest uncertainty) first

    have_binary = y_true_binary is not None and y_pred_binary is not None
    if have_binary:
        y_true_binary = np.asarray(y_true_binary).astype(int)
        y_pred_binary = np.asarray(y_pred_binary).astype(int)

    rows = []
    for cov in coverage_levels:
        k = max(1, int(round(cov * n)))
        keep = order[:k]
        acc = float(correct[keep].mean())
        row = {"coverage": cov, "n_retained": k, "n_rejected": n - k,
              "accuracy": acc, "error_rate": round(1.0 - acc, 6)}
        if have_binary:
            pos_mask = y_true_binary[keep] == positive_label
            row["n_positive_in_retained"] = int(pos_mask.sum())
            row["positive_class_recall"] = (
                float((y_pred_binary[keep][pos_mask] == positive_label).mean())
                if pos_mask.any() else None)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# STEP 7 - confidence vs MC uncertainty; high-confidence-incorrect
# --------------------------------------------------------------------------


def confidence_vs_entropy_table(det_confidence, mc_entropy, correct, filepath=None) -> pd.DataFrame:
    """One row per sample: deterministic confidence, MC predictive entropy,
    correctness - the raw material for Step 7's comparison and its
    correlation coefficient (computed by the caller via ``.corr()``, no new
    statistic invented here)."""
    out = {"deterministic_confidence": np.asarray(det_confidence, dtype=float),
          "mc_predictive_entropy": np.asarray(mc_entropy, dtype=float),
          "correct": np.asarray(correct, dtype=bool)}
    if filepath is not None:
        out["filepath"] = np.asarray(filepath)
    return pd.DataFrame(out)


def high_confidence_incorrect(det_confidence, mc_entropy, correct, filepath=None,
                              true_label=None, pred_label=None,
                              threshold=HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Every sample where the deterministic prediction was wrong AND
    deterministic confidence was >= ``threshold``.

    ``threshold`` defaults to ``gradcam_utils.HIGH_CONFIDENCE_THRESHOLD``
    (0.75) - REUSED from Phase 3's own already-documented choice, not picked
    fresh here. These are the "potentially dangerous overconfident failures"
    Step 7 asks to report; ALL qualifying rows are returned (nothing is
    truncated at this stage - truncation for a display grid happens only in
    the plotting function, Step 8).
    """
    det_confidence = np.asarray(det_confidence, dtype=float)
    mc_entropy = np.asarray(mc_entropy, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    mask = (~correct) & (det_confidence >= threshold)

    out = {"deterministic_confidence": det_confidence[mask], "mc_predictive_entropy": mc_entropy[mask]}
    if filepath is not None:
        out["filepath"] = np.asarray(filepath)[mask]
    if true_label is not None:
        out["true_label"] = np.asarray(true_label)[mask]
    if pred_label is not None:
        out["pred_label"] = np.asarray(pred_label)[mask]
    df = pd.DataFrame(out)
    return df.sort_values("deterministic_confidence", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# STEP 8 - figures
# --------------------------------------------------------------------------


def plot_entropy_by_correctness(entropy, correct, model_name, dataset_label,
                                save=True, show=True):
    """FIGURE 1 / FIGURE 5 building block: overlaid entropy histograms,
    correct vs incorrect, for one model on one dataset."""
    import matplotlib.pyplot as plt

    entropy = np.asarray(entropy, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, max(float(entropy.max()), 1e-6), 25)
    ax.hist(entropy[correct], bins=bins, alpha=0.6, color="#4c78a8",
           label=f"correct (n={int(correct.sum())})", edgecolor="black")
    ax.hist(entropy[~correct], bins=bins, alpha=0.6, color="#e45756",
           label=f"incorrect (n={int((~correct).sum())})", edgecolor="black")
    ax.set_xlabel("predictive entropy (MC Dropout mean)")
    ax.set_ylabel("count")
    ax.set_title(f"{model_name} - {dataset_label}\nentropy by correctness")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        d = model_figures_dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}_{dataset_label}_entropy_by_correctness.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_risk_coverage_curve(rc_df: pd.DataFrame, model_name, dataset_label="external",
                             save=True, show=True):
    """FIGURE 2 (MiniConvNet) / FIGURE 3 (VGG16): accuracy and error rate vs
    coverage, plus positive-class recall among retained predictions if
    available."""
    import matplotlib.pyplot as plt

    has_recall = "positive_class_recall" in rc_df.columns
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(rc_df["coverage"], rc_df["accuracy"], marker="o", color="#4c78a8", label="accuracy")
    ax.plot(rc_df["coverage"], rc_df["error_rate"], marker="s", color="#e45756", label="error rate")
    if has_recall:
        ax.plot(rc_df["coverage"], rc_df["positive_class_recall"], marker="^", color="#54a24b",
               label="malignant recall (retained)")
    ax.set_xlabel("coverage (fraction of predictions retained, least-uncertain first)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.invert_xaxis()
    ax.set_title(f"{model_name} - risk-coverage curve ({dataset_label})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        d = model_figures_dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}_{dataset_label}_risk_coverage.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_internal_vs_external_entropy(entropy_internal, entropy_external, model_name,
                                      save=True, show=True):
    """FIGURE 5: internal vs external predictive-entropy distributions,
    overlaid, for one model."""
    import matplotlib.pyplot as plt

    entropy_internal = np.asarray(entropy_internal, dtype=float)
    entropy_external = np.asarray(entropy_external, dtype=float)
    hi = max(float(entropy_internal.max()), float(entropy_external.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, hi, 25)
    ax.hist(entropy_internal, bins=bins, alpha=0.6, color="#4c78a8",
           label=f"internal (n={len(entropy_internal)})", density=True, edgecolor="black")
    ax.hist(entropy_external, bins=bins, alpha=0.6, color="#f58518",
           label=f"external (n={len(entropy_external)})", density=True, edgecolor="black")
    ax.set_xlabel("predictive entropy (MC Dropout mean)")
    ax.set_ylabel("density")
    ax.set_title(f"{model_name} - internal vs external uncertainty")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        d = model_figures_dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}_internal_vs_external_entropy.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_internal_vs_external_accuracy_recall(model_name, internal_accuracy, external_accuracy,
                                              internal_recall, external_recall,
                                              save=True, show=True):
    """FIGURE 6: grouped bars, internal vs external, for accuracy (tumour vs
    normal / malignant vs normal) and recall (internal 'tumour recall' vs
    external 'malignant recall' - the same structural comparison under two
    different label vocabularies, kept clearly labelled, not conflated)."""
    import matplotlib.pyplot as plt

    labels = ["accuracy", "recall"]
    internal_vals = [internal_accuracy, internal_recall]
    external_vals = [external_accuracy, external_recall]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width / 2, internal_vals, width, label="internal (tumour vs normal)", color="#4c78a8")
    ax.bar(x + width / 2, external_vals, width, label="external (malignant vs normal)", color="#e45756")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(internal_vals):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ax.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(external_vals):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ax.text(i + width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_title(f"{model_name} - internal vs external")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        d = model_figures_dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}_internal_vs_external_accuracy_recall.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_high_confidence_incorrect_grid(hc_df: pd.DataFrame, model_name, max_display=8,
                                        gradcam_overlays=None, img_size=(224, 224),
                                        save=True, show=True):
    """FIGURE 7: a grid of the highest-confidence INCORRECT external
    predictions.

    ``max_display`` bounds how many panels are DRAWN for readability - it does
    NOT change what is reported: :func:`high_confidence_incorrect` already
    returned every qualifying row, and this function's caller should save that
    full table regardless of how many are plotted here. If
    ``gradcam_overlays`` (a ``{filepath: overlay_array}`` dict) is supplied and
    covers a given row's ``filepath``, the overlay is shown in place of the
    raw image for that panel.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if hc_df.empty:
        print(f"{model_name}: no high-confidence-incorrect external predictions to plot.")
        return None

    subset = hc_df.head(max_display).reset_index(drop=True)
    n = len(subset)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.8 * rows), squeeze=False)

    for i, (_, r) in enumerate(subset.iterrows()):
        ax = axes[i // cols][i % cols]
        fp = r.get("filepath")
        overlay = (gradcam_overlays or {}).get(fp)
        try:
            if overlay is not None:
                ax.imshow(overlay)
                tag = " (Grad-CAM)"
            else:
                with Image.open(fp) as im:
                    ax.imshow(im.convert("RGB").resize((img_size[1], img_size[0])))
                tag = ""
        except Exception as exc:
            ax.text(0.5, 0.5, f"image unavailable\n{exc}", ha="center", va="center", fontsize=7)
            tag = ""
        ax.axis("off")
        true_l = r.get("true_label", "?")
        pred_l = r.get("pred_label", "?")
        ax.set_title(f"true={true_l} pred={pred_l}\nconf={r['deterministic_confidence']:.2f} "
                    f"H={r['mc_predictive_entropy']:.3f}{tag}", fontsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"{model_name} - high-confidence INCORRECT external predictions "
                f"({len(hc_df)} total, showing {n})", fontsize=10)
    fig.tight_layout()

    path = None
    if save:
        d = model_figures_dir(model_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{model_name}_high_confidence_incorrect_external.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


# --------------------------------------------------------------------------
# Optional: fresh Grad-CAM on the external high-confidence-incorrect images
# --------------------------------------------------------------------------


def gradcam_overlays_for_external_failures(model, hc_df: pd.DataFrame, max_n=8,
                                           img_size=(224, 224), verbose=True) -> dict:
    """Best-effort Grad-CAM overlays for up to ``max_n`` of the external
    high-confidence-incorrect images, reusing the EXACT nested-backbone-aware
    machinery already fixed in Phase 3 (``gradcam_utils.find_last_spatial_layer``
    / ``build_gradcam_submodel`` / ``compute_gradcam``).

    No Grad-CAM output from Phase 3 covers the external dataset (it only ever
    analysed the internal CT test images), so there is nothing to "connect" to
    without generating fresh overlays here - this is that generation step,
    explicitly optional (Step 8 says "if available, optionally connect"; since
    none are available for THIS image set, this produces a small new set
    instead of skipping the idea entirely). Failures are caught per-image and
    reported, never silently producing a blank/garbage overlay.
    """
    from src.gradcam_utils import build_gradcam_submodel, compute_gradcam, find_last_spatial_layer

    if hc_df.empty:
        return {}

    try:
        _, target_tensor, target_shape = find_last_spatial_layer(model)
        grad_model = build_gradcam_submodel(model, target_tensor)
    except Exception as exc:
        if verbose:
            print(f"  Grad-CAM setup failed for this model: {exc}")
            print("  Skipping optional Grad-CAM-on-external-failures section entirely.")
        return {}

    from src.gradcam_utils import load_image_for_gradcam, make_overlay

    overlays = {}
    for fp in hc_df["filepath"].head(max_n):
        try:
            batch, display = load_image_for_gradcam(fp, img_size=img_size)
            heatmap, _ = compute_gradcam(grad_model, batch)
            overlays[fp] = make_overlay(display, heatmap)
        except Exception as exc:
            if verbose:
                print(f"  Grad-CAM failed for {fp}: {exc} - skipping this image only.")
    return overlays


# --------------------------------------------------------------------------
# STEP 9 - final consolidated results table (reads everything live)
# --------------------------------------------------------------------------


def _read_internal_accuracy_this_checkpoint(model_name) -> dict:
    from src.calibration_utils import CALIBRATION_METRICS_CSV

    if not Path(CALIBRATION_METRICS_CSV).exists():
        return {"value": None, "source": "not available"}
    df = pd.read_csv(CALIBRATION_METRICS_CSV)
    row = df[(df["model"] == model_name) & (df["stage"] == "before_temperature")]
    if row.empty:
        return {"value": None, "source": "not available"}
    return {"value": float(row.iloc[0]["accuracy"]),
           "source": f"{CALIBRATION_METRICS_CSV} (this checkpoint's own test-set accuracy)"}


def _read_cv_headline(model_name) -> dict:
    """The project's official, published headline number - reported as a
    reference note alongside (never instead of) the checkpoint-specific
    internal accuracy above, since they answer different questions."""
    from src.config import RESULTS_TABLE_CSV

    if model_name == "MiniConvNet" and Path(RESULTS_TABLE_CSV).exists():
        canon = pd.read_csv(RESULTS_TABLE_CSV)
        row = canon[canon["model"].astype(str).str.startswith("MiniConvNet")]
        if len(row):
            r = row.iloc[0]
            return {"value": float(r["accuracy"]), "std": float(r.get("accuracy_std", float("nan"))),
                   "source": f"{RESULTS_TABLE_CSV} (3-fold CV headline)"}
    if model_name != "MiniConvNet" and Path(FAIR_METRICS_CSV).exists():
        fair = pd.read_csv(FAIR_METRICS_CSV)
        row = fair[(fair["model"] == model_name) & (fair["run_type"] == "finetuned")]
        if len(row):
            return {"value": float(row.iloc[0]["accuracy"]), "std": None,
                   "source": f"{FAIR_METRICS_CSV} (single fine-tuned run, no CV)"}
    return {"value": None, "std": None, "source": "not available"}


def _read_leakage_controlled_accuracy(model_name) -> dict:
    """Reads the ALREADY-COMPUTED ``summary.accuracy_mean``/``accuracy_std``
    from ``outputs/leakage/leakage_controlled_cv_results.json`` directly,
    rather than re-deriving mean/std from ``per_fold_results`` here - this
    guarantees an exact match to the number as already published (74.10% +/-
    4.24%) instead of risking a second, independently-computed aggregation
    drifting from it.
    """
    from src.config import OUTPUT_ROOT

    path = OUTPUT_ROOT / "leakage" / "leakage_controlled_cv_results.json"
    if model_name != "MiniConvNet" or not path.exists():
        return {"value": None, "std": None, "source": "not available",
               "note": ("Leakage-controlled (Option B) 3-fold CV was run for MiniConvNet only "
                        "- not attempted for baselines." if model_name != "MiniConvNet" else
                        "results file not found")}
    d = json.loads(path.read_text())
    summary = d.get("summary")
    if not summary or "accuracy_mean" not in summary:
        return {"value": None, "std": None, "source": "not available",
               "note": "'summary' block missing or incomplete in leakage_controlled_cv_results.json"}
    return {"value": float(summary["accuracy_mean"]), "std": float(summary.get("accuracy_std", 0.0)),
           "source": f"{path} (summary.accuracy_mean / accuracy_std, "
                    f"{summary.get('n_folds_valid', '?')}/{summary.get('n_folds_total', '?')} valid folds)"}


def _read_external_metrics(model_name) -> dict:
    from src.external_eval_utils import EXTERNAL_METRICS_CSV

    if not Path(EXTERNAL_METRICS_CSV).exists():
        return {"accuracy": None, "auc": None, "malignant_recall": None, "source": "not available"}
    df = pd.read_csv(EXTERNAL_METRICS_CSV)
    row = df[(df["model"] == model_name) & (df["framing"] == "malignant_vs_normal_excl_benign")]
    if row.empty:
        return {"accuracy": None, "auc": None, "malignant_recall": None, "source": "not available"}
    r = row.iloc[0]
    return {"accuracy": float(r["accuracy"]), "auc": (float(r["auc"]) if pd.notna(r["auc"]) else None),
           "malignant_recall": float(r["recall_tumor"]),
           "source": f"{EXTERNAL_METRICS_CSV} (primary framing: malignant_vs_normal_excl_benign)"}


def _read_calibration(model_name) -> dict:
    from src.calibration_utils import CALIBRATION_METRICS_CSV

    if not Path(CALIBRATION_METRICS_CSV).exists():
        return {"ece": None, "brier_score": None, "source": "not available"}
    df = pd.read_csv(CALIBRATION_METRICS_CSV)
    row = df[(df["model"] == model_name) & (df["stage"] == "before_temperature")]
    if row.empty:
        return {"ece": None, "brier_score": None, "source": "not available"}
    r = row.iloc[0]
    return {"ece": float(r["ece"]), "brier_score": float(r["brier_score"]),
           "source": f"{CALIBRATION_METRICS_CSV} (before_temperature, raw model output)"}


def build_final_results_table(model_names, params_by_model, internal_uncertainty_stats,
                              risk_coverage_tables, representative_coverage=0.8) -> pd.DataFrame:
    """Every column Step 9 asks for, each traced to a live source file (never
    hand-typed) - or explicitly marked 'not available' when the underlying
    metric genuinely was not computed for that model (e.g. leakage-controlled
    CV was MiniConvNet-only).

    ``internal_uncertainty_stats`` and ``risk_coverage_tables`` are supplied by
    the caller because they are THIS notebook's own freshly computed results
    (Steps 3-5), not something persisted by an earlier phase to read from disk.
    """
    rows = []
    for model_name in model_names:
        internal_acc = _read_internal_accuracy_this_checkpoint(model_name)
        cv_headline = _read_cv_headline(model_name)
        leakage_ctrl = _read_leakage_controlled_accuracy(model_name)
        external = _read_external_metrics(model_name)
        calib = _read_calibration(model_name)

        uinfo = internal_uncertainty_stats.get(model_name, {})
        mean_entropy = uinfo.get("mean_predictive_entropy")
        cvi = uinfo.get("correct_vs_incorrect")
        if cvi and cvi.get("mean_entropy_delta_incorrect_minus_correct") is not None:
            delta = cvi["mean_entropy_delta_incorrect_minus_correct"]
            relationship = (f"higher entropy for incorrect (delta=+{delta:.4f})" if delta > 0 else
                           f"NOT higher for incorrect (delta={delta:.4f})" if delta < 0 else
                           "no difference (delta=0)")
        else:
            relationship = "not available"

        rc = risk_coverage_tables.get(model_name)
        if rc is not None and len(rc):
            near = rc.iloc[(rc["coverage"] - representative_coverage).abs().argsort()[:1]]
            r = near.iloc[0]
            sel_perf = (f"at {int(r['coverage']*100)}% coverage: error "
                       f"{rc.iloc[0]['error_rate']*100:.1f}% -> {r['error_rate']*100:.1f}%")
        else:
            sel_perf = "not available"

        rows.append({
            "model": model_name,
            "parameters": params_by_model.get(model_name),
            "internal_accuracy_this_checkpoint": internal_acc["value"],
            "internal_accuracy_source": internal_acc["source"],
            "internal_cv_headline_accuracy": cv_headline.get("value"),
            "internal_cv_headline_std": cv_headline.get("std"),
            "leakage_controlled_accuracy": leakage_ctrl["value"],
            "leakage_controlled_std": leakage_ctrl["std"],
            "leakage_controlled_note": leakage_ctrl.get("note", ""),
            "external_accuracy": external["accuracy"],
            "external_auc": external["auc"],
            "external_malignant_recall": external["malignant_recall"],
            "ece": calib["ece"],
            "brier_score": calib["brier_score"],
            "mean_uncertainty_internal": mean_entropy,
            "uncertainty_error_relationship": relationship,
            "selective_prediction_performance": sel_perf,
        })
    return pd.DataFrame(rows)


def save_final_results_table(df: pd.DataFrame) -> Path:
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_RESULTS_TABLE_CSV, index=False)
    df.to_json(FINAL_RESULTS_TABLE_JSON, orient="records", indent=2)
    return FINAL_RESULTS_TABLE_CSV


# --------------------------------------------------------------------------
# Artefact writing - per-image predictions, metrics rows, selective-prediction
# results (Step 13 reproducibility requirements)
# --------------------------------------------------------------------------


def save_uncertainty_predictions(model_name, dataset_label, frame: dict) -> Path:
    """Per-image predictions/uncertainty values for one (model, dataset) pair.
    ``frame`` is a plain dict of equal-length arrays; every key becomes a
    column, so the internal and external callers can supply different
    (but self-describing) column sets."""
    d = model_predictions_used_dir(model_name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{model_name}_{dataset_label}_uncertainty_predictions.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


UNCERTAINTY_RECORD_COLUMNS = [
    "model", "dataset", "n_samples", "n_passes", "seed", "checkpoint_used",
    "checkpoint_substitution_note", "mean_predictive_entropy", "median_predictive_entropy",
    "mean_confidence_variance", "mean_entropy_correct", "mean_entropy_incorrect",
    "entropy_higher_for_incorrect", "n_high_confidence_incorrect",
    "high_confidence_threshold", "notes", "timestamp",
]


def record_uncertainty_result(row: dict) -> Path:
    unknown = set(row) - set(UNCERTAINTY_RECORD_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column(s): {sorted(unknown)}. Allowed: {UNCERTAINTY_RECORD_COLUMNS}")
    if not row.get("model") or not row.get("dataset"):
        raise ValueError("uncertainty rows require both 'model' and 'dataset'.")

    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if UNCERTAINTY_METRICS_CSV.exists():
        table = pd.read_csv(UNCERTAINTY_METRICS_CSV)
        for c in UNCERTAINTY_RECORD_COLUMNS:
            if c not in table.columns:
                table[c] = pd.NA
        table = table[UNCERTAINTY_RECORD_COLUMNS]
        table = table[~((table["model"] == row["model"]) & (table["dataset"] == row["dataset"]))]
    else:
        table = pd.DataFrame(columns=UNCERTAINTY_RECORD_COLUMNS)

    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[UNCERTAINTY_RECORD_COLUMNS]
    table.to_csv(UNCERTAINTY_METRICS_CSV, index=False)
    table.to_json(UNCERTAINTY_METRICS_JSON, orient="records", indent=2)
    return UNCERTAINTY_METRICS_CSV


def load_uncertainty_results() -> pd.DataFrame:
    if not UNCERTAINTY_METRICS_CSV.exists():
        return pd.DataFrame(columns=UNCERTAINTY_RECORD_COLUMNS)
    return pd.read_csv(UNCERTAINTY_METRICS_CSV)


def save_selective_prediction_results(all_rc_tables: dict) -> Path:
    """Stack every model/dataset's risk-coverage table into one tracked CSV."""
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for (model_name, dataset_label), df in all_rc_tables.items():
        d = df.copy()
        d.insert(0, "dataset", dataset_label)
        d.insert(0, "model", model_name)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(SELECTIVE_PREDICTION_CSV, index=False)
    return SELECTIVE_PREDICTION_CSV
