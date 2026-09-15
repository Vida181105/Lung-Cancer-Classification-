"""Phase 4 - calibration and uncertainty for MiniConvNet and the strongest
Phase 2 baseline (VGG16, verified programmatically - never hardcoded).

Entirely additive. Nothing here is imported by notebooks 00-11, and every
artefact goes to ``results/calibration/``. No existing result, prediction, or
checkpoint is read for writing or modified.

Hard constraints this module was written under
------------------------------------------------
* **CPU only.** No CUDA assumptions anywhere.
* **No retraining, ever.** Every function either loads an existing ``.keras``
  checkpoint and runs it forward (inference only - MC Dropout included, which
  needs stochastic forward passes, not gradient updates) or operates on
  already-computed probabilities. There is no ``.fit()`` call anywhere in this
  file.
* **Never silently substitute a different checkpoint instance's predictions.**
  :func:`locate_or_generate_predictions` implements the addendum's exact-name
  search-then-generate logic: it looks for a prediction file whose name is an
  EXACT match for the run being calibrated (e.g. ``miniconvnet_single_run``,
  ``vgg16_runB_finetuned``), reports and REJECTS any near-miss file for the
  same model under a different run name (e.g. the older
  ``vgg16_faithful_predictions.csv`` or ``miniconvnet_faithful_predictions.csv``
  from earlier phases), and only falls back to a fresh forward pass over the
  checkpoint - never to an older run's saved numbers.
* **Temperature is fit on validation data only.** :func:`fit_temperature`
  takes ``y_true_val``/``y_prob_val`` explicitly; nothing in this module lets
  a caller fit it against the test set, and the notebook is written so test
  probabilities are only ever touched for reporting, after fitting is done.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, NUM_CLASSES, PROJECT_ROOT, SEED, BATCH_SIZE
from src.config import PREDICTIONS_DIR as OUTPUTS_PREDICTIONS_DIR
from src.finetune_utils import FAIR_PREDICTIONS_DIR, CHECKPOINTS_LOCAL, FAIR_METRICS_CSV
from src.gradcam_utils import identify_strongest_finetuned_baseline   # reused, not duplicated

# --------------------------------------------------------------------------
# Paths - all new
# --------------------------------------------------------------------------

CALIBRATION_DIR = PROJECT_ROOT / "results" / "calibration"
CALIBRATION_FIGURES_DIR = CALIBRATION_DIR / "figures"
CALIBRATION_PREDICTIONS_USED_DIR = CALIBRATION_DIR / "predictions_used"

CALIBRATION_METRICS_CSV = CALIBRATION_DIR / "calibration_metrics.csv"
CALIBRATION_METRICS_JSON = CALIBRATION_DIR / "calibration_metrics.json"
MC_DROPOUT_RESULTS_JSON = CALIBRATION_DIR / "mc_dropout_results.json"


def ensure_calibration_dirs() -> dict:
    for d in (CALIBRATION_DIR, CALIBRATION_FIGURES_DIR, CALIBRATION_PREDICTIONS_USED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {"calibration_dir": str(CALIBRATION_DIR), "figures": str(CALIBRATION_FIGURES_DIR),
            "predictions_used": str(CALIBRATION_PREDICTIONS_USED_DIR)}


# --------------------------------------------------------------------------
# STEP 1 - locate the EXACT checkpoint instance's predictions, or generate
# them via a forward pass (never train, never substitute a different run)
# --------------------------------------------------------------------------


def find_saved_predictions(run_name: str, model_short_name: str) -> dict:
    """Search the project's known prediction locations for a file whose name
    is an EXACT match for ``run_name``.

    Any file that mentions the same model but under a DIFFERENT run name
    (e.g. ``vgg16_faithful_predictions.csv`` when looking for
    ``vgg16_runB_finetuned``, or ``miniconvnet_faithful_predictions.csv`` /
    ``miniconvnet_clean_predictions.csv`` when looking for
    ``miniconvnet_single_run``) is collected as a "near miss" and reported,
    never used - that is precisely the older-checkpoint substitution the
    addendum prohibits.

    ``results/gradcam/`` is deliberately NOT searched: Phase 3 only saved
    predictions for a small, seeded SUBSET of images (its Grad-CAM sample
    selection), not full test-set coverage, so it can never stand in for a
    calibration evaluation set even if a matching name happened to appear
    there.
    """
    candidates = [
        FAIR_PREDICTIONS_DIR / f"{run_name}_predictions.csv",
        OUTPUTS_PREDICTIONS_DIR / f"{run_name}_predictions.csv",
    ]
    for c in candidates:
        if c.exists():
            return {"found": True, "path": str(c), "near_misses": []}

    near_misses = []
    for d in (FAIR_PREDICTIONS_DIR, OUTPUTS_PREDICTIONS_DIR):
        if not d.exists():
            continue
        for p in sorted(d.glob(f"*{model_short_name}*_predictions.csv")):
            if p.stem != f"{run_name}_predictions":
                near_misses.append(str(p))

    return {
        "found": False, "path": None, "near_misses": near_misses,
        "reason": (f"No file named '{run_name}_predictions.csv' exists in "
                  f"{FAIR_PREDICTIONS_DIR} or {OUTPUTS_PREDICTIONS_DIR}."),
    }


def generate_predictions_via_forward_pass(checkpoint_path, split_df, split_name,
                                          one_hot, batch_size=None):
    """Load an EXISTING checkpoint and run ONE forward pass over the given
    split. This is inference, not training - there is no ``.fit()`` call
    here, only ``tf.keras.models.load_model`` + prediction.
    """
    from src.data_utils import make_dataset
    from src.evaluate_utils import predict
    from src.models import load_model_checkpoint

    frame = split_df[split_df["split"] == split_name].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"split '{split_name}' is empty in the supplied split dataframe.")

    ds = make_dataset(frame, batch_size=batch_size or BATCH_SIZE, one_hot=one_hot)
    model = load_model_checkpoint(checkpoint_path)
    y_true, y_pred, y_prob = predict(model, ds)
    return y_true, y_pred, y_prob, model, frame


def save_predictions_used(run_name, split_name, y_true, y_pred, y_prob, source) -> Path:
    """Mirror whichever predictions ended up being used (found OR generated)
    into this phase's own directory, tagged with provenance.

    This makes Phase 4 self-contained: even though ``results/fair_baseline/
    predictions/`` and freshly generated arrays live only in memory or in
    git-ignored folders, this copy - plus its provenance sidecar - survives
    for anyone re-reading this phase's results later.
    """
    ensure_calibration_dirs()
    frame = {"y_true": np.asarray(y_true).astype(int), "y_pred": np.asarray(y_pred).astype(int)}
    y_prob = np.asarray(y_prob, dtype=float)
    for i in range(min(y_prob.shape[1], NUM_CLASSES)):
        frame[f"prob_{CLASS_NAMES[i]}"] = y_prob[:, i]

    path = CALIBRATION_PREDICTIONS_USED_DIR / f"{run_name}_{split_name}_predictions_used.csv"
    pd.DataFrame(frame).to_csv(path, index=False)

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as fh:
        json.dump({"run_name": run_name, "split": split_name, "source": source,
                  "n_samples": int(len(y_true)),
                  "saved_at": datetime.now().isoformat(timespec="seconds")}, fh, indent=2)
    return path


def locate_or_generate_predictions(run_name, checkpoint_path, split_df, split_name,
                                   one_hot, model_short_name=None, verbose=True) -> dict:
    """The addendum's Step 1 logic in one call: search for the exact run's
    saved predictions; if absent, verify the checkpoint exists (never
    retrain) and generate them via a forward pass; always report exactly
    which source was used and why, and reject/print any near-miss file from
    a different run instance rather than silently using it.
    """
    model_short_name = model_short_name or run_name.split("_")[0]

    if split_name == "test":
        search = find_saved_predictions(run_name, model_short_name)
    else:
        # No phase in this project has ever saved validation-set predictions
        # for any model (Phase 2/3 only evaluated and saved on the TEST set),
        # so a search here would only ever find nothing - skip straight to
        # generation, but say so explicitly rather than pretending to search.
        search = {"found": False, "path": None, "near_misses": [],
                  "reason": (f"Validation-set predictions are never saved by any earlier "
                            f"phase in this project (only test-set predictions are persisted) "
                            f"- generating '{split_name}' predictions fresh is expected, not a "
                            f"fallback from a failed search.")}

    if search["found"]:
        df = pd.read_csv(search["path"])
        y_true = df["y_true"].to_numpy(int)
        y_pred = df["y_pred"].to_numpy(int)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if len(prob_cols) != NUM_CLASSES:
            raise ValueError(f"{search['path']} does not have {NUM_CLASSES} probability columns "
                             f"(found {len(prob_cols)}) - refusing to use it for calibration.")
        y_prob = df[prob_cols].to_numpy(float)
        source = f"SAVED FILE (exact run-name match): {search['path']}"
        model = None
    else:
        if search.get("near_misses") and verbose:
            print(f"  {len(search['near_misses'])} prediction file(s) for '{model_short_name}' were "
                  f"found under a DIFFERENT run name and are REJECTED (would calibrate the wrong "
                  f"checkpoint instance, not '{run_name}'):")
            for p in search["near_misses"]:
                print(f"    REJECTED: {p}")

        ck = Path(checkpoint_path)
        if not ck.exists():
            raise FileNotFoundError(
                f"No saved '{split_name}' predictions exist for '{run_name}' "
                f"({search.get('reason', '')}), AND its checkpoint is missing at {ck}. "
                "This phase does NOT retrain to produce a substitute - restore the checkpoint "
                "from the manual backup made after Phase 2/3 before continuing.")

        if verbose:
            print(f"  {search.get('reason', '')}")
            print(f"  Generating '{split_name}' predictions for '{run_name}' via a forward pass "
                 f"over the existing checkpoint at {ck} (inference only, no training).")
        y_true, y_pred, y_prob, model, _frame = generate_predictions_via_forward_pass(
            ck, split_df, split_name, one_hot)
        source = f"GENERATED via forward pass over checkpoint: {ck}"

    used_path = save_predictions_used(run_name, split_name, y_true, y_pred, y_prob, source)
    if verbose:
        print(f"  USING ({split_name}): {source}")
        print(f"  mirrored to: {used_path}")

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "source": source,
           "model": model, "path_used": str(used_path)}


# --------------------------------------------------------------------------
# STEP 2 - calibration metrics
# --------------------------------------------------------------------------


def expected_calibration_error(y_true, y_prob, n_bins=10) -> dict:
    """Standard (top-label) Expected Calibration Error.

    Bins predictions by their MAX predicted probability (the "confidence" the
    model reports for its own top choice), then measures, per bin, how far the
    bin's actual accuracy is from its mean confidence - weighted by how many
    predictions fall in that bin.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidence = y_prob.max(axis=1)
    prediction = y_prob.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    n = len(y_true)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                        "accuracy": None, "confidence": None})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += (count / n) * abs(acc - conf)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": count,
                    "accuracy": acc, "confidence": conf})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def brier_score(y_true, y_prob) -> float:
    """Multiclass Brier score: mean squared distance between the predicted
    probability vector and the one-hot ground truth, summed over classes."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n, k = y_prob.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def negative_log_likelihood(y_true, y_prob, eps=1e-12) -> float:
    """Mean negative log-likelihood of the true class under the predicted
    distribution. Undefined (would be infinite) at p=0, so probabilities are
    clipped away from exactly 0/1 - the same guard used everywhere probability
    logs are taken in this project.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    p_true = np.clip(y_prob[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def calibration_summary(y_true, y_prob, n_bins=10) -> dict:
    """Every calibration number Phase 4 asks for, in one dict: accuracy, mean
    confidence, ECE, Brier score, NLL - plus the raw per-bin data for the
    reliability diagram.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)

    accuracy = float(np.mean(y_pred == y_true))
    mean_confidence = float(np.mean(y_prob.max(axis=1)))
    ece_result = expected_calibration_error(y_true, y_prob, n_bins)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "confidence_minus_accuracy": round(mean_confidence - accuracy, 6),
        "ece": ece_result["ece"],
        "brier_score": brier_score(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
        "n_bins": n_bins,
        "_ece_bins": ece_result["bins"],
    }


# --------------------------------------------------------------------------
# STEP 4 - temperature scaling, fit on VALIDATION ONLY
# --------------------------------------------------------------------------


def apply_temperature(y_prob, temperature, eps=1e-12) -> np.ndarray:
    """Rescale already-softmaxed probabilities by a temperature T.

    Every model here ends in a softmax layer, so only post-softmax
    probabilities are available - not raw logits. That is not a limitation:
    if ``p = softmax(z)``, then ``z_i = log(p_i) + C`` for an unknown
    per-sample constant C that is IDENTICAL across classes i (softmax is
    shift-invariant). Substituting that into ``softmax(z / T)`` and expanding
    shows the ``exp(C / T)`` factor cancels exactly in the softmax
    normalisation, leaving::

        softmax_i(z / T) == p_i^(1/T) / sum_j p_j^(1/T)

    which is exactly what this function computes (in log-space, for numerical
    stability) - a mathematically exact reimplementation of temperature
    scaling that needs only the softmax outputs already saved/generated by
    Step 1, no access to the model's pre-softmax layer.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    log_p = np.log(np.clip(y_prob, eps, 1.0))
    scaled = log_p / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)          # numerical stability only
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def fit_temperature(y_true_val, y_prob_val, bounds=(0.05, 10.0)) -> dict:
    """Find the temperature minimising NLL on the VALIDATION set only.

    ``scipy.optimize.minimize_scalar`` is used rather than a TensorFlow
    gradient-based fit - this is a single scalar parameter and a bounded 1D
    line search is exact, simpler, and needs no TF graph construction (scipy
    is already an installed dependency of scikit-learn, which is in
    requirements.txt).

    **The test set never appears in this function.** Callers must pass
    validation-set arrays here; test-set probabilities are only ever touched
    afterwards, for reporting the already-fitted temperature's effect.
    """
    from scipy.optimize import minimize_scalar

    y_true_val = np.asarray(y_true_val).astype(int)
    y_prob_val = np.asarray(y_prob_val, dtype=float)

    def _loss(t):
        return negative_log_likelihood(y_true_val, apply_temperature(y_prob_val, t))

    result = minimize_scalar(_loss, bounds=bounds, method="bounded")
    return {
        "temperature": float(result.x),
        "val_nll_at_fitted_T": float(result.fun),
        "val_nll_at_T1": float(negative_log_likelihood(y_true_val, y_prob_val)),
        "val_n_samples": int(len(y_true_val)),
        "bounds_searched": bounds,
        "converged": bool(result.success),
    }


def before_after_comparison(y_true_test, y_prob_test, temperature, n_bins=10) -> dict:
    """Before/after ECE, Brier, NLL, mean confidence AND accuracy on the
    TEST set - computed only after the temperature has already been fitted
    on validation data.

    Includes an explicit accuracy-unchanged check: temperature scaling raises
    every class's probability to the same power ``1/T`` and renormalises, a
    strictly monotonic transform of each sample's probability vector, so it
    cannot change which class has the highest probability. Argmax - and
    therefore accuracy - is mathematically identical before and after. This is
    asserted here, not just claimed, so any drift (which would indicate a bug)
    is caught immediately rather than silently misreported.
    """
    before = calibration_summary(y_true_test, y_prob_test, n_bins)
    y_prob_calibrated = apply_temperature(y_prob_test, temperature)
    after = calibration_summary(y_true_test, y_prob_calibrated, n_bins)

    accuracy_unchanged = abs(before["accuracy"] - after["accuracy"]) < 1e-9
    return {
        "temperature": float(temperature),
        "before": before, "after": after,
        "y_prob_calibrated": y_prob_calibrated,
        "accuracy_unchanged_as_expected": accuracy_unchanged,
        "note": ("Temperature scaling is a monotonic per-sample rescaling of probabilities - "
                "it cannot and does not change any prediction's argmax, so accuracy before and "
                "after must be identical. This is a calibration-quality result only; it says "
                "nothing about, and must never be reported as, a classification-accuracy "
                "improvement." + ("" if accuracy_unchanged else
                                  " WARNING: accuracy changed - this indicates a bug, investigate "
                                  "before trusting these numbers.")),
    }


# --------------------------------------------------------------------------
# STEP 5 - MC Dropout (inference-time stochastic passes, no training)
# --------------------------------------------------------------------------


def has_batchnorm(model) -> bool:
    """Whether the model contains any BatchNormalization layer anywhere
    (including nested backbones).

    MC Dropout needs ``training=True`` at call time to keep Dropout active,
    but that flag also reactivates BatchNorm's batch-statistics behaviour if
    any BN layer is present - which would contaminate the uncertainty
    estimate with an unrelated effect. This is checked and enforced by
    :func:`mc_dropout_predict` rather than assumed; both MiniConvNet
    (BatchNorm off by design) and VGG16 (no BatchNorm layers exist in the
    architecture at all) are expected to return False here.
    """
    import tensorflow as tf

    def _walk(m):
        for layer in m.layers:
            if layer.__class__.__name__ == "BatchNormalization":
                return True
            if isinstance(layer, tf.keras.Model) and _walk(layer):
                return True
        return False

    return _walk(model)


def mc_dropout_raw_passes(model, dataset, n_passes=15, seed=SEED, verbose=True) -> dict:
    """The generic stochastic-forward-pass CORE of MC Dropout (Gal &
    Ghahramani, 2016) - ``n_passes`` forward passes with Dropout active
    (``training=True``), producing a predictive mean, predictive entropy, and
    per-sample confidence variance/std across passes, over WHATEVER label
    space ``dataset`` happens to yield.

    **This is inference only - there is no gradient computation, no optimizer
    step, and no weight update anywhere in this function.** ``training=True``
    here controls layer BEHAVIOUR (Dropout masking, BatchNorm statistics), not
    whether training happens.

    Raises if the model contains any BatchNormalization layer (see
    :func:`has_batchnorm`) rather than silently producing a contaminated
    result.

    Deliberately does NOT compute a "correct" field, unlike
    :func:`mc_dropout_predict` (below, which wraps this and adds that field
    for the internal 4-class task it was built for). This split exists so the
    exact same stochastic mechanism can be reused for the Phase 6
    uncertainty/selective-prediction experiment's EXTERNAL evaluation
    (``src/uncertainty_utils.mc_dropout_external``), where the dataset yields
    3-class IQ-OTH/NCCD labels that are not index-comparable to a 4-class
    ``argmax`` - "correctness" there has to be computed after reducing to the
    binary task, by a caller that knows about that reduction, not baked in
    here. This is a pure extraction for reuse: :func:`mc_dropout_predict`'s
    own public return value is unchanged by this refactor.

    Returns ``{n_passes, n_samples, y_labels, mean_probs, y_pred_mean,
    predictive_entropy, confidence_variance, confidence_std, stacked_probs}``
    - ``y_labels`` is named generically (not ``y_true``) because, for the
    external caller, it is not a "ground truth" in the same label space as
    ``mean_probs``'s columns.
    """
    import tensorflow as tf

    if has_batchnorm(model):
        raise RuntimeError(
            "model contains BatchNormalization layer(s); running with training=True to keep "
            "Dropout active would ALSO perturb BatchNorm's batch statistics, contaminating the "
            "uncertainty estimate. MC Dropout is not attempted for this model.")

    tf.random.set_seed(seed)
    per_pass_probs = []
    y_labels = None

    for p in range(n_passes):
        batch_probs, batch_labels = [], []
        for x, y in dataset:
            out = model(x, training=True)                 # Dropout stays active
            batch_probs.append(out.numpy())
            y_arr = y.numpy()
            if y_arr.ndim > 1:                              # one-hot -> sparse
                y_arr = np.argmax(y_arr, axis=1)
            batch_labels.append(y_arr)
        per_pass_probs.append(np.concatenate(batch_probs, axis=0))
        if p == 0:
            y_labels = np.concatenate(batch_labels, axis=0)
        if verbose:
            print(f"    MC Dropout pass {p + 1}/{n_passes} done "
                 f"({len(per_pass_probs[-1])} samples)")

    stacked = np.stack(per_pass_probs, axis=0)              # [passes, samples, classes]
    mean_probs = stacked.mean(axis=0)
    y_pred_mean = mean_probs.argmax(axis=1)

    eps = 1e-12
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    winning_class_probs_per_pass = stacked[:, np.arange(len(y_pred_mean)), y_pred_mean]
    confidence_variance = winning_class_probs_per_pass.var(axis=0)
    confidence_std = winning_class_probs_per_pass.std(axis=0)

    return {
        "n_passes": int(n_passes), "n_samples": int(len(y_labels)),
        "y_labels": y_labels, "mean_probs": mean_probs, "y_pred_mean": y_pred_mean,
        "predictive_entropy": predictive_entropy,
        "confidence_variance": confidence_variance, "confidence_std": confidence_std,
        "stacked_probs": stacked,
    }


def mc_dropout_predict(model, dataset, n_passes=15, seed=SEED, verbose=True) -> dict:
    """``n_passes`` stochastic forward passes with Dropout active
    (``training=True``), producing a predictive mean, predictive entropy, and
    per-sample confidence variance across passes.

    **This is inference only - there is no gradient computation, no optimizer
    step, and no weight update anywhere in this function.** ``training=True``
    here controls layer BEHAVIOUR (Dropout masking, BatchNorm statistics), not
    whether training happens; calling a Keras model this way is the standard
    MC Dropout technique (Gal & Ghahramani, 2016), not training.

    Raises if the model contains any BatchNormalization layer (see
    :func:`has_batchnorm`) rather than silently producing a contaminated
    result - if that happens for a given model, the caller should report MC
    Dropout as skipped for it, with the reason, per Step 5's brief.

    Implemented via :func:`mc_dropout_raw_passes` (added for Phase 6 reuse on
    the external dataset) - this function's own return value, keys, and
    numeric behaviour are UNCHANGED by that refactor: same seeding, same loop
    order, same formulas, same dict shape as before.
    """
    raw = mc_dropout_raw_passes(model, dataset, n_passes=n_passes, seed=seed, verbose=verbose)

    y_true = raw["y_labels"]
    y_pred_mean = raw["y_pred_mean"]
    correct = (y_pred_mean == y_true)

    return {
        "n_passes": raw["n_passes"], "n_samples": raw["n_samples"],
        "y_true": y_true, "y_pred_mean": y_pred_mean, "mean_probs": raw["mean_probs"],
        "predictive_entropy": raw["predictive_entropy"],
        "confidence_variance": raw["confidence_variance"], "confidence_std": raw["confidence_std"],
        "correct": correct,
    }


def mc_dropout_correct_vs_incorrect(mc_result: dict) -> dict:
    """Summary stats answering "are uncertain predictions more likely to be
    wrong?" - entropy and confidence-variance split by correctness."""
    correct = mc_result["correct"]
    entropy = mc_result["predictive_entropy"]
    conf_var = mc_result["confidence_variance"]

    def _stats(mask, label):
        if not mask.any():
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "mean_entropy": float(entropy[mask].mean()),
            "std_entropy": float(entropy[mask].std()),
            "mean_confidence_variance": float(conf_var[mask].mean()),
        }

    return {"correct": _stats(correct, "correct"), "incorrect": _stats(~correct, "incorrect")}


def print_mc_dropout_time_estimate(seconds_for_one_pass, n_passes_total, label=""):
    """Informational only - MC Dropout is inference, no stop-and-ask gate
    needed, but a multi-minute VGG16 run is worth flagging so the runner knows
    what to expect (per-pass cost varies a lot: MobileNet-class backbones are
    seconds/pass, VGG16 is tens of seconds/pass on CPU).
    """
    projected_total = seconds_for_one_pass * n_passes_total
    tag = f"[{label}] " if label else ""
    print(f"{tag}first pass took {seconds_for_one_pass:.1f}s -> "
         f"~{projected_total:.1f}s ({projected_total / 60:.1f} min) for all {n_passes_total} passes")
    return {"seconds_per_pass": round(seconds_for_one_pass, 2),
           "projected_total_seconds": round(projected_total, 1)}


# --------------------------------------------------------------------------
# STEP 6 - visualisations
# --------------------------------------------------------------------------


def plot_reliability_diagram(y_true, y_prob, model_name, method_label="", n_bins=10,
                             save=True, show=True):
    """Reliability diagram: accuracy vs confidence per bin, against the
    perfect-calibration diagonal, with a bin-count histogram beneath it."""
    import matplotlib.pyplot as plt

    summary = calibration_summary(y_true, y_prob, n_bins)
    bins = summary["_ece_bins"]
    centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    accs = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in bins]
    counts = [b["count"] for b in bins]
    width = (1.0 / n_bins) * 0.9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7.5),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.bar(centers, accs, width=width, color="#4c78a8", edgecolor="black",
           label="accuracy in bin")
    ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="perfect calibration")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0, 1); ax1.set_xlim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)
    tag = f" - {method_label}" if method_label else ""
    ax1.set_title(f"{model_name}{tag}\nreliability diagram (ECE={summary['ece']:.4f})")
    ax1.grid(alpha=0.3)

    ax2.bar(centers, counts, width=width, color="#72b7b2", edgecolor="black")
    ax2.set_xlabel("confidence (max predicted probability)")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        ensure_calibration_dirs()
        fname = f"{model_name}_{(method_label or 'raw').replace(' ', '_')}_reliability.png"
        path = CALIBRATION_FIGURES_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_confidence_distribution(y_true, y_prob, model_name, method_label="",
                                 save=True, show=True):
    """Histogram of predicted confidence (max probability) across all
    predictions."""
    import matplotlib.pyplot as plt

    confidence = np.asarray(y_prob, dtype=float).max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidence, bins=20, range=(0, 1), color="#4c78a8", edgecolor="black")
    ax.axvline(float(confidence.mean()), color="crimson", ls="--",
              label=f"mean confidence = {confidence.mean():.3f}")
    ax.set_xlabel("confidence"); ax.set_ylabel("count")
    tag = f" - {method_label}" if method_label else ""
    ax.set_title(f"{model_name}{tag} - confidence distribution")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        ensure_calibration_dirs()
        fname = f"{model_name}_{(method_label or 'raw').replace(' ', '_')}_confidence_dist.png"
        path = CALIBRATION_FIGURES_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_confidence_by_correctness(y_true, y_prob, model_name, method_label="",
                                   save=True, show=True):
    """Overlaid confidence histograms for correct vs incorrect predictions -
    the direct visual answer to "when the model is confident, is it usually
    right?"."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    confidence = y_prob.max(axis=1)
    correct = y_pred == y_true

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(confidence[correct], bins=bins, alpha=0.6, color="#4c78a8",
           label=f"correct (n={int(correct.sum())})", edgecolor="black")
    ax.hist(confidence[~correct], bins=bins, alpha=0.6, color="#e45756",
           label=f"incorrect (n={int((~correct).sum())})", edgecolor="black")
    ax.set_xlabel("confidence"); ax.set_ylabel("count")
    tag = f" - {method_label}" if method_label else ""
    ax.set_title(f"{model_name}{tag} - confidence by correctness")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        ensure_calibration_dirs()
        fname = f"{model_name}_{(method_label or 'raw').replace(' ', '_')}_confidence_by_correctness.png"
        path = CALIBRATION_FIGURES_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


def plot_uncertainty_vs_correctness(mc_result: dict, model_name, save=True, show=True):
    """Boxplots of MC-Dropout predictive entropy, split by whether the
    predictive-mean prediction was correct - the direct visual answer to
    "are uncertain predictions more likely to be wrong?"."""
    import matplotlib.pyplot as plt

    correct = mc_result["correct"]
    entropy = mc_result["predictive_entropy"]
    groups = [entropy[correct], entropy[~correct]]
    labels = [f"correct (n={int(correct.sum())})", f"incorrect (n={int((~correct).sum())})"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    nonempty_groups = [g for g in groups if len(g)]
    nonempty_labels = [l for l, g in zip(labels, groups) if len(g)]
    # Matplotlib 3.9 renamed ``labels`` to ``tick_labels``.  Support both
    # APIs so the notebook runs with either the Python 3.12 environment or
    # older course/project environments.
    try:
        bp = ax.boxplot(nonempty_groups, tick_labels=nonempty_labels,
                        patch_artist=True)
    except TypeError:
        bp = ax.boxplot(nonempty_groups, labels=nonempty_labels,
                        patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4c78a8", "#e45756"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("predictive entropy (MC Dropout mean)")
    ax.set_title(f"{model_name} - uncertainty vs correctness ({mc_result['n_passes']} passes)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = None
    if save:
        ensure_calibration_dirs()
        path = CALIBRATION_FIGURES_DIR / f"{model_name}_mc_dropout_uncertainty_vs_correctness.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return str(path) if path else None


# --------------------------------------------------------------------------
# Artefact writing
# --------------------------------------------------------------------------

CALIBRATION_RECORD_COLUMNS = [
    "model", "stage", "n_samples", "accuracy", "mean_confidence",
    "confidence_minus_accuracy", "ece", "brier_score", "nll",
    "temperature", "notes", "timestamp",
]


def record_calibration_result(row: dict) -> Path:
    """One row per (model, stage) in ``results/calibration/calibration_metrics.csv``,
    stage in {'before_temperature', 'after_temperature'}. Never overwrites a
    different model/stage's row; re-running one replaces only its own row."""
    unknown = set(row) - set(CALIBRATION_RECORD_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column(s): {sorted(unknown)}. Allowed: {CALIBRATION_RECORD_COLUMNS}")
    if not row.get("model") or not row.get("stage"):
        raise ValueError("calibration rows require both 'model' and 'stage'.")

    ensure_calibration_dirs()
    row = dict(row)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if CALIBRATION_METRICS_CSV.exists():
        table = pd.read_csv(CALIBRATION_METRICS_CSV)
        for c in CALIBRATION_RECORD_COLUMNS:
            if c not in table.columns:
                table[c] = pd.NA
        table = table[CALIBRATION_RECORD_COLUMNS]
        table = table[~((table["model"] == row["model"]) & (table["stage"] == row["stage"]))]
    else:
        table = pd.DataFrame(columns=CALIBRATION_RECORD_COLUMNS)

    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[CALIBRATION_RECORD_COLUMNS]
    table.to_csv(CALIBRATION_METRICS_CSV, index=False)
    table.to_json(CALIBRATION_METRICS_JSON, orient="records", indent=2)
    return CALIBRATION_METRICS_CSV


def load_calibration_results() -> pd.DataFrame:
    if not CALIBRATION_METRICS_CSV.exists():
        return pd.DataFrame(columns=CALIBRATION_RECORD_COLUMNS)
    return pd.read_csv(CALIBRATION_METRICS_CSV)


def save_mc_dropout_summary(all_results: dict) -> Path:
    """Persist the MC-Dropout correct-vs-incorrect summary stats (not the raw
    per-pass arrays, which are large) for every model it ran on."""
    ensure_calibration_dirs()
    serialisable = {}
    for model_name, info in all_results.items():
        serialisable[model_name] = info   # already plain dicts/floats/ints
    with open(MC_DROPOUT_RESULTS_JSON, "w") as fh:
        json.dump(serialisable, fh, indent=2, default=str)
    return MC_DROPOUT_RESULTS_JSON
