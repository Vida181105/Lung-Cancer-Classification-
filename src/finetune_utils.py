"""Phase 2 - fair re-benchmark of the four existing pretrained baselines.

Entirely additive. Nothing here is imported by notebooks 00-08, and every
artefact goes to ``results/fair_baseline/`` or ``checkpoints_local/`` - the
existing ``outputs/`` tables, predictions and reports are never touched.

Why this phase exists
---------------------
The original baseline runs were not a like-for-like contest with MiniConvNet:

* every baseline trained its **head only** (frozen ImageNet backbone), while
  MiniConvNet trained **all** of its parameters;
* every baseline used the **same** learning rate (1e-4) regardless of
  architecture, which is a reasonable head LR but a poor fine-tuning LR;
* baselines got **25 epochs**, MiniConvNet 40-60;
* baselines used sparse labels with no label smoothing, MiniConvNet one-hot
  with 0.05 smoothing;
* EfficientNetV2B0 alone additionally received inverse-frequency class weights
  as a partial-collapse remedy.

So "MiniConvNet beats ResNet50 by 0.4 points" is currently a comparison between
a fully-trained small model and a frozen feature extractor. This phase gives
each baseline a fair second chance (Run B, partial fine-tuning) and reports
whatever comes out.

Fine-tuning recipe used here
----------------------------
Run B does **not** unfreeze the whole backbone. It unfreezes the top-N layers
only, keeps every ``BatchNormalization`` layer frozen, and drops the learning
rate by 10-100x. That is the standard Keras transfer-learning recipe and it is
also the only affordable one on CPU.

One subtlety worth knowing: ``models.build_baseline(trainable_base=False)``
calls the backbone as ``base(x, training=False)``, so BN layers are traced in
inference mode. Re-enabling ``layer.trainable`` afterwards therefore does *not*
put BN back into batch-statistics mode - which is exactly what you want when
fine-tuning on ~600 images.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    NORMAL_CLASS,
    NUM_CLASSES,
    PROJECT_ROOT,
    SEED,
)

# --------------------------------------------------------------------------
# Paths - all new, none shared with outputs/
# --------------------------------------------------------------------------

FAIR_DIR = PROJECT_ROOT / "results" / "fair_baseline"
FAIR_PREDICTIONS_DIR = FAIR_DIR / "predictions"
FAIR_HISTORIES_DIR = FAIR_DIR / "histories"
FAIR_CONFUSION_DIR = FAIR_DIR / "confusion_matrices"
FAIR_CONFIGS_DIR = FAIR_DIR / "configs"
FAIR_FIGURES_DIR = FAIR_DIR / "figures"

# Checkpoints live OUTSIDE git on purpose (see .gitignore). The previous
# attempts lost every trained model because weights were never retained; this
# folder is what the runner backs up manually afterwards.
CHECKPOINTS_LOCAL = PROJECT_ROOT / "checkpoints_local"

FAIR_METRICS_CSV = FAIR_DIR / "metrics_fair_baseline.csv"
FAIR_METRICS_JSON = FAIR_DIR / "metrics_fair_baseline.json"


def ensure_fair_dirs() -> dict:
    """Create every Phase-2 directory. Safe to call repeatedly."""
    for d in (FAIR_DIR, FAIR_PREDICTIONS_DIR, FAIR_HISTORIES_DIR,
              FAIR_CONFUSION_DIR, FAIR_CONFIGS_DIR, FAIR_FIGURES_DIR,
              CHECKPOINTS_LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "fair_dir": str(FAIR_DIR),
        "predictions": str(FAIR_PREDICTIONS_DIR),
        "histories": str(FAIR_HISTORIES_DIR),
        "confusion_matrices": str(FAIR_CONFUSION_DIR),
        "configs": str(FAIR_CONFIGS_DIR),
        "figures": str(FAIR_FIGURES_DIR),
        "checkpoints_local": str(CHECKPOINTS_LOCAL),
    }


# --------------------------------------------------------------------------
# Per-model fine-tuning configuration
# --------------------------------------------------------------------------

# Deliberately NOT one learning rate for every model. Justification per row:
#
# head_lr   - Run A. With a frozen backbone only a 128-unit head is learning, so
#             a higher LR than the project's global 1e-4 converges faster inside
#             the epoch budget. VGG16 is the exception: it has no normalisation
#             layers and its features are large-magnitude, so it destabilises at
#             1e-3 and keeps the conservative 1e-4.
# ft_lr     - Run B. Backbone weights are already good; a large step destroys
#             them. 10-100x below the head LR is the standard range.
# unfreeze_top_n - how many trailing backbone layers become trainable. Chosen by
#             architecture depth AND by CPU cost: VGG16 is by far the most
#             expensive per epoch (72 s/epoch frozen), so it gets the shallowest
#             unfreeze; MobileNetV3Small is the cheapest and gets the deepest.
BASELINE_FINETUNE_CONFIG = {
    "ResNet50": {
        "head_lr": 1e-3,
        "ft_lr": 1e-5,
        "unfreeze_top_n": 12,          # roughly the conv5 residual block
        "rationale": "deep resnet; last residual block only, BN kept frozen",
    },
    "VGG16": {
        "head_lr": 1e-4,               # no BN anywhere; unstable at 1e-3
        "ft_lr": 1e-5,
        "unfreeze_top_n": 4,           # block5 conv layers
        "rationale": "no normalisation layers + slowest on CPU; shallowest unfreeze",
    },
    "EfficientNetV2B0": {
        "head_lr": 1e-3,
        "ft_lr": 1e-5,
        "unfreeze_top_n": 20,
        "rationale": "compound-scaled blocks; top block group, BN kept frozen",
    },
    "MobileNetV3Small": {
        "head_lr": 1e-3,
        "ft_lr": 1e-4,                 # small model tolerates a larger step
        "unfreeze_top_n": 20,
        "rationale": "cheapest per epoch; deepest unfreeze affordable on CPU",
    },
}

# Epoch budgets for this phase. Run A matches the project's existing baseline
# budget so Run A stays comparable to the original numbers; Run B is shorter
# because fine-tuning starts from an already-good initialisation and because
# backprop through the backbone is expensive on CPU.
EPOCHS_RUN_A = 25
EPOCHS_RUN_B = 15
EARLY_STOPPING_PATIENCE_FAIR = 5


def config_for(model_name: str) -> dict:
    if model_name not in BASELINE_FINETUNE_CONFIG:
        raise ValueError(f"Unknown baseline '{model_name}'. Phase 2 adds no new "
                         f"models; expected one of {list(BASELINE_FINETUNE_CONFIG)}.")
    return dict(BASELINE_FINETUNE_CONFIG[model_name])


# --------------------------------------------------------------------------
# Freezing / unfreezing
# --------------------------------------------------------------------------


def get_nested_backbone(model):
    """Return the pretrained backbone sub-model inside a built baseline.

    ``build_baseline()`` nests the whole ImageNet model as a single layer, so
    unfreezing means reaching into that nested model rather than walking the
    outer one.
    """
    from tensorflow.keras import Model as KModel

    candidates = [l for l in model.layers if isinstance(l, KModel)]
    if not candidates:
        raise RuntimeError("no nested backbone model found - was this built by "
                           "models.build_baseline()?")
    return max(candidates, key=lambda m: m.count_params())


def unfreeze_top_n(model, n, keep_bn_frozen=True, verbose=True) -> dict:
    """Make the last ``n`` backbone layers trainable; keep BatchNorm frozen.

    Freezing BN during fine-tuning is not a stylistic choice - with ~600
    training images, letting BN recompute batch statistics on batches of 32 is
    a well-known way to wreck a pretrained backbone.
    """
    backbone = get_nested_backbone(model)
    backbone.trainable = True

    layers = backbone.layers
    cut = max(len(layers) - int(n), 0)
    n_bn_kept = 0
    for i, layer in enumerate(layers):
        if i < cut:
            layer.trainable = False
            continue
        if keep_bn_frozen and layer.__class__.__name__ == "BatchNormalization":
            layer.trainable = False
            n_bn_kept += 1
        else:
            layer.trainable = True

    report = {
        "backbone_name": backbone.name,
        "backbone_layers": len(layers),
        "requested_unfreeze_top_n": int(n),
        "first_trainable_layer_index": cut,
        "batchnorm_layers_kept_frozen": n_bn_kept,
        "trainable_backbone_layers": sum(1 for l in layers if l.trainable),
    }
    if verbose:
        for k, v in report.items():
            print(f"  {k}: {v}")
    return report


def trainable_report(model) -> dict:
    """Total / trainable / frozen parameter counts for the efficiency table."""
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(tuple(w.shape))) for w in model.trainable_weights))
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "trainable_fraction": round(trainable / max(total, 1), 6),
    }


# --------------------------------------------------------------------------
# Checkpoints + metadata sidecars (the whole point of the addendum)
# --------------------------------------------------------------------------


def checkpoint_paths(run_name: str) -> tuple:
    """``(model.keras, model.json)`` under checkpoints_local/."""
    CHECKPOINTS_LOCAL.mkdir(parents=True, exist_ok=True)
    return (CHECKPOINTS_LOCAL / f"{run_name}.keras",
            CHECKPOINTS_LOCAL / f"{run_name}.json")


def save_checkpoint_with_metadata(model, run_name: str, model_name: str,
                                  run_type: str, metrics: dict, config: dict,
                                  epochs_run: int, extra: dict = None,
                                  verbose=True) -> dict:
    """Save weights + a sidecar JSON describing what the checkpoint actually is.

    The sidecar is what makes a checkpoint reusable months later without having
    to re-derive its provenance: which model, frozen or fine-tuned, what LR,
    how many epochs actually ran, and what it scored.

    Uses ``model.save()`` (not ``save_weights``) so the architecture travels
    with the weights and Phase 3/5 can ``load_model()`` with no rebuild.
    """
    ck_path, meta_path = checkpoint_paths(run_name)
    model.save(str(ck_path))
    size_bytes = ck_path.stat().st_size

    meta = {
        "run_name": run_name,
        "model_name": model_name,
        "run_type": run_type,                       # 'frozen' | 'finetuned' | 'scratch'
        "phase": "phase2_fair_baseline",
        "checkpoint_file": str(ck_path),
        "checkpoint_size_bytes": int(size_bytes),
        "checkpoint_size_mb": round(size_bytes / (1024 ** 2), 3),
        "keras_format": "keras_v3",
        "epochs_actually_run": int(epochs_run),
        "training_config": config,
        "final_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                              else v) for k, v in (metrics or {}).items()},
        "parameters": trainable_report(model),
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if verbose:
        print(f"  checkpoint : {ck_path}  ({meta['checkpoint_size_mb']} MB)")
        print(f"  metadata   : {meta_path}")
    return {"checkpoint": str(ck_path), "metadata": str(meta_path),
            "size_mb": meta["checkpoint_size_mb"]}


def list_local_checkpoints() -> pd.DataFrame:
    """Everything currently in checkpoints_local/, for the final backup checklist."""
    if not CHECKPOINTS_LOCAL.exists():
        return pd.DataFrame(columns=["run_name", "checkpoint", "size_mb",
                                     "metadata_present"])
    rows = []
    for ck in sorted(CHECKPOINTS_LOCAL.glob("*.keras")):
        meta = ck.with_suffix(".json")
        rows.append({
            "run_name": ck.stem,
            "checkpoint": str(ck),
            "size_mb": round(ck.stat().st_size / (1024 ** 2), 3),
            "metadata_present": meta.exists(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Metrics (STEP 5) - full set, one function
# --------------------------------------------------------------------------


def full_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Every metric Phase 2 asks for, in one dict.

    Reuses ``evaluate_utils`` so these numbers are computed by exactly the same
    code as every other result in this project.
    """
    from src.evaluate_utils import (compute_metrics, confusion, per_class_report,
                                    tumor_vs_subtype_breakdown)

    base = compute_metrics(y_true, y_pred, y_prob)
    pc = per_class_report(y_true, y_pred)
    bd = tumor_vs_subtype_breakdown(y_true, y_pred)
    cm = confusion(y_true, y_pred)

    out = dict(base)
    out["tumour_detection_accuracy"] = bd["binary_tumor_vs_healthy_accuracy"]
    out["subtype_accuracy"] = bd["subtype_accuracy_all_tumors"]
    out["most_over_predicted_class"] = bd["most_over_predicted_class"]
    for c in CLASS_NAMES:
        out[f"precision[{c}]"] = float(pc.loc[c, "precision"])
        out[f"recall[{c}]"] = float(pc.loc[c, "recall"])
        out[f"f1[{c}]"] = float(pc.loc[c, "f1-score"])
        out[f"support[{c}]"] = int(pc.loc[c, "support"])
    out["_confusion_matrix"] = cm.tolist()
    out["_per_class_frame"] = pc
    return out


def measure_inference_time(model, dataset, n_batches=5, warmup=1) -> dict:
    """Wall-clock inference cost per batch and per image, on CPU."""
    import tensorflow as tf

    batches = list(dataset.take(n_batches + warmup))
    if not batches:
        return {"inference_batches_timed": 0}
    for x, _ in batches[:warmup]:
        model.predict(x, verbose=0)

    times, n_images = [], 0
    for x, _ in batches[warmup:]:
        t0 = time.time()
        model.predict(x, verbose=0)
        times.append(time.time() - t0)
        n_images += int(tf.shape(x)[0])
    if not times:
        return {"inference_batches_timed": 0}
    total = float(np.sum(times))
    return {
        "inference_batches_timed": len(times),
        "inference_images_timed": int(n_images),
        "inference_seconds_per_batch": round(float(np.mean(times)), 4),
        "inference_ms_per_image": round(1000 * total / max(n_images, 1), 3),
    }


# --------------------------------------------------------------------------
# Artefact writing (STEP 7) - results/fair_baseline/ only
# --------------------------------------------------------------------------


def save_run_artifacts(run_name: str, y_true, y_pred, y_prob, metrics: dict,
                       history=None, config: dict = None, verbose=True) -> dict:
    """Predictions + probabilities + history + confusion matrix + config."""
    ensure_fair_dirs()
    paths = {}

    frame = {"y_true": np.asarray(y_true).astype(int),
             "y_pred": np.asarray(y_pred).astype(int)}
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        for i in range(min(y_prob.shape[1], NUM_CLASSES)):
            frame[f"prob_{CLASS_NAMES[i]}"] = y_prob[:, i]
    p = FAIR_PREDICTIONS_DIR / f"{run_name}_predictions.csv"
    pd.DataFrame(frame).to_csv(p, index=False)
    paths["predictions"] = str(p)

    cm = np.array(metrics["_confusion_matrix"])
    p = FAIR_CONFUSION_DIR / f"{run_name}_confusion.csv"
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES],
                 columns=[f"pred_{c}" for c in CLASS_NAMES]).to_csv(p)
    paths["confusion_matrix"] = str(p)

    if history is not None:
        hist = getattr(history, "history", history)
        p = FAIR_HISTORIES_DIR / f"{run_name}_history.json"
        with open(p, "w") as fh:
            json.dump({k: [float(v) for v in vals] for k, vals in hist.items()},
                      fh, indent=2)
        paths["history"] = str(p)

    if config is not None:
        p = FAIR_CONFIGS_DIR / f"{run_name}_config.json"
        with open(p, "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        paths["config"] = str(p)

    if verbose:
        for k, v in paths.items():
            print(f"  {k:18s} {v}")
    return paths


FAIR_RESULTS_COLUMNS = [
    "run_name", "model", "run_type", "phase",
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
    "cohen_kappa", "mcc", "auc_macro",
    "tumour_detection_accuracy", "subtype_accuracy",
    "total_params", "trainable_params", "trainable_fraction",
    "checkpoint_size_mb", "epochs_run", "train_seconds", "sec_per_epoch",
    "inference_ms_per_image", "learning_rate", "unfreeze_top_n",
    "status", "notes", "timestamp",
]


def record_fair_result(row: dict) -> Path:
    """One row per run in ``results/fair_baseline/metrics_fair_baseline.csv``.

    A separate file from ``outputs/results_table.csv`` by design - the original
    results are preserved untouched (STEP 1). Re-running a run replaces its own
    row rather than appending a duplicate.
    """
    unknown = set(row) - set(FAIR_RESULTS_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown column(s): {sorted(unknown)}. "
                       f"Allowed: {FAIR_RESULTS_COLUMNS}")
    if not row.get("run_name"):
        raise ValueError("fair-benchmark rows require a 'run_name'.")

    ensure_fair_dirs()
    row = dict(row)
    row.setdefault("phase", "phase2_fair_baseline")
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if FAIR_METRICS_CSV.exists():
        table = pd.read_csv(FAIR_METRICS_CSV)
        for c in FAIR_RESULTS_COLUMNS:
            if c not in table.columns:
                table[c] = pd.NA
        table = table[FAIR_RESULTS_COLUMNS]
        table = table[table["run_name"].astype(str) != str(row["run_name"])]
    else:
        table = pd.DataFrame(columns=FAIR_RESULTS_COLUMNS)

    table = pd.concat([table, pd.DataFrame([row])],
                      ignore_index=True)[FAIR_RESULTS_COLUMNS]
    table.to_csv(FAIR_METRICS_CSV, index=False)
    table.to_json(FAIR_METRICS_JSON, orient="records", indent=2)
    return FAIR_METRICS_CSV


def load_fair_results() -> pd.DataFrame:
    if not FAIR_METRICS_CSV.exists():
        return pd.DataFrame(columns=FAIR_RESULTS_COLUMNS)
    return pd.read_csv(FAIR_METRICS_CSV)


# --------------------------------------------------------------------------
# Figures (STEP 7)
# --------------------------------------------------------------------------


def plot_fair_comparison(results: pd.DataFrame, miniconvnet_ref: dict,
                         save=True, show=True) -> dict:
    """Four figures: accuracy, macro-F1, per-class F1, params-vs-performance."""
    import matplotlib.pyplot as plt

    ensure_fair_dirs()
    paths = {}
    if results.empty:
        print("no results to plot yet")
        return paths

    df = results.sort_values("accuracy", ascending=True)
    labels = df["run_name"].tolist()
    mini_acc = miniconvnet_ref.get("accuracy")
    mini_f1 = miniconvnet_ref.get("f1_macro")

    # 1. accuracy
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(df) + 1.5)))
    ax.barh(labels, df["accuracy"], color="#4c78a8")
    if mini_acc is not None:
        ax.axvline(mini_acc, ls="--", color="crimson", lw=1.6,
                   label=f"MiniConvNet {mini_acc:.4f}")
        ax.legend()
    ax.set_xlabel("test accuracy")
    ax.set_title("Fair re-benchmark - accuracy")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        p = FAIR_FIGURES_DIR / "accuracy_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths["accuracy"] = str(p)
    plt.show() if show else plt.close(fig)

    # 2. macro F1
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(df) + 1.5)))
    ax.barh(labels, df["f1_macro"], color="#72b7b2")
    if mini_f1 is not None:
        ax.axvline(mini_f1, ls="--", color="crimson", lw=1.6,
                   label=f"MiniConvNet {mini_f1:.4f}")
        ax.legend()
    ax.set_xlabel("macro F1")
    ax.set_title("Fair re-benchmark - macro F1")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        p = FAIR_FIGURES_DIR / "macro_f1_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths["macro_f1"] = str(p)
    plt.show() if show else plt.close(fig)

    # 3. per-class F1
    pc_cols = [f"f1[{c}]" for c in CLASS_NAMES if f"f1[{c}]" in results.columns]
    if pc_cols:
        fig, ax = plt.subplots(figsize=(11, max(3.5, 0.5 * len(df) + 2)))
        y = np.arange(len(df))
        h = 0.8 / len(pc_cols)
        for k, col in enumerate(pc_cols):
            ax.barh(y + k * h - 0.4, df[col], height=h,
                    label=col.replace("f1[", "").rstrip("]"))
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("per-class F1")
        ax.set_title("Fair re-benchmark - per-class F1")
        ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        if save:
            p = FAIR_FIGURES_DIR / "per_class_f1_comparison.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); paths["per_class_f1"] = str(p)
        plt.show() if show else plt.close(fig)

    # 4. parameters vs performance
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(df["total_params"], df["accuracy"], s=70, color="#e45756",
               label="fairly tuned baselines")
    for _, r in df.iterrows():
        ax.annotate(r["run_name"], (r["total_params"], r["accuracy"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if mini_acc is not None and miniconvnet_ref.get("total_params"):
        ax.scatter([miniconvnet_ref["total_params"]], [mini_acc], s=130, marker="*",
                   color="crimson", label="MiniConvNet")
    ax.set_xscale("log")
    ax.set_xlabel("total parameters (log)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Parameter count vs performance")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        p = FAIR_FIGURES_DIR / "params_vs_performance.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths["params_vs_performance"] = str(p)
    plt.show() if show else plt.close(fig)

    return paths


# --------------------------------------------------------------------------
# STEP 8 - objective conclusion
# --------------------------------------------------------------------------


def decide_conclusion(mini_accuracy, mini_std, results: pd.DataFrame,
                      metric="accuracy") -> dict:
    """Pick A/B/C/D from the numbers. No thumb on the scale.

    A - MiniConvNet still best
    B - MiniConvNet competitive (best baseline is ahead but inside 1 std)
    C - advantage gone (best baseline ahead by 1 std or more)
    D - a baseline clearly outperforms it (ahead by more than 2 std)

    ``mini_std`` is the 3-fold CV standard deviation. The band is deliberately
    generous because the baselines are single runs with **no** error bar, so a
    small nominal difference is not evidence of anything.
    """
    if results.empty:
        return {"verdict": "UNDETERMINED", "reason": "no fair-benchmark runs recorded yet"}

    best_idx = results[metric].astype(float).idxmax()
    best = results.loc[best_idx]
    best_val = float(best[metric])
    gap = best_val - float(mini_accuracy)
    std = float(mini_std) if mini_std else 0.0

    if gap <= 0:
        verdict, letter = "A - MiniConvNet remains superior", "A"
    elif gap < std:
        verdict, letter = "B - MiniConvNet remains competitive", "B"
    elif gap < 2 * std:
        verdict, letter = "C - MiniConvNet's advantage disappears after fair tuning", "C"
    else:
        verdict, letter = "D - a pretrained baseline clearly outperforms MiniConvNet", "D"

    return {
        "letter": letter,
        "verdict": verdict,
        "metric": metric,
        "miniconvnet_value": float(mini_accuracy),
        "miniconvnet_std": std,
        "best_run": str(best["run_name"]),
        "best_model": str(best["model"]),
        "best_value": best_val,
        "gap_vs_miniconvnet": round(gap, 6),
        "gap_in_std_units": (round(gap / std, 3) if std else None),
        "caveat": ("MiniConvNet's figure is a 3-fold CV mean +/- std; every baseline "
                   "here is a SINGLE run with no error bar. A gap smaller than the "
                   "CV std is not evidence of a real difference in either direction."),
    }
