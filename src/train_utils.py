"""Training helpers: seeding, compilation, class weights, callbacks, history.

The single most important thing in this file is :func:`compile_model`, which
applies ``Adam(learning_rate=1e-4, clipnorm=1.0)`` by DEFAULT. That is the fix
for the dead-ReLU collapse seen in the earlier attempt (LESSON 1) and it is not
an opt-in flag - you have to pass different values explicitly to lose it.
"""

import json
import os
import random
from pathlib import Path

import numpy as np

from src.config import (
    CLIPNORM,
    EARLY_STOPPING_PATIENCE,
    HISTORY_DIR,
    LR_MINICONVNET,
    MIN_LR,
    MODELS_DIR,
    NUM_CLASSES,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
    USE_CLASS_WEIGHTS_BY_SPLIT,
)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_global_seeds(seed=SEED):
    """Seed Python, NumPy and TensorFlow. Call once at the top of a notebook."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:      # older Keras
        pass
    return seed


def gpu_report():
    """Human-readable GPU status - print it before any long training run."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow": tf.__version__,
        "num_gpus": len(gpus),
        "gpus": [g.name for g in gpus],
        "note": "0 GPUs on Kaggle means the accelerator is not set to GPU.",
    }


# --------------------------------------------------------------------------
# Compilation (LESSON 1)
# --------------------------------------------------------------------------


def compile_model(model, lr=LR_MINICONVNET, clipnorm=CLIPNORM,
                  loss="sparse_categorical_crossentropy", metrics=None):
    """Compile with the collapse-safe optimiser defaults.

    Do NOT raise ``lr`` to 1e-3 for MiniConvNet: that is exactly the setting
    that produced flat, dead-ReLU runs in the earlier attempt.
    """
    from tensorflow.keras.optimizers import Adam

    if metrics is None:
        metrics = ["accuracy"]
    optimizer = Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def optimizer_summary(model):
    """What the model was actually compiled with - print it, don't assume."""
    opt = model.optimizer
    try:
        lr = float(opt.learning_rate.numpy())
    except Exception:
        lr = float(np.array(opt.learning_rate))
    return {
        "optimizer": type(opt).__name__,
        "learning_rate": lr,
        "clipnorm": getattr(opt, "clipnorm", None),
    }


# --------------------------------------------------------------------------
# Class weights (LESSON 5)
# --------------------------------------------------------------------------


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """Inverse-frequency weights, mean-normalised to ~1.0.

    Used by default for the ``clean`` split, where removing duplicates leaves
    the ``normal`` class much smaller than the tumour classes.
    """
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Classes {missing} have no samples in this split.")
    weights = counts.sum() / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def class_weights_for(split_variant, labels):
    """Apply the project default: weights for ``clean``, none for ``faithful``."""
    if USE_CLASS_WEIGHTS_BY_SPLIT.get(split_variant, False):
        return compute_class_weights(labels)
    return None


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------


def checkpoint_path(run_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{run_name}.keras"


def make_callbacks(run_name, monitor="val_accuracy", mode="max",
                   patience=EARLY_STOPPING_PATIENCE, checkpoint=True,
                   reduce_lr=True, csv_log=True, verbose=1):
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau + CSVLogger.

    ``restore_best_weights=True`` means the model object left behind after
    ``fit`` is the best-epoch model, which is what evaluation should use.
    """
    from tensorflow.keras import callbacks as cb

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = [cb.EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                            restore_best_weights=True, verbose=verbose)]
    if checkpoint:
        out.append(cb.ModelCheckpoint(str(checkpoint_path(run_name)),
                                      monitor=monitor, mode=mode,
                                      save_best_only=True, verbose=0))
    if reduce_lr:
        out.append(cb.ReduceLROnPlateau(monitor=monitor, mode=mode,
                                        factor=REDUCE_LR_FACTOR,
                                        patience=REDUCE_LR_PATIENCE,
                                        min_lr=MIN_LR, verbose=verbose))
    if csv_log:
        out.append(cb.CSVLogger(str(HISTORY_DIR / f"{run_name}_epochs.csv")))
    return out


# --------------------------------------------------------------------------
# History persistence / plotting
# --------------------------------------------------------------------------


def history_to_dict(history):
    """Accept a Keras ``History`` object or a plain dict."""
    hist = getattr(history, "history", history)
    return {k: [float(v) for v in vals] for k, vals in hist.items()}


def save_history(history, run_name) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{run_name}_history.json"
    with open(path, "w") as fh:
        json.dump(history_to_dict(history), fh, indent=2)
    return path


def load_history(run_name) -> dict:
    path = HISTORY_DIR / f"{run_name}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - was this run trained?")
    with open(path) as fh:
        return json.load(fh)


def plot_history(history, run_name, save=True, show=True):
    """Accuracy and loss curves side by side; returns the figure path or None."""
    import matplotlib.pyplot as plt

    from src.config import FIGURES_DIR

    hist = history_to_dict(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for key, label in (("accuracy", "train"), ("val_accuracy", "val")):
        if key in hist:
            axes[0].plot(hist[key], label=label)
    axes[0].set_title(f"{run_name} - accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key, label in (("loss", "train"), ("val_loss", "val")):
        if key in hist:
            axes[1].plot(hist[key], label=label)
    axes[1].set_title(f"{run_name} - loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = None
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"{run_name}_history.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def final_epoch_summary(history):
    """Last-epoch train/val numbers - printed after every fit call."""
    hist = history_to_dict(history)
    out = {"epochs_trained": len(hist.get("loss", []))}
    for k in ("accuracy", "val_accuracy", "loss", "val_loss"):
        if k in hist and hist[k]:
            out[f"final_{k}"] = round(hist[k][-1], 4)
    if "val_accuracy" in hist and hist["val_accuracy"]:
        out["best_val_accuracy"] = round(max(hist["val_accuracy"]), 4)
        out["best_epoch"] = int(np.argmax(hist["val_accuracy"])) + 1
    return out


def run_name_for(model_key, arch_variant=None, split_variant=None, suffix=None):
    """Consistent run naming so history/checkpoint/figure files line up."""
    parts = [model_key]
    if arch_variant:
        parts.append(arch_variant)
    if split_variant:
        parts.append(split_variant)
    if suffix:
        parts.append(suffix)
    return "_".join(str(p) for p in parts)
