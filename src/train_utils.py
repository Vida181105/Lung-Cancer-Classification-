"""Training helpers: seeding, compilation, class weights, callbacks, CPU timing.

The important function here is :func:`compile_model`, which wires in **four**
of the five lesson-2 countermeasures by default (the fifth, LeakyReLU + He
init + the wide bottleneck, lives in ``src/models.py``):

* ``Adam(learning_rate=1e-4, clipnorm=1.0)``  - LESSON 1, dead-ReLU collapse
* ``CategoricalCrossentropy(label_smoothing=0.05)`` - LESSON 2, cause E
* ``ReduceLROnPlateau(val_loss, patience=5, factor=0.5)`` - LESSON 2, cause D
  (added by :func:`make_callbacks`, on by default)

None of these is an opt-in flag; you have to pass different values explicitly
to lose them.

Because the project is **CPU-only**, this module also owns the time budgeting:
:class:`EpochTimer` logs wall-clock per epoch into every run's history, and
:func:`estimate_training_time` times a real epoch or two and extrapolates, so a
notebook can report a cost and stop for agreement instead of silently starting
a four-hour run.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from src.config import (
    CLIPNORM,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    HISTORY_DIR,
    LABEL_SMOOTHING,
    LR_MINICONVNET,
    MIN_LR,
    MODELS_DIR,
    NUM_CLASSES,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MODE,
    REDUCE_LR_MONITOR,
    REDUCE_LR_PATIENCE,
    SEED,
    TIME_ASK_THRESHOLD_MINUTES,
    TIME_PROBE_EPOCHS,
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


def compute_report():
    """What hardware this actually is. Print it before any training run.

    This project is scoped CPU-only; if a GPU ever does appear, the epoch
    budgets and the 3-fold CV decision are worth revisiting, so the fact is
    surfaced rather than assumed either way.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    return {
        "tensorflow": tf.__version__,
        "num_gpus": len(gpus),
        "gpus": [g.name for g in gpus],
        "num_cpu_devices": len(cpus),
        "os_cpu_count": os.cpu_count(),
        "note": ("CPU-only as planned - keep the scoped budgets."
                 if not gpus else
                 "A GPU is present: the CPU-scoped budgets (3-fold CV, reduced "
                 "epochs) could be revisited, but only deliberately."),
    }


# --------------------------------------------------------------------------
# Compilation (LESSON 1 + LESSON 2 causes D/E)
# --------------------------------------------------------------------------


def compile_model(model, lr=LR_MINICONVNET, clipnorm=CLIPNORM,
                  label_smoothing=LABEL_SMOOTHING, metrics=None, verbose=True):
    """Compile with the collapse-safe defaults.

    Do NOT raise ``lr`` to 1e-3 for MiniConvNet: that is exactly the setting
    that produced flat, dead-ReLU runs in the first attempt.

    ``label_smoothing > 0`` requires **one-hot targets**, because Keras only
    offers smoothing on ``CategoricalCrossentropy``. Build the datasets with
    ``one_hot=True`` (see ``data_utils.make_split_datasets``). Passing
    ``label_smoothing=0`` falls back to the sparse loss, for a deliberate
    control run.
    """
    from tensorflow.keras.losses import (CategoricalCrossentropy,
                                         SparseCategoricalCrossentropy)
    from tensorflow.keras.optimizers import Adam

    if metrics is None:
        metrics = ["accuracy"]
    optimizer = Adam(learning_rate=lr, clipnorm=clipnorm)

    if label_smoothing and label_smoothing > 0:
        loss = CategoricalCrossentropy(label_smoothing=label_smoothing)
        target_format = "one_hot"
    else:
        loss = SparseCategoricalCrossentropy()
        target_format = "sparse"

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        print(f"compiled: Adam(lr={lr}, clipnorm={clipnorm}), "
              f"{loss.__class__.__name__}(label_smoothing={label_smoothing}) "
              f"-> targets must be {target_format.upper()}")
    return model


def optimizer_summary(model):
    """What the model was actually compiled with - print it, don't assume."""
    opt = model.optimizer
    try:
        lr = float(opt.learning_rate.numpy())
    except Exception:
        lr = float(np.array(opt.learning_rate))
    loss = model.loss
    return {
        "optimizer": type(opt).__name__,
        "learning_rate": lr,
        "clipnorm": getattr(opt, "clipnorm", None),
        "loss": loss.__class__.__name__ if hasattr(loss, "__class__") else str(loss),
        "label_smoothing": float(getattr(loss, "label_smoothing", 0.0) or 0.0),
    }


# --------------------------------------------------------------------------
# Class weights (LESSON 7)
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
# Callbacks, including per-epoch wall clock (CPU budgeting)
# --------------------------------------------------------------------------


def checkpoint_path(run_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{run_name}.keras"


def _epoch_timer_class():
    """Defined lazily so importing this module never imports TensorFlow."""
    from tensorflow.keras import callbacks as cb

    class EpochTimer(cb.Callback):
        """Record wall-clock seconds per epoch into ``history`` and stdout.

        On CPU this is the difference between "the run is slow" and "the run is
        stuck", and it is what makes the time estimates in the notebooks real
        measurements rather than guesses.
        """

        def __init__(self, verbose=1):
            super().__init__()
            self.verbose = verbose
            self.epoch_seconds = []
            self._t0 = None

        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            dt = time.time() - (self._t0 or time.time())
            self.epoch_seconds.append(dt)
            if logs is not None:
                logs["epoch_seconds"] = dt
            if self.verbose:
                mean = float(np.mean(self.epoch_seconds))
                print(f"    [timer] epoch {epoch + 1}: {dt:.1f}s "
                      f"(mean {mean:.1f}s over {len(self.epoch_seconds)})")

        def summary(self):
            if not self.epoch_seconds:
                return {}
            secs = np.asarray(self.epoch_seconds, dtype=float)
            return {
                "epochs_timed": int(len(secs)),
                "total_seconds": round(float(secs.sum()), 1),
                "total_minutes": round(float(secs.sum()) / 60, 2),
                "mean_seconds_per_epoch": round(float(secs.mean()), 1),
                "max_seconds_per_epoch": round(float(secs.max()), 1),
            }

    return EpochTimer


def make_epoch_timer(verbose=1):
    """An :class:`EpochTimer` instance - keep the reference, it holds the times."""
    return _epoch_timer_class()(verbose=verbose)


def make_callbacks(run_name, timer=None, monitor=EARLY_STOPPING_MONITOR,
                   mode=EARLY_STOPPING_MODE, patience=EARLY_STOPPING_PATIENCE,
                   checkpoint=True, reduce_lr=True, csv_log=True, verbose=1):
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau + CSVLogger + timer.

    ``restore_best_weights=True`` means the model object left behind after
    ``fit`` is the best-epoch model, which is what evaluation should use.

    ``ReduceLROnPlateau`` watches ``val_loss`` (LESSON 2, cause D) while early
    stopping watches ``val_accuracy`` - deliberately different signals: the loss
    reacts sooner and is smoother on ~600 training images, so the schedule gets
    a chance to rescue a run before early stopping ends it.
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
        out.append(cb.ReduceLROnPlateau(monitor=REDUCE_LR_MONITOR,
                                        mode=REDUCE_LR_MODE,
                                        factor=REDUCE_LR_FACTOR,
                                        patience=REDUCE_LR_PATIENCE,
                                        min_lr=MIN_LR, verbose=verbose))
    if csv_log:
        out.append(cb.CSVLogger(str(HISTORY_DIR / f"{run_name}_epochs.csv")))
    out.append(timer if timer is not None else make_epoch_timer(verbose=verbose))
    return out


# --------------------------------------------------------------------------
# CPU time estimation - measure, then ask
# --------------------------------------------------------------------------


def estimate_training_time(model_fn, train_ds, val_ds, planned_epochs,
                           n_runs=1, probe_epochs=TIME_PROBE_EPOCHS,
                           class_weight=None,
                           threshold_minutes=TIME_ASK_THRESHOLD_MINUTES,
                           verbose=True) -> dict:
    """Time ``probe_epochs`` real epochs on a throwaway model, then extrapolate.

    ``model_fn`` must return a **freshly built, compiled** model: the probe
    trains it and throws it away, so the real run still starts from a clean
    initialisation and the estimate costs one or two epochs, not a warm start.

    Returns the measured seconds/epoch and the projected wall clock for
    ``planned_epochs x n_runs``, plus ``exceeds_threshold`` - which the notebook
    uses to stop and ask rather than launching a multi-hour run unannounced.
    """
    import tensorflow as tf

    probe = model_fn()
    timer = make_epoch_timer(verbose=0)
    t0 = time.time()
    probe.fit(train_ds, validation_data=val_ds, epochs=probe_epochs,
              class_weight=class_weight, callbacks=[timer], verbose=0)
    wall = time.time() - t0
    del probe
    tf.keras.backend.clear_session()

    per_epoch = float(np.mean(timer.epoch_seconds)) if timer.epoch_seconds else wall
    total_epochs = planned_epochs * n_runs
    est_seconds = per_epoch * total_epochs
    est_minutes = est_seconds / 60

    out = {
        "probe_epochs": int(probe_epochs),
        "measured_seconds_per_epoch": round(per_epoch, 1),
        "planned_epochs_per_run": int(planned_epochs),
        "n_runs": int(n_runs),
        "total_epochs_upper_bound": int(total_epochs),
        "estimated_minutes": round(est_minutes, 1),
        "estimated_hours": round(est_minutes / 60, 2),
        "threshold_minutes": threshold_minutes,
        "exceeds_threshold": bool(est_minutes > threshold_minutes),
    }
    if verbose:
        print("TIME ESTIMATE (measured, not guessed)")
        print("-" * 38)
        print(f"  measured        : {out['measured_seconds_per_epoch']}s/epoch "
              f"over {probe_epochs} real epoch(s)")
        print(f"  planned         : {planned_epochs} epochs x {n_runs} run(s) "
              f"= {total_epochs} epochs (upper bound)")
        print(f"  projected       : ~{out['estimated_minutes']} min "
              f"(~{out['estimated_hours']} h)")
        print("  early stopping usually ends runs sooner, so treat this as a ceiling.")
        if out["exceeds_threshold"]:
            print(f"\n  >>> Over the {threshold_minutes}-minute threshold. "
                  "Report this estimate and get agreement BEFORE running.")
        else:
            print(f"\n  Under the {threshold_minutes}-minute threshold - safe to proceed.")
    return out


# --------------------------------------------------------------------------
# Dead-unit probe (LESSON 2 diagnostics)
# --------------------------------------------------------------------------


def dead_unit_report(model, dataset, layer_names=None, batches=4,
                     verbose=True) -> dict:
    """Fraction of units that are <= 0 for *every* sample seen.

    The direct measurement behind lesson 2: if a partial collapse really is
    dying activations, the bottleneck layer will show a high dead fraction. With
    LeakyReLU a "dead" unit still passes a small negative gradient, so this
    reports units that never activate positively - the thing that starves a
    class - rather than assuming zero-gradient units.
    """
    import tensorflow as tf

    if layer_names is None:
        layer_names = [l.name for l in model.layers
                       if l.__class__.__name__ in ("LeakyReLU", "ReLU")]
    probe = tf.keras.Model(model.inputs,
                           [model.get_layer(n).output for n in layer_names])

    ever_positive = {n: None for n in layer_names}
    for i, (x, _) in enumerate(dataset.take(batches)):
        outs = probe(x, training=False)
        if len(layer_names) == 1:
            outs = [outs]
        for n, o in zip(layer_names, outs):
            arr = np.asarray(o)
            pos = (arr > 0).reshape(-1, arr.shape[-1]).any(axis=0)
            ever_positive[n] = pos if ever_positive[n] is None else (ever_positive[n] | pos)

    report = {}
    for n, pos in ever_positive.items():
        if pos is None:
            continue
        dead = int((~pos).sum())
        report[n] = {"units": int(pos.size), "dead_units": dead,
                     "dead_fraction": round(dead / max(pos.size, 1), 4)}
    if verbose:
        print(f"DEAD-UNIT PROBE (over {batches} batches)")
        for n, r in report.items():
            flag = "  <-- starved layer" if r["dead_fraction"] > 0.5 else ""
            print(f"  {n:16s} {r['dead_units']:5d}/{r['units']:<5d} never positive "
                  f"({r['dead_fraction']:.1%}){flag}")
    return report


# --------------------------------------------------------------------------
# History persistence / plotting
# --------------------------------------------------------------------------


def history_to_dict(history):
    """Accept a Keras ``History`` object or a plain dict."""
    hist = getattr(history, "history", history)
    return {k: [float(v) for v in vals] for k, vals in hist.items()}


def save_history(history, run_name, timer=None) -> Path:
    """Persist history JSON, with the per-epoch wall clock folded in."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist = history_to_dict(history)
    if timer is not None and getattr(timer, "epoch_seconds", None):
        hist["epoch_seconds"] = [float(s) for s in timer.epoch_seconds]
    path = HISTORY_DIR / f"{run_name}_history.json"
    with open(path, "w") as fh:
        json.dump(hist, fh, indent=2)
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
    axes[0].axhline(0.25, ls="--", lw=1, color="grey", label="chance (0.25)")
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


def final_epoch_summary(history, timer=None):
    """Last-epoch train/val numbers + wall clock - printed after every fit."""
    hist = history_to_dict(history)
    out = {"epochs_trained": len(hist.get("loss", []))}
    for k in ("accuracy", "val_accuracy", "loss", "val_loss"):
        if k in hist and hist[k]:
            out[f"final_{k}"] = round(hist[k][-1], 4)
    if "val_accuracy" in hist and hist["val_accuracy"]:
        out["best_val_accuracy"] = round(max(hist["val_accuracy"]), 4)
        out["best_epoch"] = int(np.argmax(hist["val_accuracy"])) + 1
    if timer is not None:
        out.update(timer.summary())
    elif "epoch_seconds" in hist and hist["epoch_seconds"]:
        secs = np.asarray(hist["epoch_seconds"], dtype=float)
        out["mean_seconds_per_epoch"] = round(float(secs.mean()), 1)
        out["total_minutes"] = round(float(secs.sum()) / 60, 2)
    return out


# --------------------------------------------------------------------------
# Model file size on disk (LESSON 9 - the "~6MB" claim)
# --------------------------------------------------------------------------


def measure_model_file_sizes(model, run_name="size_check", keep_files=True,
                             verbose=True) -> dict:
    """Save one model both ways and measure what each file actually costs.

    The paper states "~0.5M parameters, ~6MB", but ~499K float32 weights is only
    ~1.9MB. Adam carries two extra buffers per parameter (momentum and
    variance), so a checkpoint that includes the optimizer state should land
    near 3x the weight size. This measures both instead of arguing about it:

    * ``model.save_weights(...)`` - weights only;
    * ``model.save(...)``         - Keras's default, which also serialises the
      optimizer (and therefore its slot variables, once they exist).

    The optimizer slots are created lazily, so a model that has never taken a
    training step has no momentum/variance buffers to save. Train before calling
    this, or the two numbers come out nearly identical for the wrong reason -
    ``optimizer_slot_scalars_found`` in the printout is the guard against
    reporting that mistake as a result.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"{run_name}_weights_only.weights.h5"
    full_path = MODELS_DIR / f"{run_name}_full_with_optimizer.keras"

    model.save_weights(str(weights_path))
    model.save(str(full_path))

    n_params = int(model.count_params())
    weights_bytes = weights_path.stat().st_size
    full_bytes = full_path.stat().st_size
    mb = 1024 ** 2

    try:
        opt_vars = getattr(model.optimizer, "variables", None)
        opt_vars = opt_vars() if callable(opt_vars) else opt_vars
        optimizer_slots = int(sum(int(np.prod(tuple(v.shape))) for v in opt_vars))
    except Exception:                       # never let a probe break the check
        optimizer_slots = 0

    out = {
        "run_name": run_name,
        "total_params": n_params,
        "weights_only_path": str(weights_path),
        "weights_only_bytes": int(weights_bytes),
        "weights_only_MB": round(weights_bytes / mb, 3),
        "full_save_path": str(full_path),
        "full_save_bytes": int(full_bytes),
        "full_save_MB": round(full_bytes / mb, 3),
        "ratio_full_over_weights": round(full_bytes / max(weights_bytes, 1), 3),
        # Arithmetic the hypothesis rests on.
        "predicted_weights_MB_float32": round(n_params * 4 / mb, 3),
        "predicted_weights_plus_adam_MB": round(n_params * 4 * 3 / mb, 3),
        "optimizer_slot_scalars_found": optimizer_slots,
    }

    if verbose:
        print("MODEL FILE SIZE CHECK -", run_name)
        print("-" * 40)
        print(f"  parameters                      : {n_params:,}")
        print(f"  save_weights() -> {weights_path.name}")
        print(f"      {weights_bytes:,} bytes = {out['weights_only_MB']} MB")
        print(f"  save()         -> {full_path.name}")
        print(f"      {full_bytes:,} bytes = {out['full_save_MB']} MB")
        print(f"  ratio full/weights              : {out['ratio_full_over_weights']}")
        print(f"  predicted weights only (4B/p)   : "
              f"{out['predicted_weights_MB_float32']} MB")
        print(f"  predicted weights + Adam (12B/p): "
              f"{out['predicted_weights_plus_adam_MB']} MB")
        print(f"  optimizer slot scalars found    : {optimizer_slots:,}"
              + ("  <- 0 means the model never trained; the two sizes are NOT"
                 " comparable" if optimizer_slots == 0 else ""))
        print(f"  paper's claim                   : ~6 MB")

    if not keep_files:
        weights_path.unlink(missing_ok=True)
        full_path.unlink(missing_ok=True)
    return out


def verdict_on_size_claim(size_check, paper_claim_mb=6.0, tolerance_mb=1.0) -> str:
    """Turn the two measured sizes into a CONFIRMED / REFUTED sentence."""
    w = size_check["weights_only_MB"]
    f = size_check["full_save_MB"]
    if size_check["optimizer_slot_scalars_found"] == 0:
        return ("INCONCLUSIVE: no Adam slot variables were found, so the 'full' save carries "
                "no optimizer state. Train the model before running this check.")
    if abs(f - paper_claim_mb) <= tolerance_mb and w < paper_claim_mb - tolerance_mb:
        return (f"CONFIRMED: weights alone are {w} MB, far short of the paper's "
                f"~{paper_claim_mb} MB, while the default save() including optimizer state is "
                f"{f} MB - within {tolerance_mb} MB of the claim. The paper's figure is "
                f"consistent with a save() that includes the optimizer.")
    if abs(w - paper_claim_mb) <= tolerance_mb:
        return (f"REFUTED: the weights-only file is already {w} MB, itself close to the paper's "
                f"~{paper_claim_mb} MB, so optimizer state is not needed to explain the claim.")
    return (f"REFUTED: neither file matches the paper's ~{paper_claim_mb} MB (weights only {w} MB, "
            f"with optimizer {f} MB). The stated size must come from something else - a different "
            f"precision, serialisation format, or architecture than the one reconstructed here.")


def run_name_for(model_key, split_variant=None, suffix=None):
    """Consistent run naming so history/checkpoint/figure/prediction files line up.

    v3 has one architecture, so (unlike v2) there is no ``arch_variant``
    component - a run is identified by what actually varies: the split and an
    optional suffix (``fold2``, ``dropout_off``, ...).
    """
    parts = [model_key]
    if split_variant:
        parts.append(split_variant)
    if suffix:
        parts.append(suffix)
    return "_".join(str(p) for p in parts)
