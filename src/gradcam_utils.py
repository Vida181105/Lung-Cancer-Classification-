"""Phase 3 - Grad-CAM / Grad-CAM++ explainability for MiniConvNet and the
strongest Phase 2 baseline.

Entirely additive: nothing here is imported by notebooks 00-10, and every
artefact goes to ``results/gradcam/``. No existing result, prediction, or
checkpoint is read for writing or modified.

Hard constraints this module was written under
------------------------------------------------
* **CPU only.** A single forward + backward pass per image is cheap even on
  CPU; there is no training here, so no time-estimate-and-stop gate is needed
  - only an informational per-image timing printout (see
  :func:`print_gradcam_time_estimate`).
* **No retraining, ever.** Every function in this module either loads an
  existing ``.keras`` checkpoint or operates on already-computed predictions.
  There is no ``.fit()`` call anywhere in this file.
* **Nothing is hardcoded that can be discovered.** The strongest Phase 2
  baseline is read from ``results/fair_baseline/metrics_fair_baseline.csv``
  (see :func:`identify_strongest_finetuned_baseline`), not assumed. The target
  convolutional layer is found by walking ``model.layers`` (recursing into any
  nested backbone model) for the last layer whose output has 4 dimensions
  (see :func:`find_last_spatial_layer`), not looked up by a hardcoded name.
* **No dependency beyond what's already in requirements.txt.** Heatmap
  colourisation and resizing use matplotlib + PIL, not OpenCV.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    NORMAL_CLASS,
    NUM_CLASSES,
    PROJECT_ROOT,
    SEED,
)

# --------------------------------------------------------------------------
# Paths - all new
# --------------------------------------------------------------------------

GRADCAM_DIR = PROJECT_ROOT / "results" / "gradcam"
GRADCAM_IMAGES_DIR = GRADCAM_DIR / "images"          # original / heatmap / overlay PNGs
GRADCAM_GRIDS_DIR = GRADCAM_DIR / "grids"             # per-category visualisation grids
GRADCAM_RECORDS_CSV = GRADCAM_DIR / "gradcam_records.csv"
GRADCAM_SUMMARY_JSON = GRADCAM_DIR / "gradcam_summary.json"

CHECKPOINTS_LOCAL = PROJECT_ROOT / "checkpoints_local"
FAIR_METRICS_CSV = PROJECT_ROOT / "results" / "fair_baseline" / "metrics_fair_baseline.csv"

# --------------------------------------------------------------------------
# Selection parameters
# --------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD = 0.75
N_PER_CLASS_CATEGORY = 1        # images per (class, category) combination, per model
SELECTION_CATEGORIES = ("correct_normal_confidence", "correct_high_confidence",
                        "incorrect_normal_confidence", "incorrect_high_confidence")


def ensure_gradcam_dirs() -> dict:
    for d in (GRADCAM_DIR, GRADCAM_IMAGES_DIR, GRADCAM_GRIDS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {"gradcam_dir": str(GRADCAM_DIR), "images": str(GRADCAM_IMAGES_DIR),
            "grids": str(GRADCAM_GRIDS_DIR)}


# --------------------------------------------------------------------------
# Checkpoint discovery and verification (addendum requirement)
# --------------------------------------------------------------------------


def identify_strongest_finetuned_baseline(fair_metrics_csv=None) -> dict:
    """Read ``results/fair_baseline/metrics_fair_baseline.csv`` and return the
    ``finetuned``-type row with the highest accuracy.

    **Never hardcode which baseline "won" Phase 2** - this is read fresh every
    time so the notebook stays correct if Phase 2 is re-run with different
    results. Returns a dict with ``available=False`` and a ``reason`` if the
    file is missing or has no finetuned rows, rather than guessing.
    """
    path = Path(fair_metrics_csv or FAIR_METRICS_CSV)
    if not path.exists():
        return {"available": False,
                "reason": f"{path} not found - Phase 2 (notebook 09) must run first."}

    df = pd.read_csv(path)
    finetuned = df[df["run_type"] == "finetuned"]
    if finetuned.empty:
        return {"available": False,
                "reason": f"{path} exists but has no rows with run_type == 'finetuned'. "
                          "Phase 2's Run B may not have completed for any model."}

    best = finetuned.loc[finetuned["accuracy"].astype(float).idxmax()]
    checkpoint_path = CHECKPOINTS_LOCAL / f"{best['run_name']}.keras"
    return {
        "available": True,
        "model": str(best["model"]),
        "run_name": str(best["run_name"]),
        "accuracy": float(best["accuracy"]),
        "f1_macro": float(best["f1_macro"]) if pd.notna(best.get("f1_macro")) else None,
        "checkpoint_path": str(checkpoint_path),
        "source_csv": str(path),
        "all_finetuned_rows": finetuned[["run_name", "model", "accuracy"]]
                                      .sort_values("accuracy", ascending=False)
                                      .to_dict(orient="records"),
    }


def verify_checkpoint_exists(path, label) -> dict:
    """Whether a single checkpoint file exists. Never trains a substitute."""
    p = Path(path)
    ok = p.exists() and p.is_file()
    return {
        "label": label,
        "path": str(p),
        "exists": ok,
        "size_mb": round(p.stat().st_size / (1024 ** 2), 3) if ok else None,
        "message": (f"OK - {label} checkpoint found at {p}" if ok else
                    f"MISSING - {label} checkpoint NOT found at {p}. "
                    "This phase does NOT retrain to produce a substitute. "
                    "Restore this file from the manual backup made after Phase 2 "
                    "(see notebooks/09 and notebooks/10's final backup checklist) "
                    "before continuing this section."),
    }


def verify_required_checkpoints(primary_path, secondary_path=None,
                                secondary_label="secondary (strongest baseline)") -> dict:
    """Check both checkpoints the notebook depends on. Prints nothing itself -
    the caller (notebook cell) decides how loudly to report it."""
    primary = verify_checkpoint_exists(primary_path, "MiniConvNet (primary)")
    secondary = (verify_checkpoint_exists(secondary_path, secondary_label)
                if secondary_path else
                {"label": secondary_label, "path": None, "exists": False,
                 "message": ("MISSING - no secondary checkpoint path could be determined "
                            "(Phase 2 fair-benchmark results not found).")})
    return {
        "primary": primary,
        "secondary": secondary,
        "both_available": bool(primary["exists"] and secondary["exists"]),
    }


# --------------------------------------------------------------------------
# Programmatic target-layer discovery (never a hardcoded layer name)
# --------------------------------------------------------------------------


def find_last_spatial_layer(model, _prefix=""):
    """Return ``(full_name, output_tensor, shape)`` for the last layer in the
    model whose output has 4 dimensions (batch, H, W, C) - the standard
    Grad-CAM target: the final spatial feature map before any pooling/flatten.

    Recurses into any nested ``tf.keras.Model`` layer (the pretrained backbone
    in every baseline built by ``models.build_baseline()`` is nested this way),
    so this works identically for MiniConvNet (no nesting) and every ImageNet
    backbone (nested) without a single hardcoded layer name anywhere.

    Raises ``ValueError`` if no 4D-output layer is found at all - that is a
    real incompatibility, not something to paper over.
    """
    import tensorflow as tf

    def _collect(m, prefix):
        found = []
        for layer in m.layers:
            if isinstance(layer, tf.keras.Model):
                found.extend(_collect(layer, prefix + layer.name + "/"))
                continue
            try:
                out = layer.output
            except Exception:
                continue                          # layer has no single output tensor
            if isinstance(out, (list, tuple)):
                continue                           # multi-output layer, ambiguous - skip
            shape = getattr(out, "shape", None)
            if shape is not None and len(shape) == 4:
                found.append((prefix + layer.name, out, tuple(shape)))
        return found

    candidates = _collect(model, _prefix)
    if not candidates:
        raise ValueError(
            f"No layer with a 4-dimensional output was found in '{model.name}'. "
            "Grad-CAM needs a spatial feature map layer; this architecture may not "
            "have one reachable this way.")
    return candidates[-1]


class _NestedBackboneGradModel:
    """Callable replicating ``grad_model(image_batch, training=False) ->
    (conv_output, predictions)`` for models that wrap their pretrained
    backbone as a single nested ``tf.keras.Model`` layer.

    Why this exists (the bug this class fixes)
    --------------------------------------------
    ``models.build_baseline()`` builds every baseline (ResNet50, VGG16,
    EfficientNetV2B0, MobileNetV3Small) as::

        Input -> Lambda(preprocess_input) -> base(x, training=...) -> GAP -> Dense -> ... -> Dense

    where ``base`` is a whole pretrained Keras Functional model (e.g.
    ``keras.applications.VGG16(...)``) used as a single nested layer. When
    :func:`find_last_spatial_layer` recurses into that nested model to find its
    last 4D-output layer, the tensor it returns belongs to ``base``'s OWN
    original functional graph (rooted at ``base.input``) - Keras 3 does not
    re-trace a nested model's internals onto the outer model's input when the
    nested model is called as a layer, so that tensor is genuinely NOT
    connected to the outer model's ``inputs``. Trying to build
    ``tf.keras.Model(inputs=model.inputs, outputs=[that_tensor, model.output])``
    is exactly what raises ``"Output with path `0` is not connected to
    `inputs`"`` - this was reproduced against
    ``checkpoints_local/vgg16_runB_finetuned.keras`` and is expected for all
    four baselines, since they are all built the same way.

    MiniConvNet has no nested backbone (every layer is added directly to the
    outer model), so the direct, single-graph construction in
    :func:`build_gradcam_submodel` already works for it and is completely
    unaffected by this class - this is a fallback path, only ever reached when
    the direct construction fails.

    How it reconnects the graph
    ----------------------------
    1. The backbone's OWN input/output are used to build a small, genuinely
       connected sub-model: ``tf.keras.Model(backbone.input, [target_tensor,
       backbone.output])``. This works because both tensors live in the
       backbone's own self-contained graph.
    2. The outer model's layers before the backbone (the preprocessing
       ``Lambda``, or none at all) and after it (GAP, Dense, Dropout, ...,
       the final softmax) are replayed as plain layer calls around that
       sub-model, inside the SAME ``tf.GradientTape`` context used by
       :func:`compute_gradcam` / :func:`compute_gradcam_plusplus` - so
       gradients flow correctly from the final class score, back through the
       head layers, through the backbone sub-model, to the target conv layer.

    This reconstructs the exact original forward pass (preprocessing ->
    backbone -> head), just as two connected stages instead of one.
    """

    def __init__(self, pre_layers, backbone_grad_model, post_layers):
        self._pre_layers = list(pre_layers)
        self._backbone_grad_model = backbone_grad_model
        self._post_layers = list(post_layers)

    def __call__(self, x, training=False):
        for layer in self._pre_layers:
            x = layer(x)
        conv_output, backbone_features = self._backbone_grad_model(x, training=training)
        y = backbone_features
        for layer in self._post_layers:
            y = layer(y, training=training)
        return conv_output, y


def _find_nested_backbone(model):
    """The first top-level layer that is itself a ``tf.keras.Model``, plus the
    outer layers before and after it, in ``model.layers`` order.

    Every architecture built by ``models.build_baseline()`` has exactly one
    such nested layer (the pretrained backbone). Returns
    ``(nested_backbone, pre_layers, post_layers)``, or ``(None, None, None)``
    if the model has no nested ``tf.keras.Model`` layer at all (e.g.
    MiniConvNet), so callers can tell "flat architecture" apart from "nested
    architecture whose backbone could not be located" (which would be a real,
    reportable failure rather than something to paper over).
    """
    import tensorflow as tf

    layers = model.layers
    backbone_idx = next((i for i, l in enumerate(layers)
                        if isinstance(l, tf.keras.Model)), None)
    if backbone_idx is None:
        return None, None, None

    nested_backbone = layers[backbone_idx]
    pre_layers = [l for l in layers[:backbone_idx]
                 if not isinstance(l, tf.keras.layers.InputLayer)]
    post_layers = layers[backbone_idx + 1:]
    return nested_backbone, pre_layers, post_layers


def build_gradcam_submodel(model, target_output_tensor):
    """Callable mapping the original inputs to (conv output, class scores).

    **Flat architectures (e.g. MiniConvNet) - unchanged path.** A single
    ``tf.keras.Model(inputs=model.inputs, outputs=[target_output_tensor,
    model.output])`` works directly, because every layer sits in one graph.
    This is tried first and, when it succeeds, is returned exactly as before -
    nothing about this path has changed.

    **Nested-backbone architectures (every baseline from
    ``models.build_baseline()``) - added fallback.** The direct construction
    above raises (target tensor lives in the nested backbone's own graph, not
    the outer model's - see :class:`_NestedBackboneGradModel` for the full
    explanation). On that failure, this function locates the nested backbone
    via :func:`_find_nested_backbone` and returns a
    :class:`_NestedBackboneGradModel` instead, which reconnects
    preprocessing -> backbone -> head as two linked stages so gradients still
    flow end-to-end back to the target conv layer.

    If no nested backbone can be found either (so the failure is NOT the known
    nested-architecture case), the original exception is re-raised - callers
    should treat that as "Grad-CAM is not reliably supported for this
    architecture" (Step 3's documented-limitation path) rather than forcing a
    workaround.
    """
    import tensorflow as tf

    try:
        return tf.keras.Model(inputs=model.inputs,
                              outputs=[target_output_tensor, model.output])
    except Exception as direct_exc:
        nested_backbone, pre_layers, post_layers = _find_nested_backbone(model)
        if nested_backbone is None:
            raise direct_exc

        backbone_grad_model = tf.keras.Model(
            inputs=nested_backbone.input,
            outputs=[target_output_tensor, nested_backbone.output])
        return _NestedBackboneGradModel(pre_layers, backbone_grad_model, post_layers)


# --------------------------------------------------------------------------
# Grad-CAM and Grad-CAM++
# --------------------------------------------------------------------------


def compute_gradcam(grad_model, image_batch, class_index=None, eps=1e-8):
    """Standard Grad-CAM (Selvaraju et al. 2017) for one image.

    ``image_batch`` is a single-image batch, raw ``[0, 255]`` float32 - the
    model's own in-model ``Rescaling``/``preprocess_input`` handles scaling, so
    the caller never needs to know which preprocessing a given architecture
    expects.

    Returns ``(heatmap[h, w] in [0, 1], class_index_used)``.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("gradient of the class score w.r.t. the target layer is None - "
                           "the target layer is not on the path from input to output.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
    return heatmap.numpy(), class_index


def compute_gradcam_plusplus(grad_model, image_batch, class_index=None, eps=1e-8):
    """Grad-CAM++ (Chattopadhyay et al. 2018) for one image.

    Uses the standard first-order-gradient algebraic approximation (grads,
    grads-squared, grads-cubed combined from ONE backward pass) rather than
    literal higher-order automatic differentiation - this is the accepted
    practical implementation and is exactly as cheap as plain Grad-CAM.

    Better than Grad-CAM at highlighting multiple/diffuse regions of the same
    class, which matters here because tumour subtype cues in a CT slice are
    not always one compact blob.

    Returns ``(heatmap[h, w] in [0, 1], class_index_used)``. Raises the same
    way :func:`compute_gradcam` does if the gradient is unusable - the caller
    is expected to catch that and record the method as unreliable for this
    architecture rather than silently emitting a degenerate heatmap.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("gradient of the class score w.r.t. the target layer is None.")

    conv_output = conv_output[0]
    grads = grads[0]

    if bool(tf.reduce_any(tf.math.is_nan(grads))) or bool(tf.reduce_any(tf.math.is_inf(grads))):
        raise RuntimeError("NaN/Inf encountered in gradients - Grad-CAM++ is numerically "
                           "unreliable for this architecture/image.")

    grads_sq = tf.square(grads)
    grads_cube = grads_sq * grads
    sum_activations = tf.reduce_sum(conv_output, axis=(0, 1))          # per channel

    alpha_denom = 2.0 * grads_sq + sum_activations[None, None, :] * grads_cube
    alpha_denom = tf.where(tf.abs(alpha_denom) > eps, alpha_denom,
                           tf.ones_like(alpha_denom) * eps)
    alphas = grads_sq / alpha_denom

    alpha_norm = tf.reduce_sum(alphas, axis=(0, 1), keepdims=True)
    alphas = alphas / tf.where(tf.abs(alpha_norm) > eps, alpha_norm,
                               tf.ones_like(alpha_norm) * eps)

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    heatmap = tf.reduce_sum(conv_output * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if bool(tf.math.is_nan(max_val)) or float(max_val) <= 0:
        raise RuntimeError("Grad-CAM++ heatmap collapsed to zero/NaN for this image.")
    heatmap = heatmap / (max_val + eps)
    return heatmap.numpy(), class_index


def check_method_reliability(heatmap: np.ndarray) -> dict:
    """Cheap sanity check on a computed heatmap: not NaN, not constant, not
    degenerate. Used to decide whether to keep trusting a method for a model
    after its first successful image, without re-deriving this per call site.
    """
    finite = np.isfinite(heatmap).all()
    nonzero = float(heatmap.max()) > 0 if finite else False
    varies = bool(np.ptp(heatmap) > 1e-6) if finite else False
    return {"finite": bool(finite), "nonzero": bool(nonzero), "varies": varies,
            "reliable": bool(finite and nonzero and varies)}


# --------------------------------------------------------------------------
# Image loading, overlay, and border-energy diagnostic
# --------------------------------------------------------------------------


def load_image_for_gradcam(filepath, img_size=IMG_SIZE):
    """Decode one image exactly as the training pipeline does: raw [0,255]
    float32, resized to the model's input size. Returns ``(batch[1,H,W,3],
    display_uint8[H,W,3])`` - the batch for the model, the uint8 array for
    plotting the "original image" panel.
    """
    from PIL import Image

    with Image.open(filepath) as im:
        im = im.convert("RGB").resize((img_size[1], img_size[0]), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    display = arr.astype(np.uint8)
    batch = arr[None, ...]
    return batch, display


def colourise_heatmap(heatmap: np.ndarray, target_size) -> np.ndarray:
    """Resize a small heatmap to image size and apply a 'jet' colormap.

    Uses matplotlib + PIL only (no OpenCV, which is not in requirements.txt).
    Returns a uint8 RGB array.
    """
    import matplotlib.cm as cm
    from PIL import Image

    h = np.clip(heatmap, 0, 1)
    img = Image.fromarray((h * 255).astype(np.uint8))
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    h_resized = np.asarray(img, dtype=np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    coloured = cmap(h_resized)[:, :, :3]              # RGBA -> RGB, [0,1]
    return (coloured * 255).astype(np.uint8)


def make_overlay(original_uint8: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    """Alpha-blend a colourised heatmap over the original image."""
    coloured = colourise_heatmap(heatmap, original_uint8.shape[:2])
    blended = (original_uint8.astype(np.float32) * (1 - alpha)
              + coloured.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def border_energy_fraction(heatmap: np.ndarray, border_frac=0.12) -> float:
    """Fraction of total heatmap "energy" (positive activation mass) that
    falls in an outer border ring of the image.

    **This is a structural shortcut-learning diagnostic, not a clinical or
    localisation metric.** It answers "is the model's attention concentrated
    near the image edge (borders/corners/scanner-artifact territory) rather
    than the central tissue region" - it says nothing about tumours and is
    never used that way. See Step 7 for why an actual localisation metric
    (e.g. IoU) is not computed anywhere in this module: no ground-truth masks
    exist for this dataset.
    """
    h, w = heatmap.shape
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    mask = np.zeros_like(heatmap, dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True

    total = float(heatmap.sum())
    if total <= 0:
        return float("nan")
    return float(heatmap[mask].sum() / total)


# --------------------------------------------------------------------------
# STEP 1 - representative, non-cherry-picked sample selection
# --------------------------------------------------------------------------


def categorise_predictions(y_true, y_pred, y_prob,
                           high_conf_threshold=HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """One row per test sample: true/pred class, confidence, correctness,
    and which of the four selection categories it belongs to."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidence = y_prob.max(axis=1)
    correct = y_true == y_pred

    category = np.where(
        correct & (confidence >= high_conf_threshold), "correct_high_confidence",
        np.where(correct & (confidence < high_conf_threshold), "correct_normal_confidence",
        np.where(~correct & (confidence >= high_conf_threshold), "incorrect_high_confidence",
                 "incorrect_normal_confidence")))

    return pd.DataFrame({
        "index": np.arange(len(y_true)),
        "true_label": y_true, "pred_label": y_pred,
        "true_class": [CLASS_NAMES[i] for i in y_true],
        "pred_class": [CLASS_NAMES[i] for i in y_pred],
        "confidence": confidence, "correct": correct, "category": category,
    })


def select_representative_samples(categorised: pd.DataFrame,
                                  n_per_class_category=N_PER_CLASS_CATEGORY,
                                  seed=SEED) -> dict:
    """Seeded, reproducible selection spanning every class and every
    correctness/confidence category. **Never selects only successful
    examples** - every one of the four categories is attempted for every
    class, and any combination with zero available samples is reported as
    missing rather than silently skipped or substituted.

    Returns ``{'selected': DataFrame, 'coverage': DataFrame}`` - the coverage
    table records exactly which (class, category) pairs had 0 candidates, so
    a reader can see what could NOT be shown rather than assuming full
    coverage.
    """
    rng = np.random.default_rng(seed)
    rows, coverage = [], []

    for cls in CLASS_NAMES:
        for cat in SELECTION_CATEGORIES:
            pool = categorised[(categorised["true_class"] == cls)
                               & (categorised["category"] == cat)]
            coverage.append({"class": cls, "category": cat,
                             "n_available": int(len(pool)),
                             "n_selected": min(n_per_class_category, len(pool))})
            if pool.empty:
                continue
            take = min(n_per_class_category, len(pool))
            chosen = pool.iloc[rng.permutation(len(pool))[:take]]
            rows.append(chosen)

    selected = (pd.concat(rows, ignore_index=True) if rows
               else pd.DataFrame(columns=list(categorised.columns)))
    coverage_df = pd.DataFrame(coverage)
    return {"selected": selected, "coverage": coverage_df}


def attach_filepaths(selected: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Join the selection (by row index into the test set) back onto
    filepaths from the test dataframe. ``test_frame`` must be the exact,
    unshuffled frame the model was evaluated on (e.g. ``frames['test']`` from
    ``data_utils.make_split_datasets``) so indices line up.
    """
    out = selected.copy()
    tf_reset = test_frame.reset_index(drop=True)
    out["filepath"] = out["index"].map(tf_reset["filepath"])
    out["filename"] = out["index"].map(tf_reset["filename"])
    missing = out["filepath"].isna().sum()
    if missing:
        raise ValueError(f"{missing} selected row(s) could not be matched to a filepath - "
                         "the supplied test_frame is not aligned with the predictions.")
    return out


# --------------------------------------------------------------------------
# STEP 7 - localisation ground truth (explicitly checked, not assumed)
# --------------------------------------------------------------------------


def scan_for_localization_masks(data_root) -> dict:
    """Look for anything that could be a segmentation/localisation mask.

    Mirrors ``leakage_utils.scan_for_metadata_files``'s rigor for a different
    question: does this dataset ship tumour/lesion masks or bounding-box
    annotations at all? If nothing is found (the expected result for this
    Kaggle CT dataset, which ships classification labels only), quantitative
    localisation evaluation (e.g. IoU) is reported as NOT POSSIBLE - no
    metric is invented or approximated as a substitute.
    """
    data_root = Path(data_root)
    mask_like_patterns = ("mask", "segmentation", "seg_", "_seg", "annotation",
                          "bbox", "roi", "contour", "label_map")
    hits = []
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(pat in name for pat in mask_like_patterns):
            hits.append(str(p.relative_to(data_root)))

    available = bool(hits)
    return {
        "data_root": str(data_root),
        "mask_like_files_found": hits[:50],
        "n_mask_like_files_found": len(hits),
        "localization_masks_available": available,
        "verdict": (
            "Files matching mask/segmentation/annotation naming patterns were found - "
            "inspect them manually before assuming they are usable ground truth."
            if available else
            "NO segmentation masks, bounding boxes, or localisation annotations exist for "
            "this dataset - it ships classification labels only (class folder = label). "
            "CONSEQUENCE: quantitative localisation evaluation (e.g. IoU against ground "
            "truth) is NOT POSSIBLE and is not attempted anywhere in this analysis. Any "
            "region-level interpretation below is qualitative and must be phrased as "
            "'activations appear associated with' / 'concentrate near', never as a "
            "confirmed clinical finding."
        ),
    }


# --------------------------------------------------------------------------
# STEP 6 - saving per-image artefacts and building visualisation grids
# --------------------------------------------------------------------------


GRADCAM_RECORD_COLUMNS = [
    "record_id", "model", "method", "filepath", "filename",
    "true_class", "pred_class", "confidence", "correct", "category",
    "target_layer", "border_energy_fraction",
    "original_path", "heatmap_path", "overlay_path", "status", "timestamp",
]


def save_gradcam_record(record_id, model_name, method, row, heatmap, overlay,
                        display_image, target_layer_name, status="ok") -> dict:
    """Save original / heatmap / overlay PNGs for one image, and return the
    metadata row to accumulate into the master records CSV.

    File naming: ``{record_id}_{model}_{method}_{original|heatmap|overlay}.png`` -
    unambiguous and greppable.
    """
    import matplotlib.pyplot as plt

    ensure_gradcam_dirs()
    stem = f"{record_id}_{model_name}_{method}"
    orig_path = GRADCAM_IMAGES_DIR / f"{stem}_original.png"
    heat_path = GRADCAM_IMAGES_DIR / f"{stem}_heatmap.png"
    over_path = GRADCAM_IMAGES_DIR / f"{stem}_overlay.png"

    plt.imsave(orig_path, display_image)
    plt.imsave(heat_path, heatmap, cmap="jet")
    plt.imsave(over_path, overlay)

    bef = border_energy_fraction(heatmap) if status == "ok" else None
    return {
        "record_id": record_id, "model": model_name, "method": method,
        "filepath": row["filepath"], "filename": row["filename"],
        "true_class": row["true_class"], "pred_class": row["pred_class"],
        "confidence": float(row["confidence"]), "correct": bool(row["correct"]),
        "category": row["category"], "target_layer": target_layer_name,
        "border_energy_fraction": bef,
        "original_path": str(orig_path), "heatmap_path": str(heat_path),
        "overlay_path": str(over_path), "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def append_gradcam_records(rows: list) -> Path:
    """Append rows to ``results/gradcam/gradcam_records.csv`` (creates it with
    headers on first use). Never overwrites earlier rows for a different
    record_id/model/method combination."""
    ensure_gradcam_dirs()
    new = pd.DataFrame(rows, columns=GRADCAM_RECORD_COLUMNS)
    if GRADCAM_RECORDS_CSV.exists():
        existing = pd.read_csv(GRADCAM_RECORDS_CSV)
        key = ["record_id", "model", "method"]
        existing = existing[~existing.set_index(key).index.isin(new.set_index(key).index)]
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new
    out.to_csv(GRADCAM_RECORDS_CSV, index=False)
    return GRADCAM_RECORDS_CSV


def load_gradcam_records() -> pd.DataFrame:
    if not GRADCAM_RECORDS_CSV.exists():
        return pd.DataFrame(columns=GRADCAM_RECORD_COLUMNS)
    return pd.read_csv(GRADCAM_RECORDS_CSV)


def build_category_grid(records: pd.DataFrame, model_name, method, category,
                        max_cols=4, save=True, show=True):
    """A single figure: overlay images for every record matching
    ``model_name``/``method``/``category``, one panel per image, titled with
    true/pred/confidence. Returns the saved path (or None)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    subset = records[(records["model"] == model_name) & (records["method"] == method)
                     & (records["category"] == category)]
    if subset.empty:
        return None

    n = len(subset)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), squeeze=False)

    for i, (_, r) in enumerate(subset.reset_index(drop=True).iterrows()):
        ax = axes[i // cols][i % cols]
        img = Image.open(r["overlay_path"])
        ax.imshow(img)
        ax.axis("off")
        mark = "OK" if r["correct"] else "WRONG"
        ax.set_title(f"true={r['true_class']}\npred={r['pred_class']} "
                     f"({r['confidence']:.2f}) [{mark}]", fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"{model_name} - {method} - {category}", fontsize=11)
    fig.tight_layout()
    path = None
    if save:
        ensure_gradcam_dirs()
        path = GRADCAM_GRIDS_DIR / f"{model_name}_{method}_{category}_grid.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(path) if path else None


# --------------------------------------------------------------------------
# Informational timing (no stop gate needed - this phase is cheap)
# --------------------------------------------------------------------------


def print_gradcam_time_estimate(seconds_elapsed, n_done, n_total, label=""):
    """Print how long the remaining Grad-CAM work is projected to take.

    Purely informational, per the addendum: Grad-CAM is one forward+backward
    pass per image, cheap even on CPU, so there is no stop-and-ask gate here
    the way there is for training - just a number so the runner knows what to
    expect.
    """
    per_image = seconds_elapsed / max(n_done, 1)
    remaining = per_image * max(n_total - n_done, 0)
    tag = f"[{label}] " if label else ""
    print(f"{tag}{n_done}/{n_total} images done, {per_image:.3f}s/image, "
         f"~{remaining:.1f}s remaining for this run")
    return {"n_done": n_done, "n_total": n_total, "seconds_per_image": round(per_image, 4),
           "estimated_remaining_seconds": round(remaining, 1)}
