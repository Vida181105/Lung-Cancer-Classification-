"""Model definitions: one focused MiniConvNet, plus the frozen baselines.

The architecture decision (LESSON 3)
------------------------------------
The paper states "~0.5M parameters, ~6MB" but its layer list
(``GlobalAveragePooling2D -> Dense(64) -> Dropout -> Dense(4)``) only reaches
~106K parameters on the described backbone. The two readings cannot both be
right. v2 built both, split its compute two ways, and got a mediocre result for
each. **v3 commits to the ~0.5M Flatten reading and builds only that**, so the
CPU budget goes into making one architecture work.

Reaching ~0.5M *without* a starved bottleneck (LESSON 2)
-------------------------------------------------------
The obvious way to spend 0.5M parameters on a 14x14x128 feature map is
``Flatten(25088) -> Dense(16)``: the 25088-wide flatten eats the whole budget
and leaves a 16-unit bottleneck. That is what v2 did, and a 16-unit layer has
almost no redundancy - lose a handful of units to dying activations and entire
classes become unreachable, which is exactly the partial collapse v2 could not
fix.

v3 adds **one extra pooling layer** before the flatten:

    224 -> 112 -> 56 -> 28 -> 14 -> 7        (7 x 7 x 128 = 6272)

which cuts the flatten width 4x and buys a **64-unit** bottleneck for the same
budget - 4x the redundancy, one dense hidden layer instead of three:

    backbone           97,440
    Dense(64) head    401,732
    total             499,172      (paper: ~500,000)

Every activation is ``LeakyReLU(alpha=0.1)`` and every kernel uses He
initialisation. Neither costs a single parameter.

Preprocessing lives *inside* every model (Rescaling for MiniConvNet, the
architecture's own ``preprocess_input`` for baselines), so a model can never be
paired with the wrong input scaling.
"""

from tensorflow.keras import Model, layers

from src.config import (
    ACTIVATION,
    DENSE_UNITS,
    DROPOUT_RATE,
    EXTRA_POOL_BEFORE_FLATTEN,
    IMG_SHAPE,
    KERNEL_INITIALIZER,
    LEAKY_RELU_ALPHA,
    MINICONVNET_FILTERS,
    NUM_CLASSES,
    SEED,
)

# Paper's stated budget, and how far from it we allow ourselves to land before
# check_param_budget() complains.
PAPER_PARAM_TARGET = 500_000
PARAM_TOLERANCE = 0.05          # +/- 5%


# --------------------------------------------------------------------------
# Activation (LESSON 2, cause A)
# --------------------------------------------------------------------------


def _activation(name, activation=ACTIVATION):
    """A LeakyReLU (default) or ReLU layer. Never a fused ``activation=`` arg.

    Keeping the activation as its own named layer means ``model.summary()``
    shows exactly which one a run used - a run can't be reported without that
    being visible - and the dead-unit probe in ``train_utils`` can read the
    activation outputs by layer name.
    """
    if activation == "leaky_relu":
        try:                      # Keras 3
            return layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA, name=name)
        except TypeError:         # tf.keras <= 2.15 spells it 'alpha'
            return layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA, name=name)
    if activation == "relu":
        return layers.ReLU(name=name)
    raise ValueError(f"Unknown activation '{activation}'. "
                     "Expected 'leaky_relu' or 'relu'.")


# --------------------------------------------------------------------------
# MiniConvNet - the single v3 architecture
# --------------------------------------------------------------------------


def build_miniconvnet(num_classes=NUM_CLASSES, input_shape=IMG_SHAPE,
                      filters=MINICONVNET_FILTERS, dense_units=DENSE_UNITS,
                      dropout_rate=DROPOUT_RATE, activation=ACTIVATION,
                      kernel_initializer=KERNEL_INITIALIZER,
                      extra_pool=EXTRA_POOL_BEFORE_FLATTEN,
                      use_batchnorm=False, name="miniconvnet"):
    """The v3 MiniConvNet: ~499K parameters, LeakyReLU, He init, wide head.

    ``dropout_rate=0`` drops the Dropout layer entirely (used by the ablation
    notebook). ``use_batchnorm=True`` is a deliberate experiment, not a default
    - it changes the parameter count, so log it with a ``config_note``.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation=None,
                          kernel_initializer=kernel_initializer,
                          name=f"conv{i}")(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = _activation(f"act{i}", activation)(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    # LESSON 2, cause B: this pool is what makes the wide bottleneck affordable.
    if extra_pool:
        x = layers.MaxPooling2D(2, name="pool_extra")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation=None,
                     kernel_initializer=kernel_initializer, name="dense_1")(x)
    x = _activation("act_dense_1", activation)(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=SEED, name="dropout_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_initializer=kernel_initializer,
                           name="predictions")(x)

    return Model(inputs, outputs, name=name)


def count_params(model):
    """(total, trainable, non_trainable, approx float32 size in MB)."""
    import numpy as np

    total = int(model.count_params())
    # np.prod over the shape tuple works on both tf.keras (TF<=2.15) and
    # Keras 3 variables, unlike shape.num_elements().
    trainable = int(sum(int(np.prod(tuple(w.shape))) for w in model.trainable_weights))
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "approx_MB_float32": round(total * 4 / (1024 ** 2), 2),
    }


def check_param_budget(model, target=PAPER_PARAM_TARGET, tolerance=PARAM_TOLERANCE,
                       verbose=True) -> dict:
    """Confirm the built model actually lands on the paper's ~0.5M budget.

    The whole architecture argument in the README rests on this number, so it
    is asserted by the code rather than trusted to arithmetic in a comment.
    """
    total = int(model.count_params())
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    ok = lo <= total <= hi
    out = {"total_params": total, "target": target,
           "within_tolerance": bool(ok),
           "pct_of_target": round(100 * total / target, 2)}
    if verbose:
        print(f"parameter budget: {total:,} vs paper's ~{target:,} "
              f"({out['pct_of_target']}%) -> "
              + ("OK" if ok else f"OUT OF RANGE (allowed {lo:,.0f}-{hi:,.0f})"))
    return out


def flatten_width(model) -> int:
    """Width of the flatten layer - the number the bottleneck argument is about."""
    import numpy as np

    return int(np.prod(tuple(model.get_layer("flatten").input.shape[1:])))


def architecture_summary(model) -> dict:
    """One dict capturing every choice a reader would ask about."""
    acts = [l.__class__.__name__ for l in model.layers
            if l.__class__.__name__ in ("LeakyReLU", "ReLU")]
    dense = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    return {
        "params": int(model.count_params()),
        "flatten_width": flatten_width(model),
        "bottleneck_units": int(dense[0].units) if dense else None,
        "activation": acts[0] if acts else None,
        "n_activation_layers": len(acts),
        "kernel_initializer": dense[0].kernel_initializer.__class__.__name__
                              if dense else None,
        "dropout_layers": sum(1 for l in model.layers
                              if l.__class__.__name__ == "Dropout"),
    }


# --------------------------------------------------------------------------
# Pretrained baselines (frozen backbones - CPU-scoped, LESSON: see README)
# --------------------------------------------------------------------------

DEFAULT_BASELINES = ["ResNet50", "VGG16", "MobileNetV3Small", "EfficientNetV2B0"]


def _baseline_spec(name):
    """Return ``(constructor, preprocess_input)`` for a baseline name."""
    from tensorflow.keras import applications as apps

    registry = {
        "ResNet50": (apps.ResNet50, apps.resnet50.preprocess_input),
        "VGG16": (apps.VGG16, apps.vgg16.preprocess_input),
        "MobileNetV3Small": (apps.MobileNetV3Small,
                             apps.mobilenet_v3.preprocess_input),
        "EfficientNetV2B0": (apps.EfficientNetV2B0,
                             apps.efficientnet_v2.preprocess_input),
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline '{name}'. Expected one of "
                         f"{list(registry)}. The extended set from v2 "
                         "(VGG19/InceptionV3/ConvNeXtTiny) is out of scope on CPU.")
    return registry[name]


def build_baseline(name, num_classes=NUM_CLASSES, input_shape=IMG_SHAPE,
                   trainable_base=False, dropout_rate=0.3, dense_units=128,
                   weights="imagenet"):
    """ImageNet-pretrained backbone + small classification head.

    ``trainable_base=False`` (feature extraction, head-only training) is the
    only mode used in this project: full fine-tuning of a 23M-parameter
    backbone is not affordable on CPU, and feature extraction is the standard
    comparison protocol the paper's table uses anyway.

    Each architecture's own ``preprocess_input`` is applied *inside* the model,
    so the shared ``[0, 255]`` data pipeline is correct for every model here.
    Note ``MobileNetV3``'s ``preprocess_input`` is a documented no-op (it
    rescales internally), so applying it is harmless.
    """
    constructor, preprocess_input = _baseline_spec(name)

    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)

    base = constructor(include_top=False, weights=weights,
                       input_shape=input_shape)
    base.trainable = trainable_base
    x = base(x, training=False if not trainable_base else None)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=SEED, name="dropout_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name=f"{name.lower()}_transfer")
