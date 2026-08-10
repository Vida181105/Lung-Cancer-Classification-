"""Model definitions: the two MiniConvNet head variants and the baselines.

Architecture ambiguity (LESSON 2)
---------------------------------
The paper states ~0.5M parameters / ~6MB, but the head described in the earlier
reconstruction (GlobalAveragePooling -> Dense(64) -> Dropout -> Dense(4)) only
reaches ~106K parameters on the same backbone. A Flatten(25088) -> Dense(16) x3
-> Dense(4) head lands at ~0.5M. Both are therefore built here as named
variants and both are trained and reported. Neither is silently preferred.

Shared backbone: 4 conv blocks with filters (16, 32, 64, 128), each
Conv2D(3x3, same, ReLU) + MaxPool(2x2). 224 -> 112 -> 56 -> 28 -> 14, so the
final feature map is 14 x 14 x 128 = 25088 units, which is exactly the Flatten
width the paper's head implies.

Preprocessing lives *inside* every model (Rescaling for MiniConvNet, the
architecture's own ``preprocess_input`` for baselines), so a model can never be
paired with the wrong input scaling.
"""

from tensorflow.keras import Model, layers

from src.config import (
    DROPOUT_RATE,
    FLATTEN_DENSE_UNITS,
    GAP_DENSE_UNITS,
    IMG_SHAPE,
    LEAKY_RELU_ALPHA,
    MINICONVNET_ACTIVATION,
    MINICONVNET_FILTERS,
    NUM_CLASSES,
    SEED,
)


def _activation_layer(activation, name):
    """ReLU (paper-faithful default) or LeakyReLU (LESSON 11 fallback).

    Returns ``None`` when the activation can be folded into the ``Conv2D``/
    ``Dense`` layer itself, so the ReLU path builds exactly the same graph it
    did before this option existed. Neither choice adds parameters.
    """
    if activation in (None, "relu"):
        return None
    if activation == "leaky_relu":
        try:                      # Keras 3
            return layers.LeakyReLU(negative_slope=LEAKY_RELU_ALPHA, name=name)
        except TypeError:         # tf.keras <= 2.15 spells it 'alpha'
            return layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA, name=name)
    raise ValueError(f"Unknown activation '{activation}'. "
                     "Expected 'relu' or 'leaky_relu'.")


def _dense_block(x, units, activation, name):
    """Dense + activation, keeping the ReLU path a single fused layer.

    The narrow ``Dense(16)`` layers of the flatten head are where a dead unit
    costs the most - lose a few of 16 and whole classes become unreachable,
    which is exactly the partial collapse of LESSON 11.
    """
    act = _activation_layer(activation, name=f"{name}_act")
    x = layers.Dense(units, activation=None if act is not None else "relu",
                     name=name)(x)
    return act(x) if act is not None else x

# --------------------------------------------------------------------------
# MiniConvNet
# --------------------------------------------------------------------------


def _miniconvnet_backbone(inputs, filters=MINICONVNET_FILTERS,
                          use_batchnorm=False,
                          activation=MINICONVNET_ACTIVATION):
    """Conv/Pool stack shared by both head variants.

    ``use_batchnorm`` defaults to False to keep the parameter count matching
    the paper's reconstruction; enable it only as a deliberate experiment
    (log it as such in experiments_log.csv).

    ``activation`` defaults to ``'relu'`` (the paper's reading). Pass
    ``'leaky_relu'`` as the LESSON 11 fallback for a run that still collapses
    partially - dead ReLU units have zero gradient forever, leaky ones can
    recover. The parameter count is identical either way.
    """
    act = _activation_layer(activation, name="act0")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same",
                          activation=None if act is not None else "relu",
                          name=f"conv{i}")(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn{i}")(x)
        if act is not None:
            x = _activation_layer(activation, name=f"act{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    return x


def build_miniconvnet_gap(num_classes=NUM_CLASSES, input_shape=IMG_SHAPE,
                          dropout_rate=DROPOUT_RATE,
                          dense_units=GAP_DENSE_UNITS,
                          filters=MINICONVNET_FILTERS, use_batchnorm=False,
                          activation=MINICONVNET_ACTIVATION,
                          name="miniconvnet_gap"):
    """GAP-head variant: GAP -> Dense(64) -> Dropout -> Dense(4).

    ~106K parameters - i.e. it does NOT reach the paper's stated ~0.5M.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = _miniconvnet_backbone(inputs, filters, use_batchnorm, activation)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = _dense_block(x, dense_units, activation, "dense_1")
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=SEED, name="dropout_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inputs, outputs, name=name)


def build_miniconvnet_flatten(num_classes=NUM_CLASSES, input_shape=IMG_SHAPE,
                              dropout_rate=DROPOUT_RATE,
                              dense_units=FLATTEN_DENSE_UNITS,
                              filters=MINICONVNET_FILTERS, use_batchnorm=False,
                              activation=MINICONVNET_ACTIVATION,
                              name="miniconvnet_flatten"):
    """Flatten-head variant: Flatten(25088) -> Dense(16) x3 -> Dense(4).

    ~0.5M parameters - matches the paper's stated size. Dropout sits after the
    first (by far the largest) dense layer.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = _miniconvnet_backbone(inputs, filters, use_batchnorm, activation)
    x = layers.Flatten(name="flatten")(x)
    x = _dense_block(x, dense_units, activation, "dense_1")
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=SEED, name="dropout_1")(x)
    x = _dense_block(x, dense_units, activation, "dense_2")
    x = _dense_block(x, dense_units, activation, "dense_3")
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inputs, outputs, name=name)


MINICONVNET_BUILDERS = {
    "gap": build_miniconvnet_gap,
    "flatten": build_miniconvnet_flatten,
}


def build_miniconvnet(arch_variant, **kwargs):
    """Dispatch on ``'gap'`` / ``'flatten'`` so notebooks can loop over both."""
    if arch_variant not in MINICONVNET_BUILDERS:
        raise ValueError(
            f"Unknown arch_variant '{arch_variant}'. "
            f"Expected one of {list(MINICONVNET_BUILDERS)}."
        )
    return MINICONVNET_BUILDERS[arch_variant](**kwargs)


def count_params(model):
    """(total, trainable, non_trainable, approx_size_MB_float32)."""
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


# --------------------------------------------------------------------------
# Pretrained baselines
# --------------------------------------------------------------------------

# name -> (keras application module attr, application class name)
DEFAULT_BASELINES = ["ResNet50", "VGG16", "MobileNetV3Small", "EfficientNetV2B0"]
EXTENDED_BASELINES = ["VGG19", "InceptionV3", "ConvNeXtTiny"]
ALL_BASELINES = DEFAULT_BASELINES + EXTENDED_BASELINES


def _baseline_spec(name):
    """Return ``(constructor, preprocess_input)`` for a baseline name."""
    from tensorflow.keras import applications as apps

    registry = {
        "ResNet50": (apps.ResNet50, apps.resnet50.preprocess_input),
        "VGG16": (apps.VGG16, apps.vgg16.preprocess_input),
        "VGG19": (apps.VGG19, apps.vgg19.preprocess_input),
        "MobileNetV3Small": (apps.MobileNetV3Small,
                             apps.mobilenet_v3.preprocess_input),
        "EfficientNetV2B0": (apps.EfficientNetV2B0,
                             apps.efficientnet_v2.preprocess_input),
        "InceptionV3": (apps.InceptionV3, apps.inception_v3.preprocess_input),
        "ConvNeXtTiny": (apps.ConvNeXtTiny, apps.convnext.preprocess_input),
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline '{name}'. Expected one of "
                         f"{list(registry)}.")
    return registry[name]


def build_baseline(name, num_classes=NUM_CLASSES, input_shape=IMG_SHAPE,
                   trainable_base=False, dropout_rate=0.3, dense_units=128,
                   weights="imagenet"):
    """ImageNet-pretrained backbone + small classification head.

    ``trainable_base=False`` (default) is feature-extraction transfer learning,
    which is what the paper's comparison table uses. The architecture's own
    ``preprocess_input`` is applied inside the model, so the dataset can keep
    feeding raw [0, 255] pixels for every model in the project.

    Note: ``MobileNetV3`` and ``ConvNeXt`` already rescale internally; their
    ``preprocess_input`` is a documented no-op, so applying it is harmless.
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
