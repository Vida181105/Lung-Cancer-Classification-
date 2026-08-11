"""Shared, non-runnable logic for the v3 MiniConvNet lung-cancer replication.

Modules
-------
config          hyperparameters, seeds, class order, anti-collapse defaults, paths
data_utils      dataset resolution/indexing, leakage audit, splits, tf.data
models          build_miniconvnet (single ~0.5M architecture), baseline builders
train_utils     seeding, compile_model (clipnorm + label smoothing), callbacks,
                CPU time estimation, dead-unit probe, model file-size check
evaluate_utils  detect_collapse (full + partial), saved predictions, metrics,
                tumour-vs-subtype breakdown, results routing

Notebooks import from here; nothing in this package trains anything on import.
"""

__version__ = "3.0.0"
