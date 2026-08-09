"""Shared, non-runnable logic for the MiniConvNet lung-cancer replication.

Modules
-------
config          hyperparameters, seeds, class order, output paths
data_utils      dataset resolution/indexing, leakage audit, splits, tf.data
models          build_miniconvnet_gap / build_miniconvnet_flatten, baselines
train_utils     seeding, compile_model (clipnorm default), class weights, callbacks
evaluate_utils  detect_collapse, metrics, confusion matrices, results routing

Notebooks import from here; nothing in this package trains anything on import.
"""

__version__ = "1.0.0"
