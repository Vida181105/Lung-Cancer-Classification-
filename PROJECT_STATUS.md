# Project status - MiniConvNet lung CT classification

Last updated: after writing Phase 6 (uncertainty-aware / selective prediction). This document is a
status tracker, not a results write-up - see the individual phase notebooks and `results/`/`outputs/`
for actual numbers, all of which are read live from disk (never hand-typed) by every notebook that
reports them.

**Nothing in Phase 6 has been executed as of this update** - `src/uncertainty_utils.py` and
`notebooks/15_uncertainty_selective_prediction.ipynb` were written and validated (syntax, cross-module
imports) but not run; the checkpoints, external dataset, and TensorFlow environment they depend on
exist only on the machine that will actually run them.

---

## COMPLETED

- **Paper replication** - MiniConvNet (~499K params) built and trained; CT dataset headline
  70.59% ± 12.00% (3-fold CV, `outputs/results_table.csv`); LC25000 second-dataset validation 96.11%
  (`outputs/results_table_lc25000.csv`).
- **Duplicate / leakage analysis** (Phase 1) - 153 exact duplicates, 891 additional near-duplicate
  pairs, 363 images in duplicate-contaminated groups, no patient IDs available, image-group-level
  leakage control implemented (`src/leakage_utils.py`, `outputs/leakage/`).
- **Leakage-controlled evaluation** (Option B) - `StratifiedGroupKFold`, 3-fold CV,
  74.10% ± 4.24% (`outputs/leakage/leakage_controlled_cv_results.json`), zero group violations
  verified. **No standalone checkpoint was produced by this run** (confirmed: no `.save()` call
  anywhere in `src/leakage_cv_utils.py` or notebook 14) - Phase 6 discloses and works around this.
- **Fair baseline comparison** (Phase 2) - ResNet50, VGG16, EfficientNetV2B0, MobileNetV3Small, frozen
  and fine-tuned (`src/finetune_utils.py`, `results/fair_baseline/`). Fine-tuned VGG16 is the
  strongest baseline (77.14%), identified programmatically, never hardcoded.
- **Explainability** (Phase 3) - Grad-CAM and Grad-CAM++ for MiniConvNet and VGG16
  (`src/gradcam_utils.py`, `results/gradcam/`), including the fix for VGG16's nested-backbone graph
  disconnection. Found one possible border-artifact shortcut-learning concern
  (`border_energy_fraction` up to 1.000 for one MiniConvNet record).
- **Calibration / uncertainty** (Phase 4) - ECE, Brier score, temperature scaling (fit on validation
  only), MC Dropout (`src/calibration_utils.py`, `results/calibration/`). MiniConvNet ECE 16.57%,
  VGG16 ECE 2.65%; MC Dropout showed higher predictive entropy for incorrect predictions internally.
- **External validation** (Phase 5) - independent IQ-OTH/NCCD dataset, task reduced to malignant vs
  normal (benign excluded from scoring, three framings computed and reported)
  (`src/external_eval_utils.py`, `results/external_eval/`). MiniConvNet external accuracy 46.06%,
  AUC 0.522, malignant recall 8.2%; VGG16 external accuracy 56.09%, AUC 0.641, malignant recall 33.5%.
- **BatchNorm ablation** - completed, negative result, reverted from the working tree (its two commits
  are reverted in git history; not repeated).

## NEWLY COMPLETED (code written this session - not yet executed)

- **Uncertainty vs error** - per-image MC Dropout entropy re-computed (Phase 4 only persisted
  aggregates) and compared between correct/incorrect predictions, on both the internal test set and
  the external dataset (`src/uncertainty_utils.py`: `correct_vs_incorrect_stats`, `uncertainty_bins`).
- **Selective prediction / risk-coverage** - standard 100%-to-10% coverage grid, accuracy/error/
  malignant-recall-among-retained at each level, for both models, both datasets
  (`risk_coverage_curve`, `results/uncertainty/selective_prediction_results.csv`).
- **Internal vs external uncertainty analysis** - entropy and accuracy/recall compared on a shared
  binary-reduced footing (tumour-vs-normal internally, malignant-vs-normal externally), with
  cautious, non-overclaiming language built into the notebook's own markdown.
- **Confidence vs MC uncertainty** - correlation between deterministic softmax confidence and MC
  predictive entropy; dedicated high-confidence-incorrect analysis (threshold reused from Phase 3,
  not re-picked) on both datasets, with optional fresh Grad-CAM on the external failures.
- **Final consolidated results table** - `results/uncertainty/final_results_table.csv`, every column
  read live from its source file or explicitly marked "not available".
- **The leakage-controlled-checkpoint substitution** - resolved and disclosed per the addendum:
  MiniConvNet results in Phase 6 use `checkpoints_local/miniconvnet_single_run.keras` (the same
  instance analysed in Phases 3-4), not a checkpoint derived from Option B, which never saved one.

New files: `src/uncertainty_utils.py`, `notebooks/15_uncertainty_selective_prediction.ipynb`,
`PROJECT_STATUS.md` (this file). One additive refactor: `src/calibration_utils.py` gained
`mc_dropout_raw_passes()` (the stochastic-pass core extracted from `mc_dropout_predict()` for reuse
on external data); `mc_dropout_predict()`'s own return value is unchanged.

## REMAINING

- **Run notebook 15** on a machine with the checkpoints, the external dataset, and TensorFlow -
  nothing above has produced a single number yet.
- **Fill in Step 10's interpretation** in that notebook from the actual computed answers once run.
- Everything else in the numbered list above is genuinely complete; there is no other planned phase
  beyond Phase 6 as of this update.
