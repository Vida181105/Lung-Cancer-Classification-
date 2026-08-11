# MiniConvNet — NSCLC CT classification (replication)

A clean-room replication of a lightweight CNN for non-small cell lung cancer (NSCLC)
subtype classification from chest CT slices, benchmarked against standard pretrained
architectures.

> **Paper**
> Baqir, M.A., Qayyum, S., Ashfaq, N. et al. *A lightweight CNN for enhanced non-small
> cell lung cancer classification using CT scan image.* **Scientific Reports 16**, 12985 (2026).
> DOI: [10.1038/s41598-026-41401-w](https://doi.org/10.1038/s41598-026-41401-w)

**What the paper claims**

| Claim | Value |
|---|---|
| MiniConvNet test accuracy | ~96% |
| MiniConvNet size | ~0.5M parameters, ~6 MB |
| Task | 4-class: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal |
| Dataset | Kaggle *Chest CT-Scan images* (Mohamed Hany), ~1000 CT slices |
| Comparison | MiniConvNet vs. standard ImageNet-pretrained backbones |

**What this repo does**: reproduces that pipeline end to end on Kaggle — dataset audit,
preprocessing checks, MiniConvNet training (both plausible architecture readings), a
dropout ablation, 5-fold cross-validation, pretrained baselines, and an evaluation
notebook that explains any gap versus the paper using the confusion matrix.

**Status**: code complete, results not yet produced. Every results table and every
"Results" section below is an empty template — no numbers have been pre-filled.

---

## Institutional memory — lessons from the earlier attempt

This is the **second, clean attempt** at this replication. The first accumulated several
real bugs during iterative debugging. Each lesson below is built into the code from day
one rather than left to be rediscovered. **This section is permanent documentation — do
not delete it after the first successful run.**

### 1. MiniConvNet collapses with default Adam settings

Symptom: loss and accuracy go flat — identical values for many consecutive epochs — and
never recover. Root cause: dead-ReLU collapse.

**Fix, applied by default and not as an opt-in flag**: `Adam(learning_rate=1e-4, clipnorm=1.0)`
for all MiniConvNet training, in
[`src/train_utils.py`](src/train_utils.py) → `compile_model()`. **Do not use `1e-3` as the
default learning rate for MiniConvNet.**

### 2. The paper's architecture description is internally ambiguous

The earlier reconstruction described a head of
`GlobalAveragePooling2D → Dense(64) → Dropout → Dense(4)`, but on the shared backbone that
head totals only **~106K parameters** — it cannot reach the paper's own stated "~0.5M
params, ~6MB". A `Flatten(25088) → Dense(16) → Dense(16) → Dense(16) → Dense(4)` head does
land at **~0.5M**.

Both are therefore built as two named, clearly labelled variants in
[`src/models.py`](src/models.py):

| Builder | Head | Params |
|---|---|---|
| `build_miniconvnet_gap` | GAP → Dense(64) → Dropout → Dense(4) | ~106K |
| `build_miniconvnet_flatten` | Flatten(25088) → Dense(16)×3 → Dense(4) | ~0.5M |

Shared backbone: 4 blocks of `Conv2D(f, 3×3, same, ReLU) + MaxPool(2×2)` with
`f = (16, 32, 64, 128)`, so 224 → 112 → 56 → 28 → 14 and the final feature map is
14 × 14 × 128 = **25088**, exactly the Flatten width the paper's head implies.

Exact counts from the layer arithmetic: backbone 97,440 + GAP head 8,516 = **105,956**;
backbone + Flatten head 402,036 = **499,476**. Note that 499,476 float32 weights is ~1.9 MB,
not the paper's stated ~6 MB — so even the `flatten` reading only matches the paper on
parameter count, not on file size. Worth mentioning when you discuss the ambiguity.

**Both are trained and both are reported.** Neither is silently picked.

#### 2a. Where the paper's "~6 MB" comes from — measured, not assumed

The arithmetic behind the hypothesis: Adam carries two extra buffers per parameter (momentum and
variance), so a checkpoint saved *with the optimizer* costs ~3× the weights.

| quantity | value |
|---|---|
| `flatten` parameters | 499,476 |
| weights only, float32 (4 B/param) | ~1.91 MB |
| weights + Adam state (12 B/param) | ~5.72 MB |
| paper's stated size | ~6 MB |

Keras's default `model.save()` serialises the optimizer; `model.save_weights()` does not. So the
paper's figure is consistent with a default `save()`, and inconsistent with weights alone.

**This is measured, not argued.** `train_utils.measure_model_file_sizes()` saves the same trained
model both ways and prints the two file sizes; notebook 02 calls it on run B (the `flatten` variant)
right after training, and prints a CONFIRMED / REFUTED verdict against the ~6 MB claim. It requires a
model that has actually trained — Adam's slot variables are created lazily, so an untrained model has
no optimizer state to save and both files would come out the same size for the wrong reason. The
printout's `optimizer_slot_scalars_found` is the guard against reporting that mistake as a result.

> **[CHOICE] Measured numbers pending.** The check is wired up but has not yet been executed on a
> trained model. Run notebook 02 through run B, then replace this note with the two printed sizes
> — `save_weights()` = _ MB, `save()` = _ MB — and state plainly whether they confirm or refute the
> optimizer-state explanation. Do not paraphrase or round the numbers from the arithmetic above;
> only the measured file sizes settle it.

### 3. Always run automatic collapse detection

Never trust a raw accuracy number without it. A dead network with a frozen output looks
artificially *stable* — zero variance across epochs — and can otherwise pass for a good,
consistent result.

[`src/evaluate_utils.py`](src/evaluate_utils.py) → `detect_collapse()` flags a run when:

1. training loss or accuracy is flat (std < `1e-4`) across a trailing window of 10 epochs, **or**
2. Cohen's kappa or MCC comes out at ~0.0 on evaluation, **or**
3. the model predicts a single class for every test sample.

Any flagged run is written to the results CSVs with `status = INVALID_collapsed` instead of
being reported as a real number. This is applied to **every** MiniConvNet run, both
architecture variants, every CV fold, both ablation arms and every baseline.

These three checks are necessary but not sufficient — see **lesson 11** for the milder failure they
miss.

### 4. The dataset has known train/test leakage

153 duplicate files were found previously, concentrated in the `normal` class. Duplicate
detection (MD5 over file bytes) and deduplication are built in from the start, producing
**two split variants**:

| Variant | Definition | Use |
|---|---|---|
| `faithful` | the dataset's own `train/valid/test` folders, duplicates included as-is | the paper-comparable split |
| `clean` | content-hash deduplicated, leakage-free, group-aware stratified 70/10/20 re-split | robustness / generalisation experiment |

**The `clean` variant's results are never a direct replication of the paper's number.**
They are labelled as a robustness experiment in the README, in the notebooks, in the
figures and in the `config_note` column of `experiments_log.csv`.

### 5. The `clean` variant has severe class imbalance

Removing duplicates shrinks `normal` far more than the tumour classes. Inverse-frequency
`class_weight` is therefore **on by default for `clean`** and off for `faithful`
(`USE_CLASS_WEIGHTS_BY_SPLIT` in [`src/config.py`](src/config.py), applied via
`train_utils.class_weights_for()`).

### 6. Kaggle puts the dataset at `/kaggle/input/<slug>/`, not `Data/`

[`src/data_utils.py`](src/data_utils.py) → `resolve_data_root()` detects the environment
and auto-matches the dataset folder by name pattern, raising a message that lists what it
actually found if the match fails. **Every notebook that loads data uses this resolver — no
hardcoded paths anywhere.**

The dataset's folder naming is also inconsistent: `train/` and `valid/` use long staging
names (`adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`) while `test/` uses short ones
(`adenocarcinoma`). `normalize_class_name()` maps both onto the canonical class list.

### 7. Keep results tables separated from the start

Don't let debugging-run clutter build up in the main table:

| File | Contents |
|---|---|
| `outputs/results_table.csv` | exactly **one row per model** — the canonical/final result for each |
| `outputs/experiments_log.csv` | **every** experimental/debugging run, tagged with a required `config_note` |
| `outputs/ablation_dropout.csv` | the dropout-on vs. dropout-off comparison, its own clean 2-row table |

`evaluate_utils.record_result(row, kind=...)` routes each row to the correct file
(`canonical` / `experiment` / `ablation`) and replaces rather than appends when a model
already has a canonical row — nobody has to remember which file a run belongs in. Unknown
column names raise, so a typo can never vanish into an unwritten column.

### 8. Report MiniConvNet as the 5-fold CV mean ± std

A single favourable split previously looked much better than the true average. Both are
computed, but **the CV mean ± std is the number that goes in the main comparison table and
the paper-comparison section** (notebook 04 writes it; single runs from notebook 02 stay in
`experiments_log.csv`).

### 9. The confusion matrix is the explanation, not a diagnostic afterthought

In the earlier attempt the model separated tumour-vs-healthy almost perfectly (~97%) but
confused the three tumour subtypes with each other, over-predicting adenocarcinoma. That —
not a training bug — was the gap versus the paper.

`evaluate_utils.tumor_vs_subtype_breakdown()` computes tumour-vs-healthy accuracy and
subtype accuracy **separately**, plus which class is most over-predicted;
`interpret_breakdown()` turns that into written prose that lands in
`outputs/reports/comparison.md`. This is a first-class output of notebook 06.

### 10. Git-LFS is not installed by default

Large `.keras`/`.h5` files risk being committed as raw blobs. [`.gitattributes`](.gitattributes)
declares LFS tracking for model files, and `models/` is git-ignored until LFS is confirmed
working. **Run `git lfs install` before committing any trained model**, then remove the
`models/` lines from [`.gitignore`](.gitignore).

### 11. A model can collapse *partially* — and that passes every check in lesson 3

Found by reading the raw confusion matrices in `outputs/reports/comparison.md` by hand, after the
first full training round had already reported every run as `ok`:

```
miniconvnet_gap_faithful           kappa 0.0617, status "ok"
[[33  0 87  0]
 [22  0 29  0]
 [ 0  0 54  0]
 [31  0 59  0]]
     ^     ^  two all-zero COLUMNS: classes the model never predicts, at all
```

The model is only ever choosing between 2 of the 4 classes. It is not flat, its kappa is small but
comfortably nonzero, and it predicts more than one class — so **none** of lesson 3's three checks
fire, and a broken model is logged as a real, merely-weak result. The same pattern appears in
`miniconvnet_flatten_faithful` (2 dead columns) and `miniconvnet_flatten_clean` (1 dead column).

**Fix**: [`src/evaluate_utils.py`](src/evaluate_utils.py) → `detect_partial_collapse()` counts the
distinct classes ever predicted on the evaluation set and flags the run if that is fewer than
`NUM_CLASSES`. It is called **from inside `detect_collapse()`**, so every existing call site — all
four notebook-02 runs, both ablation arms, all ten CV folds, every baseline, and the checkpoint
re-scoring in notebook 06 — is covered, with no second check wired separately anywhere.

| status | meaning |
|---|---|
| `ok` | reportable |
| `INVALID_collapsed` | flat curve, kappa/MCC ~0, or a single predicted class |
| `INVALID_partial_collapse` | predicts some but not all classes; can occur at clearly nonzero kappa |

The two tags are kept **distinct** so the failure modes stay separable in the results tables;
`valid_only()` and the CV "exclude collapsed folds" rule reject both.

**Fix, second half — save the raw predictions.** This check could not be applied to the existing runs
because only aggregate metrics had been saved, so confirming it meant re-reading confusion matrices
out of a markdown report. Every training notebook now calls `evaluate_utils.save_predictions()`,
writing one small CSV per run to `outputs/predictions/` (tracked in git — it is evidence, not
clutter). `recheck_saved_predictions()` re-applies the current detector to all of them in seconds,
with no retraining. **Any new diagnostic should be applied through that path first.**

**Fallback if a re-run still collapses partially**: a ReLU unit whose pre-activation is negative for
every input has exactly zero gradient and never recovers; enough dead units in the `flatten` head's
narrow `Dense(16)` layers makes whole classes unreachable. Set
`MINICONVNET_ACTIVATION = 'leaky_relu'` in [`src/config.py`](src/config.py) (alpha 0.1, applied to
both backbone and head, **zero change to the parameter count**), restart the kernel, and re-run once.

> **[CHOICE] `relu` remains the default.** It is the faithful reading of the paper, and the
> replication's headline numbers must come from the faithful architecture. `leaky_relu` is a
> documented, opt-in remedy for a specific diagnosed failure — not a free accuracy knob. Do not run
> both and report the better one; any run using it carries `activation=leaky_relu` in its
> `notes` / `config_note`.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitattributes                     LFS tracking for .keras/.h5
├── .gitignore
├── Data/                              local dataset (git-ignored; on Kaggle it lives in /kaggle/input)
├── src/
│   ├── __init__.py
│   ├── config.py                      hyperparameters, seeds, class order, paths
│   ├── data_utils.py                  resolve_data_root, indexing, leakage audit, splits, tf.data
│   ├── models.py                      build_miniconvnet_gap / _flatten, baseline builders
│   ├── train_utils.py                 compile_model (clipnorm default), class weights, callbacks, model-size check
│   └── evaluate_utils.py              detect_collapse (+ partial), saved predictions, metrics, confusion matrices, results routing
├── notebooks/
│   ├── 00_dataset_audit.ipynb         class counts, duplicate/leakage detection, writes both splits
│   ├── 01_preprocessing_check.ipynb   resize/scale/augmentation checks, sample images, leakage re-check
│   ├── 02_train_miniconvnet.ipynb     2 architectures × 2 splits = 4 runs, all collapse-checked; + model file-size check (LESSON 2a)
│   ├── 03_ablation_dropout.ipynb      dropout on vs off, full epoch budget
│   ├── 04_cross_validation.ipynb      5-fold CV, both architectures → canonical MiniConvNet rows; §3b prints the re-run cost first
│   ├── 05_train_baselines.ipynb       ResNet50, VGG16, MobileNetV3Small, EfficientNetV2B0 (+ optional extended set)
│   ├── 06_evaluate_and_compare.ipynb  results tables, confusion matrices, paper comparison
│   └── 07_gradcam_optional.ipynb      optional extension — skip if short on time
├── models/                            trained weights (git-ignored until LFS is set up)
└── outputs/
    ├── figures/                       history curves, confusion matrices, comparison plots
    ├── history/                       per-run history JSON + per-epoch CSV
    ├── predictions/                   per-run raw y_true/y_pred/y_prob (LESSON 11; tracked in git)
    ├── reports/                       dataset_audit.md, comparison.md
    ├── splits/                        faithful_split.csv, clean_split.csv
    ├── results_table.csv              one row per model (canonical)
    ├── experiments_log.csv            every run, with config_note
    └── ablation_dropout.csv           2-row dropout comparison
```

**Design rule**: every stage a person actually runs is a notebook, split into many small
single-purpose cells that print their intermediate state. Only shared, non-runnable logic
(model definitions, config, utilities) lives in `src/*.py` and is imported by the notebooks.

---

## Running on Kaggle

1. **New Notebook** → *File* → *Import Notebook*, or clone this repo in the first cell:
   ```python
   !git clone https://github.com/<your-user>/<your-repo>.git
   %cd <your-repo>/notebooks
   ```
2. **Attach the dataset**: right sidebar → *Add Input* → *Datasets* → search
   **"Chest CT-Scan images"** (by Mohamed Hany) → *Add*. It mounts read-only under
   `/kaggle/input/`; `resolve_data_root()` finds it automatically.
3. **Set the accelerator**: right sidebar → *Session options* → *Accelerator* → **GPU**
   (T4 ×2 or P100). Notebook 02 prints the GPU status — if it reports 0 GPUs, fix this
   before training.
4. **Enable internet** (needed only for notebook 05, which downloads ImageNet weights):
   *Session options* → *Internet* → **On**.
5. **Run the notebooks in order**:

   | Order | Notebook | Needs GPU | Rough cost |
   |---|---|---|---|
   | 1 | `00_dataset_audit.ipynb` | no | ~1 min |
   | 2 | `01_preprocessing_check.ipynb` | no | ~2 min |
   | 3 | `02_train_miniconvnet.ipynb` | yes | 4 training runs |
   | 4 | `03_ablation_dropout.ipynb` | yes | 2 training runs |
   | 5 | `04_cross_validation.ipynb` | yes | 10 training runs (longest) |
   | 6 | `05_train_baselines.ipynb` | yes | 4 runs (+3 optional) |
   | 7 | `06_evaluate_and_compare.ipynb` | no | ~2 min |
   | 8 | `07_gradcam_optional.ipynb` | no | optional |

   Notebook 00 **must** run first — it writes the split definitions every later notebook
   loads. Notebooks 02–05 can be run in separate Kaggle sessions; results accumulate in
   `outputs/`.

6. **Persisting results**: Kaggle wipes everything outside `/kaggle/working` at session
   end. Commit the notebook (*Save Version* → *Save & Run All*) or download `outputs/`
   before the session expires.

### Running locally

```bash
pip install -r requirements.txt
# place the Kaggle dataset in ./Data/ with train/ valid/ test/ subfolders
jupyter notebook notebooks/
```

Everything except the training epoch counts is identical; the same notebooks work in both
places because of `resolve_data_root()`.

---

## Results

*All tables below are empty templates. Fill them from the generated CSVs after running the
notebooks — do not hand-write numbers that the pipeline did not produce.*

> **Current state of `outputs/` (as of the lesson-11 fix).** The CSVs and `comparison.md` in
> `outputs/` hold results from the training round that ran *before* partial-collapse detection
> existed, so several rows carry `status = ok` that the current detector would reject — at minimum
> `miniconvnet_gap_faithful`, `miniconvnet_flatten_faithful` and `miniconvnet_flatten_clean`, whose
> confusion matrices have all-zero columns. **Those files have deliberately not been hand-edited**;
> they will be overwritten by the corrected pipeline on the next run and nothing is to be corrected
> by typing. The dropout ablation (notebook 03) is the first thing to re-run; the CV folds
> (notebook 04) could not be checked without a re-run because that round saved no per-fold
> predictions — notebook 04 §3b now reports that and estimates the cost before anything starts.

### Main comparison — `outputs/results_table.csv`

One row per model. MiniConvNet rows are 5-fold CV mean ± std (LESSON 8); baselines are
single feature-extraction runs on the `faithful` split.

| model | arch_variant | split_variant | params | accuracy | accuracy_std | precision_macro | recall_macro | f1_macro | cohen_kappa | mcc | auc_macro | epochs | n_runs | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniConvNet-gap (5-fold CV) | gap | faithful | | | | | | | | | | | | |
| MiniConvNet-flatten (5-fold CV) | flatten | faithful | | | | | | | | | | | | |
| ResNet50 | transfer_frozen | faithful | | | | | | | | | | | | |
| VGG16 | transfer_frozen | faithful | | | | | | | | | | | | |
| MobileNetV3Small | transfer_frozen | faithful | | | | | | | | | | | | |
| EfficientNetV2B0 | transfer_frozen | faithful | | | | | | | | | | | | |

### Experiment log — `outputs/experiments_log.csv`

Every single run, including the four notebook-02 runs, all ten CV folds, and anything else
tried during debugging. Same columns as above plus `config_note`.

| model | arch_variant | split_variant | accuracy | f1_macro | cohen_kappa | mcc | status | config_note |
|---|---|---|---|---|---|---|---|---|
| miniconvnet_gap_faithful | gap | faithful | | | | | | GAP head, faithful split, single run |
| miniconvnet_flatten_faithful | flatten | faithful | | | | | | Flatten head, faithful split, single run |
| miniconvnet_gap_clean | gap | clean | | | | | | **clean split — robustness experiment, NOT a replication** |
| miniconvnet_flatten_clean | flatten | clean | | | | | | **clean split — robustness experiment, NOT a replication** |
| … | | | | | | | | |

### Dropout ablation — `outputs/ablation_dropout.csv`

| run_name | arch_variant | split_variant | dropout_enabled | dropout_rate | accuracy | f1_macro | cohen_kappa | mcc | epochs | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ablation_flatten_faithful_dropout_on | flatten | faithful | True | 0.5 | | | | | | |
| ablation_flatten_faithful_dropout_off | flatten | faithful | False | 0.0 | | | | | | |

### Split-variant comparison

How much of the `faithful` accuracy was duplicate-driven leakage:

| architecture | faithful accuracy | clean accuracy | drop |
|---|---|---|---|
| gap | | | |
| flatten | | | |

*Reminder: the `clean` column is a robustness experiment, not a replication of the paper.*

---

## Paper vs. replication

*Template — fill in after running the notebooks. Notebook 06 generates
`outputs/reports/comparison.md` with the computed numbers and `TODO:` markers matching this
section.*

### Headline

| model | paper accuracy | our accuracy | paper params | our params |
|---|---|---|---|---|
| MiniConvNet | 0.96 | | ~0.5M | |
| ResNet50 | | | | |
| VGG16 | | | | |
| MobileNetV3Small | | | | |
| EfficientNetV2B0 | | | | |

Our MiniConvNet figure is the **5-fold CV mean ± std of the `flatten` variant on the
`faithful` split** — the paper-sized architecture on the paper-comparable data.

### Architecture ambiguity — which reading is right?

*Fill in: does the `gap` (~106K) or the `flatten` (~0.5M) variant come closer to the paper's
reported accuracy, and does the answer change the conclusion about the paper's stated model
size?*

### Confusion-matrix explanation of the gap

Notebook 06 computes these separately for each trained model:

| model | overall accuracy | tumour-vs-healthy | subtype accuracy | detection − subtype | most over-predicted |
|---|---|---|---|---|---|
| miniconvnet_gap_faithful | | | | | |
| miniconvnet_flatten_faithful | | | | | |
| miniconvnet_gap_clean | | | | | |
| miniconvnet_flatten_clean | | | | | |

*Fill in the interpretation:*

- Is the residual error a **tumour-detection** problem or a **subtype-discrimination**
  problem? (A large positive `detection − subtype` gap means the latter.)
- Which class absorbs the confusions? (The earlier attempt: adenocarcinoma.)
- How much of the difference versus the paper is explained by the confusion structure
  rather than by training quality?

### Leakage and generalisation

*Fill in: how far accuracy drops on the deduplicated `clean` split, and what that implies
about the published number, which was computed on the same duplicated data.*

### What would close the gap

*Fill in after seeing the results.*

---

## Notes and caveats

- **Pooled cross-validation on `faithful` shares duplicated images across folds.** That is
  a property of the paper-comparable protocol, not a bug in notebook 04; the leakage-free
  counterpart is the `clean`-split experiment. It is stated in the notebook and in the
  `notes` column of every CV row.
- **`InceptionV3` officially expects 299×299 inputs.** The optional extended baseline runs
  it at the shared 224×224 with `include_top=False`, deliberately, so every model sees
  identical data. Report that if you quote it.
- **`BatchNormalization` is off by default** in the MiniConvNet backbone so the parameter
  counts match the paper's reconstruction. Enabling it is a legitimate experiment — log it
  in `experiments_log.csv` with a `config_note`.
- **Datasets yield raw `float32` pixels in `[0, 255]`**; rescaling and each baseline's
  `preprocess_input` happen *inside* the model, so a model can never be paired with the
  wrong input scaling.

## Licence and attribution

Dataset: *Chest CT-Scan images* by Mohamed Hany, via Kaggle — see the dataset page for its
licence terms. Method replicated from Baqir et al. (2026), cited above; this repository is
an independent reimplementation and is not affiliated with the authors.
