# MiniConvNet — NSCLC CT classification (replication, attempt v3)

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
| MiniConvNet test accuracy | ~96% (96.6% cross-validated) |
| MiniConvNet size | ~0.5M parameters, ~6 MB |
| Task | 4-class: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal |
| Dataset | Kaggle *Chest CT-Scan images* (Mohamed Hany), ~1000 CT slices (paper: 900) |
| Comparison | MiniConvNet vs. standard ImageNet-pretrained backbones |

**This is the third attempt.** Two earlier attempts are frozen under [`archive/`](archive/) and both
hit the same wall: MiniConvNet trains unstably and lands nowhere near 96%. The best *verified*
result so far is **~50–52% via 5-fold CV** (v2, `archive/outputs_v2/results_table.csv`), and even
that came from runs later shown to be partially broken. v3 is a focused rebuild: one architecture,
every known fix applied from day one, and a CPU-only budget spent on making a single design work
rather than surveying several.

**Status**: code complete, **nothing has been run yet**. Every results table below is an empty
template. No numbers have been pre-filled, and none will be typed by hand.

---

## Hard constraint: CPU only

There is no GPU for this project at any point, and every scope decision below follows from that.

| Decision | v2 did | v3 does | Why |
|---|---|---|---|
| Architecture | 2 variants (~106K and ~0.5M) | **1** (~499K, the Flatten reading) | don't split CPU time two ways |
| Cross-validation | 5-fold | **3-fold** | ~40% less training, still a genuine mean ± std |
| Baselines | 4 + 3 optional | **4**, frozen backbones only | full fine-tuning of a 23M-param backbone is not affordable |
| Timing | assumed | **measured** | every notebook times a real epoch and extrapolates |

Three rules the notebooks enforce rather than merely mention:

1. **Measure before you commit.** Before any run estimated over **20 minutes**,
   `train_utils.estimate_training_time()` trains a throwaway model for one real epoch, extrapolates,
   and prints a stop-and-agree warning. No notebook launches a long run unannounced.
2. **Log wall clock everywhere.** `train_utils.EpochTimer` records seconds per epoch into every run's
   history JSON and into the `notes` column of every results row, so the time budget is visible
   rather than reconstructed afterwards.
3. **Prefer the cheaper design.** Thoroughness that cannot finish is not thoroughness.

---

## Institutional memory — 11 lessons, applied from day one

These are carried forward from the two earlier attempts and built into the code before the first
run, not rediscovered. **This section is permanent documentation — do not delete it after a
successful run.** The evidence behind each lesson lives in [`archive/`](archive/).

### 1. Dead-ReLU / full training collapse

Symptom: loss and accuracy go flat — identical values for many consecutive epochs — and never
recover. Root cause: Adam at too high a learning rate kills neurons permanently (output zero
forever), freezing the network.

**Fix, applied by default and not as an opt-in flag**: `Adam(learning_rate=1e-4, clipnorm=1.0)` in
[`src/train_utils.py`](src/train_utils.py) → `compile_model()`. **Do not use `1e-3` for MiniConvNet.**

This one worked — v2 saw no full collapses. It did not fix lesson 2.

### 2. Partial collapse — the unresolved bug from attempt 2

Even with lesson 1 applied, v2's MiniConvNet repeatedly learned to predict **only 2 of 4 classes**,
across all four configurations tried. Verified from raw prediction files, not summary metrics:

```
miniconvnet_gap_faithful           kappa 0.0617, logged as status "ok"
[[33  0 87  0]
 [22  0 29  0]
 [ 0  0 54  0]
 [31  0 59  0]]
     ^     ^   large.cell.carcinoma and squamous.cell.carcinoma: never predicted, ever
```

**This is the main problem v3 has to solve.** Because iterating one fix at a time is unaffordable on
CPU, all five suspected contributors are addressed **simultaneously, as defaults**:

| | Cause | v3 default | Where |
|---|---|---|---|
| A | ReLU units dying (zero gradient forever) | `LeakyReLU(alpha=0.1)` **everywhere** | [`src/models.py`](src/models.py) |
| B | Bottleneck too narrow for any redundancy | **`Dense(64)`**, not v2's `Dense(16)`×3 | [`src/models.py`](src/models.py) |
| C | Initialisation left to Keras's Glorot default | **`he_normal`** (correct for ReLU-family) | [`src/models.py`](src/models.py) |
| D | One fixed learning rate for the whole run | **`ReduceLROnPlateau(val_loss, patience=5, factor=0.5)`** | [`src/train_utils.py`](src/train_utils.py) |
| E | Early overconfident commitment to a subset of classes | **`CategoricalCrossentropy(label_smoothing=0.05)`** | [`src/train_utils.py`](src/train_utils.py) |

> **[CHOICE] LeakyReLU is the v3 default, not a fallback.** v2 documented it as something to try
> *after* a failure and never got there. Applying it up front costs zero parameters and zero
> accuracy risk; deferring it costs a full CPU training round to find out.

How B is afforded without breaking the parameter budget is the architecture note in lesson 3.
`train_utils.dead_unit_report()` measures the mechanism directly (units that never activate
positively), so if a partial collapse ever returns, the cause is diagnosed rather than guessed.

### 3. Architecture ambiguity — and v3's commitment

The paper's own layer list (`GlobalAveragePooling2D → Dense(64) → Dropout → Dense(4)`) totals only
**~106K parameters** on the described backbone, contradicting the paper's stated "~0.5M params, ~6MB".
Both readings cannot be right.

> **[CHOICE] v3 builds only the ~0.5M Flatten reading.** It is the one the paper's own size claim
> supports, and building one architecture properly beats building two mediocre ones on a CPU budget.
> The ambiguity is documented here and reported in the comparison; it is not silently resolved.

**The trap in the obvious implementation.** A 4-block backbone leaves a 14×14×128 feature map, so
`Flatten(25088) → Dense(16)` spends the entire budget on the flatten and leaves a **16-unit**
bottleneck — v2's design, and exactly the fragility of lesson 2's cause B. v3 adds **one extra
pooling layer** before the flatten:

```
224 → 112 → 56 → 28 → 14 → 7        7 × 7 × 128 = 6272   (v2: 25088)
```

which cuts the flatten width 4× and buys a **64-unit** bottleneck for the same budget:

| Component | Parameters |
|---|---|
| backbone, 4 × (Conv2D(f, 3×3, same) + LeakyReLU + MaxPool), f = (16, 32, 64, 128) | 97,440 |
| `Flatten(6272) → Dense(64)` | 401,472 |
| `Dense(4)` softmax | 260 |
| **total** | **499,172** (paper: ~500,000) |

`models.check_param_budget()` asserts this at build time, so the claim cannot drift away from the
code. Notebook 02 prints it before training.

### 4. Automated collapse detection — both modes, from the first run

Never trust a raw accuracy number without it. [`src/evaluate_utils.py`](src/evaluate_utils.py) →
`detect_collapse()` flags:

| Status | Trigger |
|---|---|
| `INVALID_collapsed` | training loss/accuracy flat (std < `1e-4`) over a trailing 10-epoch window, **or** Cohen's kappa / MCC ~0, **or** a single class predicted for every sample |
| `INVALID_partial_collapse` | **fewer than all 4 classes ever predicted** on the evaluation set — the check v2 lacked, which let a kappa-0.06 model with two dead columns report as `ok` |
| `ok` | reportable |

`detect_partial_collapse()` is called **from inside** `detect_collapse()`, so every call site is
covered and there is no second check to remember to wire up. The two tags stay distinct so the
failure modes remain separable in the results tables; `valid_only()` and the CV fold-exclusion rule
reject both. **Any flagged run is excluded from every reported mean — never silently included.**

### 5. Save raw predictions for every run

v2 could not re-audit its own finished runs when the partial-collapse check was written, because only
aggregate metrics had been saved; confirming the bug meant reading confusion matrices out of a
markdown report by hand.

Every training notebook calls `evaluate_utils.save_predictions()` from the first run onward, writing
`y_true`, `y_pred` and per-class probabilities to `outputs/predictions/<run>_predictions.csv`
(tracked in git — it is evidence, not clutter). `recheck_saved_predictions()` re-applies the current
detector to all of them in seconds. **On a CPU-only project, a retrain is not an affordable
substitute for having kept the evidence.**

### 6. The dataset has known train/test leakage

153 duplicate files, ~95% of them in the `normal` class. Duplicate detection (MD5 over file bytes)
and deduplication are built in from the start, producing **two split variants**:

| Variant | Definition | Use |
|---|---|---|
| `faithful` | the dataset's own `train/valid/test` folders, duplicates included as-is | the paper-comparable split |
| `clean` | content-hash deduplicated, leakage-free, group-aware stratified 70/10/20 re-split | robustness / generalisation experiment |

**The `clean` variant's results are never a direct replication of the paper's number.** They are
labelled as a robustness experiment in the README, the notebooks, the figures and the `config_note`
column. The clean split's leakage-free property is *asserted*, not assumed
(`data_utils.assert_no_leakage()`).

### 7. The `clean` variant has severe class imbalance

Deduplication shrinks `normal` far more than the tumour classes. Inverse-frequency `class_weight` is
therefore **on by default for `clean`** and off for `faithful` (`USE_CLASS_WEIGHTS_BY_SPLIT` in
[`src/config.py`](src/config.py), applied via `train_utils.class_weights_for()`).

### 8. Keep results tables separated from the start

| File | Contents |
|---|---|
| `outputs/results_table.csv` | exactly **one row per model** — the canonical/final result |
| `outputs/experiments_log.csv` | **every** run, tagged with a required `config_note` |
| `outputs/ablation_dropout.csv` | the dropout comparison, its own clean 2-row table |

`evaluate_utils.record_result(row, kind=...)` routes each row and replaces rather than appends when a
model already has a canonical row. Unknown column names raise, so a typo can never vanish into an
unwritten column. The three files are created **headers-only** by notebook 00; numbers only ever
arrive from a notebook that actually trained something.

### 9. The ~6 MB claim — verify it, don't hypothesise

The paper says ~6 MB; ~499K float32 weights is only ~1.9 MB. Adam keeps two extra buffers per
parameter (momentum + variance), so a checkpoint saved *with* the optimizer costs ~3× the weights:

| Quantity | Value |
|---|---|
| parameters | 499,172 |
| weights only, float32 (4 B/param) | ~1.90 MB |
| weights + Adam state (12 B/param) | ~5.71 MB |
| paper's stated size | ~6 MB |

Keras's default `model.save()` serialises the optimizer; `save_weights()` does not.
`train_utils.measure_model_file_sizes()` saves the **trained** model both ways and measures the two
files; `verdict_on_size_claim()` turns them into a CONFIRMED / REFUTED sentence. Notebook 02 §7 runs
it. Adam's slot variables are created lazily, so an untrained model would produce two similar files
for the wrong reason — `optimizer_slot_scalars_found` is the guard against reporting that as a result.

> **[CHOICE] Measured numbers pending.** Nothing has been run yet. After notebook 02 §7, replace this
> note with the two printed sizes — `save_weights()` = _ MB, `save()` = _ MB — and the verdict.
> Do not substitute the predicted values in the table above for measured ones.

### 10. Report the cross-validated mean ± std, not a single run

A single favourable split can look much better than the true average; this happened in v2. The
**3-fold CV mean ± std from notebook 03 is the headline number**; single runs from notebook 02 stay
in `experiments_log.csv`.

> **[CHOICE] 3-fold, not 5-fold — a disclosed CPU-budget decision.** Three folds still give a genuine
> distributional estimate over independent test partitions, which is what lesson 10 is for, at ~40%
> less training. **State the fold count whenever this number is quoted**; never present it as a
> 5-fold result.

Caveat to state alongside it: pooled CV on `faithful` shares duplicated images across folds. That is
a property of the paper-comparable protocol, not a bug — the leakage-free counterpart is the
`clean`-split run in notebook 02.

### 11. Report tumour-vs-healthy and subtype accuracy separately, for every model

The most useful diagnostic across both previous attempts. **Every** model tested so far, including
the baselines that worked well, separates tumour from healthy tissue at **92–99%** while telling the
three tumour subtypes apart at only **35–66%**:

| v2 model | tumour vs healthy | subtype |
|---|---|---|
| ResNet50 | 0.984 | 0.655 |
| VGG16 | 0.965 | 0.571 |
| MobileNetV3Small | 0.991 | 0.521 |
| EfficientNetV2B0 | 0.997 | 0.513 |

`evaluate_utils.tumor_vs_subtype_breakdown()` computes both plus the most over-predicted class;
`interpret_breakdown()` turns that into prose for the report; the numbers land in the
`binary_tumor_acc` / `subtype_acc` columns of **every** results row. **This is likely the project's
most defensible finding regardless of what MiniConvNet's final number turns out to be.**

### Also carried forward (v2 lessons that still apply)

- **Kaggle mounts the dataset at `/kaggle/input/<slug>/`, not `Data/`.** `data_utils.resolve_data_root()`
  auto-detects the environment and matches the folder by name pattern. No notebook hardcodes a path.
  The dataset's folder naming is also inconsistent (`train/`+`valid/` use long staging names,
  `test/` short ones); `normalize_class_name()` maps both onto the canonical class list.
- **Git-LFS is not installed by default.** [`.gitattributes`](.gitattributes) declares LFS tracking
  for `*.keras`/`*.h5`, and `models/` stays git-ignored until `git lfs install` has been run.

---

## What "try to improve accuracy" means here — and its limit

Every technique in lesson 2 is applied in good faith to genuinely close the gap toward 96%. But:

- **No tuning against the test set.** Architecture and hyperparameter decisions are validated on the
  validation split only. The test set (or held-out CV fold) is evaluated once, at the end, per
  configuration.
- **No unbounded search.** If, with all five lesson-2 measures applied together, CV accuracy is still
  far below the paper (e.g. well under 70–75%) and further changes are not moving it, **stop and
  report the plateau as a finding** — with the confusion-matrix and subtype-accuracy evidence for
  where the model actually struggles. **2–3 focused attempts is the right amount of effort, not
  10+.** On CPU, an open-ended search for the magic combination is not a good use of the time.
- **Do not claim the paper is "replicated" if the number is far off.** Report the actual result and
  the evidence for the difference.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitattributes                     LFS tracking for .keras/.h5
├── .gitignore
├── archive/                           FROZEN v1/v2 attempt - evidence for the lessons above; never edit
│   ├── README_v2.md
│   ├── notebooks_v2/, src_v2/, outputs_v2/
├── Data/                              local dataset (git-ignored; on Kaggle it lives in /kaggle/input)
├── src/
│   ├── config.py                      hyperparameters, seeds, class order, anti-collapse defaults, paths
│   ├── data_utils.py                  resolve_data_root, indexing, leakage audit, splits, tf.data
│   ├── models.py                      build_miniconvnet (one ~499K architecture), baseline builders
│   ├── train_utils.py                 compile_model (clipnorm + label smoothing), callbacks + LR schedule,
│   │                                  CPU time estimation, EpochTimer, dead-unit probe, file-size check
│   └── evaluate_utils.py              detect_collapse (full + partial), prediction saving, metrics,
│                                      tumour-vs-subtype breakdown, results routing
├── notebooks/
│   ├── 00_dataset_audit.ipynb         class counts, duplicate/leakage audit, both splits, headers-only CSVs
│   ├── 01_preprocessing_check.ipynb   geometry/dtype/range, label alignment, one-hot, augmentation preview
│   ├── 02_train_miniconvnet.ipynb     the focused build: faithful + clean runs, dead-unit probe, file-size check
│   ├── 03_cross_validation.ipynb      3-fold CV on faithful → the canonical MiniConvNet row
│   ├── 04_ablation_dropout.ipynb      dropout on vs off, same architecture, same budget
│   ├── 05_train_baselines.ipynb       ResNet50, VGG16, MobileNetV3Small, EfficientNetV2B0 (frozen)
│   └── 06_evaluate_and_compare.ipynb  collapse audit, tumour-vs-subtype table, paper comparison, comparison.md
├── models/                            trained weights (git-ignored until LFS is set up)
└── outputs/
    ├── figures/                       history curves, confusion matrices, comparison plots
    ├── history/                       per-run history JSON (incl. seconds/epoch) + per-epoch CSV
    ├── predictions/                   per-run raw y_true/y_pred/y_prob (LESSON 5; tracked in git)
    ├── reports/                       dataset_audit.md, comparison.md
    ├── splits/                        faithful_split.csv, clean_split.csv
    ├── results_table.csv              one row per model (canonical)
    ├── experiments_log.csv            every run, with config_note
    └── ablation_dropout.csv           2-row dropout comparison
```

**Design rule**: every stage a person actually runs is a notebook, split into small single-purpose
cells that print their intermediate state (image counts, `model.summary()`, seconds per epoch, final
metrics, collapse status), each preceded by a markdown cell saying what "looks right". Only shared,
non-runnable logic lives in `src/*.py`.

---

## Running

```bash
pip install -r requirements.txt
# place the Kaggle "Chest CT-Scan images" dataset in ./Data/ with train/ valid/ test/ subfolders
jupyter notebook notebooks/
```

Run in order — **notebook 00 must run first**, it writes the split definitions every later notebook
loads. Check the results of each before starting the next.

| Order | Notebook | Trains? | Cost on CPU |
|---|---|---|---|
| 1 | `00_dataset_audit.ipynb` | no | ~1 min (hashing ~1000 files) |
| 2 | `01_preprocessing_check.ipynb` | no | ~2 min |
| 3 | `02_train_miniconvnet.ipynb` | 2 runs | §3 measures and reports before starting |
| 4 | `03_cross_validation.ipynb` | 3 runs | §4 measures and reports before starting |
| 5 | `04_ablation_dropout.ipynb` | 2 runs | §1 measures and reports before starting |
| 6 | `05_train_baselines.ipynb` | 4 runs | §2 measures; heaviest notebook — one model per cell |
| 7 | `06_evaluate_and_compare.ipynb` | no | ~1 min (reads CSVs only) |

The same notebooks work on Kaggle unchanged (`resolve_data_root()` finds `/kaggle/input/<slug>/`);
attach the dataset as an input and enable internet for notebook 05's ImageNet weight download.

---

## Results

*Empty templates. Fill them from the generated CSVs after running the notebooks — do not hand-write
numbers the pipeline did not produce.*

### Main comparison — `outputs/results_table.csv`

MiniConvNet is the **3-fold CV mean ± std** (LESSON 10); baselines are single frozen-backbone runs on
the `faithful` split.

| model | params | accuracy | accuracy_std | f1_macro | cohen_kappa | binary_tumor_acc | subtype_acc | n_runs | status |
|---|---|---|---|---|---|---|---|---|---|
| MiniConvNet (3-fold CV) | 499,172 | | | | | | | | |
| ResNet50 | | | | | | | | | |
| VGG16 | | | | | | | | | |
| MobileNetV3Small | | | | | | | | | |
| EfficientNetV2B0 | | | | | | | | | |

### Single runs — `outputs/experiments_log.csv`

| run | split | accuracy | f1_macro | cohen_kappa | classes predicted | status |
|---|---|---|---|---|---|---|
| miniconvnet_faithful | faithful | | | | | |
| miniconvnet_clean | clean *(robustness experiment, NOT a replication)* | | | | | |
| cv_faithful_fold1–3 | faithful | | | | | |

#### Fold-to-fold variance: why fold 3 is lower

The 3-fold CV std (±0.12) is driven almost entirely by fold 3: **0.769 / 0.781 / 0.568**. Analysis of
the saved per-fold prediction files (`outputs/predictions/cv_faithful_fold{1,2,3}_predictions.csv`),
no retraining involved:

- **Not a class-balance effect.** `StratifiedKFold` gives the three test partitions near-identical
  distributions — adenocarcinoma 33.6–33.9%, large cell 18.6–18.9%, normal 21.3–21.6%, squamous
  25.8–26.1%. Fold 3's test set is not harder by composition.
- **Not a collapse.** All three folds predict all four classes; fold 3 is `ok`, not
  `INVALID_partial_collapse`.
- **The drop is concentrated, not spread evenly.** Per-class recall, folds 1/2 → fold 3:

  | class | fold 1 | fold 2 | fold 3 |
  |---|---|---|---|
  | large.cell.carcinoma | 0.661 | 0.790 | **0.302** |
  | normal | 1.000 | 0.986 | **0.761** |
  | adenocarcinoma | 0.690 | 0.841 | 0.634 |
  | squamous.cell.carcinoma | 0.759 | 0.523 | 0.517 |

  Large cell carcinoma — the smallest tumour class — accounts for most of it: 42 of its 63 test
  images are lost to adenocarcinoma (23) and squamous (19). Secondly, and unusually for this project,
  detection itself degrades: 14 `normal` images are called adenocarcinoma, dropping fold 3's
  tumour-vs-healthy accuracy to 0.922 against 0.973 and 0.976 for the other two folds.
- **Fold 3 was also the only fold to stop early**: 22 epochs against the full 40-epoch budget used by
  folds 1 and 2, which never triggered early stopping and were therefore still capped by the budget
  rather than converged.

**Conclusion, stated no more strongly than the data supports**: fold 3's lower accuracy is *not*
attributable to an unluckier class distribution, and it is not a collapse. It is a weaker model whose
loss is concentrated in the minority tumour subtype, coinciding with it being the only fold to
early-stop at roughly half the training of the others. Whether the early stop caused the weaker
result or merely reflects a validation split that plateaued sooner **cannot be settled from the saved
data** — the per-fold history JSONs are git-ignored and no validation curves were retained. Beyond
that, this is consistent with ordinary fold-to-fold variance on ~1000 images.

### Dropout ablation — `outputs/ablation_dropout.csv`

| run_name | dropout | accuracy | f1_macro | cohen_kappa | classes predicted | status |
|---|---|---|---|---|---|---|
| ablation_faithful_dropout_on | 0.5 | | | | | |
| ablation_faithful_dropout_off | 0.0 | | | | | |

### Leakage: faithful vs clean

| split | accuracy | drop |
|---|---|---|
| faithful | | — |
| clean | | |

### CPU wall clock

| notebook | measured s/epoch | total |
|---|---|---|
| 02 (2 runs) | | |
| 03 (3 folds) | | |
| 04 (2 arms) | | |
| 05 (4 baselines) | | |

---

## Paper vs. replication

*Template — fill in from `outputs/reports/comparison.md`, which notebook 06 generates with the
computed numbers and matching `TODO:` markers.*

### Headline

| model | paper accuracy | our accuracy | paper params | our params |
|---|---|---|---|---|
| MiniConvNet | 0.96 | | ~0.5M | 499,172 |
| ResNet50 | | | | |
| VGG16 | | | | |
| MobileNetV3Small | | | | |
| EfficientNetV2B0 | | | | |

Our MiniConvNet figure is the **3-fold CV mean ± std on the `faithful` split** — the paper-sized
architecture on the paper-comparable data, at a disclosed reduced fold count.

### The two tasks, separated (LESSON 11)

| model | overall | tumour vs healthy | subtype | detection − subtype | most over-predicted |
|---|---|---|---|---|---|
| MiniConvNet (pooled CV) | | | | | |
| miniconvnet_faithful | | | | | |
| miniconvnet_clean | | | | | |
| ResNet50 | | | | | |
| VGG16 | | | | | |
| MobileNetV3Small | | | | | |
| EfficientNetV2B0 | | | | | |

*Fill in: is the residual error a tumour-detection problem or a subtype-discrimination problem? Which
class absorbs the confusions? Does the pattern match the 92–99% / 35–66% split seen in v2?*

### Did the v3 fixes solve the partial collapse?

*Fill in: did any run trip `INVALID_partial_collapse`? What did the dead-unit probe (notebook 02 §6)
report for the 64-unit bottleneck? If the collapse is gone, which of the five measures can be
credited — and can that be told apart at all, given they were applied together?*

### Architecture ambiguity

*Fill in: does the ~0.5M reading reach the paper's accuracy? Combined with the measured file size
(lesson 9), what does that say about which architecture the paper actually trained?*

### Leakage and generalisation

*Fill in: how far accuracy drops on the deduplicated `clean` split, and what that implies about the
published number, which was computed on the same duplicated data.*

### Honest verdict

*Fill in. If the number is far below 96%, say so plainly and give the evidence — the plateau, the
confusion structure, the subtype gap. **Do not describe the paper as replicated.***

---

## Notes and caveats

- **The 900 vs 1000 image discrepancy** (documented, no further investigation needed): the paper
  reports 900 images, the public dataset now has ~1000. The three tumour classes match the paper
  exactly; the entire surplus is in `normal`, which is also ~95% duplicate-contaminated. Most likely
  the dataset was expanded with additional (largely duplicate) normal-class scans after publication.
  Notebook 00 checks this against the paper's per-class counts and prints a CONFIRMED line.
- **Pooled cross-validation on `faithful` shares duplicated images across folds** — a property of the
  paper-comparable protocol, stated in the notebook and in the `notes` column of every CV row.
- **Datasets yield raw `float32` pixels in `[0, 255]`**; rescaling and each baseline's
  `preprocess_input` happen *inside* the model, so a model can never be paired with the wrong input
  scaling.
- **MiniConvNet trains on one-hot targets**, because label smoothing (lesson 2, cause E) has no sparse
  counterpart in Keras. `compile_model()` prints which format it expects; baselines use sparse labels
  and no smoothing, since smoothing is a MiniConvNet anti-collapse measure and not part of the
  baseline comparison protocol.
- **`BatchNormalization` is off by default** in the backbone so the parameter count matches the
  paper's stated budget. Enabling it is a legitimate experiment — log it with a `config_note`.

## Licence and attribution

Dataset: *Chest CT-Scan images* by Mohamed Hany, via Kaggle — see the dataset page for its licence
terms. Method replicated from Baqir et al. (2026), cited above; this repository is an independent
reimplementation and is not affiliated with the authors.
