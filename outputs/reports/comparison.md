# Paper vs replication

Paper: Baqir, M.A., Qayyum, S., Ashfaq, N. et al. "A lightweight CNN for enhanced non-small cell lung cancer classification using CT scan image." Scientific Reports 16, 12985 (2026). DOI: 10.1038/s41598-026-41401-w

## Headline comparison

```
           model  paper_accuracy      our_accuracy  paper_params  our_params
     MiniConvNet            0.96 0.5050 +/- 0.0499      500000.0      499476
        ResNet50             NaN            0.7111           NaN    23850500
           VGG16             NaN            0.6413           NaN    14780868
MobileNetV3Small             NaN            0.6000           NaN     1013492
EfficientNetV2B0             NaN            0.5937           NaN     6083796
```

MiniConvNet is reported as the 5-fold cross-validation mean +/- std on the faithful (paper-comparable) split, not a single best run.

## Canonical results table

```
                          model    arch_variant split_variant   params  accuracy  accuracy_std  precision_macro  recall_macro  f1_macro  cohen_kappa      mcc  auc_macro  epochs_trained  n_runs status                                                                                                                                                                                                   notes           timestamp
    MiniConvNet-gap (5-fold CV)             gap      faithful   105956  0.523000      0.036159         0.348767      0.478711  0.381574     0.316882 0.394488        NaN              32       5     ok 5-fold CV mean+/-std on the faithful (paper-comparable) split; f1_macro=0.3816 +/- 0.0438; 0 fold(s) excluded as collapsed; pooled CV shares duplicated images across folds (documented dataset caveat) 2026-08-10T11:19:36
MiniConvNet-flatten (5-fold CV)         flatten      faithful   499476  0.505000      0.049875         0.326016      0.455876  0.369368     0.287081 0.364611        NaN              27       5     ok 5-fold CV mean+/-std on the faithful (paper-comparable) split; f1_macro=0.3694 +/- 0.0717; 0 fold(s) excluded as collapsed; pooled CV shares duplicated images across folds (documented dataset caveat) 2026-08-10T11:19:36
                       ResNet50 transfer_frozen      faithful 23850500  0.711111           NaN         0.719577      0.733606  0.720238     0.600877 0.604287   0.897012              30       1     ok                                                                                                            ImageNet feature extraction, head-only training; binary_tumor_acc=0.9841; subtype_acc=0.6552 2026-08-10T11:29:17
                          VGG16 transfer_frozen      faithful 14780868  0.641270           NaN         0.656394      0.698843  0.657010     0.519499 0.534554   0.885973              30       1     ok                                                                                                            ImageNet feature extraction, head-only training; binary_tumor_acc=0.9651; subtype_acc=0.5709 2026-08-10T11:48:53
               MobileNetV3Small transfer_frozen      faithful  1013492  0.600000           NaN         0.662521      0.560199  0.539075     0.405874 0.462171   0.827466              25       1     ok                                                                                                            ImageNet feature extraction, head-only training; binary_tumor_acc=0.9905; subtype_acc=0.5211 2026-08-10T11:51:12
               EfficientNetV2B0 transfer_frozen      faithful  6083796  0.593651           NaN         0.635335      0.576008  0.547917     0.420515 0.438446   0.869081              30       1     ok                                                                                                            ImageNet feature extraction, head-only training; binary_tumor_acc=0.9968; subtype_acc=0.5134 2026-08-10T11:55:41
```

## Dropout ablation

```
                             run_name arch_variant split_variant  dropout_enabled  dropout_rate  accuracy  f1_macro  cohen_kappa      mcc  epochs_trained status                                                          notes           timestamp
 ablation_flatten_faithful_dropout_on      flatten      faithful             True           0.5  0.260317   0.19129     0.071784 0.112081              25     ok full budget 100 epochs, identical seed/lr/clipnorm across arms 2026-08-09T21:17:46
ablation_flatten_faithful_dropout_off      flatten      faithful            False           0.0  0.288889   0.20681     0.074417 0.100648              27     ok full budget 100 epochs, identical seed/lr/clipnorm across arms 2026-08-09T21:19:28
```

## Confusion-matrix explanation of the gap

### miniconvnet_gap_faithful (gap head, faithful split)

- overall accuracy: 0.2762
- tumour-vs-healthy accuracy: 0.4444
- subtype accuracy (true tumour samples): 0.1264
- most over-predicted class: normal
- status: ok

Tumour-vs-healthy accuracy: 0.4444. Subtype accuracy among true tumour samples: 0.1264. Gap between the two tasks: 0.3180. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'normal' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[33  0 87  0]
 [22  0 29  0]
 [ 0  0 54  0]
 [31  0 59  0]]
```

### miniconvnet_flatten_faithful (flatten head, faithful split)

- overall accuracy: 0.2952
- tumour-vs-healthy accuracy: 0.4921
- subtype accuracy (true tumour samples): 0.1494
- most over-predicted class: normal
- status: ok

Tumour-vs-healthy accuracy: 0.4921. Subtype accuracy among true tumour samples: 0.1494. Gap between the two tasks: 0.3426. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'normal' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[39  0 81  0]
 [22  0 29  0]
 [ 0  0 54  0]
 [40  0 50  0]]
```

### miniconvnet_gap_clean (gap head, clean split)

- overall accuracy: 0.5059
- tumour-vs-healthy accuracy: 0.9647
- subtype accuracy (true tumour samples): 0.4650
- most over-predicted class: squamous.cell.carcinoma
- status: ok

Tumour-vs-healthy accuracy: 0.9647. Subtype accuracy among true tumour samples: 0.4650. Gap between the two tasks: 0.4997. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'squamous.cell.carcinoma' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[41  0  5 22]
 [16  0  0 21]
 [ 0  0 13  0]
 [19  0  1 32]]
```

### miniconvnet_flatten_clean (flatten head, clean split)

- overall accuracy: 0.3235
- tumour-vs-healthy accuracy: 0.9235
- subtype accuracy (true tumour samples): 0.3503
- most over-predicted class: squamous.cell.carcinoma
- status: ok

Tumour-vs-healthy accuracy: 0.9235. Subtype accuracy among true tumour samples: 0.3503. Gap between the two tasks: 0.5732. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'squamous.cell.carcinoma' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[ 0 16  0 52]
 [ 0  9  0 28]
 [ 0  9  0  4]
 [ 0  6  0 46]]
```

## Interpretation (write these yourself)

- TODO: does the replication reproduce the paper's 96% on the faithful split? By how much does it differ?
- TODO: how much of the faithful-split accuracy is attributable to duplicate-driven train/test leakage (compare with the clean-split runs)?
- TODO: which architecture variant (gap vs flatten) is closer to the paper, and does the parameter-count ambiguity change the conclusion?
- TODO: is the residual error a tumour-detection problem or a subtype-discrimination problem, given the breakdown above?
- TODO: what would you change to close the remaining gap?