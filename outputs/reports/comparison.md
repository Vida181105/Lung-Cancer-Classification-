# Paper vs replication (v3)

Paper: Baqir, M.A., Qayyum, S., Ashfaq, N. et al. "A lightweight CNN for enhanced non-small cell lung cancer classification using CT scan image." Scientific Reports 16, 12985 (2026). DOI: 10.1038/s41598-026-41401-w

Scope: CPU-only. One MiniConvNet architecture (~499K params, the Flatten reading); 3-fold cross-validation; four frozen-backbone baselines. Each of these is a disclosed budget decision, not an oversight.

## Headline comparison

```
           model  paper_accuracy             our_accuracy  paper_params  our_params
     MiniConvNet            0.96        0.7059 +/- 0.1200      500000.0      499172
        ResNet50             NaN                   0.7016           NaN    23850500
           VGG16             NaN                   0.6000           NaN    14780868
MobileNetV3Small             NaN                   0.6349           NaN     1013492
EfficientNetV2B0             NaN INVALID_partial_collapse           NaN     6083796
```

## Canonical results table

```
                  model     arch_variant split_variant   params  accuracy  accuracy_std  precision_macro  recall_macro  f1_macro  cohen_kappa      mcc  auc_macro  binary_tumor_acc  subtype_acc  epochs_trained  n_runs                   status                                                                                                                                                                                                                                                                                                           notes           timestamp
MiniConvNet (3-fold CV) miniconvnet_500k      faithful   499172  0.705936      0.119965         0.732669      0.705325  0.708897     0.595953 0.601217        NaN          0.956984     0.648504              34       3                       ok 3-fold CV mean+/-std on the faithful (paper-comparable) split [3 folds not 5: disclosed CPU-budget decision]; f1_macro=0.7089 +/- 0.1253; 0 fold(s) excluded (0 collapsed, 0 partial); activation=leaky_relu, label_smoothing=0.05; pooled CV shares duplicated images across folds (documented dataset caveat) 2026-08-11T14:07:33
               ResNet50  transfer_frozen      faithful 23850500  0.701587           NaN         0.724636      0.698475  0.702566     0.576849 0.585116   0.895020          0.984127     0.643678              25       1                       ok                                                                                                                                                                                                                             ImageNet feature extraction, head-only training; 17.6s/epoch, 7.33 min total on CPU 2026-08-11T15:16:30
                  VGG16  transfer_frozen      faithful 14780868  0.600000           NaN         0.605732      0.651334  0.609381     0.463504 0.477362   0.858507          0.952381     0.521073              25       1                       ok                                                                                                                                                                                                                            ImageNet feature extraction, head-only training; 72.1s/epoch, 30.02 min total on CPU 2026-08-11T15:52:57
       MobileNetV3Small  transfer_frozen      faithful  1013492  0.634921           NaN         0.628766      0.587119  0.560904     0.463858 0.490615   0.845319          0.987302     0.563218              25       1                       ok                                                                                                                                                                                                                              ImageNet feature extraction, head-only training; 3.9s/epoch, 1.63 min total on CPU 2026-08-11T16:02:00
       EfficientNetV2B0  transfer_frozen      faithful  6083796  0.631746           NaN         0.536152      0.580787  0.547008     0.455642 0.488910   0.845775          0.990476     0.559387              25       1 INVALID_partial_collapse                                                                                                                                                                                                                               ImageNet feature extraction, head-only training; 6.5s/epoch, 2.7 min total on CPU 2026-08-11T16:18:24
```

## Dropout ablation

```
                     run_name split_variant  dropout_enabled  dropout_rate  accuracy  f1_macro  cohen_kappa      mcc  n_predicted_classes  epochs_trained status                                                                                                            notes           timestamp
 ablation_faithful_dropout_on      faithful             True           0.5  0.438095  0.432296     0.292754 0.339739                    4              34     ok budget 60 epochs, identical seed/lr/clipnorm/label-smoothing across arms; activation=leaky_relu; 3.73 min on CPU 2026-08-11T14:48:30
ablation_faithful_dropout_off      faithful            False           0.0  0.419048  0.408355     0.263821 0.304152                    4              21     ok budget 60 epochs, identical seed/lr/clipnorm/label-smoothing across arms; activation=leaky_relu; 2.39 min on CPU 2026-08-11T15:00:59
```

## Collapse audit (LESSON 4)

Two distinct failure tags. `INVALID_collapsed`: flat training curve, kappa/MCC ~0, or a single predicted class. `INVALID_partial_collapse`: the model never predicts some of the four classes - whole all-zero columns in the confusion matrix - which can happen at a clearly nonzero kappa and was the unresolved bug of the v2 attempt. Neither is a reportable result.

```
                     run_name  accuracy  cohen_kappa  n_predicted_classes never_predicted_classes                   status
ablation_faithful_dropout_off  0.419048     0.263821                    4                                               ok
 ablation_faithful_dropout_on  0.438095     0.292754                    4                                               ok
            cv_faithful_fold1  0.769461     0.686862                    4                                               ok
            cv_faithful_fold2  0.780781     0.699700                    4                                               ok
            cv_faithful_fold3  0.567568     0.401296                    4                                               ok
           cv_faithful_pooled  0.706000     0.597010                    4                                               ok
    efficientnetv2b0_faithful  0.631746     0.455642                    3    large.cell.carcinoma INVALID_partial_collapse
            miniconvnet_clean  0.705882     0.579749                    4                                               ok
         miniconvnet_faithful  0.406349     0.262674                    4                                               ok
    mobilenetv3small_faithful  0.634921     0.463858                    4                                               ok
            resnet50_faithful  0.701587     0.576849                    4                                               ok
               vgg16_faithful  0.600000     0.463504                    4                                               ok
```

## The two tasks, separated (LESSON 11)

Tumour-vs-healthy accuracy against subtype accuracy, for every run:

```
                          run    n  overall_accuracy  tumor_vs_healthy  subtype_accuracy  detection_minus_subtype     most_over_predicted      never_predicted                   status
            cv_faithful_fold2  333            0.7808            0.9760            0.7241                   0.2518          adenocarcinoma                    -                       ok
            cv_faithful_fold1  334            0.7695            0.9731            0.7061                   0.2669                  normal                    -                       ok
           cv_faithful_pooled 1000            0.7060            0.9570            0.6484                   0.3086          adenocarcinoma                    -                       ok
            miniconvnet_clean  170            0.7059            0.9471            0.7134                   0.2337    large.cell.carcinoma                    -                       ok
            resnet50_faithful  315            0.7016            0.9841            0.6437                   0.3404          adenocarcinoma                    -                       ok
    mobilenetv3small_faithful  315            0.6349            0.9873            0.5632                   0.4241          adenocarcinoma                    -                       ok
    efficientnetv2b0_faithful  315            0.6317            0.9905            0.5594                   0.4311          adenocarcinoma large.cell.carcinoma INVALID_partial_collapse
               vgg16_faithful  315            0.6000            0.9524            0.5211                   0.4313 squamous.cell.carcinoma                    -                       ok
            cv_faithful_fold3  333            0.5676            0.9219            0.5153                   0.4067          adenocarcinoma                    -                       ok
 ablation_faithful_dropout_on  315            0.4381            0.8317            0.3333                   0.4984    large.cell.carcinoma                    -                       ok
ablation_faithful_dropout_off  315            0.4190            0.6698            0.2989                   0.3710                  normal                    -                       ok
         miniconvnet_faithful  315            0.4063            0.7238            0.2950                   0.4288                  normal                    -                       ok
```

## Confusion matrices

### cv_faithful_pooled

- overall accuracy: 0.7060
- tumour-vs-healthy accuracy: 0.9570
- subtype accuracy (true tumour samples): 0.6484
- most over-predicted class: adenocarcinoma
- classes never predicted: none
- status: ok

Tumour-vs-healthy accuracy: 0.9570. Subtype accuracy among true tumour samples: 0.6484. Gap between the two tasks: 0.3086. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'adenocarcinoma' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[244  26  15  53]
 [ 44 109   5  29]
 [ 15   0 197   3]
 [ 92   7   5 156]]
```

### miniconvnet_clean

- overall accuracy: 0.7059
- tumour-vs-healthy accuracy: 0.9471
- subtype accuracy (true tumour samples): 0.7134
- most over-predicted class: large.cell.carcinoma
- classes never predicted: none
- status: ok

Tumour-vs-healthy accuracy: 0.9471. Subtype accuracy among true tumour samples: 0.7134. Gap between the two tasks: 0.2337. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'large.cell.carcinoma' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[44 10  3 11]
 [ 6 30  0  1]
 [ 0  5  8  0]
 [12  1  1 38]]
```

### miniconvnet_faithful

- overall accuracy: 0.4063
- tumour-vs-healthy accuracy: 0.7238
- subtype accuracy (true tumour samples): 0.2950
- most over-predicted class: normal
- classes never predicted: none
- status: ok

Tumour-vs-healthy accuracy: 0.7238. Subtype accuracy among true tumour samples: 0.2950. Gap between the two tasks: 0.4288. The model separates tumour from healthy far better than it separates the three tumour subtypes, so the headline accuracy gap versus the paper is a subtype-discrimination problem, not a detection problem. 'normal' is the most over-predicted class and absorbs most of the subtype confusions.

```
[[13 49 51  7]
 [ 0 41  9  1]
 [ 3  0 51  0]
 [ 8 35 24 23]]
```

## Interpretation (write these yourself)

- TODO: how far is the replication from the paper's 96%, and is the remaining gap a tumour-detection problem or a subtype-discrimination problem (see the table above)?
- TODO: how much of the faithful-split accuracy was duplicate-driven leakage (faithful vs clean)?
- TODO: did the five lesson-2 measures eliminate the v2 partial collapse? What did the dead-unit probe in notebook 02 say?
- TODO: how does a ~499K-parameter model trained from scratch compare with frozen ImageNet backbones 2-50x its size, on identical data?
- TODO: if the result plateaued below the paper, say so plainly with this evidence. Do not describe the paper as replicated.