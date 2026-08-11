# Dataset audit

Data root: `C:\Users\shrey\OneDrive\sem7\Medical Image\Project\Lung-Cancer-Classification-new\Data`

## Class counts (ours vs paper)

```
                         ours  paper  difference
adenocarcinoma            338    338           0
large.cell.carcinoma      187    187           0
normal                    215    115         100
squamous.cell.carcinoma   260    260           0
```

Total ours=1000, paper=900, difference=100 (all of it in `normal` if the CONFIRMED line printed above).

## Duplication and leakage

- n_files: 1000
- n_unique_hashes: 847
- n_duplicate_groups: 59
- n_files_in_duplicate_groups: 212
- n_redundant_files: 153
- n_leaked_files: 120
- n_train_test_leaked_files: 101
- duplicates_per_class: {'adenocarcinoma': 2, 'large.cell.carcinoma': 1, 'normal': 204, 'squamous.cell.carcinoma': 5}
- leaked_per_class: {'adenocarcinoma': 0, 'large.cell.carcinoma': 0, 'normal': 120, 'squamous.cell.carcinoma': 0}

## Splits written

### faithful (paper-comparable, duplicates kept)

```
split                    train  val  test  total
class                                           
adenocarcinoma             195   23   120    338
large.cell.carcinoma       115   21    51    187
normal                     148   13    54    215
squamous.cell.carcinoma    155   15    90    260
```

### clean (deduplicated, leakage-free - robustness experiment, NOT a replication)

```
split                    train  val  test  total
class                                           
adenocarcinoma             235   34    68    337
large.cell.carcinoma       131   19    37    187
normal                      46    7    13     66
squamous.cell.carcinoma    180   25    52    257
```
