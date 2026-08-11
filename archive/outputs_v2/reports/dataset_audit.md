# Dataset audit

- data root: `/mnt/c/Users/shrey/OneDrive/sem7/Medical Image/Project/Lung-Cancer-Classification-/Data`
- total images: 1000
- unique content hashes: 847
- duplicate groups: 59
- files in duplicate groups: 212
- redundant (removable) files: 153
- files leaking across splits: 120
- files leaking specifically train<->test: 101

## Duplicates per class

- adenocarcinoma: 2
- large.cell.carcinoma: 1
- normal: 204
- squamous.cell.carcinoma: 5

## Class counts per original split

```
orig_split               test  train  val  total
class                                           
adenocarcinoma            120    195   23    338
large.cell.carcinoma       51    115   21    187
normal                     54    148   13    215
squamous.cell.carcinoma    90    155   15    260
```

## faithful split (paper-comparable, duplicates kept)

```
split                    train  val  test  total
class                                           
adenocarcinoma             195   23   120    338
large.cell.carcinoma       115   21    51    187
normal                     148   13    54    215
squamous.cell.carcinoma    155   15    90    260
```

## clean split (deduplicated, leakage-free - robustness experiment, NOT a replication)

```
split                    train  val  test  total
class                                           
adenocarcinoma             235   34    68    337
large.cell.carcinoma       131   19    37    187
normal                      46    7    13     66
squamous.cell.carcinoma    180   25    52    257
```

- clean train imbalance ratio (max/min): 5.11
- class weighting is enabled by default for `clean` training runs.