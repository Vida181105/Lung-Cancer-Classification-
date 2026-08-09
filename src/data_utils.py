"""Dataset discovery, leakage audit, split construction and tf.data pipelines.

Design notes
------------
* Every notebook resolves the dataset through :func:`resolve_data_root`, so the
  same code runs locally (``Data/``) and on Kaggle (``/kaggle/input/<slug>/``).
* The dataset ships with duplicate files that straddle the train/test boundary
  (LESSON 4). :func:`audit_leakage` quantifies this and
  :func:`build_clean_split` removes it.
* TensorFlow is imported lazily inside the pipeline helpers so that the audit
  notebook (which needs no TF) starts instantly.
* Datasets always yield **raw float32 pixels in [0, 255]**. Rescaling /
  ``preprocess_input`` is done *inside* the model (see ``src/models.py``), which
  makes it impossible to pair a model with the wrong preprocessing.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BATCH_SIZE,
    CLASS_FOLDER_PREFIXES,
    CLASS_NAMES,
    CLEAN_SPLIT_FRACTIONS,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    LOCAL_DATA_DIR,
    SEED,
    SPLITS_DIR,
)

# --------------------------------------------------------------------------
# Environment / path resolution
# --------------------------------------------------------------------------


def resolve_data_root():
    """Return the directory that contains ``train/``, ``valid/``, ``test/``.

    On Kaggle the dataset is mounted read-only under ``/kaggle/input/<slug>/``
    and the slug is chosen by whoever attached it, so it is auto-detected by
    name pattern. Locally we fall back to the repo's ``Data/`` folder.
    """
    if os.path.exists("/kaggle/input"):
        candidates = os.listdir("/kaggle/input")
        for c in candidates:
            if "ctscan" in c.lower() or "ct-scan" in c.lower() or "chest" in c.lower():
                root = Path("/kaggle/input") / c
                return _descend_to_split_root(root)
        raise FileNotFoundError(
            "Running on Kaggle but couldn't auto-detect the dataset folder. "
            f"Found: {candidates}. Check the exact folder name and update "
            "resolve_data_root()."
        )
    if not LOCAL_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Local dataset folder not found at {LOCAL_DATA_DIR}. Download the "
            "Kaggle 'Chest CT-Scan images' dataset into it, or run on Kaggle "
            "with the dataset attached as an input."
        )
    return _descend_to_split_root(LOCAL_DATA_DIR)


def _descend_to_split_root(root: Path) -> Path:
    """Some Kaggle mirrors nest the split folders one level deeper."""
    root = Path(root)
    if any((root / s).is_dir() for s in ("train", "test")):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any((child / s).is_dir() for s in ("train", "test")):
            return child
    return root


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def normalize_class_name(folder_name: str) -> str:
    """Map a raw class folder name onto one of ``CLASS_NAMES``.

    ``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`` -> ``adenocarcinoma``.
    Longest prefix wins so that ``large.cell.carcinoma`` is never mistaken for
    another class.
    """
    name = folder_name.strip().lower()
    matches = [c for c, p in CLASS_FOLDER_PREFIXES.items() if name.startswith(p)]
    if not matches:
        raise ValueError(
            f"Unrecognised class folder '{folder_name}'. Expected one of "
            f"{list(CLASS_FOLDER_PREFIXES)} (optionally with a staging suffix)."
        )
    return max(matches, key=len)


def index_dataset(data_root=None) -> pd.DataFrame:
    """Walk the dataset and return one row per image file.

    Columns: ``filepath, filename, class, label, orig_split, folder``.
    ``orig_split`` uses the dataset's own folder names normalised to
    ``train`` / ``val`` / ``test``.
    """
    data_root = Path(data_root or resolve_data_root())
    split_alias = {"train": "train", "valid": "val", "validation": "val",
                   "val": "val", "test": "test"}

    rows = []
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        split = split_alias.get(split_dir.name.strip().lower())
        if split is None:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = normalize_class_name(class_dir.name)
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append({
                    "filepath": str(f),
                    "filename": f.name,
                    "class": cls,
                    "label": CLASS_NAMES.index(cls),
                    "orig_split": split,
                    "folder": class_dir.name,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No images found under {data_root}. Path resolution probably "
            "failed - check resolve_data_root()."
        )
    return df.reset_index(drop=True)


def class_counts(df: pd.DataFrame, split_col="orig_split") -> pd.DataFrame:
    """Class x split count table, always in canonical class order."""
    table = (df.pivot_table(index="class", columns=split_col, values="filepath",
                            aggfunc="count", fill_value=0)
               .reindex(CLASS_NAMES, fill_value=0))
    table["total"] = table.sum(axis=1)
    return table


# --------------------------------------------------------------------------
# Duplicate / leakage audit (LESSON 4)
# --------------------------------------------------------------------------


def file_hash(path, chunk_size=1 << 20) -> str:
    """MD5 of the raw file bytes (content identity, not pixel identity)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def add_hashes(df: pd.DataFrame, progress_every=200) -> pd.DataFrame:
    """Add a ``hash`` column. This reads every file once (no training)."""
    df = df.copy()
    hashes = []
    for i, p in enumerate(df["filepath"], start=1):
        hashes.append(file_hash(p))
        if progress_every and i % progress_every == 0:
            print(f"  hashed {i}/{len(df)} files")
    df["hash"] = hashes
    return df


def find_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to a content hash that appears more than once."""
    if "hash" not in df.columns:
        raise KeyError("Call add_hashes(df) first.")
    counts = df["hash"].value_counts()
    dup_hashes = counts[counts > 1].index
    return df[df["hash"].isin(dup_hashes)].sort_values(["hash", "orig_split"])


def audit_leakage(df: pd.DataFrame) -> dict:
    """Quantify duplication and cross-split leakage.

    Returns a dict with the headline numbers plus per-class breakdowns. A
    "leaked" file is a duplicate whose hash occurs in more than one split -
    i.e. the model can memorise it in training and be rewarded at test time.
    """
    dups = find_duplicate_groups(df)
    n_groups = dups["hash"].nunique()
    n_dup_files = len(dups)
    n_redundant = n_dup_files - n_groups  # files removable without data loss

    spans = df.groupby("hash")["orig_split"].nunique()
    leaked_hashes = spans[spans > 1].index
    leaked = df[df["hash"].isin(leaked_hashes)]

    train_test = df.groupby("hash")["orig_split"].agg(set)
    tt_hashes = [h for h, s in train_test.items()
                 if "train" in s and "test" in s]
    train_test_leak = df[df["hash"].isin(tt_hashes)]

    return {
        "n_files": len(df),
        "n_unique_hashes": df["hash"].nunique(),
        "n_duplicate_groups": int(n_groups),
        "n_files_in_duplicate_groups": int(n_dup_files),
        "n_redundant_files": int(n_redundant),
        "n_leaked_files": int(len(leaked)),
        "n_train_test_leaked_files": int(len(train_test_leak)),
        "duplicates_per_class": dups["class"].value_counts()
                                   .reindex(CLASS_NAMES, fill_value=0).to_dict(),
        "leaked_per_class": leaked["class"].value_counts()
                                .reindex(CLASS_NAMES, fill_value=0).to_dict(),
        "duplicate_rows": dups,
        "leaked_rows": leaked,
    }


# --------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------


def build_faithful_split(df: pd.DataFrame) -> pd.DataFrame:
    """The paper-comparable split: the dataset's own folders, untouched.

    Duplicates are deliberately kept. Results from this split are the ones that
    can be compared with the paper; they are also the ones affected by the
    known train/test leakage.
    """
    out = df.copy()
    out["split"] = out["orig_split"]
    out["split_variant"] = "faithful"
    return out.reset_index(drop=True)


def build_clean_split(df: pd.DataFrame, fractions=CLEAN_SPLIT_FRACTIONS,
                      seed=SEED) -> pd.DataFrame:
    """Deduplicated, leakage-free, group-aware stratified split.

    Group = content hash. All copies of an image form one group; exactly one
    representative per group survives, so no group can straddle two splits and
    the leakage-free property holds by construction.

    NOT a replication of the paper's protocol - label results from this split
    as a robustness / generalisation experiment everywhere.
    """
    from sklearn.model_selection import train_test_split

    if "hash" not in df.columns:
        raise KeyError("Call add_hashes(df) first - clean split is hash-based.")

    # Deterministic representative per group: first path in sorted order.
    reps = (df.sort_values("filepath")
              .drop_duplicates(subset="hash", keep="first")
              .reset_index(drop=True))

    train_frac, val_frac, test_frac = fractions
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1.0"

    train_df, rest = train_test_split(
        reps, train_size=train_frac, stratify=reps["class"],
        random_state=seed, shuffle=True)
    rel_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest, train_size=rel_val, stratify=rest["class"],
        random_state=seed, shuffle=True)

    train_df, val_df, test_df = (train_df.copy(), val_df.copy(), test_df.copy())
    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out["split_variant"] = "clean"

    # Hard guarantee, cheap to check, catastrophic if wrong.
    assert_no_leakage(out)
    return out.reset_index(drop=True)


def assert_no_leakage(split_df: pd.DataFrame):
    """Raise if any content hash appears in more than one split."""
    if "hash" not in split_df.columns:
        raise KeyError("assert_no_leakage needs a 'hash' column.")
    spans = split_df.groupby("hash")["split"].nunique()
    bad = spans[spans > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} content hashes appear in more than one split - "
            "the split is NOT leakage-free."
        )
    return True


SPLIT_COLUMNS = ["filepath", "filename", "class", "label", "orig_split",
                 "folder", "hash", "split", "split_variant"]


def save_split(split_df: pd.DataFrame, name: str) -> Path:
    """Persist a split definition to ``outputs/splits/<name>_split.csv``."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}_split.csv"
    cols = [c for c in SPLIT_COLUMNS if c in split_df.columns]
    split_df[cols].to_csv(path, index=False)
    return path


def load_split(name: str) -> pd.DataFrame:
    """Load a split saved by :func:`save_split` (notebook 00 must run first).

    Filepaths are re-resolved against the *current* data root, so a split
    definition produced locally still works on Kaggle and vice versa.
    """
    path = SPLITS_DIR / f"{name}_split.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/00_dataset_audit.ipynb first - "
            "it writes the faithful and clean split definitions."
        )
    df = pd.read_csv(path)
    return rebase_filepaths(df)


def rebase_filepaths(df: pd.DataFrame, data_root=None) -> pd.DataFrame:
    """Rewrite stored absolute paths onto the current data root.

    Keeps the trailing ``<split>/<class folder>/<file>`` portion of each stored
    path, which is stable across environments.
    """
    data_root = Path(data_root or resolve_data_root())
    df = df.copy()

    def _rebase(p):
        parts = Path(p).parts
        if len(parts) < 3:
            return p
        tail = Path(*parts[-3:])
        return str(data_root / tail)

    df["filepath"] = df["filepath"].map(_rebase)
    missing = [p for p in df["filepath"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files from the saved split are missing under "
            f"{data_root}. First missing: {missing[0]}"
        )
    return df


def split_counts(split_df: pd.DataFrame) -> pd.DataFrame:
    """Class x split counts for a built split (train/val/test order enforced)."""
    table = class_counts(split_df, split_col="split")
    ordered = [c for c in ["train", "val", "test"] if c in table.columns]
    return table[ordered + ["total"]]


def imbalance_ratio(split_df: pd.DataFrame, split="train") -> float:
    """max/min class count in a split - >2 means class weighting matters."""
    counts = split_df[split_df["split"] == split]["class"].value_counts()
    return float(counts.max() / max(counts.min(), 1))


# --------------------------------------------------------------------------
# tf.data pipelines
# --------------------------------------------------------------------------


def get_augmenter(seed=SEED):
    """Light geometric augmentation. CT scans are not flipped vertically and
    are not heavily rotated, so the transforms stay conservative."""
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, fill_mode="nearest", seed=seed),
            layers.RandomZoom(0.10, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=seed),
        ],
        name="augmentation",
    )


def _decode_image(path, img_size):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)          # raw [0, 255]; model rescales
    img.set_shape((img_size[0], img_size[1], 3))
    return img


def make_dataset(df: pd.DataFrame, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                 shuffle=False, augment=False, seed=SEED, one_hot=False,
                 drop_remainder=False):
    """Build a ``tf.data.Dataset`` of ``(image[0..255], label)`` from a frame.

    Pass ``shuffle=augment=True`` for training sets and leave both False for
    val/test so that predictions stay aligned with ``df['label']``.
    """
    import tensorflow as tf

    if df.empty:
        raise ValueError("make_dataset received an empty dataframe.")

    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    if one_hot:
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_NAMES))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode_image(p, img_size), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if augment:
        augmenter = get_augmenter(seed=seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def make_split_datasets(split_df: pd.DataFrame, img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE, augment_train=True, seed=SEED):
    """Convenience wrapper: returns ``(train_ds, val_ds, test_ds, frames)``."""
    frames = {s: split_df[split_df["split"] == s].reset_index(drop=True)
              for s in ("train", "val", "test")}
    for s, f in frames.items():
        if f.empty:
            raise ValueError(f"Split '{s}' is empty - check the split definition.")
    train_ds = make_dataset(frames["train"], img_size, batch_size,
                            shuffle=True, augment=augment_train, seed=seed)
    val_ds = make_dataset(frames["val"], img_size, batch_size)
    test_ds = make_dataset(frames["test"], img_size, batch_size)
    return train_ds, val_ds, test_ds, frames


def labels_of(df: pd.DataFrame) -> np.ndarray:
    """Ground-truth label vector in dataframe order (== dataset order)."""
    return df["label"].astype(int).to_numpy()
