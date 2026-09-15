"""Phase 1 - data-leakage / duplicate analysis for the CT dataset.

Entirely additive: nothing in this module is imported by notebooks 00-07, and
every artefact it writes goes to ``outputs/leakage/`` or to a new, distinctly
named split file. No existing result is ever overwritten.

Design constraints this module was written under
------------------------------------------------
* **CPU only.** No TensorFlow is needed for steps 1-4 and 6-7: identification,
  hashing, near-duplicate search and split construction are pure
  numpy/pandas/PIL. TensorFlow is imported lazily, inside one function, only if
  a checkpoint evaluation is actually requested.
* **No neural embeddings.** Near-duplicate detection uses a DCT perceptual hash
  (pHash) implemented here in numpy, so there is no new dependency and no
  model forward pass. Pair search is O(n^2) on 64-bit integers, which is
  ~500k comparisons for this 1,000-image dataset - milliseconds - and stays
  practical to roughly 20k images.
* **Never assume a filename is a patient ID.** :func:`inspect_patient_identifiers`
  reports what identifier-like structure exists and explicitly refuses to treat
  it as a patient identifier without corroborating metadata.

Terminology used consistently throughout
----------------------------------------
``exact duplicate``   identical file bytes (MD5), already computed by
                      ``data_utils.add_hashes``.
``near duplicate``    perceptually similar but not byte-identical: Hamming
                      distance <= ``PHASH_THRESHOLD`` between 64-bit pHashes.
``image group``       a connected component over the union of exact-duplicate
                      and near-duplicate relations. This is the grouping unit
                      for the leakage-controlled split - it is an *image*
                      group, NOT a patient group.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    OUTPUT_ROOT,
    SEED,
    SPLITS_DIR,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# All Phase 1 artefacts live here, so nothing can collide with outputs/ files
# produced by notebooks 00-07.
LEAKAGE_DIR = OUTPUT_ROOT / "leakage"

# pHash geometry. 8x8 low-frequency block of a 32x32 DCT -> 64-bit hash.
PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4

# Hamming distance at or below which two pHashes are called near-duplicates.
# 5/64 is the conventional conservative threshold: it catches re-saves, minor
# crops and compression variants while rarely joining genuinely distinct
# slices. Sensitivity to this choice is reported by phash_threshold_sweep().
PHASH_THRESHOLD = 5

# Split fractions for the leakage-controlled split. Chosen to match the
# project's existing `clean` split so the two are comparable.
LEAKAGE_SPLIT_FRACTIONS = (0.70, 0.10, 0.20)

# Filename patterns worth reporting in step 1. None of these is treated as a
# patient identifier - they are reported so a human can judge.
_ID_LIKE_PATTERNS = {
    "pure_digits": r"^\d+$",
    "digits_with_copy_suffix": r"^\d+\s*\(\d+\)$",
    "leading_digit_block": r"^\d{4,}",
    "contains_patient_word": r"(?i)(patient|subject|case|pid)",
}

METADATA_EXTENSIONS = (".csv", ".json", ".xml", ".txt", ".xlsx", ".tsv")
DICOM_EXTENSIONS = (".dcm", ".dicom")


def ensure_leakage_dir() -> Path:
    LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    return LEAKAGE_DIR


# --------------------------------------------------------------------------
# STEP 1 - patient identifiers
# --------------------------------------------------------------------------


def scan_for_metadata_files(data_root) -> dict:
    """Look for any sidecar metadata or DICOM files under the dataset root.

    The Kaggle 'Chest CT-Scan images' dataset ships as plain PNG/JPG in class
    folders, so the expected result is *nothing found*. This function exists so
    that conclusion is a measurement rather than an assumption.
    """
    data_root = Path(data_root)
    meta_files, dicom_files = [], []
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in METADATA_EXTENSIONS:
            meta_files.append(str(p.relative_to(data_root)))
        elif suf in DICOM_EXTENSIONS:
            dicom_files.append(str(p.relative_to(data_root)))
    return {
        "data_root": str(data_root),
        "n_metadata_files": len(meta_files),
        "metadata_files": meta_files[:50],
        "n_dicom_files": len(dicom_files),
        "dicom_files": dicom_files[:50],
    }


def probe_dicom_patient_ids(dicom_paths, limit=25) -> dict:
    """Read PatientID from DICOM headers, if there are any DICOMs and pydicom.

    Returns a dict whose ``available`` key is the only thing callers should
    branch on. pydicom is NOT in requirements.txt; its absence is reported, not
    raised, because the expected state for this dataset is "no DICOMs at all".
    """
    dicom_paths = list(dicom_paths or [])
    if not dicom_paths:
        return {"available": False,
                "reason": "no DICOM files found under the dataset root"}
    try:
        import pydicom
    except ImportError:
        return {"available": False,
                "reason": ("DICOM files exist but pydicom is not installed "
                           "(not in requirements.txt). Install pydicom to read "
                           "PatientID headers.")}

    ids = []
    for p in dicom_paths[:limit]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            ids.append(str(getattr(ds, "PatientID", "")) or None)
        except Exception as exc:                      # unreadable header
            ids.append(f"<unreadable: {type(exc).__name__}>")
    real = [i for i in ids if i and not str(i).startswith("<")]
    return {
        "available": bool(real),
        "n_probed": len(ids),
        "n_with_patient_id": len(real),
        "distinct_patient_ids": len(set(real)),
        "sample": real[:10],
    }


def inspect_patient_identifiers(df: pd.DataFrame, data_root=None) -> dict:
    """Decide, with evidence, whether genuine patient IDs exist.

    **This function will not call a filename a patient ID.** A numeric filename
    stem is an image index unless something external (a manifest, a DICOM
    header, a dataset card) says otherwise. Treating it as a patient ID and
    then "grouping by patient" would manufacture a false guarantee - the exact
    failure this whole phase exists to detect.

    Returns a dict including ``patient_ids_available`` (bool) and
    ``verdict`` (a sentence safe to paste into a report).
    """
    import re

    filenames = df["filename"].astype(str)
    stems = filenames.str.rsplit(".", n=1).str[0]

    pattern_hits = {}
    for name, pat in _ID_LIKE_PATTERNS.items():
        pattern_hits[name] = int(stems.str.match(pat).sum())

    # If a filename stem were a patient ID, we would expect far fewer distinct
    # stems than images (many slices per patient). Report the ratio; do not
    # conclude from it.
    n_images = len(df)
    n_distinct_stems = int(stems.nunique())
    stem_reuse_ratio = round(n_distinct_stems / max(n_images, 1), 4)

    # Structural candidates: any column that could carry an identifier.
    available_columns = list(df.columns)
    identifier_columns = [c for c in available_columns
                          if c.lower() in ("patient", "patient_id", "subject",
                                           "subject_id", "case_id", "study_uid",
                                           "series_uid")]

    meta = scan_for_metadata_files(data_root) if data_root else {
        "n_metadata_files": None, "n_dicom_files": None,
        "note": "data_root not supplied; filesystem not scanned"}
    dicom = probe_dicom_patient_ids(
        [Path(meta["data_root"]) / p for p in meta.get("dicom_files", [])]
        if meta.get("n_dicom_files") else [])

    patient_ids_available = bool(identifier_columns) or bool(dicom.get("available"))

    if patient_ids_available:
        verdict = ("Genuine patient identifiers ARE available "
                   f"(columns={identifier_columns}, dicom={dicom.get('available')}). "
                   "Patient-level grouped splitting is therefore defensible.")
    else:
        verdict = (
            "NO genuine patient identifiers are available. The dataset provides "
            "only filepath, filename, class folder and original split; there is "
            "no manifest, no DICOM header, and no patient/subject column. "
            "Filename stems are numeric but are image indices, not patient IDs, "
            "and are NOT treated as such here. "
            "CONSEQUENCE: patient-level leakage CANNOT be verified or excluded. "
            "Any claim of 'patient-aware' or 'patient-level' evaluation would be "
            "unsupported. The strongest defensible grouping is image-level "
            "(exact + near-duplicate groups), which is what this analysis builds."
        )

    return {
        "n_images": n_images,
        "n_distinct_filename_stems": n_distinct_stems,
        "stem_reuse_ratio": stem_reuse_ratio,
        "filename_pattern_hits": pattern_hits,
        "filename_examples": filenames.head(8).tolist(),
        "columns_available": available_columns,
        "identifier_columns_found": identifier_columns,
        "metadata_scan": meta,
        "dicom_probe": dicom,
        "patient_ids_available": patient_ids_available,
        "patient_level_split_defensible": patient_ids_available,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# STEP 2 - exact duplicates
# --------------------------------------------------------------------------


def file_md5(path, chunk_size=1 << 20) -> str:
    """MD5 over raw file bytes. Mirrors data_utils.file_hash()."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_md5(df: pd.DataFrame, progress_every=200) -> pd.DataFrame:
    """Add a ``hash`` column if the frame does not already carry one."""
    if "hash" in df.columns:
        return df
    df = df.copy()
    hashes = []
    for i, p in enumerate(df["filepath"], start=1):
        hashes.append(file_md5(p))
        if progress_every and i % progress_every == 0:
            print(f"  MD5 {i}/{len(df)}")
    df["hash"] = hashes
    return df


def exact_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format table of every file that shares its MD5 with another file.

    Columns: ``group_id, hash, group_size, filepath, filename, class, split,
    n_splits_spanned, splits_spanned``.
    """
    if "hash" not in df.columns:
        raise KeyError("call ensure_md5(df) first")
    counts = df["hash"].value_counts()
    dup_hashes = counts[counts > 1].index
    dups = df[df["hash"].isin(dup_hashes)].copy()
    if dups.empty:
        return pd.DataFrame(columns=["group_id", "hash", "group_size", "filepath",
                                     "filename", "class", "split",
                                     "n_splits_spanned", "splits_spanned"])

    order = {h: i + 1 for i, h in enumerate(sorted(dup_hashes))}
    dups["group_id"] = dups["hash"].map(order)
    dups["group_size"] = dups["hash"].map(counts)

    split_col = "split" if "split" in dups.columns else "orig_split"
    spans = dups.groupby("hash")[split_col].agg(lambda s: sorted(set(s)))
    dups["splits_spanned"] = dups["hash"].map(lambda h: "|".join(spans[h]))
    dups["n_splits_spanned"] = dups["hash"].map(lambda h: len(spans[h]))
    if split_col != "split":
        dups["split"] = dups[split_col]

    cols = ["group_id", "hash", "group_size", "filepath", "filename", "class",
            "split", "n_splits_spanned", "splits_spanned"]
    return dups.sort_values(["group_id", "filepath"])[cols].reset_index(drop=True)


def duplicate_pair_matrix(df: pd.DataFrame, split_col="split",
                          group_col="hash") -> pd.DataFrame:
    """Count duplicate PAIRS by the split-pair they connect.

    Produces the six numbers the brief asks for - within train / within val /
    within test / train-val / train-test / val-test - by enumerating every
    unordered pair inside every duplicate group. Cross-split cells are the ones
    that constitute leakage; within-split cells are redundancy, not leakage.
    """
    from itertools import combinations

    splits = sorted(df[split_col].dropna().unique())
    mat = pd.DataFrame(0, index=splits, columns=splits, dtype=int)

    for _, grp in df.groupby(group_col):
        if len(grp) < 2:
            continue
        for a, b in combinations(grp[split_col].tolist(), 2):
            lo, hi = sorted([a, b])
            mat.loc[lo, hi] += 1
            if lo != hi:
                mat.loc[hi, lo] += 1          # keep the matrix symmetric
    return mat


def summarise_pair_matrix(mat: pd.DataFrame) -> dict:
    """Flatten the pair matrix into the named counts used in the report."""
    def cell(a, b):
        if a in mat.index and b in mat.columns:
            return int(mat.loc[a, b])
        return 0

    within = {s: cell(s, s) for s in mat.index}
    cross = {}
    seen = set()
    for a in mat.index:
        for b in mat.columns:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            cross[f"{a}<->{b}"] = cell(a, b)
    return {
        "within_split_pairs": within,
        "cross_split_pairs": cross,
        "total_within": int(sum(within.values())),
        "total_cross": int(sum(cross.values())),
    }


# --------------------------------------------------------------------------
# STEP 3 - near duplicates (perceptual hash, CPU only, no embeddings)
# --------------------------------------------------------------------------


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, so no scipy dependency is required."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0, :] *= np.sqrt(1 / n)
    m[1:, :] *= np.sqrt(2 / n)
    return m


def phash(path, hash_size=PHASH_SIZE, highfreq_factor=PHASH_HIGHFREQ_FACTOR) -> int:
    """64-bit DCT perceptual hash of one image.

    Grayscale -> 32x32 -> 2D DCT -> keep the top-left 8x8 low-frequency block ->
    threshold at the block median (excluding the DC term, which encodes only
    overall brightness). Robust to re-compression, small crops and mild
    resizing; sensitive to genuine anatomical difference.
    """
    from PIL import Image

    size = hash_size * highfreq_factor
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float64)

    d = _dct_matrix(size)
    dct = d @ arr @ d.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])            # drop DC before thresholding
    bits = (low > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def add_phashes(df: pd.DataFrame, progress_every=200) -> pd.DataFrame:
    """Add a ``phash`` column (uint64). One decode per image; CPU-cheap."""
    df = df.copy()
    values, failures = [], []
    for i, p in enumerate(df["filepath"], start=1):
        try:
            values.append(phash(p))
        except Exception as exc:
            values.append(0)
            failures.append((p, f"{type(exc).__name__}: {exc}"))
        if progress_every and i % progress_every == 0:
            print(f"  pHash {i}/{len(df)}")
    df["phash"] = pd.Series(values, index=df.index, dtype="uint64")
    if failures:
        print(f"  WARNING: {len(failures)} image(s) failed to hash; "
              "they are given phash=0 and excluded from pairing.")
        for p, e in failures[:5]:
            print("   ", p, e)
    df.attrs["phash_failures"] = failures
    return df


def _popcount_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        t[i] = bin(i).count("1")
    return t


def hamming_distance_matrix(hashes: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between 64-bit hashes, vectorised.

    O(n^2) in memory and time. At n=1,000 that is a 1M-entry uint8 matrix (1 MB)
    computed in well under a second. Above roughly 20,000 images this should be
    replaced with a BK-tree or LSH bucketing; a guard raises before that point
    rather than silently thrashing.
    """
    n = len(hashes)
    if n > 20000:
        raise MemoryError(
            f"{n} images would need an {n}x{n} distance matrix. Use a BK-tree or "
            "band/LSH bucketing instead of the brute-force path.")
    b = np.frombuffer(np.asarray(hashes, dtype=">u8").tobytes(),
                      dtype=np.uint8).reshape(n, 8)
    tbl = _popcount_table()
    xor = b[:, None, :] ^ b[None, :, :]
    return tbl[xor].sum(axis=2).astype(np.uint8)


def near_duplicate_pairs(df: pd.DataFrame, threshold=PHASH_THRESHOLD,
                         exclude_exact=True, split_col="split") -> pd.DataFrame:
    """Every image pair within ``threshold`` Hamming distance.

    ``exclude_exact=True`` removes pairs that are already byte-identical, so
    this table reports *additional* near-duplicate evidence beyond step 2
    rather than restating it.
    """
    work = df.reset_index(drop=True)
    if "phash" not in work.columns:
        raise KeyError("call add_phashes(df) first")

    dist = hamming_distance_matrix(work["phash"].to_numpy())
    iu = np.triu_indices(len(work), k=1)
    d = dist[iu]
    keep = d <= threshold
    i_idx, j_idx = iu[0][keep], iu[1][keep]

    rows = []
    for i, j, dd in zip(i_idx, j_idx, d[keep]):
        a, b = work.iloc[i], work.iloc[j]
        same_bytes = ("hash" in work.columns) and (a["hash"] == b["hash"])
        if exclude_exact and same_bytes:
            continue
        sa = a.get(split_col, None)
        sb = b.get(split_col, None)
        rows.append({
            "hamming_distance": int(dd),
            "exact_duplicate": bool(same_bytes),
            "same_class": a["class"] == b["class"],
            "cross_split": (sa is not None and sb is not None and sa != sb),
            "split_a": sa, "split_b": sb,
            "class_a": a["class"], "class_b": b["class"],
            "filename_a": a["filename"], "filename_b": b["filename"],
            "filepath_a": a["filepath"], "filepath_b": b["filepath"],
        })
    cols = ["hamming_distance", "exact_duplicate", "same_class", "cross_split",
            "split_a", "split_b", "class_a", "class_b",
            "filename_a", "filename_b", "filepath_a", "filepath_b"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["hamming_distance", "filename_a"]).reset_index(drop=True)


def phash_threshold_sweep(df: pd.DataFrame, thresholds=(0, 2, 4, 5, 6, 8, 10)) -> pd.DataFrame:
    """How many pairs each threshold would call near-duplicates.

    Reported so the choice of PHASH_THRESHOLD is visible and auditable rather
    than a magic number: if the pair count explodes between 5 and 8, the
    threshold is doing real work; if it is flat, the finding is robust.
    """
    dist = hamming_distance_matrix(df["phash"].to_numpy())
    iu = np.triu_indices(len(df), k=1)
    d = dist[iu]
    same_bytes = None
    if "hash" in df.columns:
        h = df["hash"].to_numpy()
        same_bytes = h[iu[0]] == h[iu[1]]
    rows = []
    for t in thresholds:
        m = d <= t
        rows.append({
            "threshold": t,
            "pairs_total": int(m.sum()),
            "pairs_excluding_exact": int((m & ~same_bytes).sum())
                                     if same_bytes is not None else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Image groups (exact + near duplicates), via union-find
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_image_groups(df: pd.DataFrame, near_pairs: pd.DataFrame) -> pd.DataFrame:
    """Add ``image_group`` - connected components over duplicate relations.

    Two images share a group if they are byte-identical OR within the pHash
    threshold, transitively. This is the unit that must not straddle a split
    boundary.

    **This is an image group, not a patient group.** Two slices from the same
    patient that look different will land in different groups; this grouping
    bounds duplicate-driven leakage, not patient-driven leakage.
    """
    work = df.reset_index(drop=True).copy()
    idx_of_path = {p: i for i, p in enumerate(work["filepath"])}
    uf = _UnionFind(len(work))

    if "hash" in work.columns:
        for _, grp in work.groupby("hash"):
            ids = grp.index.tolist()
            for other in ids[1:]:
                uf.union(ids[0], other)

    if near_pairs is not None and len(near_pairs):
        for a, b in zip(near_pairs["filepath_a"], near_pairs["filepath_b"]):
            ia, ib = idx_of_path.get(a), idx_of_path.get(b)
            if ia is not None and ib is not None:
                uf.union(ia, ib)

    roots = [uf.find(i) for i in range(len(work))]
    remap = {r: gid for gid, r in enumerate(sorted(set(roots)), start=1)}
    work["image_group"] = [remap[r] for r in roots]
    return work


def group_summary(df: pd.DataFrame, group_col="image_group") -> dict:
    sizes = df[group_col].value_counts()
    return {
        "n_images": int(len(df)),
        "n_groups": int(sizes.size),
        "n_singleton_groups": int((sizes == 1).sum()),
        "n_multi_image_groups": int((sizes > 1).sum()),
        "largest_group_size": int(sizes.max()) if len(sizes) else 0,
        "n_images_in_multi_groups": int(sizes[sizes > 1].sum()),
    }


# --------------------------------------------------------------------------
# STEP 4 - leakage-controlled split
# --------------------------------------------------------------------------


def build_leakage_controlled_split(df: pd.DataFrame, group_col="image_group",
                                   fractions=LEAKAGE_SPLIT_FRACTIONS,
                                   seed=SEED) -> pd.DataFrame:
    """Group-aware stratified split: no image group spans two splits.

    Greedy assignment - groups are shuffled deterministically, ordered largest
    first, and each is placed in whichever split is furthest below its target
    quota for that group's majority class. This keeps class balance close to
    target while making the no-straddle property hold *by construction*, which
    is then asserted by :func:`assert_no_group_leakage`.

    **Naming discipline**: the returned frame's ``split_variant`` is
    ``leakage_controlled_image_group`` - never "patient" - so a downstream
    reader cannot mistake it for a patient-level guarantee.
    """
    rng = np.random.default_rng(seed)
    work = df.reset_index(drop=True).copy()

    groups = []
    for gid, grp in work.groupby(group_col):
        cls = grp["class"].mode().iloc[0]
        groups.append({"group": gid, "size": len(grp), "class": cls})
    gdf = pd.DataFrame(groups)
    gdf = gdf.iloc[rng.permutation(len(gdf))]
    gdf = gdf.sort_values("size", ascending=False, kind="mergesort")

    train_f, val_f, test_f = fractions
    targets = {"train": train_f, "val": val_f, "test": test_f}
    per_class_total = work["class"].value_counts().to_dict()
    quota = {s: {c: targets[s] * n for c, n in per_class_total.items()}
             for s in targets}
    filled = {s: {c: 0.0 for c in per_class_total} for s in targets}

    assignment = {}
    for _, row in gdf.iterrows():
        c, n = row["class"], row["size"]
        deficits = {s: quota[s][c] - filled[s][c] for s in targets}
        chosen = max(deficits, key=lambda s: (deficits[s], targets[s]))
        assignment[row["group"]] = chosen
        filled[chosen][c] += n

    work["split"] = work[group_col].map(assignment)
    work["split_variant"] = "leakage_controlled_image_group"
    assert_no_group_leakage(work, group_col=group_col)
    return work


def assert_no_group_leakage(split_df: pd.DataFrame, group_col="image_group",
                            split_col="split") -> bool:
    """Raise if any image group appears in more than one split."""
    spans = split_df.groupby(group_col)[split_col].nunique()
    bad = spans[spans > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} image group(s) span more than one split - the split is "
            "NOT leakage-controlled.")
    return True


def split_class_table(split_df: pd.DataFrame) -> pd.DataFrame:
    t = (split_df.pivot_table(index="class", columns="split", values="filepath",
                              aggfunc="count", fill_value=0)
                 .reindex(CLASS_NAMES, fill_value=0))
    ordered = [c for c in ("train", "val", "test") if c in t.columns]
    t = t[ordered]
    t["total"] = t.sum(axis=1)
    return t


def save_leakage_controlled_split(split_df: pd.DataFrame,
                                  name="leakage_controlled") -> Path:
    """Write to ``outputs/splits/<name>_split.csv`` - a NEW filename.

    The existing ``faithful_split.csv`` and ``clean_split.csv`` are never
    touched; this refuses to write if the target already exists under a name
    that would collide with them.
    """
    if name in ("faithful", "clean"):
        raise ValueError("refusing to overwrite an existing project split")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}_split.csv"
    cols = [c for c in ["filepath", "filename", "class", "label", "orig_split",
                        "folder", "hash", "phash", "image_group", "split",
                        "split_variant"] if c in split_df.columns]
    split_df[cols].to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# STEP 5 - evaluation of an EXISTING checkpoint (never trains)
# --------------------------------------------------------------------------


def find_existing_checkpoints(models_dir=None) -> list:
    """Every reusable model file on disk. Empty list = nothing to evaluate."""
    from src.config import MODELS_DIR

    d = Path(models_dir or MODELS_DIR)
    if not d.exists():
        return []
    out = []
    for pat in ("*.keras", "*.h5", "*.hdf5"):
        out.extend(sorted(str(p) for p in d.glob(pat)))
    return out


def checkpoint_availability_report(models_dir=None) -> dict:
    """Whether step 5 can run at all, and what it means if it cannot."""
    cks = find_existing_checkpoints(models_dir)
    from src.config import MODELS_DIR

    return {
        "models_dir": str(models_dir or MODELS_DIR),
        "models_dir_exists": Path(models_dir or MODELS_DIR).exists(),
        "n_checkpoints": len(cks),
        "checkpoints": cks,
        "can_evaluate_without_retraining": bool(cks),
        "note": ("A checkpoint is available; the leakage-controlled evaluation can "
                 "proceed with no training."
                 if cks else
                 "NO checkpoint exists. models/ is git-ignored and no weights were "
                 "retained from any previous run, so the trained MiniConvNet cannot "
                 "be re-scored on the new split. Producing a leakage-controlled "
                 "number REQUIRES retraining, which this phase must not start "
                 "without explicit approval."),
    }


def evaluate_checkpoint_on_split(checkpoint_path, split_df, split="test",
                                 batch_size=None, one_hot=True,
                                 run_name="leakage_controlled_eval") -> dict:
    """Score an existing model on a split. **Loads weights; never trains.**

    Deliberately reuses the project's own evaluation pipeline
    (``evaluate_utils``) so the numbers are computed exactly as every other
    result in this repo was. TensorFlow is imported here, lazily, so the rest of
    this module runs on a machine with no TF installed.

    IMPORTANT caveat the caller must carry into any comparison: a model trained
    on the ORIGINAL split has very likely seen images that sit in the
    leakage-controlled TEST split. Re-scoring an old checkpoint therefore gives
    an *upper bound* on leakage-controlled performance, not a clean estimate.
    The only clean estimate comes from retraining under the new split.
    """
    from src.config import BATCH_SIZE
    from src.data_utils import make_dataset
    from src.evaluate_utils import (compute_metrics, confusion, detect_collapse,
                                    per_class_report, predict, save_predictions,
                                    tumor_vs_subtype_breakdown)
    from src.models import load_model_checkpoint

    frame = split_df[split_df["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"split '{split}' is empty")

    ds = make_dataset(frame, batch_size=batch_size or BATCH_SIZE, one_hot=one_hot)
    model = load_model_checkpoint(checkpoint_path)           # load only
    y_true, y_pred, y_prob = predict(model, ds)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    collapse = detect_collapse(kappa=metrics["cohen_kappa"], mcc=metrics["mcc"],
                               y_pred=y_pred)
    breakdown = tumor_vs_subtype_breakdown(y_true, y_pred)
    per_class = per_class_report(y_true, y_pred)
    cm = confusion(y_true, y_pred)

    save_predictions(run_name, y_true, y_pred, y_prob,
                     meta={"split_variant": "leakage_controlled_image_group",
                           "checkpoint": str(checkpoint_path),
                           "evaluated_split": split,
                           "caveat": ("checkpoint was trained on the ORIGINAL split; "
                                      "this is an upper bound, not a clean estimate")})
    import tensorflow as tf
    tf.keras.backend.clear_session()

    return {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "n_eval": int(len(y_true)),
        "metrics": metrics,
        "per_class_f1": {CLASS_NAMES[i]: float(per_class.loc[CLASS_NAMES[i], "f1-score"])
                         for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "tumor_detection_accuracy": breakdown["binary_tumor_vs_healthy_accuracy"],
        "subtype_accuracy": breakdown["subtype_accuracy_all_tumors"],
        "status": collapse["status"],
    }


def metrics_from_saved_predictions(run_name):
    """Recompute the full metric set for an ALREADY-SAVED run. Zero compute.

    Used to restate the ORIGINAL-split performance in exactly the same format as
    the leakage-controlled numbers, so the two columns of the comparison table
    are computed by identical code.
    """
    from src.evaluate_utils import (compute_metrics, confusion, load_predictions,
                                    per_class_report, tumor_vs_subtype_breakdown)

    y_true, y_pred, y_prob = load_predictions(run_name)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    per_class = per_class_report(y_true, y_pred)
    breakdown = tumor_vs_subtype_breakdown(y_true, y_pred)
    return {
        "run_name": run_name,
        "n_eval": int(len(y_true)),
        "metrics": metrics,
        "per_class_f1": {c: float(per_class.loc[c, "f1-score"]) for c in CLASS_NAMES},
        "confusion_matrix": confusion(y_true, y_pred).tolist(),
        "tumor_detection_accuracy": breakdown["binary_tumor_vs_healthy_accuracy"],
        "subtype_accuracy": breakdown["subtype_accuracy_all_tumors"],
    }


def comparison_table(original: dict, controlled: dict = None) -> pd.DataFrame:
    """Side-by-side original vs leakage-controlled, with deltas where possible."""
    def row(label, key, sub=None):
        def get(d):
            if d is None:
                return None
            return d["metrics"].get(key) if sub is None else d.get(sub)
        o, c = get(original), get(controlled)
        return {"metric": label, "original": o, "leakage_controlled": c,
                "delta": (None if (o is None or c is None) else round(c - o, 6))}

    rows = [
        row("accuracy", "accuracy"),
        row("precision_macro", "precision_macro"),
        row("recall_macro", "recall_macro"),
        row("f1_macro", "f1_macro"),
        row("cohen_kappa", "cohen_kappa"),
        row("mcc", "mcc"),
        row("tumour_detection_accuracy", None, "tumor_detection_accuracy"),
        row("subtype_accuracy", None, "subtype_accuracy"),
    ]
    for c in CLASS_NAMES:
        o = original["per_class_f1"].get(c) if original else None
        cc = controlled["per_class_f1"].get(c) if controlled else None
        rows.append({"metric": f"f1[{c}]", "original": o, "leakage_controlled": cc,
                     "delta": (None if (o is None or cc is None) else round(cc - o, 6))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# STEP 6 - artefact writing (new files only)
# --------------------------------------------------------------------------


def write_leakage_outputs(duplicate_groups: pd.DataFrame,
                          near_duplicate_groups: pd.DataFrame,
                          split_assignments: pd.DataFrame,
                          leakage_report: pd.DataFrame,
                          summary: dict) -> dict:
    """Write the five Phase-1 artefacts into ``outputs/leakage/``.

    Every path is new. Nothing under ``outputs/`` produced by notebooks 00-07 is
    read for writing or overwritten, and the function refuses to clobber an
    existing Phase-1 file without an explicit caller decision.
    """
    ensure_leakage_dir()
    paths = {
        "duplicate_groups": LEAKAGE_DIR / "duplicate_groups.csv",
        "near_duplicate_groups": LEAKAGE_DIR / "near_duplicate_groups.csv",
        "split_assignments": LEAKAGE_DIR / "split_assignments.csv",
        "leakage_report": LEAKAGE_DIR / "leakage_report.csv",
        "leakage_summary": LEAKAGE_DIR / "leakage_summary.json",
    }
    duplicate_groups.to_csv(paths["duplicate_groups"], index=False)
    near_duplicate_groups.to_csv(paths["near_duplicate_groups"], index=False)
    split_assignments.to_csv(paths["split_assignments"], index=False)
    leakage_report.to_csv(paths["leakage_report"], index=False)

    summary = dict(summary)
    summary["written_at"] = datetime.now().isoformat(timespec="seconds")
    with open(paths["leakage_summary"], "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {k: str(v) for k, v in paths.items()}


def build_leakage_report_rows(patient_info: dict, exact_summary: dict,
                              near_summary: dict, group_info: dict,
                              split_table: pd.DataFrame) -> pd.DataFrame:
    """Tidy long-format report: one row per finding, safe to paste into a paper."""
    rows = [
        {"section": "patient_ids", "check": "patient_ids_available",
         "value": patient_info["patient_ids_available"],
         "note": patient_info["verdict"]},
        {"section": "patient_ids", "check": "identifier_columns_found",
         "value": ", ".join(patient_info["identifier_columns_found"]) or "none",
         "note": "columns in the index that could carry a subject identifier"},
        {"section": "patient_ids", "check": "dicom_available",
         "value": patient_info["dicom_probe"].get("available"),
         "note": patient_info["dicom_probe"].get("reason", "")},
        {"section": "exact", "check": "n_images", "value": exact_summary["n_images"],
         "note": ""},
        {"section": "exact", "check": "n_unique_md5",
         "value": exact_summary["n_unique_hashes"], "note": ""},
        {"section": "exact", "check": "n_duplicate_groups",
         "value": exact_summary["n_duplicate_groups"], "note": ""},
        {"section": "exact", "check": "n_files_in_duplicate_groups",
         "value": exact_summary["n_files_in_duplicate_groups"], "note": ""},
        {"section": "exact", "check": "n_redundant_files",
         "value": exact_summary["n_redundant_files"],
         "note": "removable with no information loss"},
        {"section": "exact", "check": "cross_split_duplicate_pairs",
         "value": exact_summary["pairs"]["total_cross"],
         "note": str(exact_summary["pairs"]["cross_split_pairs"])},
        {"section": "exact", "check": "within_split_duplicate_pairs",
         "value": exact_summary["pairs"]["total_within"],
         "note": str(exact_summary["pairs"]["within_split_pairs"])},
        {"section": "near", "check": "phash_threshold",
         "value": near_summary["threshold"], "note": "Hamming distance over 64-bit pHash"},
        {"section": "near", "check": "near_duplicate_pairs_excl_exact",
         "value": near_summary["n_pairs"], "note": ""},
        {"section": "near", "check": "near_duplicate_cross_split_pairs",
         "value": near_summary["n_cross_split"], "note": ""},
        {"section": "near", "check": "near_duplicate_cross_class_pairs",
         "value": near_summary["n_cross_class"],
         "note": "near-identical images with DIFFERENT labels - inspect manually"},
        {"section": "groups", "check": "n_image_groups",
         "value": group_info["n_groups"], "note": "exact + near, connected components"},
        {"section": "groups", "check": "n_multi_image_groups",
         "value": group_info["n_multi_image_groups"], "note": ""},
        {"section": "groups", "check": "n_images_in_multi_groups",
         "value": group_info["n_images_in_multi_groups"], "note": ""},
        {"section": "groups", "check": "largest_group_size",
         "value": group_info["largest_group_size"], "note": ""},
    ]
    for cls in split_table.index:
        rows.append({"section": "controlled_split", "check": f"count[{cls}]",
                     "value": int(split_table.loc[cls, "total"]),
                     "note": " ".join(f"{c}={int(split_table.loc[cls, c])}"
                                      for c in split_table.columns if c != "total")})
    return pd.DataFrame(rows, columns=["section", "check", "value", "note"])
