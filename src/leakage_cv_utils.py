"""Robustness check: 3-fold CV on Phase 1's leakage-controlled split.

Purpose (read this before anything else)
------------------------------------------------------------------------------
**This is not an attempt to improve MiniConvNet's reported accuracy.** The
existing headline (70.59% +/- 12.00%, 3-fold CV on the `faithful` pooled
split, see `outputs/results_table.csv`) was measured under a fold-
construction scheme that this project's own audit flagged as a
methodological risk: plain `StratifiedKFold` has no awareness of
duplicate/near-duplicate images, so a fold's "test" portion can share an
image - or a near-identical one - with that same fold's training portion, a
leak that can inflate the reported accuracy for reasons that have nothing to
do with how well the model actually generalises.

Phase 1 already built and verified a leakage-controlled split
(`outputs/leakage/split_assignments.csv`, `image_group` column - exact and
near-duplicate images collapsed into one group each, guaranteed never to
straddle a partition boundary) but never evaluated the CV headline against
it. This module makes that evaluation possible, changing ONLY which images
can appear together in a fold - architecture, hyperparameters, epoch budget
and every other part of the training configuration are identical to the
original headline run in `notebooks/03_cross_validation.ipynb`.

Everything here is additive: nothing in this module is imported by
notebooks 00-13, and it writes only to `outputs/leakage/
leakage_controlled_cv_results.csv` / `.json` - `outputs/results_table.csv`
and every other existing results file are read (for the headline comparison)
but never opened for writing.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_TABLE_CSV
from src.leakage_utils import LEAKAGE_DIR

SPLIT_ASSIGNMENTS_CSV = LEAKAGE_DIR / "split_assignments.csv"
LEAKAGE_CONTROLLED_CV_RESULTS_CSV = LEAKAGE_DIR / "leakage_controlled_cv_results.csv"
LEAKAGE_CONTROLLED_CV_RESULTS_JSON = LEAKAGE_DIR / "leakage_controlled_cv_results.json"

REQUIRED_COLUMNS = ["filepath", "filename", "class", "label", "split", "hash",
                   "image_group", "split_variant"]
EXPECTED_SPLIT_VARIANT = "leakage_controlled_image_group"

# Read live from outputs/results_table.csv wherever possible
# (load_headline_reference() below) - this is only the fallback used if that
# file is absent on the machine running the evaluation, so a stale hardcoded
# number is never silently preferred over a real one.
_FALLBACK_HEADLINE = {
    "accuracy_mean": 0.705936, "accuracy_std": 0.119965, "f1_macro_mean": 0.708897,
    "binary_tumor_acc": 0.956984, "subtype_acc": 0.648504,
    "source": "hardcoded fallback (outputs/results_table.csv not found on this machine)",
}


# --------------------------------------------------------------------------
# STEP 1 - load and verify the EXISTING leakage-controlled split, never build
# a new one
# --------------------------------------------------------------------------


def load_leakage_controlled_split(path=None) -> dict:
    """Load and verify Phase 1's leakage-controlled split. Never constructs a
    new one - if this fails, the caller must stop and report, not proceed.

    Returns ``{'ok': True, 'df': <rebased dataframe>, 'path': str}`` on
    success, or ``{'ok': False, 'reason': <message>}`` on any verification
    failure: missing file, missing required column, an unexpected
    `split_variant` tag, or a filepath that does not resolve on this machine.
    """
    path = Path(path or SPLIT_ASSIGNMENTS_CSV)
    if not path.exists():
        return {"ok": False,
               "reason": (f"{path} does not exist. Run notebooks/08_leakage_analysis.ipynb "
                          "first - it builds and writes this split. This module does NOT "
                          "construct a new split as a substitute.")}

    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return {"ok": False,
               "reason": f"{path} is missing expected column(s): {missing_cols}. Not using it."}

    variants = df["split_variant"].dropna().unique().tolist()
    if variants != [EXPECTED_SPLIT_VARIANT]:
        return {"ok": False,
               "reason": (f"{path}'s split_variant column is {variants}, expected exactly "
                          f"['{EXPECTED_SPLIT_VARIANT}']. Not using it - this may not be the "
                          "file this evaluation expects.")}

    from src.data_utils import rebase_filepaths
    try:
        df = rebase_filepaths(df)
    except FileNotFoundError as exc:
        return {"ok": False, "reason": f"filepaths in {path} could not be rebased: {exc}"}

    return {"ok": True, "df": df.reset_index(drop=True), "path": str(path)}


# --------------------------------------------------------------------------
# STEP 2 - verify the CV folds themselves preserve the leakage control
# --------------------------------------------------------------------------


def assert_no_group_leakage_in_folds(pool: pd.DataFrame, folds) -> dict:
    """Verify that no `image_group` straddles a fold's train/test boundary -
    i.e. that `StratifiedGroupKFold` actually did its job on THIS pooling,
    for THIS run. Returns a report rather than silently trusting the library
    call; a non-empty `violations` list means the CV is NOT actually
    leakage-controlled and the result should not be reported as such.
    """
    violations = []
    for i, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_groups = set(pool.iloc[tr_idx]["image_group"])
        te_groups = set(pool.iloc[te_idx]["image_group"])
        overlap = tr_groups & te_groups
        if overlap:
            violations.append({"fold": i, "n_overlapping_groups": len(overlap)})
    return {"clean": len(violations) == 0, "violations": violations,
           "n_folds_checked": len(folds)}


# --------------------------------------------------------------------------
# STEP 4 - comparison against the existing headline
# --------------------------------------------------------------------------


def load_headline_reference() -> dict:
    """The existing 3-fold-CV headline, read live from
    `outputs/results_table.csv` wherever possible - never a silently-stale
    hardcoded number."""
    if Path(RESULTS_TABLE_CSV).exists():
        canon = pd.read_csv(RESULTS_TABLE_CSV)
        row = canon[canon["model"].astype(str).str.startswith("MiniConvNet")]
        if len(row):
            r = row.iloc[0]
            return {
                "accuracy_mean": float(r["accuracy"]), "accuracy_std": float(r["accuracy_std"]),
                "f1_macro_mean": float(r["f1_macro"]),
                "binary_tumor_acc": float(r["binary_tumor_acc"]),
                "subtype_acc": float(r["subtype_acc"]),
                "source": f"{RESULTS_TABLE_CSV} ('{r['model']}' row, live)",
            }
    return dict(_FALLBACK_HEADLINE)


def compare_to_headline(leakage_controlled_summary: dict, headline: dict = None) -> dict:
    """Side-by-side comparison, plus the explicit 'within one std dev?'
    verdict this evaluation exists to answer.

    A lower leakage-controlled number is never framed as a failure here - see
    the ``verdict`` text, which states plainly that a confirmed-lower,
    leakage-controlled number is the more honest one if that is what the
    data shows.
    """
    headline = headline or load_headline_reference()
    lc_acc = leakage_controlled_summary["accuracy_mean"]
    lc_std = leakage_controlled_summary["accuracy_std"]
    diff = lc_acc - headline["accuracy_mean"]
    within_headline_std = abs(diff) <= headline["accuracy_std"]

    return {
        "leakage_controlled_accuracy_mean": lc_acc,
        "leakage_controlled_accuracy_std": lc_std,
        "headline_accuracy_mean": headline["accuracy_mean"],
        "headline_accuracy_std": headline["accuracy_std"],
        "difference": round(diff, 6),
        "within_headline_std_dev": bool(within_headline_std),
        "headline_source": headline["source"],
        "verdict": (
            "The leakage-controlled result is WITHIN the original headline's own standard "
            "deviation - i.e. not a meaningfully different result given the noise already "
            "present in a 3-fold estimate. This is consistent with (though does not by itself "
            "prove) the original 70.59% figure being reasonably robust to the leakage found in "
            "Phase 1."
            if within_headline_std else
            "The leakage-controlled result falls OUTSIDE the original headline's own standard "
            "deviation - a genuine, notable change, not noise. If lower, this is the MORE HONEST "
            "number, not a regression or a failure - it should be treated as the figure of "
            "record for any claim about how well MiniConvNet actually generalises within this "
            "dataset, and the difference should be reported plainly as evidence that pooled, "
            "non-group-aware CV was overstating performance."
        ),
    }


# --------------------------------------------------------------------------
# STEP 3 / 6 - artefact writing (new files only, never overwrites results_table.csv)
# --------------------------------------------------------------------------


def write_leakage_controlled_cv_results(per_fold_df: pd.DataFrame, summary: dict,
                                        comparison: dict, group_leakage_check: dict,
                                        extra: dict = None) -> dict:
    """Write the two new, clearly-labelled output files this task specifies.
    Never touches `outputs/results_table.csv` or any other existing results
    file."""
    LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    per_fold_df.to_csv(LEAKAGE_CONTROLLED_CV_RESULTS_CSV, index=False)

    payload = {
        "task": "Leakage-controlled 3-fold CV robustness check (Option B)",
        "purpose": ("Confirm or honestly revise confidence in the existing MiniConvNet CV "
                   "headline, using Phase 1's verified leakage-controlled split. NOT an "
                   "attempt to improve the reported accuracy."),
        "per_fold_results": per_fold_df.to_dict(orient="records"),
        "summary": summary,
        "comparison_to_headline": comparison,
        "group_leakage_check_on_folds": group_leakage_check,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    with open(LEAKAGE_CONTROLLED_CV_RESULTS_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    return {"csv": str(LEAKAGE_CONTROLLED_CV_RESULTS_CSV),
           "json": str(LEAKAGE_CONTROLLED_CV_RESULTS_JSON)}


def load_leakage_controlled_cv_results() -> dict:
    if not LEAKAGE_CONTROLLED_CV_RESULTS_JSON.exists():
        return {}
    with open(LEAKAGE_CONTROLLED_CV_RESULTS_JSON) as fh:
        return json.load(fh)
