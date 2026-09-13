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

# BatchNorm-ablation output - a separate pair of files (never overwrites Option
# B's own results above). Added for the BatchNorm-vs-Option-B comparison; not
# used by anything else in this module.
LEAKAGE_CONTROLLED_CV_BATCHNORM_RESULTS_CSV = LEAKAGE_DIR / "leakage_controlled_cv_batchnorm_results.csv"
LEAKAGE_CONTROLLED_CV_BATCHNORM_RESULTS_JSON = LEAKAGE_DIR / "leakage_controlled_cv_batchnorm_results.json"

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


def load_option_b_reference() -> dict:
    """Option B's own leakage-controlled 3-fold-CV result, read live from
    `outputs/leakage/leakage_controlled_cv_results.json` wherever possible -
    never a silently-stale hardcoded number, mirroring
    :func:`load_headline_reference`'s pattern for a different reference file.

    This is the correct comparison point for the BatchNorm ablation (which
    must be compared against Option B's leakage-controlled result, not the
    original pooled-CV headline) - `outputs/leakage/
    leakage_controlled_cv_results.json` is only ever READ here, never written.
    """
    if LEAKAGE_CONTROLLED_CV_RESULTS_JSON.exists():
        with open(LEAKAGE_CONTROLLED_CV_RESULTS_JSON) as fh:
            payload = json.load(fh)
        s = payload.get("summary", {})
        if "accuracy_mean" in s:
            return {
                "accuracy_mean": float(s["accuracy_mean"]), "accuracy_std": float(s["accuracy_std"]),
                "f1_macro_mean": float(s.get("f1_macro_mean", float("nan"))),
                "binary_tumor_acc": float(s.get("binary_tumor_acc_mean", float("nan"))),
                "subtype_acc": float(s.get("subtype_acc_mean", float("nan"))),
                "source": f"{LEAKAGE_CONTROLLED_CV_RESULTS_JSON} ('summary' block, live)",
            }
    return dict(_FALLBACK_OPTION_B)


def compare_to_option_b(batchnorm_summary: dict, option_b: dict = None) -> dict:
    """Side-by-side comparison of the BatchNorm variant against Option B's
    own leakage-controlled result - same shape and logic as
    :func:`compare_to_headline`, kept as a separate function rather than
    reused directly so the printed verdict text refers to the correct
    comparison point (Option B, not the original pooled-CV headline) instead
    of a misleadingly-worded reused message.

    Neither direction is framed as a failure: BatchNorm doing nothing, or
    making things slightly worse, is exactly as reportable a result as an
    improvement - see the ``verdict`` text.
    """
    option_b = option_b or load_option_b_reference()
    bn_acc = batchnorm_summary["accuracy_mean"]
    bn_std = batchnorm_summary["accuracy_std"]
    diff = bn_acc - option_b["accuracy_mean"]
    within_option_b_std = abs(diff) <= option_b["accuracy_std"]

    return {
        "batchnorm_accuracy_mean": bn_acc, "batchnorm_accuracy_std": bn_std,
        "option_b_accuracy_mean": option_b["accuracy_mean"],
        "option_b_accuracy_std": option_b["accuracy_std"],
        "difference": round(diff, 6),
        "within_option_b_std_dev": bool(within_option_b_std),
        "option_b_source": option_b["source"],
        "verdict": (
            "BatchNorm's result is WITHIN Option B's own standard deviation - i.e. not a "
            "meaningfully different result given the noise already present in a 3-fold "
            "estimate. BatchNorm neither helps nor hurts in any way distinguishable from "
            "ordinary fold-to-fold variance for this architecture and dataset size."
            if within_option_b_std else
            ("BatchNorm's result falls OUTSIDE Option B's own standard deviation, ABOVE it - a "
             "genuine, notable improvement, not noise. This is evidence BatchNorm helps training "
             "stability or accuracy for this architecture, beyond what the existing anti-collapse "
             "measures (LeakyReLU, He init, the widened bottleneck) already provide."
             if diff > 0 else
             "BatchNorm's result falls OUTSIDE Option B's own standard deviation, BELOW it - a "
             "genuine, notable regression, not noise. Report this plainly: BatchNorm measurably "
             "hurts this architecture at this dataset size, most plausibly because "
             "BatchNorm's running-statistics estimates are noisy with the small per-fold batch "
             "counts here, adding instability rather than removing it. This is a legitimate, "
             "informative ablation result, not a failed experiment.")
        ),
    }


# Fallback if outputs/leakage/leakage_controlled_cv_results.json is absent on
# the machine running the BatchNorm ablation - the value on record at the time
# this fallback was written (see git history for provenance).
_FALLBACK_OPTION_B = {
    "accuracy_mean": 0.7410464356572142, "accuracy_std": 0.042424258899348866,
    "f1_macro_mean": 0.7496416378845843, "binary_tumor_acc": 0.9540288791785798,
    "subtype_acc": 0.7057266532128338,
    "source": "hardcoded fallback (outputs/leakage/leakage_controlled_cv_results.json not found)",
}


# --------------------------------------------------------------------------
# STEP 3 / 6 - artefact writing (new files only, never overwrites results_table.csv)
# --------------------------------------------------------------------------


def write_leakage_controlled_cv_results(per_fold_df: pd.DataFrame, summary: dict,
                                        comparison: dict, group_leakage_check: dict,
                                        extra: dict = None, csv_path=None, json_path=None,
                                        task_label=None, purpose_text=None) -> dict:
    """Write the leakage-controlled-CV output files.

    Defaults to Option B's own file pair
    (``LEAKAGE_CONTROLLED_CV_RESULTS_CSV``/``_JSON``) exactly as before, so
    every existing caller (e.g. `notebooks/14_leakage_controlled_cv.ipynb`)
    is completely unaffected. ``csv_path``/``json_path`` let a different
    caller (e.g. the BatchNorm ablation) redirect output to its own,
    separately-named files instead of overwriting Option B's - pass
    :data:`LEAKAGE_CONTROLLED_CV_BATCHNORM_RESULTS_CSV`/``_JSON`` for that.
    ``task_label``/``purpose_text`` similarly override the JSON payload's
    descriptive fields so a different ablation's file doesn't claim to be
    "Option B" internally.
    """
    csv_path = Path(csv_path or LEAKAGE_CONTROLLED_CV_RESULTS_CSV)
    json_path = Path(json_path or LEAKAGE_CONTROLLED_CV_RESULTS_JSON)

    LEAKAGE_DIR.mkdir(parents=True, exist_ok=True)
    per_fold_df.to_csv(csv_path, index=False)

    payload = {
        "task": task_label or "Leakage-controlled 3-fold CV robustness check (Option B)",
        "purpose": purpose_text or (
            "Confirm or honestly revise confidence in the existing MiniConvNet CV "
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
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    return {"csv": str(csv_path), "json": str(json_path)}


def load_leakage_controlled_cv_results() -> dict:
    if not LEAKAGE_CONTROLLED_CV_RESULTS_JSON.exists():
        return {}
    with open(LEAKAGE_CONTROLLED_CV_RESULTS_JSON) as fh:
        return json.load(fh)


def load_batchnorm_cv_results() -> dict:
    if not LEAKAGE_CONTROLLED_CV_BATCHNORM_RESULTS_JSON.exists():
        return {}
    with open(LEAKAGE_CONTROLLED_CV_BATCHNORM_RESULTS_JSON) as fh:
        return json.load(fh)
