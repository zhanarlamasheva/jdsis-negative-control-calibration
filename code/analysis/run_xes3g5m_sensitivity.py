"""Repeated-split and treatment-threshold sensitivity analysis for XES3G5M A/B.

The script writes a compact JSON summary. It never overwrites the prespecified
primary result file. Each run retains a user-disjoint 80/20 holdout, derives
all thresholds from training users, and fits CATE before observing held-out
outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_xes3g5m_designs import ArticleXESDesigns, ROOT


DEFAULT_SEEDS = (42, 123, 456, 789, 20260626)
DEFAULT_QUANTILES = (0.40, 0.50, 0.60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "xes3g5m_sensitivity.json")
    parser.add_argument("--qini-permutations", type=int, default=1000)
    args = parser.parse_args()

    runs = []
    for seed in DEFAULT_SEEDS:
        analysis = ArticleXESDesigns(args.data_dir, seed=seed)
        cohort = analysis.select_cohort(analysis.load_full())
        train_frame, test_frame = analysis.user_split(cohort)
        train_rows = analysis.make_rows(train_frame)
        test_rows = analysis.make_rows(test_frame)
        for quantile in DEFAULT_QUANTILES:
            thresholds = analysis.thresholds(train_rows, quantile)
            for design in ("a", "b"):
                result = analysis.run_design(
                    train_rows, test_rows, thresholds, design, args.qini_permutations
                )
                runs.append({
                    "seed": seed,
                    "threshold_quantile": quantile,
                    "design": design,
                    "uncalibrated_ate": result["uncalibrated"]["ate"],
                    "uncalibrated_ci95": result["uncalibrated"]["ci95"],
                    "sign_aware_ate": result["sign_aware"]["ate"],
                    "sign_aware_ci95": result["sign_aware"]["ci95"],
                    "qini": result["qini_empirical_null"]["observed"],
                    "qini_empirical_null_p": result["qini_empirical_null"]["p_value"],
                    "qini_permutation_p": result["qini_permutation_test"]["p_value"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": {"seeds": DEFAULT_SEEDS, "threshold_quantiles": DEFAULT_QUANTILES,
                     "qini_permutations": args.qini_permutations,
                     "holdout": "user-disjoint 80/20 for every run"},
        "runs": runs,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
