# Re-analysis scripts for the reviewer report

All scripts use the same estimator as the manuscript (5-fold group cross-fit AIPW with
XGBoost, common support [0.02, 0.98], student-cluster variance, sign-aware equal pooling).
Requirements: Python 3.10+, numpy, pandas, scikit-learn, xgboost.

| Script | Reviewer comment | Input |
|---|---|---|
| `reanalysis_oulad.py` | 1 (timing), 4 (sign stability on common support), 5 (intervals) | raw OULAD CSVs |
| `reanalysis_xes.py` | 2 (fixed windows, reduced panel), 6 (sample flow) | `question_level/train_valid_sequences_quelevel.csv` |
| `run_control_only_factor.py` | 3 (control-only factor scenario) | none |

## Run

    python code/analysis/reanalysis_oulad.py --data /path/to/OULAD --out results_oulad
    python code/analysis/reanalysis_oulad.py --data /path/to/OULAD --out results_oulad_boot --pipeline-boot 200   # optional, slow
    python code/analysis/reanalysis_xes.py --source /path/to/train_valid_sequences_quelevel.csv --out results_xes
    python code/simulations/run_control_only_factor.py code/simulations 500     # path to run_binary_validation.py

Each script writes a JSON file with every number and a Markdown summary.

## Where the numbers go in the manuscript (yellow [__] placeholders)

* Table 2, raw and covariance intervals: `designs.reference.targets.*.raw_ci` and `cov_ci`
  (or `covariance_interval_diagnostic` in the existing `oulad_common_design.json` /
  `xes_raw_source_common_design.json`; XES raw intervals from `diagnostic.targets.*.ci95`).
* Table 3, last two columns and target rows: `designs.reference.sign_stability_common_support`
  (`marginal.*.sign_retention`, `residualised.*.sign_retention`).
* Section 4.4 text: `coding_audit`, `landmark_flow`, `orientation_changes`.
* Table 5: `designs.{reference,landmark,landmark_strict}` → `n`, `n_common_support`,
  `targets.not_withdrawn.{raw, raw_ci, corrected, mom_ci, cov_ci}`, `loco.{raw_rmse, corrected_rmse}`.
* Section 5.1 / 5.4 counts and Table 8: `provenance`, `modes.*.flow`, `modes.fixed.designs.*`.
* Section 7.2, Design B2 sentence: summarise `modes.fixed.designs.B2`.

Check first that the `reference` design reproduces Table 2 closely (cohort = 2013B and
2013J presentations; the manuscript used processed inputs, so small differences are expected).
If the landmark estimates differ materially, Sections 4.1, 7.2 and the abstract need updating.
