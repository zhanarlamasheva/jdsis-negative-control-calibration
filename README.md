# When can negative controls correct bias in educational data mining?

Code, saved outputs and simulations for the article
*When Can Negative Controls Correct Bias in Educational Data Mining? Conditions, Diagnostics and Limits of Sign-Aware Calibration*
(manuscript under review).

The repository contains no student records. OULAD and XES3G5M must be obtained from their original publishers.

## Data

| Dataset | Files used | Where to put them |
|---|---|---|
| OULAD | `studentInfo.csv`, `studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`, `studentVle.csv` | any folder, passed as `--data` |
| XES3G5M | `question_level/train_valid_sequences_quelevel.csv` (SHA-256 `52bb050f…66ec83`) | any path, passed as `--source` |

OULAD analyses use the 2013B and 2013J presentations (5,859 registrations).

## Environment

Python 3.10+ with `numpy`, `pandas`, `scikit-learn` and `xgboost` (see `requirements.txt`).
All real-data scripts use seed 20260626; the sign bootstrap uses 2,000 student-cluster draws.

## Reproducing the real-data results

```bash
python code/analysis/reanalysis_oulad.py --data /path/to/OULAD --out results/real_data
python code/analysis/reanalysis_xes.py --source /path/to/train_valid_sequences_quelevel.csv --out results/real_data
python code/reporting/table4_drop_one.py results/real_data/oulad_reanalysis.json
python code/reporting/figure2_real_panels.py results/real_data figure2_panels_CD.png
```

Each re-analysis script writes a JSON file with every reported number and a Markdown summary.
The JSON files used for the article are saved in `results/real_data/`. Do not use `--fast` for reported results.

## Where each number in the article comes from

| Article | Script | Output and fields |
|---|---|---|
| Table 2, OULAD rows; Section 4.1 | `reanalysis_oulad.py` | `oulad_reanalysis.json` → `designs.reference.{n_common_support, n_students, targets.*}` |
| Table 2, XES3G5M rows; Section 5.1 | `reanalysis_xes.py` | `xes_reanalysis.json` → `modes.fractional.designs.{A,B2}.{flow, full_panel.target}` |
| Table 3 | `reanalysis_oulad.py` | `designs.reference.sign_stability_common_support.{marginal, residualised}` |
| Section 4.2, stability rule (+0.140) | `reanalysis_oulad.py` | `designs.reference.stable_panel` |
| Table 4; Section 4.2 pseudo-target intervals | `table4_drop_one.py` | computed from `designs.reference.estimates` |
| Section 4.3 thresholds | `reanalysis_oulad.py` | `designs.reference.targets.not_withdrawn.{corrected, mom_ci}` |
| Section 4.4 text | `reanalysis_oulad.py` | `coding_audit`, `landmark_flow`, `designs.*.sign_stability_common_support.orientation_changes` |
| Table 5 | `reanalysis_oulad.py` | `designs.{reference, landmark, landmark_strict}` |
| Table 6 | both re-analysis scripts | OULAD: `designs.reference.loco`; XES3G5M: `modes.fractional.designs.*.full_panel.loco` |
| Table 7; Section 5.3 | `run_xes3g5m_sensitivity.py` | console and saved JSON |
| Table 8; Section 5.4 | `reanalysis_xes.py` | `modes.fixed` |
| Section 5.2 cleaning counts | `reanalysis_xes.py` | `provenance` |
| Table 9 (binary simulation) | `code/simulations/run_binary_validation.py` | written to `binary_results/` next to the script; not saved in this repository |
| Table 10 (Gaussian stress test) | `code/simulations/run_experiments.py` | console |
| Table 11 (binary AIPW stress test) | `code/simulations/run_binary_stress.py` | `results/binary_results/` (seed 20261031) |
| Control-only factor scenario (Section 6) | `code/simulations/run_control_only_factor.py` | console |
| Tables 12–13 (two-factor OULAD experiment) | `code/simulations/run_oulad_two_factors.py` | `results/oulad_two_factors/` |
| Figure 2, panels C–D | `code/reporting/figure2_real_panels.py` | `results/figure/figure2_audit_summary.png`, from both re-analysis JSON files |
| Other figures | `code/reporting/` | `results/figure/` |

## Repository structure

- `code/analysis/` — real-data analyses: re-analysis scripts, shared estimator (`nc_common.py`), XES3G5M sensitivity analysis.
- `code/simulations/` — known-truth, stress-test and semi-synthetic experiments.
- `code/reporting/` — table and figure generation.
- `code/data/`, `code/models/` — loaders and models from the larger project this work grew out of; they are not needed to reproduce the article.
- `results/real_data/` — saved re-analysis outputs used in the article.
- `results/binary_results/`, `results/oulad_two_factors/` — saved simulation outputs.
- `results/figure/` — figures.

## Known limitations

- The re-analysis scripts are the sole source of OULAD and XES3G5M numbers in the article. An earlier pipeline built on processed inputs gave slightly different values (for example, 5,629 instead of 5,563 OULAD registrations on common support); those values are no longer reported.
- Several scripts in `code/reporting/` and `code/analysis/` (for example, `integrate_results.py` and `run_real_data_audit.py`) belong to that earlier pipeline and expect intermediate files that are not included.
- Real-data intervals are nominal; a full-pipeline bootstrap is available through `reanalysis_oulad.py --pipeline-boot N` but was not run for the article.
