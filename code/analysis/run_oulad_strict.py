"""Strict, raw-data OULAD reproduction package.

Runs the temporal 2013/2014 uplift estimator, the six-control calibration, and
multiple-treatment sensitivity scripts. Writes a single article-facing JSON plus
checksums and a manifest. CATE fitting is train-only in the committed foundation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EDNET = ROOT.parent / "education_models"
OUT = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    commands = [
        [sys.executable, "-m", "models.uplift_oulad"],
        [sys.executable, "-m", "scripts.uplift_evalue_robustness"],
        [sys.executable, "-m", "scripts.uplift_pos_multiT"],
    ]
    for command in commands:
        subprocess.run(command, cwd=EDNET, check=True)

    metrics = json.loads((EDNET / "results" / "oulad" / "uplift_metrics.json").read_text())
    controls = json.loads((EDNET / "results" / "oulad" / "uplift_evalue_robustness.json").read_text())
    multit = json.loads((EDNET / "results" / "oulad" / "uplift_pos_multiT.json").read_text())
    primary = {"dataset": "OULAD", "estimand": "ATE of on-time first-assessment submission on retention",
               "temporal_split": "2013 train; 2014 holdout for Qini/AUUC",
               "uncalibrated": metrics, "negative_control_calibration": controls["nc_calibration"],
               "negative_controls": controls["negative_control_panel"],
               "qini_empirical_null": controls["qini_calibration"], "multiple_treatment": multit}
    OUT.mkdir(exist_ok=True)
    (OUT / "oulad_primary_strict.json").write_text(json.dumps(primary, indent=2))

    inputs = [EDNET / "data" / "oulad" / "processed" / name for name in ("daily_vle.parquet", "student_target.parquet")]
    checksums = {str(path.relative_to(EDNET)): sha256(path) for path in inputs}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}
    manifest = {"seed": 42, "k_folds": 5, "foundation": "education_models/models/uplift_oulad.py",
                "commands": [" ".join(c) for c in commands], "python": sys.version,
                "platform": platform.platform(), "packages": versions, "input_sha256": checksums,
                "outputs": ["oulad_primary_strict.json", "../education_models/results/oulad/uplift_metrics.json",
                            "../education_models/results/oulad/uplift_evalue_robustness.json",
                            "../education_models/results/oulad/uplift_pos_multiT.json"]}
    (OUT / "oulad_manifest_strict.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT / 'oulad_primary_strict.json'}")
    print(f"wrote {OUT / 'oulad_manifest_strict.json'}")


if __name__ == "__main__":
    main()
