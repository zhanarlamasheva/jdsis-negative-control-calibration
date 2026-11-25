"""Panel-level uncertainty for the OULAD negative-control calibration.

Addresses two reviewer asks that the point estimates alone do not answer:

  (1) Bootstrap standard errors on the empirical-null parameters mu and tau
      themselves. With only K = 6 controls, the null is estimated with real
      uncertainty; a cluster bootstrap over the control panel quantifies it.

  (2) Leave-one-control-out (LOO) stability. Dropping each control in turn and
      recomputing the calibrated ATE shows whether the conclusion (naive ->
      false significant, sign-aware -> honest null) hinges on any single
      control or is a property of the whole panel.

Inputs are the six OULAD control estimates and the target effect, taken from
the frozen pipeline outputs (results/oulad/uplift_evalue_robustness.json and
uplift_metrics.json). No model refit; this is panel-level resampling only.

CPU, seconds. Output: results/panel_bootstrap_loo.json + printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "panel_bootstrap_loo.json"
PRIMARY = Path(__file__).parent / "results" / "oulad_primary_strict.json"

if not PRIMARY.exists():
    raise FileNotFoundError("Run run_oulad_strict.py before panel_bootstrap_loo.py")
_primary = json.loads(PRIMARY.read_text())
_controls = _primary["negative_controls"]["controls"]
NAMES = [row["outcome"] for row in _controls]
TAU_K = np.asarray([row["ate"] for row in _controls], dtype=float)
SE_K = np.asarray([row["se"] for row in _controls], dtype=float)
S_K = np.asarray([row["eng_sign"] for row in _controls], dtype=float)
OBS_ATE = float(_primary["uncalibrated"]["ate"])
OBS_SE = float(_primary["uncalibrated"]["se"])
Z = 1.959963985


def mom_null(taus: np.ndarray, ses: np.ndarray) -> tuple[float, float]:
    """Schuemie method-of-moments empirical null (mu, tau_resid)."""
    mu = float(taus.mean())
    tau2 = max(0.0, float(taus.var(ddof=1) - np.mean(ses**2)))
    return mu, np.sqrt(tau2)


def calibrate(taus: np.ndarray, ses: np.ndarray, oriented: bool,
              signs: np.ndarray = S_K) -> dict:
    t = (signs * taus) if oriented else taus
    mu, tau = mom_null(t, ses)
    # sign-aware subtracts mu on the target's (positive) convention.
    est = OBS_ATE - mu
    se = np.sqrt(OBS_SE**2 + tau**2)
    lo, hi = est - Z * se, est + Z * se
    return {"mu": mu, "tau": tau, "ate": est, "lo": lo, "hi": hi,
            "sig": not (lo <= 0 <= hi)}


def bootstrap(B: int = 10000, seed: int = 20260626) -> dict:
    """Cluster bootstrap over the K controls -> SE of mu and tau."""
    rng = np.random.default_rng(seed)
    out = {}
    for oriented, key in [(False, "naive"), (True, "sign_aware")]:
        mus, taus_, ates = [], [], []
        for _ in range(B):
            idx = rng.integers(0, len(TAU_K), len(TAU_K))
            c = calibrate(TAU_K[idx], SE_K[idx], oriented, S_K[idx])
            mus.append(c["mu"]); taus_.append(c["tau"]); ates.append(c["ate"])
        out[key] = {
            "mu_se": float(np.std(mus, ddof=1)),
            "mu_ci": [float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))],
            "tau_se": float(np.std(taus_, ddof=1)),
            "ate_ci_boot": [float(np.percentile(ates, 2.5)), float(np.percentile(ates, 97.5))],
        }
    return out


def loo() -> dict:
    """Leave-one-control-out calibrated ATE for both schemes."""
    out = {}
    for oriented, key in [(False, "naive"), (True, "sign_aware")]:
        rows = []
        for k in range(len(TAU_K)):
            keep = [j for j in range(len(TAU_K)) if j != k]
            c = calibrate(TAU_K[keep], SE_K[keep], oriented, S_K[keep])
            rows.append({"dropped": NAMES[k], "ate": round(c["ate"], 4),
                         "sig": c["sig"]})
        ates = [r["ate"] for r in rows]
        out[key] = {"rows": rows, "ate_min": min(ates), "ate_max": max(ates),
                    "always_sig": all(r["sig"] for r in rows),
                    "never_sig": not any(r["sig"] for r in rows)}
    return out


if __name__ == "__main__":
    res = {
        "full": {"naive": calibrate(TAU_K, SE_K, False),
                 "sign_aware": calibrate(TAU_K, SE_K, True)},
        "bootstrap": bootstrap(),
        "loo": loo(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"json -> {OUT}\n")

    for k in ("naive", "sign_aware"):
        f = res["full"][k]; b = res["bootstrap"][k]; l = res["loo"][k]
        print(f"[{k}] mu={f['mu']:+.4f} (boot SE {b['mu_se']:.4f}, CI {b['mu_ci'][0]:+.4f}..{b['mu_ci'][1]:+.4f})"
              f"  tau={f['tau']:.4f} (boot SE {b['tau_se']:.4f})")
        print(f"        calibrated ATE={f['ate']:+.4f} CI[{f['lo']:+.4f},{f['hi']:+.4f}] sig={f['sig']}")
        print(f"        LOO ATE range [{l['ate_min']:+.4f},{l['ate_max']:+.4f}] "
              f"always_sig={l['always_sig']} never_sig={l['never_sig']}\n")
