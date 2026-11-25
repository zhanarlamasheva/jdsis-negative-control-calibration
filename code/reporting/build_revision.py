import csv
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

root = Path(__file__).resolve().parent
source = root.parent / "editorial_revision"
started = perf_counter()
log_path = root / "simulation.log"

def log(message):
    stamp = datetime.now(timezone(timedelta(hours=5))).strftime("%H:%M")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")

seed = 20261116
n = 30000
rng = np.random.default_rng(seed)
log(f"CPU Gaussian simulation started: seed={seed}, repetitions per scenario={n}")
rows = []
truth_sign = np.array([1, -1, 1, -1, 1, -1])
scenarios = [
    ("Matched", [1]*6, 0),
    ("Unequal, mean one", [.25,.5,.75,1.25,1.5,1.75], 0),
    ("Weak controls", [.2,.3,.4,.5,.6,.7], 0),
    ("Sign error", [1]*6, .25),
    ("Dominant control", [6,1,1,1,1,1], 0),
    ("Dominant + sign error", [6,1,1,1,1,1], .25),
]
for index, (scenario, loadings, p) in enumerate(scenarios, 1):
    target = .06 + rng.normal(0, .025, n)
    controls = .06*np.array(loadings)*truth_sign + rng.normal(0, .02, (n,6))
    signs = truth_sign * np.where(rng.random((n,6)) < p, -1, 1)
    for name, panel in [("Raw", None), ("Unaligned", controls), ("Aligned", controls*signs)]:
        estimate = target if panel is None else target - panel.mean(axis=1)
        spread = 0 if panel is None else np.maximum(0, panel.var(axis=1, ddof=1)-.02**2)
        cover = np.mean(abs(estimate) <= 1.96*np.sqrt(.025**2+spread))
        rows.append((scenario, name, float(estimate.mean()), float(cover), float(1-cover)))
    elapsed = perf_counter()-started
    rate = index*n/elapsed
    log(f"{index}/{len(scenarios)} scenarios, {rate:.0f} repetitions/s, ETA {(len(scenarios)-index)*n/rate:.2f}s")
with (root/"simulation.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["scenario","method","bias","coverage","false_positive_rate"])
    writer.writerows(rows)

def table(caption, label, spec, header, body):
    return (r"\begin{table}[htbp]\centering\small"+"\n"+
            rf"\caption{{{caption}}}\label{{{label}}}"+"\n"+
            rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}\toprule"+"\n"+
            header+r"\\\midrule"+"\n"+
            "\n".join(line+r" \\" for line in body)+"\n"+
            r"\bottomrule\end{tabularx}\end{table}"+"\n")

sim = table("Gaussian stress test at a true effect of zero. Each method column gives bias, 95\% coverage, and false-positive rate, respectively.",
            "tab:gaussian-stress", "Xlll", "Scenario & Raw & Unaligned & Aligned",
            [scenario+" & "+" & ".join(f"{b:+.3f}, {c:.1%}, {f:.1%}".replace("%",r"\%") for _,_,b,c,f in rows[i:i+3])
             for i,(scenario,_,_,_,_) in enumerate(rows) if i%3==0])

data = json.loads((source/"evidence/oulad_common_design.json").read_text())["diagnostic"]
target = data["targets"]["not_withdrawn"]
controls = list(data["controls"].values())
values = np.array([c["estimate"] for c in controls])
se = np.array([c["se"] for c in controls])
signs = np.array(target["orientations"])
oriented = values*signs
def omitted(keep):
    estimate = target["uncalibrated"]-oriented[keep].mean()
    spread = max(0, oriented[keep].var(ddof=1)-(se[keep]**2).mean())
    margin = 1.96*np.sqrt(target["se"]**2+spread)
    return estimate, estimate-margin, estimate+margin
assert np.allclose(omitted(np.ones(6,dtype=bool)),
                   [target["calibrated"],*target["manuscript_interval"]])
names = ["Disability","Male indicator","Age 35+","Higher education","IMD 50+","Previous attempts"]
loo = table("OULAD common design, non-withdrawal. The proxy correlation is with pre-deadline clicks. The final columns recompute the signed correction and nominal interval after dropping the row's control.",
            "tab:target-loo","Xrrrrl",r"Control & Contrast & Proxy $r$ & Sign & Adjusted & Nominal interval",
            [f"{name} & {values[j]:+.3f} & {controls[j]['proxy_correlation']:+.3f} & {signs[j]:+d} & {a:+.3f} & [{lo:+.3f}, {hi:+.3f}]"
             for j,name in enumerate(names) for a,lo,hi in [omitted(np.arange(6)!=j)]])

for pattern in ("*.tex","jedm.cls","audit_summary.png"):
    for path in (source/"manuscript_revised").glob(pattern):
        shutil.copy2(path, root/path.name)
path = root/"main.tex"
content = path.read_text(encoding="utf-8")
content = "\n".join(line for line in content.splitlines() if "Working revision." not in line)+"\n"
path.write_text(content,encoding="utf-8")
path = root/"sections.tex"
content = path.read_text(encoding="utf-8")
position = r"\subsection{Exact conditions for improvement}"
content = content.replace(position,r"""For nonzero target bias $b_Y$, write the oriented population control bias as
$s_k b_k=\lambda_k b_Y$. Positive $\lambda_k$ give directional agreement. Equal pooling recovers the magnitude of $b_Y$ when $K^{-1}\sum_k\lambda_k=1$; departures leave residual bias $b_Y(1-K^{-1}\sum_k\lambda_k)$. When $b_Y=0$, exact mean correction instead requires the oriented biases to average zero. The measured proxy correlations motivate orientation but cannot identify $\lambda_k$. In OULAD, engagement and opportunity to study could link submission timing and subsequent retention to these controls via support needs, social constraints, prior preparation, deprivation or study history. These pathways can differ by control. The positive target proxy association, together with the individual control proxy associations in Table~\ref{tab:target-loo}, determines the relative signs.

"""+position)
content = content.replace(r"\input{orientation_results.tex}",r"""\input{orientation_results.tex}
\input{target_loo.tex}
This deletion analysis keeps the target fit and other signs fixed. In the archived design, omitting previous attempts changed the signed estimate from $+0.040$ to $+0.050$ and the nominal interval then excluded zero. The common-design audit changes the adjustment set and estimates a different adjusted contrast; its deletion results illustrate how panel sensitivity depends on the design as well as on the chosen controls.
""")
content = content.replace(r"\input{simulation_results.tex}",r"""\input{simulation_results.tex}

\subsection{Stress test of magnitude and orientation}
We extended the known-truth evaluation with a Gaussian experiment at the estimate level. The true effect is zero and the target estimator has population bias $0.06$ and standard error $0.025$. Six independent control estimates have standard errors $0.020$, alternating true orientations, and population magnitudes $0.06\lambda_k$. Sampling errors are mutually independent in this experiment. We evaluated $\lambda_k=1$, unequal loadings averaging one $(.25,.50,.75,1.25,1.50,1.75)$, weak controls $(.20,.30,.40,.50,.60,.70)$, and a single dominant control $(6,1,1,1,1,1)$. A separate perturbation reverses each orientation independently with probability $0.25$, with and without the dominant control. We used 30,000 draws per scenario and compared the uncorrected Wald interval with the same nominal MoM pooling interval for each correction.

\input{stress_results.tex}
Unequal magnitudes can preserve the mean correction when they average to the target bias. Weak controls leave residual bias despite correct orientation. A dominant control can overcorrect the point estimate while inflating the pooled interval, so coverage alone can conceal poor correction. At a true zero effect, empirical coverage and false-positive rejection are complementary. This experiment isolates the pooling step under fixed target and control standard errors. The binary simulations above additionally exercise outcome modelling and cross-fitting.
""")
path.write_text(content,encoding="utf-8")
(root/"stress_results.tex").write_text(sim,encoding="utf-8")
(root/"target_loo.tex").write_text(loo,encoding="utf-8")
log("Simulation, table generation and full-panel formula check complete")
