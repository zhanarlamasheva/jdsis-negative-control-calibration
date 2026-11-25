import csv
from pathlib import Path

out = Path(__file__).resolve().parent / "oulad_two_factors"
with (out/"summary.csv").open() as stream:
    rows = list(csv.DictReader(stream))

methods = {"uncalibrated":"Raw", "manuscript_mom_raw":"Ordinary", "sign_mom":"Signed"}
panels = {"matched":"Matched", "unequal_proportional":"Proportional", "unequal_mixed":"Mixed"}
table = [
    r"\begin{center}\footnotesize",
    r"\begin{tabular}{llrrrrr}\toprule",
    r"Panel & Method & ATE & Bias & Coverage & Width & Reject zero \\\midrule",
]
for row in rows:
    if row["method"] in methods:
        label = panels[row["panel"]] if row["method"]=="uncalibrated" else ""
        truth,bias,coverage,width,reject = (
            float(row[key]) for key in ["truth","bias","coverage","width","reject_zero"]
        )
        table.append(
            f"{label} & {methods[row['method']]} & {truth:.3f} & {bias:+.3f} & "
            f"{coverage:.3f} & {width:.3f} & {reject:.3f}" + r" \\"
        )
table += [r"\bottomrule\end{tabular}", r"\end{center}",
          r"Rejecting zero measures false positives at ATE zero and power otherwise. "
          r"This benchmark tests the chosen confounding structures on OULAD covariates. "
          r"Empirical control comparability still requires a substantive argument."]
(out/"results.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
