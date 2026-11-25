from pathlib import Path
import csv

root = Path(__file__).resolve().parent
with (root/"binary_results/summary.csv").open() as stream:
    rows = list(csv.DictReader(stream))
methods = {"uncalibrated":"Uncorrected", "manuscript_mom_raw":"Ordinary", "sign_mom":"Signed"}
lines = [
    r"\begin{center}\small",
    r"\begin{tabular}{llrrrrr}\toprule",
    r"Panel & Method & ATE & Bias & Coverage & Reject zero & Width \\\midrule",
]
for row in rows:
    if row["method"] not in methods:
        continue
    values = [float(row[key]) for key in ("true_ate","bias","coverage","rejection_rate","mean_width")]
    ate,bias,cover,reject,width = values
    lines.append(f"{row['scenario'].capitalize()} & {methods[row['method']]} & {ate:.3f} & {bias:+.3f} & {cover:.3f} & {reject:.3f} & {width:.3f}" + r" \\")
lines.extend([r"\bottomrule\end{tabular}", r"\end{center}"])
(root/"binary_stress_results.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
path = root/"main.tex"
text = path.read_text(encoding="utf-8").replace(r"\end{document}",r"\input{appendices.tex}"+"\n"+r"\end{document}")
path.write_text(text,encoding="utf-8")
path = root/"sections.tex"
text = path.read_text(encoding="utf-8")
needle = r"\subsection{What remains unverified}"
text = text.replace(needle,r"""In the binary AIPW extension (Appendix~\ref{app:binary-stress}), weak controls left signed bias of $+.075$ and a 66.2\% false-positive rate under the null. A dominant control yielded bias of $-.024$ with 99.6\% coverage and 2.34 times the uncorrected interval width. With a true effect of .105, the signed interval rejected zero in 96.0\% of matched-panel runs and 9.2\% of dominant-panel runs.

"""+needle)
path.write_text(text,encoding="utf-8")
