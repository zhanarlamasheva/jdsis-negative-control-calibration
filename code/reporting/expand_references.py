from pathlib import Path
import re

root = Path(__file__).resolve().parent
original = (root.parent/"manuscript_updated/manuscript.tex").read_text(encoding="utf-8")
refs = (root/"references.tex").read_text(encoding="utf-8")
refs = re.sub(r"\\bibitem\[.*?\]\{liu2023\}.*?(?=\\end\{thebibliography\})","",refs,flags=re.S)
selection = {
  6:("vanderweele2017","VanderWeele and Ding",2017),
  11:("fazil2024","Fazil et al.",2024),
  17:("zhang2025","Zhang et al.",2025),
  18:("weidlich2022","Weidlich et al.",2022),
  20:("kennedy2023","Kennedy",2023),
  22:("abdelrahman2023","Abdelrahman et al.",2023),
  23:("yang2024","Yang et al.",2024),
  25:("cohausz2022","Cohausz",2022),
  27:("ramaswami2023","Ramaswami et al.",2023),
  30:("inoue2025","Inoue et al.",2025),
  31:("idowu2024","Idowu et al.",2024),
  32:("palanci2024","Palanci et al.",2024),
  40:("inoue2024","Inoue et al.",2024),
}
entries = []
for number,(key,author,year) in selection.items():
    line = next(line for line in original.splitlines() if line.startswith(r"\item ["+str(number)+"]"))
    body = re.sub(r"^\\item \[\d+\] ","",line).split(r"\pdfstartlink")[0].strip()
    doi = re.search(r"doi:([^\\]+)",line).group(1)
    if number==18:
        body=body.replace("183-195","183-199")
    if number==32:
        body=body.replace("14321-14352","22629-22650").replace("Yilmaz, R.","Yilmaz, R. M.")
    if number==30:
        body=body.replace("Ito, Y., Goto, A.,","Ito, Y., Goto, A.,").replace("Komukai, S., Ito", "Komukai, S., Ito")
    entries.append(r"\bibitem[\protect\citeauthoryear{"+author+"}{"+author+"}{"+str(year)+"}]{"+key+"}\n"+body+r" \url{https://doi.org/"+doi+"}.\n")
entries.append(r"""\bibitem[\protect\citeauthoryear{Tchetgen Tchetgen et al.}{Tchetgen Tchetgen et al.}{2024}]{proximal2024}
Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., and Miao, W. 2024. An introduction to proximal causal inference. \textit{Statistical Science} 39(3), 375--390. \url{https://doi.org/10.1214/23-STS911}.
""")
refs=refs.replace(r"\end{thebibliography}","\n".join(entries)+"\n"+r"\end{thebibliography}")
(root/"references.tex").write_text(refs,encoding="utf-8")
path=root/"sections.tex"
text=path.read_text(encoding="utf-8")
text=text.replace(r"\cite{liu2023}",r"\footnote{Liu et al. (2023), XES3G5M, NeurIPS Datasets and Benchmarks. Dataset publication: \url{https://doi.org/10.52202/075280-1429}.}")
text=text.replace(r"\section{Background and related work}",r"""Recent educational studies examine engagement-based prediction \cite{fazil2024}, dashboard interventions \cite{ramaswami2023} and algorithmic bias in progress monitoring \cite{idowu2024}. A review of distance-education learning analytics describes the breadth of applications \cite{palanci2024}, while knowledge-tracing research provides the context for the XES analysis \cite{abdelrahman2023}. These settings motivate explicit treatment, outcome and timing definitions.

\section{Background and related work}""")
text=text.replace(r"\subsection{Negative controls: detection versus correction}",r"""\subsection{Negative controls: detection versus correction}
The recent scoping review by \cite{yang2024} distinguishes bias detection, calibration and bias correction, each with its own assumptions. Proximal inference formalizes relationships between imperfect confounder proxies through identifying restrictions \cite{proximal2024}. Negative-control-calibrated difference-in-differences additionally uses temporal structure \cite{zhang2025}. The present pooling diagnostics concern the transport of residual bias across outcomes.

""")
text=text.replace(r"\subsection{Estimation}",r"""\subsection{Estimation}
DAG-based discussions in learning analytics emphasize confounding, overcontrol and collider bias \cite{weidlich2022}, and causal explanation frameworks make intervention assumptions explicit in student-success analysis \cite{cohausz2022}. Confounder selection follows substantive timing and causal relationships \cite{inoue2025}. Doubly robust heterogeneous-effect estimation addresses nuisance estimation under identification conditions \cite{kennedy2023}. A recent review of heterogeneous-effect methods in randomized trials further motivates assessing effect heterogeneity against a clearly specified design \cite{inoue2024}.

""")
text=text.replace("The E-value for the uncalibrated contrast",r"The E-value \cite{vanderweele2017} for the uncalibrated contrast")
# Trim redundant archived benchmark display to accommodate the expanded bibliography.
a=text.index(r"\begin{center}\small",text.index(r"\subsection{Archived known-truth benchmarks}"))
b=text.index(r"\subsection{What the evidence supports}",a)
text=text[:a]+text[b:]
# Remove the redundant E-value illustration, retaining the estimate and interpretation.
a=text.index(r"\begin{center}",text.index(r"\subsection{OULAD sensitivity to hidden confounding}"))
b=text.index(r"\end{center}",a)+len(r"\end{center}")
text=text[:a]+text[b:]
path.write_text(text,encoding="utf-8")
# Ensure actual clickable DOI links, including the dataset footnote.
main=root/"main.tex"
maintext=main.read_text(encoding="utf-8")
maintext=maintext.replace(r"\usepackage{amsmath,amssymb,tabularx,longtable,graphicx}",r"\usepackage{amsmath,amssymb,tabularx,longtable,graphicx}"+"\n"+r"\usepackage[hidelinks]{hyperref}")
main.write_text(maintext,encoding="utf-8")
keys=re.findall(r"\\bibitem\[.*?\]\{([^}]+)\}",refs)
cited=set()
for cite in re.findall(r"\\cite\{([^}]+)\}",text):
    cited.update(cite.split(","))
assert len(keys)==23, len(keys)
assert set(keys)<=cited, set(keys)-cited
years=list(map(int,re.findall(r"\\citeauthoryear\{[^}]*\}\{[^}]*\}\{(\d{4})\}",refs)))
print("Journal articles:",len(keys),"Recent:",sum(y>=2022 for y in years),"Fraction:",sum(y>=2022 for y in years)/len(years))
