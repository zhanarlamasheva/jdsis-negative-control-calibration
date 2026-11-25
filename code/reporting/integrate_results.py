from pathlib import Path
import re

root = Path(__file__).resolve().parent
main = root/"main.tex"
sections = root/"sections.tex"
archive = (root/"appendices.tex").read_text(encoding="utf-8")
chunks = re.split(r"\\clearpage\s*(?:\\appendix\s*)?",archive)[1:]
chunks = [part.replace(r"\section{",r"\subsection{",1) for part in chunks]
assert len(chunks)==7
text=sections.read_text(encoding="utf-8")
text=text.replace(r"\section{Cross-domain evaluation in XES3G5M}","".join(chunks[:2])+"\n"+r"\section{Cross-domain evaluation in XES3G5M}")
text=text.replace(r"\section{Discussion}",chunks[5]+"\n"+r"\section{Simulation results}"+"\n"+"".join(chunks[2:4]))
text=text.replace(r"\subsection{What remains unverified}",chunks[6]+chunks[4]+"\n"+r"\section{Discussion}"+"\n"+r"\subsection{What remains unverified}")
text=text.replace(r"Appendix~\ref{app:binary-stress}",r"Section~\ref{sec:binary-stress}").replace("app:binary-stress","sec:binary-stress")
sections.write_text(text,encoding="utf-8")
main.write_text(main.read_text(encoding="utf-8").replace("\\input{appendices.tex}\n",""),encoding="utf-8")
