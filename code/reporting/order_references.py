from pathlib import Path
import re
root=Path(__file__).resolve().parent
def expanded(name):
    text=(root/name).read_text(encoding="utf-8")
    return re.sub(r"\\input\{([^}]+)\}",lambda m:expanded(m[1] if m[1].endswith(".tex") else m[1]+".tex"),text)
text=expanded("main.tex").split(r"\begin{thebibliography}")[0]
order=[]
for group in re.findall(r"\\cite(?:[a-zA-Z]*)?(?:\[[^\]]*\])?\{([^}]+)\}",text):
    for key in group.split(","):
        if key.strip() not in order:
            order.append(key.strip())
path=root/"references.tex"
refs=path.read_text(encoding="utf-8")
chunks=re.split(r"(?=\\bibitem)",refs)
entries={}
for chunk in chunks[1:]:
    key=re.search(r"\]\{([^}]+)\}",chunk)[1]
    entries[key]=chunk.split(r"\end{thebibliography}")[0].strip()
assert set(order)==set(entries),(set(order)-set(entries),set(entries)-set(order))
path.write_text(chunks[0]+"\n\n".join(entries[k] for k in order)+"\n"+r"\end{thebibliography}"+"\n",encoding="utf-8")
print(f"Ordered {len(order)} references by first citation")
