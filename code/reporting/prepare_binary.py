from pathlib import Path

here = Path(__file__).resolve().parent
base = here.parent / "effectiveness_validation" / "run_binary_validation.py"
text = base.read_text(encoding="utf-8")
start = text.index("    scenarios=")
end = text.index("    for s in scenarios:", start)
text = text[:start] + """    scenarios=[
        dict(name='matched',w=[1.,0.],ly=[.8,0.],loadings=[.8]*6),
        dict(name='weak',w=[1.,0.],ly=[.8,0.],loadings=[.15,.20,.25,.30,.35,.40]),
        dict(name='dominant',w=[1.,0.],ly=[.8,0.],loadings=[3.2,.8,.8,.8,.8,.8])]
""" + text[end:]
text = text.replace("ec=(u@np.array(s['lc'])+.4*x)[:,None]*SIGNS",
                    "ec=(u[:,0,None]*np.array(s['loadings'])+.4*x[:,None])*SIGNS")
text = text.replace("(ROOT/'BINARY_PROTOCOL_RU.md').read_bytes()", 
                    "(ROOT.parent/'effectiveness_validation'/'BINARY_PROTOCOL_RU.md').read_bytes()")
text = text.replace("import time", "import time\nfrom datetime import datetime, timedelta, timezone")
text = text.replace("def main():", """def log(*parts, **kwargs):
    stamp = datetime.now(timezone(timedelta(hours=5))).strftime('%H:%M')
    line = stamp + ' ' + ' '.join(map(str, parts))
    print(line, flush=True)
    with (ROOT/'binary_progress.log').open('a', encoding='utf-8') as stream:
        stream.write(line + '\\n')


def main():""")
text = text.replace("    print(", "    log(").replace("    log(line, flush=True)", "    print(line, flush=True)")
text = text.replace("            draws={}; trims=[]", """            cell_start=time.perf_counter()
            log(s['name'], f'alpha={alpha:g}, CPU, seed={args.seed}, 0/{args.replications}')
            draws={}; trims=[]""")
text = text.replace("            base=np.array", """                if (rep+1)%100==0:
                    rate=(rep+1)/(time.perf_counter()-cell_start)
                    log(s['name'], f'alpha={alpha:g}, {rep+1}/{args.replications}, {rate:.1f} rep/s, ETA {(args.replications-rep-1)/rate:.1f}s')
            base=np.array""")
(here/"run_binary_stress.py").write_text(text, encoding="utf-8")
