import subprocess, sys, re
R='/Users/andrewhendel/CascadeProjects/mojolearn'
def git(*a): return subprocess.run(['git','-C',R,*a],capture_output=True).stdout.decode('utf-8','replace')
GEN={'MOJOLEARN_BENCH_INSTALLED','MOJOLEARN_NUMERIC_MODE','MOJOLEARN_VENDOR','MOJOLEARN_COMPILE_JOBS'}
code_cache={}; doc_cache={}
mainsrc=None
def on_main(d):
    if d not in code_cache:
        code_cache[d]=bool(git('grep','-l','-F','-e',d,'origin/main','--','*.mojo','*.py','*.sh').strip())
    return code_cache[d]
TOK=re.compile(r'\b(KEPT|KEEP|DEFAULT|DROPPED[-a-z]*|HOLD[-a-z]*|HELD|NEUTRAL|SUPERSEDED|PARKED|OPEN|READY-AB|RECORD|PENDING|Pending|untried|LOSER|removed)\b')
def doc(d):
    k=d.replace('MOJOLEARN_','')
    if k not in doc_cache:
        out=git('grep','-h','-w','-e',k,'origin/main','--','docs/apple-fast/EXPERIMENTS.md','docs/TOMBSTONES.md','docs/plans','docs/apple-fast','grid_controls','core/six_lane_experiment_guards.mojo')
        toks={}
        for l in out.splitlines():
            for t in TOK.findall(l): toks[t]=toks.get(t,0)+1
        doc_cache[k]=toks
    return doc_cache[k]
for b in sys.stdin.read().split():
    rb='origin/'+b
    shas=[l[2:] for l in git('cherry','origin/main',rb).splitlines() if l.startswith('+ ')]
    defs=set()
    for s in shas:
        for ln in git('show','--format=','--unified=0',s,'--','*.mojo','*.py','*.sh').splitlines():
            if ln.startswith('+') and not ln.startswith('+++'):
                defs.update(re.findall(r'MOJOLEARN_[A-Z0-9_]{4,}[A-Z0-9]',ln))
    defs-=GEN
    res=[]
    for d in sorted(defs):
        m=on_main(d); t=doc(d)
        res.append(f"{d.replace('MOJOLEARN_','')}:{'M' if m else '-'}:{','.join(sorted(t,key=lambda x:-t[x])[:3])}")
    print(b+'\t'+' '.join(res)[:1500],flush=True)
