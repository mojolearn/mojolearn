import subprocess, sys, re
R='/Users/andrewhendel/CascadeProjects/mojolearn'
def git(*a): return subprocess.run(['git','-C',R,*a],capture_output=True).stdout.decode('utf-8','replace')
def grep_docs(pat, paths=('docs','experiments/removed','core/six_lane_experiment_guards.mojo','grid_controls')):
    out=git('grep','-l','-F','-e',pat,'origin/main','--',*paths)
    return [l.split(':',1)[1] for l in out.splitlines()]
DEF=re.compile(r'MOJOLEARN_[A-Z0-9_]{4,}')
for b in sys.stdin.read().split():
    rb='origin/'+b
    short=b.split('/')[-1]
    mb=git('merge-base','origin/main',rb).strip()
    # only use own commits' diff for defines (ahead commits)
    d=git('diff','--unified=0',mb,rb,'--','*.mojo','*.py','*.sh')
    added=set(); 
    for ln in d.splitlines():
        if ln.startswith('+') and not ln.startswith('+++'):
            added.update(DEF.findall(ln))
    # restrict to defines that look like experiments
    exp=sorted(x for x in added if re.search(r'_FAST_|_IDENTICAL_|_EXP|_AB_|_OFF$|_ON$',x))[:40]
    st=[]
    for x in exp:
        inmain=bool(git('grep','-l','-F','-e',x,'origin/main','--','*.mojo','*.py','*.sh').strip())
        docs=grep_docs(x,('docs/apple-fast/EXPERIMENTS.md','docs/TOMBSTONES.md'))
        tag=('M' if inmain else '-')+('E' if any('EXPERIMENTS' in p for p in docs) else '')+('T' if any('TOMBSTONES' in p for p in docs) else '')
        st.append(f'{x}={tag}')
    nd=grep_docs(short) if len(short)>=6 else []
    print('\t'.join([b,str(len(exp)),' '.join(st)[:900],';'.join(nd[:8])]),flush=True)
