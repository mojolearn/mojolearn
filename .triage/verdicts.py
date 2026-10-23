import subprocess, sys, re
R='/Users/andrewhendel/CascadeProjects/mojolearn'
def git(*a): return subprocess.run(['git','-C',R,*a],capture_output=True).stdout.decode('utf-8','replace')
V=re.compile(r'\b(KEPT|KEEP|DEFAULT|DROPPED[-A-Za-z]*|HOLD[-A-Za-z]*|HELD|SUPERSEDED|PARKED|PORTED|ported|merged|MERGED|LOSER|loser|RECORD|PASS|FAIL|READY-AB|WIN|rejected|retired|abandoned|landed)\b')
for b in sys.stdin.read().split():
    short=b.split('/')[-1]
    pat=r'(lane|fix|finish|board|archive)/[a-z0-9/-]*\b'+re.escape(short)+r'\b' if len(short)<6 else re.escape(short)+r'([^a-z0-9-]|$)'
    out=git('grep','-n','-E','-e',pat,'origin/main','--','docs','experiments/removed','AGENTS.md')
    hits=[]
    for l in out.splitlines():
        _,path,ln,txt=l.split(':',3)
        # exact short match (avoid prefix of longer names like -r1)
        if re.search(re.escape(short)+r'-[a-z0-9]',txt) and not re.search(re.escape(short)+r'([^a-z0-9-]|$)',txt): continue
        hits.append((path,ln,txt))
    files=sorted({h[0] for h in hits})
    vs={}
    for p,ln,t in hits:
        for m in V.findall(t): vs[m]=vs.get(m,0)+1
    pri=[h for h in hits if 'EXPERIMENTS' in h[0] or 'TOMBSTONES' in h[0] or 'LEDGER' in h[0]] or hits
    first=f'{pri[0][0]}:{pri[0][1]}' if pri else ''
    snippet=re.sub(r'\s+',' ',pri[0][2])[:220] if pri else ''
    print('\t'.join([b,str(len(hits)),first,' '.join(f'{k}{v}' for k,v in sorted(vs.items(),key=lambda x:-x[1])[:6]),';'.join(files[:6]),snippet]),flush=True)
