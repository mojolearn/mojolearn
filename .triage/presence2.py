import subprocess, sys, re, pickle, os
R='/Users/andrewhendel/CascadeProjects/mojolearn'
def git(*a, inp=None):
    return subprocess.run(['git','-C',R,*a],capture_output=True,input=inp).stdout
EXT=re.compile(r'\.(mojo|py|sh|toml|md|yml|yaml|txt|tsv|c|h|cpp|metal)$')
SKIP=re.compile(r'^(bench/results/|experiments/results|.*\.json$)')
tree=git('ls-tree','-r','-l','origin/main').decode().splitlines()
objs=[]
for l in tree:
    meta,path=l.split('\t',1); parts=meta.split()
    if parts[1]!='blob' or parts[3]=='-' : continue
    if int(parts[3])>2_000_000 or not EXT.search(path) or SKIP.match(path): continue
    objs.append((parts[2],path))
out=git('cat-file','--batch',inp=('\n'.join(o for o,_ in objs)+'\n').encode())
G=set(); pos=0
for o,p in objs:
    nl=out.index(b'\n',pos); size=int(out[pos:nl].split()[2])
    for ln in out[nl+1:nl+1+size].decode('utf-8','replace').splitlines():
        s=ln.strip()
        if len(s)>=25: G.add(s)
    pos=nl+1+size+1
for b in sys.stdin.read().split():
    rb='origin/'+b
    shas=[l[2:] for l in git('cherry','origin/main',rb).decode().splitlines() if l.startswith('+ ')]
    tot=hit=0; miss={}; files=set()
    for s in shas:
        d=git('show','--format=','--unified=0','--no-color',s).decode('utf-8','replace')
        cur=None
        for ln in d.splitlines():
            if ln.startswith('+++ '):
                cur=ln[6:] if ln.startswith('+++ b/') else None
                if cur: files.add(cur)
                continue
            if cur is None or SKIP.match(cur) or not EXT.search(cur) or not cur.endswith(('.mojo','.py','.sh')): continue
            if ln.startswith('+') and not ln.startswith('+++'):
                t=ln[1:].strip()
                if len(t)<25: continue
                tot+=1; h=t in G; hit+=h
                if not h: miss[cur]=miss.get(cur,0)+1
    top=sorted(miss.items(),key=lambda x:-x[1])[:5]
    print('\t'.join([b,str(len(shas)),str(tot),str(hit),f'{(hit/tot if tot else -1):.2f}',str(len(files)),';'.join(f'{f}:{n}' for f,n in top)]),flush=True)
