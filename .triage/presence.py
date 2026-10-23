import subprocess, sys, os, re
R='/Users/andrewhendel/CascadeProjects/mojolearn'
def git(*a, inp=None):
    return subprocess.run(['git','-C',R,*a],capture_output=True,input=inp).stdout
EXT=re.compile(r'\.(mojo|py|sh|toml|md|yml|yaml|txt|tsv|c|h|cpp|metal)$')
SKIP=re.compile(r'^(bench/results/|experiments/results|.*\.json$)')
# global main line set
tree=git('ls-tree','-r','-l','origin/main').decode().splitlines()
objs=[]
for l in tree:
    meta,path=l.split('\t',1); parts=meta.split()
    if parts[1]!='blob' or parts[3]=='-' : continue
    if int(parts[3])>2_000_000 or not EXT.search(path) or SKIP.match(path): continue
    objs.append((parts[2],path))
inp=('\n'.join(o for o,_ in objs)+'\n').encode()
out=git('cat-file','--batch',inp=inp)
G=set(); pos=0
for o,p in objs:
    nl=out.index(b'\n',pos); hdr=out[pos:nl].split(); size=int(hdr[2])
    data=out[nl+1:nl+1+size]; pos=nl+1+size+1
    for ln in data.decode('utf-8','replace').splitlines():
        s=ln.strip()
        if len(s)>=25: G.add(s)
sys.stderr.write(f'main lines {len(G)} files {len(objs)}\n')
for b in sys.stdin.read().split():
    rb='origin/'+b
    mb=git('merge-base','origin/main',rb).decode().strip()
    d=git('diff','--unified=0','--no-color',mb,rb).decode('utf-8','replace')
    cur=None; tot=0; hit=0; codetot=0; codehit=0; miss={}
    for ln in d.splitlines():
        if ln.startswith('+++ '):
            cur=ln[6:] if ln.startswith('+++ b/') else None; continue
        if cur is None or SKIP.match(cur) or not EXT.search(cur): continue
        if ln.startswith('+') and not ln.startswith('+++'):
            s=ln[1:].strip()
            if len(s)<25: continue
            tot+=1; h= s in G; hit+=h
            if cur.endswith(('.mojo','.py','.sh')):
                codetot+=1; codehit+=h
                if not h: miss[cur]=miss.get(cur,0)+1
    top=sorted(miss.items(),key=lambda x:-x[1])[:4]
    print('\t'.join([b,str(tot),str(hit),str(codetot),str(codehit),';'.join(f'{f}:{n}' for f,n in top)]),flush=True)
