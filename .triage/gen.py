import csv, subprocess, collections
R='/Users/andrewhendel/CascadeProjects/mojolearn'; D='/Users/andrewhendel/mojolearn-wt/triage-old-lanes/.triage/'
def git(*a): return subprocess.run(['git','-C',R,*a],capture_output=True).stdout.decode('utf-8','replace')
fin=[r for r in csv.reader(open(D+'final.tsv'),delimiter='\t')]
def files(b):
    shas=[l[2:] for l in git('cherry','origin/main','origin/'+b).splitlines() if l.startswith('+ ')]
    if not shas:
        mb=git('merge-base','origin/main','origin/'+b).strip()
        fs=git('diff','--name-only',mb,'origin/'+b).split()
    else:
        fs=set()
        for s in shas: fs.update(git('show','--format=','--name-only',s).split())
        fs=sorted(fs)
    tops=collections.Counter('/'.join(f.split('/')[:2]) if '/' in f else f for f in fs)
    return len(fs), ', '.join(f'{k}' for k,_ in tops.most_common(3))
lines=[]
for b,d,ahead,uniq,v,e,rec in fin:
    n,t=files(b)
    e=e.replace('|','/')
    lines.append(f'| `{b}` | {d} | {ahead} ({uniq}) | {n}: {t} | {v} | {e} | {rec} |')
open(D+'table.md','w').write('\n'.join(lines)+'\n')
print(len(lines))
