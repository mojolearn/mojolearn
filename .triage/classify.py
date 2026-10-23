import csv, re, sys
D='/Users/andrewhendel/mojolearn-wt/triage-old-lanes/.triage/'
def load(f): return {r[0]:r for r in csv.reader(open(D+f),delimiter='\t')}
meta=load('meta.tsv'); p2=load('presence2.tsv'); ver=load('verdicts.tsv'); ch=load('cherry.tsv'); ev=load('evidence.tsv')
CA='docs/identical/candidate-audit/candidate-audit-2026-10-04.md'
FENCE='docs/plans/host-route-fence-2026-10-09.md'
FIN='docs/plans/FINISH_LEFTOVERS_20261021.md'
rows=[]
for b,m in meta.items():
    date,ahead=m[1],m[2]; uniq=int(ch[b][1]); pres=float(p2[b][4]); vline=ver[b]
    nhits=int(vline[1]); first=vline[2]; vw=vline[3]; files=vline[4]
    short=b.split('/')[-1]
    v=r=e=''
    if b.startswith('finish/'):
        v,e,r='a',f'{FIN} table: (a)','retire'
    elif uniq==0:
        v,e,r='a',f'git cherry: all {m[2]} commits patch-equal on main','retire'
    elif b.startswith('archive/20261006/') and int(m[4])>700:
        v,e,r='a',f'{FENCE}:16-29 (same code reached main via bd4c36dbe); unique-line presence {pres:.2f}','retire'
    elif pres>=0.9:
        v,e,r='a',f'{pres:.0%} of unique added code lines present on main','retire'
    elif re.search(r'(^|-)measure(-|$)|stage-diag|ab-measure|apple-verify|-sweep$|merged-check|merged-owed-check|par-harness|pass4-measure|s17-measure|ivf-filter-fix-measure|neural-pass\d+-on-\d+',short):
        v,e,r='b',f'measurement/job harness ({CA}: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts','retire'
    elif nhits and ('TOMBSTONES' in first or 'EXPERIMENTS' in first or 'LEDGER' in first):
        v,e,r='b',f'{first} [{vw}]','retire'
    else:
        v,e,r='?',(first+' '+vw).strip(),''
    rows.append([b,date,ahead,str(uniq),v,e,r,f'{pres:.2f}',p2[b][6][:150]])
with open(D+'class.tsv','w') as f:
    for x in sorted(rows): f.write('\t'.join(x)+'\n')
from collections import Counter
print(Counter(x[4] for x in rows))
