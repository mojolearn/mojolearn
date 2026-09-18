import re,sys,collections,os
def ops(p):
    c=collections.Counter()
    for l in open(p):
        l=l.strip()
        if not l or l.startswith(('.',';','//')) or l.endswith(':'): continue
        op=l.split()[0]
        c[op]+=1
    return c
FP=re.compile(r'^v_(fma|fmac|mac|mul|add|sub|min|max|cmp_class|div|rcp|sqrt|cvt).*_f(16|32|64)|^v_pk_')
n_same=0
for f in sorted(os.listdir('gcn-main')):
    if not f.endswith('.amdgcn'): continue
    a,b=ops('gcn-main/'+f),ops('gcn-branch/'+f)
    fa={k:v for k,v in a.items() if FP.match(k)}; fb={k:v for k,v in b.items() if FP.match(k)}
    if a!=b:
        print(f, 'FP-op multiset equal' if fa==fb else 'FP-op multiset DIFFERS', {k:(a[k],b[k]) for k in set(a)|set(b) if a[k]!=b[k]})
    else: n_same+=1
print('kernels with identical opcode multisets:', n_same)
