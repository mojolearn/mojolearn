# A float32 model of x_decomp/jacobi_par.mojo's arithmetic (tiny n): index math, signs, convergence.
import numpy as np
f = np.float32
def first(r,b,m): return r if b==0 else (r+b)%(m-1)
def second(r,b,m): return m-1 if b==0 else (r+m-1-b)%(m-1)
def cs_of(app,aqq,apq):
    if apq == 0: return f(1),f(0)
    theta = f((aqq-app)/(f(2)*apq))
    root = f(np.sqrt(f(theta*theta+f(1))))
    t = f(1)/f(theta+root) if theta>=0 else f(-1)/f(root-theta)
    c = f(1)/f(np.sqrt(f(t*t+f(1))))
    return c, f(t*c)
def check_pairs(m):
    seen=set()
    for r in range(m-1):
        used=set()
        for b in range(m//2):
            p,q=sorted((first(r,b,m),second(r,b,m)))
            assert p!=q and p not in used and q not in used
            used|={p,q}; seen.add((p,q))
    assert len(seen)==m*(m-1)//2, (m,len(seen))
for m in (2,4,6,8,20,50,256): check_pairs(m)
print("pairs ok")
def eigh_par(A, tol=1e-7, sweeps=30):
    n=A.shape[0]; m=n+(n%2); h=m//2
    a=A.astype(f).copy(); v=np.eye(n,dtype=f)
    for sw in range(sweeps+1):
        d=np.diag(a).astype(np.float64); off=(a.astype(np.float64)**2).sum()-(d**2).sum()
        if off <= tol*tol*(off+(d**2).sum()): return np.diag(a).copy(), v, sw
        if sw==sweeps: return None
        for r in range(m-1):
            pr=[]; cs=[]
            for b in range(h):
                p,q=sorted((first(r,b,m),second(r,b,m))); pr.append((p,q))
                cs.append(cs_of(a[p,p],a[q,q],a[p,q]) if q<n else (f(1),f(0)))
            na=a.copy()
            for i in range(h):
                pi,qi=pr[i]; ci,si=cs[i]
                for j in range(i,h):
                    pj,qj=pr[j]; cj,sj=cs[j]
                    if i==j:
                        if qi<n:
                            tt=f(si/ci)
                            na[pi,pi]=a[pi,pi]-tt*a[pi,qi]; na[qi,qi]=a[qi,qi]+tt*a[pi,qi]; na[pi,qi]=0; na[qi,pi]=0
                        continue
                    vi=qi<n; vj=qj<n
                    b00=a[pi,pj]; b01=a[pi,qj] if vj else f(0); b10=a[qi,pj] if vi else f(0); b11=a[qi,qj] if (vi and vj) else f(0)
                    t00=cj*b00-sj*b01; t01=sj*b00+cj*b01; t10=cj*b10-sj*b11; t11=sj*b10+cj*b11
                    n00=ci*t00-si*t10; n01=ci*t01-si*t11; n10=si*t00+ci*t10; n11=si*t01+ci*t11
                    na[pi,pj]=n00; na[pj,pi]=n00
                    if vj: na[pi,qj]=n01; na[qj,pi]=n01
                    if vi: na[qi,pj]=n10; na[pj,qi]=n10
                    if vi and vj: na[qi,qj]=n11; na[qj,qi]=n11
            a=na
            for j in range(h):
                pj,qj=pr[j]; cj,sj=cs[j]
                if qj<n:
                    vp=v[:,pj].copy(); vq=v[:,qj].copy()
                    v[:,pj]=cj*vp-sj*vq; v[:,qj]=sj*vp+cj*vq
def svd_par(R, tol=9.5367431640625e-07, sweeps=60):
    n=R.shape[0]; m=n+(n%2); h=m//2
    rt=R.T.astype(f).copy(); vt=np.eye(n,dtype=f)
    for sw in range(sweeps):
        rots=0
        for r in range(m-1):
            for b in range(h):
                p,q=sorted((first(r,b,m),second(r,b,m)))
                if q>=n: continue
                app=f(rt[p]@rt[p]); aqq=f(rt[q]@rt[q]); apq=f(rt[p]@rt[q])
                if abs(apq) > f(tol)*(np.sqrt(app)*np.sqrt(aqq)):
                    c,s=cs_of(app,aqq,apq); rots+=1
                    rp=rt[p].copy(); rq=rt[q].copy(); rt[p]=c*rp-s*rq; rt[q]=s*rp+c*rq
                    vp=vt[p].copy(); vq=vt[q].copy(); vt[p]=c*vp-s*vq; vt[q]=s*vp+c*vq
        if rots==0:
            return np.sqrt((rt.astype(f)**2).sum(1)), vt.T.copy(), sw+1
    return None
g=np.random.default_rng(3)
for n in (7,40,65):
    B=g.standard_normal((n,n)).astype(f); S=B+B.T
    w,v,sw=eigh_par(S)
    ref=np.linalg.eigvalsh(S.astype(np.float64))
    o=np.argsort(w); w=w[o]; v=v[:,o]
    S64=S.astype(np.float64); v64=v.astype(np.float64)
    print("eigh",n,"sweeps",sw,"w",np.max(np.abs(w-ref))/np.max(np.abs(ref)),"res",np.linalg.norm(S64@v64-v64*w)/np.linalg.norm(S64),"orth",np.linalg.norm(v64.T@v64-np.eye(n))/np.sqrt(n))
    C=g.standard_normal((n,max(n//2,1))).astype(f); G=C@C.T; G=(G+G.T)*f(0.5)
    w,v,sw=eigh_par(G); ref=np.linalg.eigvalsh(G.astype(np.float64)); o=np.argsort(w)
    print("eigh gram",n,"sweeps",sw,"w",np.max(np.abs(w[o]-ref))/np.max(np.abs(ref)))
    A=g.standard_normal((n,n)).astype(f); R=np.triu(np.linalg.qr(A)[1]).astype(f)
    s,V,sw=svd_par(R); ref=np.linalg.svd(R.astype(np.float64),compute_uv=False)
    o=np.argsort(-s); s=s[o]; V=V[:,o].astype(np.float64)
    print("svd",n,"sweeps",sw,"srel",np.max(np.abs(s-ref)/ref),"orth",np.linalg.norm(V.T@V-np.eye(n))/np.sqrt(n),"Av",np.max(np.abs(np.linalg.norm(R.astype(np.float64)@V,axis=0)-s))/ref[0])
