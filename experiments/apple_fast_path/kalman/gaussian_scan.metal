// SPDX-License-Identifier: Apache-2.0
// Experimental Apple-only source. NOT COMPILED, EXECUTED, OR MEASURED.
// Gaussian filtering monoid: Sarkka/Garcia-Fernandez, arXiv:1905.13002.
// All matrices here use row-major fixed-stride storage, unlike production Mojo.
#include <metal_stdlib>
using namespace metal;
constant uint R = 8;
struct Vec { float x[8]; };
struct Mat { float x[64]; };
struct Factor { Mat A, C, J; Vec b, eta; uint bad; };
struct Model { Mat T, Q, P0; Vec Z, alpha0, drift; uint rd, nobs, n_diff; };
struct Stage { uint count, stride, block_size, reserved; };
struct Innovation { float pred, residual, variance, log_variance, quadratic; uint bad; };
inline Mat zero() { Mat a; for(uint i=0;i<64;++i) a.x[i]=0; return a; }
inline Vec vz() { Vec a; for(uint i=0;i<8;++i) a.x[i]=0; return a; }
inline Mat eye(uint n) { Mat a=zero(); for(uint i=0;i<n;++i) a.x[i*R+i]=1; return a; }
inline Mat add(Mat a, Mat b) { for(uint i=0;i<64;++i) a.x[i]+=b.x[i]; return a; }
inline Vec va(Vec a, Vec b) { for(uint i=0;i<8;++i) a.x[i]+=b.x[i]; return a; }
inline Vec neg(Vec a) { for(uint i=0;i<8;++i) a.x[i]=-a.x[i]; return a; }
inline Mat transpose(Mat a,uint n) { Mat b=zero(); for(uint i=0;i<n;++i) for(uint j=0;j<n;++j) b.x[i*R+j]=a.x[j*R+i]; return b; }
inline Mat mul(Mat a,Mat b,uint n) { Mat c=zero(); for(uint i=0;i<n;++i) for(uint j=0;j<n;++j) for(uint k=0;k<n;++k) c.x[i*R+j]+=a.x[i*R+k]*b.x[k*R+j]; return c; }
inline Vec mv(Mat a,Vec b,uint n) { Vec c=vz(); for(uint i=0;i<n;++i) for(uint j=0;j<n;++j) c.x[i]+=a.x[i*R+j]*b.x[j]; return c; }
inline float dotv(Vec a,Vec b,uint n) { float v=0; for(uint i=0;i<n;++i) v+=a.x[i]*b.x[i]; return v; }
inline Factor identity(uint n) { Factor f; f.A=eye(n); f.C=zero(); f.J=zero(); f.b=vz(); f.eta=vz(); f.bad=0; return f; }
// Experimental pivoted solve. No inverse of singular Q or C is required.
// Tiny pivots deliberately refuse this candidate instead of regularizing it.
inline Mat inverse(Mat a,uint n,thread uint &bad) {
    Mat b=eye(n);
    for(uint k=0;k<n;++k) {
        uint p=k;
        for(uint i=k+1;i<n;++i) if(abs(a.x[i*R+k])>abs(a.x[p*R+k])) p=i;
        float pivot=a.x[p*R+k];
        if(!isfinite(pivot)||abs(pivot)<1e-20f) {bad=1;return zero();}
        for(uint j=0;j<n;++j) {
            float t=a.x[k*R+j];a.x[k*R+j]=a.x[p*R+j];a.x[p*R+j]=t;
            t=b.x[k*R+j];b.x[k*R+j]=b.x[p*R+j];b.x[p*R+j]=t;
        }
        for(uint j=0;j<n;++j) {a.x[k*R+j]/=pivot;b.x[k*R+j]/=pivot;}
        for(uint i=0;i<n;++i) if(i!=k) {
            float s=a.x[i*R+k];
            for(uint j=0;j<n;++j) {a.x[i*R+j]-=s*a.x[k*R+j];b.x[i*R+j]-=s*b.x[k*R+j];}
        }
    }
    return b;
}
// Ordered composition: left observations occur BEFORE right observations.
inline Factor compose(Factor left,Factor right,uint n) {
    Factor o=identity(n); o.bad=left.bad|right.bad;
    if(o.bad) return o;
    Mat g=inverse(add(eye(n),mul(left.C,right.J,n)),n,o.bad);
    // Use a second solve: avoids assuming rounded C and J are symmetric.
    Mat h=inverse(add(eye(n),mul(right.J,left.C,n)),n,o.bad);
    if(o.bad) return o;
    Mat rg=mul(right.A,g,n), lt=transpose(left.A,n);
    o.A=mul(rg,left.A,n);
    o.b=va(mv(rg,va(left.b,mv(left.C,right.eta,n)),n),right.b);
    o.C=add(mul(mul(rg,left.C,n),transpose(right.A,n),n),right.C);
    o.eta=va(mv(mul(lt,h,n),va(right.eta,neg(mv(right.J,left.b,n))),n),left.eta);
    o.J=add(mul(mul(mul(lt,h,n),right.J,n),left.A,n),left.J);
    for(uint i=0;i<n;++i) {
        if(!isfinite(o.b.x[i])||!isfinite(o.eta.x[i])) o.bad=1;
        for(uint j=0;j<n;++j) if(!isfinite(o.A.x[i*R+j])||!isfinite(o.C.x[i*R+j])||!isfinite(o.J.x[i*R+j])) o.bad=1;
    }
    return o;
}
// Caller supplies y with exogenous observation intercept already subtracted.
// t=0 uses the actual production P0/alpha0, including finite diffuse kappa.
// Later leaves condition transition noise on y_t; Q may be rank deficient.
kernel void kalman_scan_leaves(device const float *y [[buffer(0)]],
    constant Model &m [[buffer(1)]],device Factor *out [[buffer(2)]],uint t [[thread_position_in_grid]]) {
    if(t>=m.nobs) return;
    Factor f=identity(min(m.rd,R));
    if(m.rd<1||m.rd>R||m.n_diff>=m.nobs||!isfinite(y[t])) {f.bad=1;out[t]=f;return;}
    Mat p=t==0?m.P0:m.Q;
    Vec base=t==0?m.alpha0:m.drift;
    Mat transition=t==0?zero():m.T;
    Vec pz=mv(p,m.Z,m.rd);
    float s=dotv(m.Z,pz,m.rd);
    float residual=y[t]-dotv(m.Z,base,m.rd);
    if(!(s>0)||!isfinite(s)) {f.bad=1;out[t]=f;return;}
    Vec ht=mv(transpose(transition,m.rd),m.Z,m.rd);
    f.A=transition; f.C=p; f.b=base;
    for(uint i=0;i<m.rd;++i) {
        float k=pz.x[i]/s;
        f.b.x[i]+=k*residual;
        f.eta.x[i]=ht.x[i]*residual/s;
        for(uint j=0;j<m.rd;++j) {
            f.A.x[i*R+j]-=k*ht.x[j];
            f.C.x[i*R+j]-=k*pz.x[j];
            f.J.x[i*R+j]=ht.x[i]*ht.x[j]/s;
        }
    }
    out[t]=f;
}
// Candidate K1: inclusive Hillis-Steele scan; ping-pong nonaliasing buffers.
// Encode stride=1,2,4,... < nobs, with a device-buffer dependency per pass.
kernel void kalman_scan_step(device const Factor *in [[buffer(0)]],
    device Factor *out [[buffer(1)]],constant Model &m [[buffer(2)]],
    constant Stage &s [[buffer(3)]],uint t [[thread_position_in_grid]]) {
    if(t>=s.count||m.rd<1||m.rd>R) return;
    out[t]=t<s.stride?in[t]:compose(in[t-s.stride],in[t],m.rd);
}
// Candidate K2: each lane serially scans a short time block; then scan only
// block totals with kalman_scan_step. O(n log(n/B)) wide matrix traffic is
// replaced by O(n + (n/B)log(n/B)); local sequential depth is B.
kernel void kalman_scan_blocks(device const Factor *leaves [[buffer(0)]],
    device Factor *local [[buffer(1)]],device Factor *totals [[buffer(2)]],
    constant Model &m [[buffer(3)]],constant Stage &s [[buffer(4)]],uint block [[thread_position_in_grid]]) {
    uint first=block*s.block_size;
    if(first>=m.nobs||s.block_size==0||m.rd<1||m.rd>R) return;
    Factor f=identity(m.rd);
    for(uint t=first;t<min(first+s.block_size,m.nobs);++t) {f=compose(f,leaves[t],m.rd);local[t]=f;}
    totals[block]=f;
}
kernel void kalman_scan_carry(device const Factor *local [[buffer(0)]],
    device const Factor *totals_prefix [[buffer(1)]],device Factor *out [[buffer(2)]],
    constant Model &m [[buffer(3)]],constant Stage &s [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=m.nobs||s.block_size==0||m.rd<1||m.rd>R) return;
    uint block=t/s.block_size;
    out[t]=block==0?local[t]:compose(totals_prefix[block-1],local[t],m.rd);
}
// Reconstruct innovation statistics from filtered prefix t-1. Output mean
// and variance use the original observation convention (observe then evolve).
kernel void kalman_scan_innovations(device const float *y [[buffer(0)]],
    device const Factor *prefix [[buffer(1)]],constant Model &m [[buffer(2)]],
    device Innovation *out [[buffer(3)]],uint t [[thread_position_in_grid]]) {
    if(t>=m.nobs||m.rd<1||m.rd>R) return;
    Vec a=m.alpha0; Mat p=m.P0;
    uint bad=prefix[t].bad;
    if(t>0) {a=va(mv(m.T,prefix[t-1].b,m.rd),m.drift);p=add(mul(mul(m.T,prefix[t-1].C,m.rd),transpose(m.T,m.rd),m.rd),m.Q);}
    Innovation v;v.pred=dotv(m.Z,a,m.rd);v.residual=y[t]-v.pred;
    v.variance=dotv(m.Z,mv(p,m.Z,m.rd),m.rd);
    v.bad=bad|uint(!(v.variance>0)||!isfinite(v.variance)||!isfinite(v.residual));
    v.log_variance=0;v.quadratic=0;
    if(!v.bad&&t>=m.n_diff) {v.log_variance=log(v.variance);v.quadratic=v.residual*v.residual/v.variance;}
    out[t]=v;
}
// Keep reduction order serial as a control: scan changes filtering arithmetic
// but this candidate does not additionally introduce a likelihood sum tree.
// status carries the production sign convention: negative = diffuse step.
kernel void kalman_scan_likelihood(device const Innovation *terms [[buffer(0)]],
    constant Model &m [[buffer(1)]],device float *loglike [[buffer(2)]],
    device int *status [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(tid!=0) return;
    if(m.n_diff>=m.nobs) {status[0]=2147483647;return;}
    float l=0,q=0;status[0]=0;
    for(uint t=0;t<m.nobs;++t) {
        if(terms[t].bad) {status[0]=t<m.n_diff?-int(t+1):int(t+1);return;}
        if(t>=m.n_diff) {l+=terms[t].log_variance;q+=terms[t].quadratic;}
    }
    float n=float(m.nobs-m.n_diff);
    loglike[0]=-0.5f*(l+n*(q/n+1.8378770664093453f));
}
// Final state is the NEXT predictive state, matching production after nobs.
kernel void kalman_scan_final_state(device const Factor *prefix [[buffer(0)]],
    constant Model &m [[buffer(1)]],device Vec *alpha [[buffer(2)]],
    device Mat *covariance [[buffer(3)]],device uint *status [[buffer(4)]],uint tid [[thread_position_in_grid]]) {
    if(tid!=0) return;
    if(m.nobs==0||m.rd<1||m.rd>R) {status[0]=1;return;}
    Factor f=prefix[m.nobs-1];status[0]=f.bad;
    if(f.bad) return;
    alpha[0]=va(mv(m.T,f.b,m.rd),m.drift);
    covariance[0]=add(mul(mul(m.T,f.C,m.rd),transpose(m.T,m.rd),m.rd),m.Q);
}
// Candidate K3: scalar noiseless-observation specialization. Observing x_t
// fixes it to y_t, so alpha_(t+1)=T*y_t+drift and P_(t+1)=Q exactly in real
// arithmetic. No convergence threshold, scan workspace, or warm-up guess.
// Only rd=1, n_diff=0, Z=[1], finite observations. Reuse likelihood reduction.
kernel void kalman_scalar_innovations(device const float *y [[buffer(0)]],
    constant Model &m [[buffer(1)]],device Innovation *out [[buffer(2)]],uint t [[thread_position_in_grid]]) {
    if(t>=m.nobs) return;
    Innovation v;v.pred=0;v.residual=0;v.variance=0;v.log_variance=0;v.quadratic=0;
    v.bad=uint(m.rd!=1||m.n_diff!=0||m.Z.x[0]!=1.0f);
    if(v.bad) {out[t]=v;return;}
    v.pred=t==0?m.alpha0.x[0]:m.T.x[0]*y[t-1]+m.drift.x[0];
    v.variance=t==0?m.P0.x[0]:m.Q.x[0];v.residual=y[t]-v.pred;
    v.bad=uint(!(v.variance>0)||!isfinite(v.variance)||!isfinite(v.residual));
    if(!v.bad) {v.log_variance=log(v.variance);v.quadratic=v.residual*v.residual/v.variance;}
    out[t]=v;
}
kernel void kalman_scalar_final_state(device const float *y [[buffer(0)]],
    constant Model &m [[buffer(1)]],device Vec *alpha [[buffer(2)]],
    device Mat *covariance [[buffer(3)]],device uint *status [[buffer(4)]],uint tid [[thread_position_in_grid]]) {
    if(tid!=0) return;
    status[0]=uint(m.nobs==0||m.rd!=1||m.n_diff!=0||m.Z.x[0]!=1.0f);
    if(status[0]) return;
    Vec a=vz();a.x[0]=m.T.x[0]*y[m.nobs-1]+m.drift.x[0];
    status[0]=uint(!isfinite(a.x[0])||!(m.Q.x[0]>0)||!isfinite(m.Q.x[0]));
    if(status[0]) return;
    alpha[0]=a;covariance[0]=m.Q;
}
