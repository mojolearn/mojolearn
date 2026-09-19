/* SPDX-License-Identifier: Apache-2.0
 * Host arithmetic derived from checks/numerics.mojo. No platform libm calls.
 * Build with contraction disabled; only the explicit fma sites fuse.
 * Linux baseline is x86-64-v3; macOS baseline is Apple M1, both with FMA.
 */
#include <stdint.h>
#include <string.h>
#include <float.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "portable math requires IEEE binary64"
#endif
static uint64_t bits(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }
static double value(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }
static double fm(double a, double b, double c) {
#if defined(__x86_64__)
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
#elif defined(__aarch64__) || defined(__arm64__)
    double out;
    __asm__("fmadd %d0, %d1, %d2, %d3" : "=w"(out) : "w"(a), "w"(b), "w"(c));
    return out;
#else
#error "portable math supports the wheel's x86-64-v3 and arm64 baselines"
#endif
}
/* Hardware sqrt is correctly rounded on both supported baselines. */
double mojolearn_sqrt(double x) {
#if defined(__x86_64__)
    return _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)));
#else
    double out;
    __asm__("fsqrt %d0, %d1" : "=w"(out) : "w"(x));
    return out;
#endif
}
static double log_fraction(double input, int *exponent, double *fraction) {
    uint64_t u=bits(input); int e=0;
    if ((u >> 52)==0) { input *= 18014398509481984.0; u=bits(input); e=-54; }
    e += (int)((u >> 52)&0x7ff)-1022;
    double m=value((u&UINT64_C(0x000fffffffffffff))|UINT64_C(0x3fe0000000000000));
    double z,y,x;
    if (e>2 || e<-2) {
        if (m<0.70710678118654752440) { --e; z=m-0.5; y=fm(0.5,z,0.5); }
        else { z=m-0.5; z=z-0.5; y=fm(0.5,m,0.5); }
        x=z/y; z=x*x;
        double r=fm(-7.89580278884799154124e-1,z,1.63866645699558079767e1);
        r=fm(r,z,-6.41409952958715622951e1);
        double q=z+-3.56722798256324312549e1;
        q=fm(q,z,3.12093766372244180303e2); q=fm(q,z,-7.69691943550460008604e2);
        y=x*(z*r/q);
    } else {
        if (m<0.70710678118654752440) { --e; x=fm(2.0,m,-1.0); }
        else x=m-1.0;
        z=x*x;
        double p=fm(1.01875663804580931796e-4,x,4.97494994976747001425e-1);
        p=fm(p,x,4.70579119878881725854e0); p=fm(p,x,1.44989225341610930846e1);
        p=fm(p,x,1.79368678507819816313e1); p=fm(p,x,7.70838733755885391666e0);
        double q=x+1.12873587189167450590e1;
        q=fm(q,x,4.52279145837532221105e1); q=fm(q,x,8.29875266912776603211e1);
        q=fm(q,x,7.11544750618563894466e1); q=fm(q,x,2.31251620126765340583e1);
        y=x*(z*p/q);
        /* log64 adds exponent compensation BEFORE this fma. */
    }
    *exponent=e; *fraction=x; return y;
}
static int special_log(double x, double *out) {
    if (x!=x) { *out=x; return 1; }
    if (x==0) { *out=value(UINT64_C(0xfff0000000000000)); return 1; }
    if (x<0) { *out=value(UINT64_C(0x7ff8000000000000)); return 1; }
    if (bits(x)==UINT64_C(0x7ff0000000000000)) { *out=x; return 1; }
    return 0;
}
double mojolearn_log(double input) {
    double out; if (special_log(input,&out)) return out;
    /* Preserve the original branch BEFORE mantissa normalization adjusts e. */
    uint64_t u=bits(input); int raw_e=0;
    if ((u>>52)==0) { u=bits(input*18014398509481984.0); raw_e=-54; }
    raw_e+=(int)((u>>52)&0x7ff)-1022;
    int e; double x; double y=log_fraction(input,&e,&x);
    y=fm((double)e,-2.121944400546905827679e-4,y);
    if (!(raw_e>2 || raw_e<-2)) y=fm(x*x,-0.5,y);
    y=y+x;
    return fm((double)e,0.693359375,y);
}
double mojolearn_log2(double input) {
    double out; if (special_log(input,&out)) return out;
    uint64_t u=bits(input); int raw_e=0;
    if ((u>>52)==0) { u=bits(input*18014398509481984.0); raw_e=-54; }
    raw_e+=(int)((u>>52)&0x7ff)-1022;
    int e; double x; double y=log_fraction(input,&e,&x);
    if (!(raw_e>2 || raw_e<-2)) y=fm(x*x,-0.5,y);
    out=y*0.44269504088896340735992;
    out=fm(x,0.44269504088896340735992,out); out=out+y; out=out+x;
    return out+(double)e;
}
float mojolearn_log2f(float x) { return (float)mojolearn_log2((double)x); }
double mojolearn_log10(double x) { return mojolearn_log(x)*0.434294481903251827651; }
double mojolearn_exp(double x) {
    if (x!=x) return x;
    if (x>709.782712893384) return value(UINT64_C(0x7ff0000000000000));
    if (x<-708.3964185322641) return 0.0;
    double t=x*1.4426950408889634+0.5;
    int ki=(int)t; if ((double)ki>t) --ki;
    double k=(double)ki;
    double r=fm(k,-6.93145751953125e-1,x); r=fm(k,-1.42860682030941723212e-6,r);
    double xx=r*r;
    double p=fm(1.26177193074810590878e-4,xx,3.02994407707441961300e-2);
    p=fm(p,xx,9.99999999999999999910e-1); p=p*r;
    double q=fm(3.00198505138664455042e-6,xx,2.52448340349684104192e-3);
    q=fm(q,xx,2.27265548208155028766e-1); q=fm(q,xx,2.00000000000000000009e0);
    double y=p/(q-p); y=fm(2.0,y,1.0);
    int k1=ki>>1,k2=ki-k1;
    y=y*value((uint64_t)(k1+1023)<<52);
    return y*value((uint64_t)(k2+1023)<<52);
}
/* Exact binary decomposition/scaling used by the bundled host runtime. */
double mojolearn_frexp(double x, int *exponent) {
    uint64_t u=bits(x), magnitude=u&UINT64_C(0x7fffffffffffffff);
    int e=(int)(magnitude>>52);
    *exponent=0;
    if (!magnitude || e==2047) return x;
    if (!e) { x*=18014398509481984.0; u=bits(x); e=(int)((u>>52)&2047)-54; }
    *exponent=e-1022;
    return value((u&UINT64_C(0x800fffffffffffff))|UINT64_C(0x3fe0000000000000));
}
double mojolearn_ldexp(double x, int exponent) {
    uint64_t u=bits(x), sign=u&UINT64_C(0x8000000000000000);
    uint64_t magnitude=u&UINT64_C(0x7fffffffffffffff);
    if (!magnitude || (magnitude>>52)==2047) return x;
    int e; double fraction=mojolearn_frexp(x,&e);
    int64_t target=(int64_t)e+exponent+1022;
    uint64_t mantissa=(bits(fraction)&UINT64_C(0xfffffffffffff))|UINT64_C(0x10000000000000);
    if (target>=2047) return value(sign|UINT64_C(0x7ff0000000000000));
    if (target>0) return value(sign|((uint64_t)target<<52)|(mantissa&UINT64_C(0xfffffffffffff)));
    if (target < -52) return value(sign);
    int shift=(int)(1-target);
    uint64_t rounded=mantissa>>shift, rem=mantissa&((UINT64_C(1)<<shift)-1);
    uint64_t half=UINT64_C(1)<<(shift-1);
    rounded+=(rem>half || (rem==half && (rounded&1)));
    return value(sign|rounded);
}
double mojolearn_modf(double x, double *integer) {
    uint64_t u=bits(x), sign=u&UINT64_C(0x8000000000000000);
    int e=(int)((u>>52)&2047)-1023;
    if (e<0) { *integer=value(sign); return x; }
    if (e>=52) { *integer=x; return x!=x ? x : value(sign); }
    uint64_t mask=(UINT64_C(1)<<(52-e))-1;
    *integer=value(u&~mask);
    if (!(u&mask)) return value(sign);
    return x-*integer;
}
long mojolearn_lround(double x) {
    /* C leaves domain errors implementation-defined; use the x86/arm
       integer-indefinite value without undefined float-to-integer casts. */
    if (x!=x || x>=9223372036854775808.0 || x< -9223372036854775808.0)
        return (-9223372036854775807L-1L);
    double integer; double fraction=mojolearn_modf(x,&integer);
    long result=(long)integer;
    if (fraction>=0.5) ++result;
    if (fraction<=-0.5) --result;
    return result;
}
