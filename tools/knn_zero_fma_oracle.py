#!/usr/bin/env python3
"""Independent integer RN-even FMA oracle for the opt-in kNN repair gate."""
import random
import struct
from pathlib import Path


def decode(w):
    e = (w >> 23) & 255
    return ((-1 if w >> 31 else 1) * ((w & 0x7fffff) | 0x800000), e - 150) if e else (0, -149)


def oracle(a, b, c):
    am, ae = decode(a); bm, be = decode(b); cm, ce = decode(c)
    pe = ae + be
    unit = min(pe, ce)
    value = (am * bm << (pe-unit)) + (cm << (ce-unit))
    if not value:
        return 0x80000000 if am * bm == cm == 0 and ((a^b)&c&0x80000000) else 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    exponent = value.bit_length()-1+unit
    quantum = max(exponent-23, -149)
    shift = quantum-unit
    if shift > 0:
        q, r = divmod(value, 1 << shift)
        halfway = 1 << (shift-1)
        q += r > halfway or (r == halfway and q & 1)
    else:
        q = value << -shift
    if quantum == -149 and q < 0x800000:
        return sign  # output FTZ
    if q >= 0x1000000:
        q >>= 1; quantum += 1
    biased = quantum+23+127
    return sign | (0x7f800000 if biased >= 255 else (biased << 23) | (q & 0x7fffff))


def is_exact_subnormal(a, b, c):
    am, ae = decode(a); bm, be = decode(b); cm, ce = decode(c)
    pe = ae+be
    unit = min(pe, ce)
    value = abs((am*bm << (pe-unit)) + (cm << (ce-unit)))
    return not value or (unit < -126 and value < (1 << (-126-unit)))


def fixtures():
    rows=set()
    def put(a,b,c):
        for sa,sb,sc in [(0,0,0),(0x80000000,0,0),(0,0x80000000,0x80000000),(0x80000000,0x80000000,0x80000000)]:
            aa,bb,cc=a^sa,b^sb,c^sc
            rows.add((aa,bb,cc,oracle(aa,bb,cc),int(is_exact_subnormal(aa,bb,cc))))
    for a in range(0x3f7ffff0,0x3f800011):
        for b in [0x800000,0x800001,0xffffff,0x1000000]:put(a,b,0)
    for ce in range(1,41):
        for cm in [0x800000,0x800001,0xbfffff,0xc00000,0xffffff]:
            c=(ce<<23)|(cm&0x7fffff)
            for ae in range(1,126):
                be=104-ae
                if be<1:continue
                for m in [0x7ffffe,0x7fffff,0,1]:
                    put((ae<<23)|m,(be<<23),c)
    rng=random.Random(2099)
    for _ in range(6000):
        ae=rng.randrange(95,145);be=rng.randrange(1,45)
        a=(ae<<23)|rng.randrange(1<<23);b=(be<<23)|rng.randrange(1<<23)
        p=oracle(a,b,0)&0x7fffffff
        for delta in [-1,0,1]:
            c=max(0,min(0x7f7fffff,p+delta))|0x80000000
            put(a,b,c)
    return sorted(rows)

if __name__=='__main__':
    rows=fixtures()
    path=Path('bench/knn_zero_fma_oracle.txt')
    path.write_text(''.join(' '.join(map(str,r))+'\n' for r in rows))
    print(path,len(rows))
