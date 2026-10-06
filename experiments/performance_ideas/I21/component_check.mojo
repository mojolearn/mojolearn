# SPDX-License-Identifier: Apache-2.0
"""Tail component batches with independently spelled canonical E-step."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from mixture.checks.gmm_check import _device_e_step
from mixture.checks.gmm_oracle import oracle_e_step
from mixture.checks.estep import gmm_component_batch_width,GMM_COMPONENT_BATCH

def _same(a: List[Float32],b: List[Float32]) raises:
    if len(a)!=len(b):
        raise Error("I21 component batch shape moved")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i])!=bitcast[DType.uint32](b[i]):
            raise Error("I21 component batch bits moved at "+String(i))

def check_component_batches() raises:
    var ctx = DeviceContext()
    var quiet = IdentityTrace.disabled()
    for k in [1,3,9,17]:
        for d in [3,17]:
            var n = 129
            var x = List[Float32]()
            var means = List[Float32]()
            var prec = List[Float32]()
            var ldet = List[Float32]()
            var lw = List[Float32]()
            for i in range(n*d):
                x.append(Float32((i*19)%127-63)*Float32(0.03125))
            for c in range(k):
                for j in range(d):
                    means.append(Float32((c+j)%7-3)*Float32(0.0625))
                    for t in range(d):
                        prec.append(Float32(1) if j==t else Float32(0))
                ldet.append(Float32(0))
                lw.append(Float32(-2))
            var dev = _device_e_step(ctx,x,means,prec,ldet,lw,n,d,k,quiet,String("I21batch"))
            var expected = oracle_e_step(x,means,prec,ldet,lw,n,d,k,quiet,String("I21batch"))
            _same(dev.mahal,expected.mahal)
            _same(dev.wlp,expected.wlp)
            _same(dev.rowmax,expected.rowmax)
            _same(dev.lse,expected.lse)
            _same(dev.logresp,expected.logresp)
            if bitcast[DType.uint32](dev.meanll)!=bitcast[DType.uint32](expected.meanll):
                raise Error("I21 component batch likelihood moved")
            var width = gmm_component_batch_width(n,d,k)
            if width<=0 or width>8 or (k>1 and width<=1):
                raise Error("I21 component batch budget witness failed")
            print("I21 component k=",k,"d=",d,"width=",width,"candidate=",GMM_COMPONENT_BATCH)
    if gmm_component_batch_width(2147483647,2147483647,17)!=0:
        raise Error("I21 component scratch/index bound missing")
