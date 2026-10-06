# SPDX-License-Identifier: Apache-2.0
"""Actual IVF search with planted CSR occupancy: all rows in one list,
uniform lists, heavy skew, empty lists and odd feature tails. Every probe
is selected and flat, staged, grouped schedules compare all returned bits.
Changing nprobe/index training is excluded. Opt-in balanced tasks split long lists into fixed work units without
changing probe order, per-point arithmetic, filtering or final tie order."""
from std.os import setenv, unsetenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from ivf.checks.ivf_check import _plant_index, _search

def check(ctx: DeviceContext, dim: Int, occupancy: Int) raises:
    var n = 519
    var lists = 17
    var nq = 33
    var x = List[Float32]()
    var labels = List[UInt32]()
    var centers = List[Float32](length=lists*dim,fill=Float32(0))
    var queries = List[Float32](length=nq*dim,fill=Float32(0))
    for row in range(n):
        var label = 0 if occupancy==0 else row%lists if occupancy==1 else 0 if row%7!=0 else row%lists
        labels.append(UInt32(label))
        for f in range(dim):
            x.append(Float32((row%71+f*7)%37-18)/Float32(16))
    var index = _plant_index(ctx,x,labels,centers,n,dim,lists)
    var expected_i = List[UInt32]()
    var expected_d = List[UInt32]()
    for arm in range(3):
        if not setenv("MOJOLEARN_IVF_SCAN_GROUPED", "1" if arm==2 else "0", True) or not setenv("MOJOLEARN_IVF_SCAN_STAGED", "1" if arm==1 else "0", True):
            raise Error("I16 could not select scan arm")
        var result = _search(ctx,index,queries,nq,17,lists)
        for q in range(nq):
            if result.n_candidates[q]!=Int32(n):
                raise Error("I16 selected lists dropped candidates")
        for i in range(nq*17):
            var word = bitcast[DType.uint32](result.distances[i])
            if arm==0:
                expected_i.append(result.indices[i])
                expected_d.append(word)
            elif expected_i[i]!=result.indices[i] or expected_d[i]!=word:
                raise Error("I16 occupancy schedule changed top-k bits")
        print("I16 scan_arm=",arm,"dim=",dim,"occupancy=",occupancy,"candidates=",n)
    _ = unsetenv("MOJOLEARN_IVF_SCAN_GROUPED")
    _ = unsetenv("MOJOLEARN_IVF_SCAN_STAGED")

def main() raises:
    var ctx = DeviceContext()
    for d in [7, 33, 65]:
        for occupancy in range(3):
            check(ctx,d,occupancy)
    print("I16 PASS skew_empty_uniform cases=9 scan_arms=3")
