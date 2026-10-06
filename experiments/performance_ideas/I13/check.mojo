# SPDX-License-Identifier: Apache-2.0
"""Same production canonicalizer at ragged degrees and dense skew,
duplicates/int32 extremes, empty rows, CSR capacity sentinels.
Runs the restored bounded global schedule or opt-in compact bucket route.
No pathological quadratic baseline; operation bound is explicit."""
from max.gpu.host import DeviceContext
from neighbors.checks.rbc_canonical_merge_check import check

def main() raises:
    var ctx = DeviceContext()
    var sizes: List[Int] = [0,1,2,7,31,257,3,513,2,1025,0]
    check(ctx,sizes,False,False)
    check(ctx,sizes,True,False)
    check(ctx,sizes,False,True)
    var skew = List[Int](length=4099,fill=3)
    skew[2037] = 16385
    check(ctx,skew,True,False)
    check(ctx,[0,0],False,False)
    print("I13 PASS compact_degree_buckets cases=5 max_skew=16385 short_rows=4098")
