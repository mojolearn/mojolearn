# SPDX-License-Identifier: Apache-2.0
"""AMD qualification of real degree-bounded independent graph tasks.

Uses the canonical row scheduler's skew/dense/tail/extreme-value fixtures.
Each task writes independent CSR slots; canonical order and reach are gated.
Also qualifies actual node histogram task construction with the shared N07
integer arithmetic, skewed node row spans, and refused malformed offsets.
Neither primitive qualification substitutes for a complete forest fit.
"""
from max.gpu.host import DeviceContext
from experiments.performance_ideas.A07.production_check import run_checks as check_production
from neighbors.checks.rbc_canonical_merge_check import check
from experiments.performance_ideas.A07.histogram_check import run_checks


def main() raises:
    check_production()
    var ctx=DeviceContext()
    check(ctx,[0,1,2,7,31,257,3,513,2,1025,0],False,False)
    check(ctx,[0,1,2,7,31,257,3,513,2,1025,0],True,False)
    check(ctx,[0,1,2,7,31,257,3,513,2,1025,0],False,True)
    var skew=List[Int](length=4099,fill=3)
    skew[2037]=16385
    check(ctx,skew,True,False)
    check(ctx,[0,0],False,False)
    print("A07_GRAPH_TASKS_PASS")
    run_checks()
