# SPDX-License-Identifier: Apache-2.0
"""Public retained KDE score_samples timing; no oracle or identity rerun."""
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from kde.checks.kde_check import _train_fixture,_query_fixture,_weight_fixture
from kde.resident_fit import kde_fit_prepare,kde_score_samples_resident,kde_fit_release

def main() raises:
    var n=gemm_step_env_int("AB_ROWS",100000)
    var q=gemm_step_env_int("AB_QUERIES",1024)
    var d=gemm_step_env_int("AB_FEATURES",8)
    var train=_train_fixture(n,d,29)
    var query=_query_fixture(train,n,q,d,37)
    var weights=_weight_fixture(n,41)
    var output=List[Float32](length=q,fill=Float32(0))
    var handle=kde_fit_prepare(train.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),n,d,Float32(0.25),String("gaussian"),String("euclidean"),weights,True)
    for phase in range(2):
        var start=perf_counter_ns()
        _=kde_score_samples_resident(handle,query.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),q,d,Float32(0.25),String("gaussian"),String("euclidean"),output.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
        var elapsed=perf_counter_ns()-start
        print("MEASURE id=I20 phase="+String(phase)+" rows="+String(n)+" queries="+String(q)+" features="+String(d)+" elapsed_ns="+String(elapsed))
    kde_fit_release(handle)
