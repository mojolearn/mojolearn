# SPDX-License-Identifier: Apache-2.0
"""precision_recall_curve on the host column (`metrics/mojolearn_metrics_host_run.mojo`).

One fixture: a single curve over 40 points, nothing independent to split
across cores (the per-threshold comparison is 40 x 40 integer counts).
"""
from metrics.mojolearn_metrics_host_run import precision_recall_curve_run
from checks.native_tests.metrics_selection_host_p2_cases import PrcCalls, check_curve


struct HostPrc(PrcCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run(y, scores, n, precision, recall, thresholds)


def run_checks() raises:
    check_curve[HostPrc]()
    print("PASS metrics_selection_host_p2_cpu: precision_recall_curve counts at every threshold")
