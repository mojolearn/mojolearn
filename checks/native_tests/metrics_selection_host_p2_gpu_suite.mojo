# SPDX-License-Identifier: Apache-2.0
"""precision_recall_curve on the device column (`metrics/mojolearn_metrics_run.mojo`).

One fixture on the process's one metrics device context; the ranking keeps
its own device parallelism.
"""
from metrics.mojolearn_metrics_run import precision_recall_curve_run
from checks.native_tests.metrics_selection_host_p2_cases import PrcCalls, check_curve


struct DevicePrc(PrcCalls):
    @staticmethod
    def curve(y: List[Int32], scores: List[Float32], n: Int, precision: MutPointer[Float32, MutUntrackedOrigin],
              recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return precision_recall_curve_run(y, scores, n, precision, recall, thresholds)


def run_checks() raises:
    check_curve[DevicePrc]()
    print("PASS metrics_selection_host_p2_gpu: precision_recall_curve counts at every threshold")
