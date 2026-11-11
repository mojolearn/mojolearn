# SPDX-License-Identifier: Apache-2.0
"""Native validation of the existing Python-style class index range."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def svc_score_codes_valid(codes: List[Int64], classes: Int) -> Bool:
    var count = len(codes)
    if count == 0:
        return True
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    var failures = List[Int](length=units, fill=0)
    var source = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()))
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm source, imm flags, imm count, imm classes, imm chunk}:
        var failed = False
        for row in range(unit * chunk, min((unit + 1) * chunk, count)):
            failed = failed or source[row] < -Int64(classes) or source[row] >= Int64(classes)
        flags[unit] = Int(failed)
    host_parallelize(task, units)
    var result = True
    for flag in failures:  # metadata: one status per independent partition
        result = result and flag == 0
    _ = codes
    return result
