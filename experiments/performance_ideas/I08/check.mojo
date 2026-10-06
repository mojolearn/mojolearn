# SPDX-License-Identifier: Apache-2.0
"""Exercise actual Mamba2 forward and host replay at chunk tails,
multiple shipped state-width cases and alternating workspace sizes.
Separate rollback builds attribute tiling and triangular dead work."""
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from mamba.checks.mamba2_check import run_pair, compare_dumps, total_moved

def main() raises:
    var ctx = DeviceContext()
    var lengths: List[Int] = [63, 65, 131, 63]
    var cases_checked = 0
    for case_id in range(3):
        for i in range(len(lengths)):
            var trace = IdentityTrace.disabled()
            var pair = run_pair(ctx, case_id, 1, lengths[i], trace, "I08_" + String(case_id) + "_" + String(i))
            var diffs = compare_dumps(pair[0], pair[1], False)
            if total_moved(diffs) != 0:
                raise Error("I08 SSD tile replay differs at case " + String(case_id))
            cases_checked += 1
    print("I08 PASS full_stage_pairs=", cases_checked)
