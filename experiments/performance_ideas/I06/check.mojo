# SPDX-License-Identifier: Apache-2.0
"""Actual fused attention launches, eager bits, and launch-token reach on
neighboring non-board tails, real KV-sharing ratios, mask/decode boundaries.
No timing here: adversarial fixtures qualify semantics only."""
from max.gpu.host import DeviceContext
from transformer.checks.transformer_fused_check import FusedCase, run_case
from transformer.impl.llama.fused_attention import FUSED_RAN, fused_attention_arm_parse

# NEVER RUN — PENDING VALIDATION
def main() raises:
    var ctx = DeviceContext()
    var arms = List[String]()
    arms.append("stash_tiled_fgrid_r32_qres_pf")
    arms.append("stash_tiled_fgrid_r32_qres_pf_kvgrid_r32")
    var lengths: List[Int] = [31, 33, 79]
    var launched = 0
    var backward = 0
    var kv = 0
    var checks = 0
    for ratio in range(1, 4):
        var nh = 2 * ratio
        for i in range(len(lengths)):
            var length = lengths[i]
            var c = FusedCase("I06_gqa" + String(ratio) + "_tail" + String(length), 1, length, nh, 2, 64, 19, 7, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN)
            for a in range(len(arms)):
                if run_case(ctx, c, fused_attention_arm_parse(arms[a]), launched, backward, kv) != 0:
                    raise Error("I06 attention schedule changed canonical bits")
                checks += 1
    if launched == 0 or backward == 0 or kv == 0:
        raise Error("I06 candidate launch tokens absent")
    print("I06 PASS cases=", checks, "arm_launches=", launched, "backward=", backward, "kv=", kv)
