# SPDX-License-Identifier: Apache-2.0
"""Full forward/backward fixture paths include retained exponent stash,
recomputed state, and clean wrapper replay; repeated shape changes stress
scratch aliasing. Existing rejected packed/alias paths remain controls."""
from max.gpu.host import DeviceContext
from transformer.checks.transformer_fused_check import FusedCase, run_case
from transformer.impl.llama.fused_attention import FUSED_RAN, fused_attention_arm_parse

def main() raises:
    var ctx = DeviceContext()
    var arms: List[String] = ["stash_tiled_fgrid_r32_qres_pf_kvrecompute", "stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32", "stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32"]
    var lengths: List[Int] = [47, 83, 47]
    var launched = 0
    var backward = 0
    var kv = 0
    for i in range(len(lengths)):
        var l = lengths[i]
        var c = FusedCase("I07_lifetime" + String(i), 1, l, 4, 1, 64, 0, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN)
        for a in range(len(arms)):
            if run_case(ctx, c, fused_attention_arm_parse(arms[a]), launched, backward, kv) != 0:
                raise Error("I07 stored/recomputed state changed bits")
    if backward == 0:
        raise Error("I07 backward candidate did not run")
    print("I07 PASS shape_reuse=3 arms=3 backward_launches=", backward)
