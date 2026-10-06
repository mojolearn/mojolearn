# SPDX-License-Identifier: Apache-2.0
"""Full forward/backward fixture paths include retained exponent stash,
recomputed state, and clean wrapper replay; repeated shape changes stress
scratch aliasing. Existing rejected packed/alias paths remain controls."""
from max.gpu.host import DeviceContext
from experiments.performance_ideas.I07.state_cost import state_cost
from transformer.checks.transformer_fused_check import FusedCase, run_case
from transformer.impl.llama.fused_attention import ATTN_V1_RECOMPUTE_BACKWARD, FUSED_RAN, fused_attention_arm_parse

# NEVER RUN — PENDING MEASUREMENT
def main() raises:
    var ctx = DeviceContext()
    # Model selection requests retention; a globally forced-recompute
    # control must actually retain zero cells and run the plain backward.
    print("I07 profile forced_recompute=", ATTN_V1_RECOMPUTE_BACKWARD)
    var arms: List[String] = ["stash_tiled_fgrid_r32_qres_pf_kvrecompute", "stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32", "stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32"]
    var lengths: List[Int] = [47, 83, 47]
    var launched = 0
    var backward = 0
    var kv = 0
    for i in range(len(lengths)):
        var l = lengths[i]
        var c = FusedCase("I07_lifetime" + String(i), 1, l, 4, 1, 64, 0, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN)
        # Explicit experiments cover memory-starved recomputation and a
        # retained-state budget; the model chooses actual canonical arms.
        for budget in [UInt64(0),UInt64(1<<30)]:
            var estimate = state_cost(1,4,l,l,64,budget,UInt64(1),UInt64(1),UInt64(8))
            var retain = estimate[2]
            if retain!=(budget!=0):
                raise Error("I07 byte budget policy witness failed")
            var selected = 1 if retain else 0
            if run_case(ctx,c,fused_attention_arm_parse(arms[selected]),launched,backward,kv)!=0:
                raise Error("I07 model-selected retained/recompute state moved bits")
            print("I07 model bytes=",estimate[0]," recompute_units=",estimate[1]," requested_retain=",retain)
        for a in range(len(arms)):
            if run_case(ctx, c, fused_attention_arm_parse(arms[a]), launched, backward, kv) != 0:
                raise Error("I07 stored/recomputed state changed bits")
    if backward == 0:
        raise Error("I07 backward candidate did not run")
    print("I07 PASS shape_reuse=3 arms=3 backward_launches=", backward)
