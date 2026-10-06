# SPDX-License-Identifier: Apache-2.0
"""I04 scalar prerequisite probe; never installs a new numerical default.

Alternative leaves are intentionally different versions, not schedules.
Adversarial cancellation, signed zeros and subnormals expose their error and
bits. Device implementation stays blocked until I01-I03 prove a fold limit.
"""
from std.memory import bitcast
from gemm.host.gemm_oracle import gemm_oracle_at_leaf
from gemm.contract import contract_leaf_size
from gemm.checks.gemm_step_arms import _value


def main() raises:
    var ks: List[Int] = [127,128,129,255,257,513,1025]
    var leaves: List[Int] = [64,128,256]
    for fixture in range(3):
        for ki in range(len(ks)):
            var k = ks[ki]
            var a = List[Float32]()
            var b = List[Float32]()
            for p in range(k):
                var v = _value(p,17)
                var w = _value(p,31)
                if fixture == 1:
                    v = Float32(-0.0)
                    w = Float32(1.0)
                elif fixture == 2:
                    v = bitcast[DType.float32](UInt32(0x00800001))
                    w = Float32(0.5)
                a.append(v)
                b.append(w)
            var baseline = gemm_oracle_at_leaf(a,b,0,1,1,k,contract_leaf_size(k))
            for li in range(len(leaves)):
                var candidate = gemm_oracle_at_leaf(a,b,0,1,1,k,leaves[li])
                print("FOLD_PROBE fixture="+String(fixture)+" k="+String(k)
                      +" leaf="+String(leaves[li])
                      +" baseline_bits="+hex(bitcast[DType.uint32](baseline[0]))
                      +" candidate_bits="+hex(bitcast[DType.uint32](candidate[0])))
    print("FOLD_PROFILE_PREREQUISITE_UNPROVEN no_device_profile_enabled")
