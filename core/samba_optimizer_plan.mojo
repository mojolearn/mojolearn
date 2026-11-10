# SPDX-License-Identifier: Apache-2.0
"""Samba's packed optimizer metadata, retaining its existing numerical ops."""
from training.optimizer_public_common import OptHyper
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


@fieldwise_init
struct SambaOptimizerPlan(Copyable, Movable):
    var tensors: Int
    var step: Int
    var hyper: OptHyper

    def floats(self) -> List[Float64]:
        return [self.hyper.lr, self.hyper.beta1, self.hyper.beta2, self.hyper.eps,
            self.hyper.weight_decay, self.hyper.momentum, self.hyper.dampening, self.hyper.max_norm]


def samba_optimizer_run(entry: NativeDescriptorCall, operation: Int, addresses: List[Int],
    plan: SambaOptimizerPlan, handle: Int, io: Int) raises -> Int:
    var integers = List[Int]()
    if operation != 25:
        integers.append(handle)
    if operation == 27:
        integers.append(io)
    integers.append(plan.tensors)
    integers.append(plan.hyper.kind)
    integers.append(plan.step)
    integers.append(plan.hyper.nesterov)
    integers.append(Int(plan.hyper.maximize))
    return native_descriptor_call(entry, operation, addresses, integers, plan.floats())
