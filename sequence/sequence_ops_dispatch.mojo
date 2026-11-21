# SPDX-License-Identifier: Apache-2.0
# The sequence layer operations of the whole-model descriptor that run on
# either executor (sequence/neural_model_dispatch.mojo SEQ_LAYER_NORM,
# SEQ_MOE_FORWARD, SEQ_MOE_BACKWARD), shared by the host and the device
# dispatch, and the host image's refusal of the eleven operations that need
# device tensors. RUN only: typed words in, one word out, no Python object and
# no Python import (lane dw-autograd-seq, 2026-11-21).
#
# Per operation (aa = addresses, pp = ints, ff = floats; `result` is the one
# word the descriptor returns). The lists are the lists the typed kernels have
# always taken, unchanged:
#   SEQ_LAYER_NORM    aa [x, weight or 0, bias or 0, y out, dy or 0, dx out,
#                         dweight out, dbias out]
#                     pp [M, D, has weight, has bias, backward(, keep stats)]
#                     ff [eps]                 (sequence/ops_impl.mojo layer_norm_impl)
#   SEQ_MOE_FORWARD   aa [x, router, gate_up, down, y, logits, selected, weights]
#                     pp [T, D, F, E, k, renormalise(, resident weight handle)]
#                     (moe_forward_impl; a seventh word > 0 is a device-only
#                     form, served by sequence/sequence_ops_device_dispatch.mojo)
#   SEQ_MOE_BACKWARD  aa / pp as `moe_backward_addrs` (autograd/route_ops.mojo)
# Not split across cores here: each operation is one kernel call; the kernels
# own their own parallel form.
from core.native_descriptor_call import native_descriptor_words
from autograd.route_ops import moe_backward_addrs
from sequence.exec_trait import Exec
from sequence.ops_impl import layer_norm_impl, moe_forward_impl
from sequence.neural_model_dispatch import (
    SEQ_LAYER_NORM, SEQ_MOE_FORWARD, SEQ_MOE_BACKWARD, SEQ_TENSOR_ALLOC, SEQ_TENSOR_COPY,
    SEQ_LAYER_NORM_DEV, SEQ_LAYER_NORM_DEV_ALLOC, SEQ_MOE_FORWARD_DEV, SEQ_MOE_FORWARD_DEV_ALLOC,
    SEQ_MOE_WEIGHTS_PUT, SEQ_MOE_WEIGHTS_FREE,
)


def sequence_ops_operation(op: Int) -> Bool:
    """True for the fourteen sequence operations of the descriptor."""
    return op >= SEQ_LAYER_NORM and op <= SEQ_MOE_WEIGHTS_FREE


def sequence_ops_floats(ff: Int, nf: Int) raises -> List[Float64]:
    if nf < 0 or (nf > 0 and ff == 0):
        raise Error("native sequence: invalid scalar span")
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var result = List[Float64](capacity=nf)
    for i in range(nf):  # driver-loop: over one layer's option scalars
        result.append(source[i])
    return result^


def sequence_ops_execute[E: Exec](mut ex: E, op: Int, aa: Int, na: Int, pp: Int, np: Int,
                                  ff: Int, nf: Int) raises -> Int:
    """The three operations both images serve, on the caller's executor."""
    var addresses = native_descriptor_words(aa, na)
    var ints = native_descriptor_words(pp, np)
    if op == SEQ_LAYER_NORM:
        return layer_norm_impl(ex, addresses, ints, sequence_ops_floats(ff, nf))
    if nf != 0:
        raise Error("native sequence: the MoE operations take no scalar descriptors")
    if op == SEQ_MOE_FORWARD:
        return moe_forward_impl(ex, addresses, ints)
    if op == SEQ_MOE_BACKWARD:
        return moe_backward_addrs(ex, addresses, ints, List[Int]())
    raise Error("native sequence: unknown operation")


def sequence_ops_host_refusal(op: Int, np: Int, pp: Int) raises:
    """The host (CPU) image has no host kernel for resident tensors, the
    device forms of LayerNorm and the MoE forward, or resident MoE weights.
    Each is refused in the words the doors used when the entry was absent.
    Returns without raising for the three operations the host image serves
    (a resident weight handle on the MoE operations is refused here too)."""
    if op == SEQ_LAYER_NORM:
        return
    if op == SEQ_MOE_FORWARD or op == SEQ_MOE_BACKWARD:
        if np == 7 and native_descriptor_words(pp, np)[6] > 0:
            raise Error("RuntimeError: MoEBlock: this binding holds no resident weights; pass host weights")
        return
    if op >= SEQ_TENSOR_ALLOC and op < SEQ_TENSOR_COPY:
        raise Error("RuntimeError: this native sequence binding has no device tensors")
    if op == SEQ_TENSOR_COPY:
        raise Error("RuntimeError: native binding has no device-to-device tensor copy")
    if op == SEQ_LAYER_NORM_DEV or op == SEQ_LAYER_NORM_DEV_ALLOC:
        raise Error("RuntimeError: layer_norm: this binding has no device LayerNorm; pass a host input")
    if op == SEQ_MOE_FORWARD_DEV or op == SEQ_MOE_FORWARD_DEV_ALLOC:
        raise Error("RuntimeError: MoEBlock: this binding has no device forward; pass a host input")
    if op == SEQ_MOE_WEIGHTS_PUT or op == SEQ_MOE_WEIGHTS_FREE:
        raise Error("RuntimeError: MoEBlock: this binding holds no resident weights; pass host weights")
    raise Error("native sequence: unknown operation")
