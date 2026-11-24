# SPDX-License-Identifier: Apache-2.0
# The device image's sequence operations of the whole-model descriptor
# (sequence/neural_model_dispatch.mojo SEQ_LAYER_NORM .. SEQ_MOE_WEIGHTS_FREE).
# RUN only: typed words in, typed words out, no Python object and no Python
# import (lane dw-autograd-seq, 2026-11-21). The three operations both images
# serve go to sequence/sequence_ops_dispatch.mojo; the eleven below need
# device tensors and exist in this image only.
#
# Per operation (aa = addresses, pp = ints, ff = floats; `result` is the one
# word the descriptor returns):
#   SEQ_MOE_FORWARD / SEQ_MOE_BACKWARD with a seventh int word > 0: that word
#       is a SEQ_MOE_WEIGHTS_PUT handle whose device copies are read instead
#       of aa[1..3] (the branch the Python entries of the same name held).
#       SEQ_MOE_BACKWARD may carry three more words (pp 7..9): resident x, dy
#       and dx handles read and written in place (aa[0], aa[4], aa[5] unread).
#   SEQ_TENSOR_ALLOC      pp [n, zero]            result = the new handle (>= 1)
#   SEQ_TENSOR_FREE       pp [handle]             result = the handle
#   SEQ_TENSOR_UPLOAD     aa [host floats]  pp [handle, n]
#   SEQ_TENSOR_DOWNLOAD   aa [host floats out]  pp [handle, n]
#   SEQ_TENSOR_COPY       pp [dst handle, src handle, n]
#   SEQ_LAYER_NORM_DEV    aa [weight or 0, bias or 0, dweight out, dbias out]
#                         pp [H, the H handles..., the option words...]
#                         ff [eps]                result = M * D
#   SEQ_LAYER_NORM_DEV_ALLOC  aa [handles out (H words), then the four above]
#                         pp [H, the H handles..., the H sizes..., the option
#                             words...]   ff [eps]
#                         a slot whose size word is positive is a new resident
#                         tensor of that many floats; the H handles the layer
#                         ran on are written to `handles out`. result = H.
#   SEQ_MOE_FORWARD_DEV   pp [H, the H handles..., the option words...]
#   SEQ_MOE_FORWARD_DEV_ALLOC aa [handles out (H words)]
#                         pp [H, the H handles..., the H sizes..., the option
#                             words...]            result = H
#   SEQ_MOE_WEIGHTS_PUT   aa [router, gate_up, down]  pp [E, D, F]
#                         result = the weights' handle (> 0)
#   SEQ_MOE_WEIGHTS_FREE  pp [handle]
#
# OWNERSHIP. A handle SEQ_TENSOR_ALLOC returns belongs to the caller, which
# releases it with exactly one SEQ_TENSOR_FREE. The tensors an `_ALLOC` form
# makes belong to this operation until it returns: on a refusal it releases
# every one it made (`seq_tensor_free_made`) and the caller receives none;
# on success they are the caller's, as if it had allocated them. A
# SEQ_MOE_WEIGHTS_PUT handle belongs to the caller until its one
# SEQ_MOE_WEIGHTS_FREE. The registries are not locked: the door calls every
# one of these with the GIL held (as the Python entries ran).
#
# Not split across cores here: each operation is one registry step or one
# kernel call on the device queue.
from core.native_descriptor_call import native_descriptor_words
from autograd.arg_words import fptr_word
from autograd.route_ops import moe_backward_addrs, moe_backward_io
from sequence.exec_device import DeviceExec
from sequence.ops_impl import moe_forward_run_impl
from sequence.seq_tensor import seq_tensor_alloc, seq_tensor_free, seq_tensor_upload, seq_tensor_download, seq_tensor_copy, seq_tensor_ptr
from sequence.sequence_seq_tensor_run import seq_tensor_alloc_slots, seq_tensor_free_made
from sequence.layernorm_dev import layer_norm_dev
from sequence.moe_dev import moe_forward_dev
from sequence.moe_weights import moe_weights_put, moe_weights_ptrs, moe_weights_free
from sequence.sequence_ops_dispatch import sequence_ops_execute, sequence_ops_floats
from sequence.neural_model_dispatch import (
    SEQ_LAYER_NORM, SEQ_MOE_FORWARD, SEQ_MOE_BACKWARD, SEQ_TENSOR_ALLOC, SEQ_TENSOR_FREE,
    SEQ_TENSOR_UPLOAD, SEQ_TENSOR_DOWNLOAD, SEQ_TENSOR_COPY, SEQ_LAYER_NORM_DEV,
    SEQ_LAYER_NORM_DEV_ALLOC, SEQ_MOE_FORWARD_DEV, SEQ_MOE_FORWARD_DEV_ALLOC,
    SEQ_MOE_WEIGHTS_PUT, SEQ_MOE_WEIGHTS_FREE,
)


def _seq_slice(words: List[Int], start: Int, count: Int) raises -> List[Int]:
    if start < 0 or count < 0 or start + count > len(words):
        raise Error("native sequence: a word list runs past the descriptor")
    var result = List[Int](capacity=count)
    for i in range(count):  # driver-loop: over one layer call's handle / option words
        result.append(words[start + i])
    return result^


def _seq_handles_out(address: Int, handles: List[Int]) raises:
    if address == 0:
        raise Error("native sequence: null handle result buffer")
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(handles)):  # driver-loop: over one layer call's handle slots (at most six)
        dst[i] = handles[i]


def sequence_ops_device_execute(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> Int:
    if op == SEQ_LAYER_NORM:
        var norm_ex = DeviceExec()
        return sequence_ops_execute(norm_ex, op, aa, na, pp, np, ff, nf)
    var addresses = native_descriptor_words(aa, na)
    var ints = native_descriptor_words(pp, np)
    if op == SEQ_MOE_FORWARD or op == SEQ_MOE_BACKWARD:
        var ex = DeviceExec()
        if (np != 7 and not (np == 10 and op == SEQ_MOE_BACKWARD)) or ints[6] <= 0:
            return sequence_ops_execute(ex, op, aa, na, pp, np, ff, nf)
        if nf != 0:
            raise Error("native sequence: the MoE operations take no scalar descriptors")
        var w = moe_weights_ptrs(ints[6])
        if op == SEQ_MOE_BACKWARD:
            var resident: List[Int] = [Int(w[0]), Int(w[1]), Int(w[2])]
            if np == 10:
                # lane vdf-seq (2026-11-24): resident x, dy and dx (words 7..9,
                # seq_tensor handles of T x D floats) are read and written in
                # place; nothing crosses the bus for them.
                var td = ints[0] * ints[1]
                var io: List[Int] = [Int(seq_tensor_ptr(ints[7], td, "x")), Int(seq_tensor_ptr(ints[8], td, "dy")),
                    Int(seq_tensor_ptr(ints[9], td, "dx"))]
                var head = _seq_slice(ints, 0, 7)
                return moe_backward_io(ex, addresses, head, resident, io)
            return moe_backward_addrs(ex, addresses, ints, resident)
        # `moe_forward`'s argument checks, in its words.
        if na != 8:
            raise Error("moe_forward: requires 8 addresses and 7 integer parameters")
        var T = ints[0]
        var D = ints[1]
        var F = ints[2]
        var En = ints[3]
        var k = ints[4]
        if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
            raise Error("moe_forward: T, D, F, E >= 1 and 1 <= k <= E")
        return moe_forward_run_impl(ex, addresses, T, D, F, En, k, ints[5], w[0], w[1], w[2])
    if op == SEQ_TENSOR_ALLOC:
        if na != 0 or np != 2 or nf != 0:
            raise Error("native sequence: invalid tensor allocation descriptor")
        return seq_tensor_alloc(ints[0], ints[1] != 0)
    if op == SEQ_TENSOR_FREE:
        if na != 0 or np != 1 or nf != 0:
            raise Error("native sequence: invalid tensor release descriptor")
        return seq_tensor_free(ints[0])
    if op == SEQ_TENSOR_UPLOAD or op == SEQ_TENSOR_DOWNLOAD:
        if na != 1 or np != 2 or nf != 0:
            raise Error("native sequence: invalid tensor transfer descriptor")
        if op == SEQ_TENSOR_UPLOAD:
            return seq_tensor_upload(ints[0], addresses[0], ints[1])
        return seq_tensor_download(ints[0], addresses[0], ints[1])
    if op == SEQ_TENSOR_COPY:
        if na != 0 or np != 3 or nf != 0:
            raise Error("native sequence: invalid tensor copy descriptor")
        return seq_tensor_copy(ints[0], ints[1], ints[2])
    if op == SEQ_LAYER_NORM_DEV:
        if na != 4 or np < 1 or ints[0] < 0:
            raise Error("native sequence: invalid device LayerNorm descriptor")
        var count = ints[0]
        return layer_norm_dev(_seq_slice(ints, 1, count), addresses,
                              _seq_slice(ints, 1 + count, np - 1 - count), sequence_ops_floats(ff, nf))
    if op == SEQ_LAYER_NORM_DEV_ALLOC:
        if na != 5 or np < 1 or ints[0] < 0:
            raise Error("native sequence: invalid device LayerNorm descriptor")
        var count = ints[0]
        var handles = _seq_slice(ints, 1, count)
        var sizes = _seq_slice(ints, 1 + count, count)
        var options = _seq_slice(ints, 1 + 2 * count, np - 1 - 2 * count)
        var floats = sequence_ops_floats(ff, nf)
        if addresses[0] == 0:
            raise Error("native sequence: null handle result buffer")
        var made = seq_tensor_alloc_slots(handles, sizes)
        try:
            _ = layer_norm_dev(handles, _seq_slice(addresses, 1, 4), options, floats)
        except error:
            seq_tensor_free_made(made)
            raise error^
        _seq_handles_out(addresses[0], handles)
        return count
    if op == SEQ_MOE_FORWARD_DEV:
        if na != 0 or np < 1 or ints[0] < 0 or nf != 0:
            raise Error("native sequence: invalid device MoE descriptor")
        var count = ints[0]
        return moe_forward_dev(_seq_slice(ints, 1, count), _seq_slice(ints, 1 + count, np - 1 - count))
    if op == SEQ_MOE_FORWARD_DEV_ALLOC:
        if na != 1 or np < 1 or ints[0] < 0 or nf != 0 or addresses[0] == 0:
            raise Error("native sequence: invalid device MoE descriptor")
        var count = ints[0]
        var handles = _seq_slice(ints, 1, count)
        var sizes = _seq_slice(ints, 1 + count, count)
        var options = _seq_slice(ints, 1 + 2 * count, np - 1 - 2 * count)
        var made = seq_tensor_alloc_slots(handles, sizes)
        try:
            _ = moe_forward_dev(handles, options)
        except error:
            seq_tensor_free_made(made)
            raise error^
        _seq_handles_out(addresses[0], handles)
        return count
    if op == SEQ_MOE_WEIGHTS_PUT:
        if na != 3 or np != 3 or nf != 0:
            raise Error("moe_weights_put: requires 3 addresses and [E, D, F]")
        return moe_weights_put(fptr_word(addresses[0], "router"), fptr_word(addresses[1], "gate_up_proj"),
                               fptr_word(addresses[2], "down_proj"), ints[0], ints[1], ints[2])
    if op == SEQ_MOE_WEIGHTS_FREE:
        if na != 0 or np != 1 or nf != 0:
            raise Error("native sequence: invalid MoE weight release descriptor")
        moe_weights_free(ints[0])
        return 0
    raise Error("native sequence: unknown operation")
