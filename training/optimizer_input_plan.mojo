# SPDX-License-Identifier: Apache-2.0
"""Native optimizer buffer-layout and gradient-shape plans."""
from bindings.tensor_view import TensorView, base_format, shape_text
from std.memory import memcpy
from core.classical_pandas_impl import pd_raw_at
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _SRC = " (python/mojolearn/_training_impl.py)"
comptime _MIXED_PRECISION = "MIXED PRECISION IS NOT COVERED. Every buffer on this surface is float32 (loss contract section 1, optimizer contract section 1) and there is no float16 or bfloat16 anywhere in the two profiles: no master-weight copy, no loss scaler, no autocast, and no gated measurement of what a half-precision accumulation would do to the bits. A float16 array is REFUSED BY NAME here rather than upcast, because upcasting it silently would answer a question about float32 and label it as an answer about float16."


def same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over shape words
        if a[i] != b[i]:
            return False
    return True



def check_views(views: List[TensorView], name: String, where: String, inplace: Bool) raises:
    """`check_tensors` over views already read (by `as_seq_views` or
    `_check_grads`), in the same order with the same messages."""
    for i in range(len(views)):  # driver-loop: over parameter buffers
        _check_view(views[i], i, name, where, inplace)



def _check_view(v: TensorView, i: Int, name: String, where: String, inplace: Bool) raises:
    """One tensor's `_check_dtype` refusals."""
    var code = base_format(v.fmt)
    if code == "e" or v.dtype == "float16" or v.dtype == "bfloat16":
        raise Error("NotImplementedError: mojolearn." + where + ": " + String(_MIXED_PRECISION))
    if code != "f":
        raise Error(
            "TypeError: mojolearn." + where + ": " + name + "[" + String(i) + "] has dtype " + v.dtype
            + "; this surface is float32 only (loss contract section 1, optimizer contract section "
            + "1) and refuses rather than casts, because a cast changes which arithmetic answered "
            + "and leaves the label unchanged" + _SRC
        )
    if v.count() < 1:
        raise Error(
            "ValueError: mojolearn." + where + ": " + name + "[" + String(i) + "] is empty, shape "
            + shape_text(v.shape) + "; an empty tensor is REFUSED and not skipped -- the clip would "
            + "fold a zero-length GEMM, which no gate in this tree has run (training/estimator.mojo)"
        )
    if inplace and (not v.c_contig or v.readonly):
        raise Error(
            "ValueError: mojolearn." + where + ": " + name + "[" + String(i) + "] must be "
            + "C-contiguous and writable; it is written back IN PLACE (the torch.optim contract, the "
            + "trailing underscore on clip_grad_norm_) and a strided or read-only buffer cannot be. "
            + "Pass np.ascontiguousarray(...) of it, or a writable mojolearn.Array "
            + "(python/mojolearn/_training_impl.py, DEVIATION 2403)"
        )



def _addrs_of(views: List[TensorView], mut addrs: List[Int], mut nbytes: List[Int]):
    for j in range(len(views)):  # driver-loop: over parameter buffers
        addrs.append(views[j].addr)
        nbytes.append(views[j].nbytes)


def optimizer_gradient_count(parameters: Int, gradients: Int, owner: String, operation: String) raises:
    if gradients != parameters:
        raise Error("ValueError: mojolearn." + owner + "." + operation + ": " + String(gradients)
            + " gradients for " + String(parameters) + " parameter tensors")


def optimizer_gradient_shape(parameter: TensorView, gradient: TensorView, index: Int,
    owner: String, operation: String) raises:
    if not same_shape(parameter.shape, gradient.shape):
        raise Error("ValueError: mojolearn." + owner + "." + operation + ": grads[" + String(index)
            + "] has shape " + shape_text(gradient.shape) + ", params[" + String(index)
            + "] has " + shape_text(parameter.shape))


def optimizer_accumulation_shapes(probes: List[List[TensorView]]) raises -> Int:
    if len(probes) == 0:
        raise Error("ValueError: mojolearn.accumulate_grads: no microbatches")
    var count = len(probes[0])
    for i in range(len(probes)):  # driver-loop: microbatch tensor layouts
        var actual = len(probes[i])
        if actual != count:
            raise Error("ValueError: mojolearn.accumulate_grads: microbatch " + String(i) + " has " + String(actual)
                + " tensors, microbatch 0 has " + String(count))
        for k in range(actual):  # driver-loop: parameter buffer shapes
            if not same_shape(probes[i][k].shape, probes[0][k].shape):
                raise Error("ValueError: mojolearn.accumulate_grads: microbatch " + String(i) + " tensor "
                    + String(k) + " has shape " + shape_text(probes[i][k].shape)
                    + ", microbatch 0 has " + shape_text(probes[0][k].shape))
    return count


def optimizer_contiguous_address(view: TensorView, mut keep: List[List[Float32]]) raises -> Int:
    """Borrow contiguous input; gather strided FP32 bytes in parallel otherwise.

    The boundary retains the admitted export. Work splits by output elements
    using the existing host worker policy; no Python buffer is re-exported and
    no arithmetic/conversion changes the copied floating-point bits.
    """
    if view.c_contig:
        return view.addr
    var rank = len(view.shape)
    if len(view.strides) != rank or view.itemsize != 4:
        raise Error("native optimizer: invalid strided float32 metadata")
    var count = 1
    for axis in range(rank):  # driver-loop: admitted tensor dimension words
        var extent = view.shape[axis]
        if extent < 0 or (extent > 0 and count > 9223372036854775807 // extent):
            raise Error("native optimizer: strided tensor size overflow")
        count *= extent
    if count > 9223372036854775807 // 4 or view.nbytes != count * 4 or (count > 0 and view.addr == 0):
        raise Error("native optimizer: invalid strided float32 storage")
    var store = List[Float32](length=max(count, 1), fill=Float32(0))
    var destination = Int(store.unsafe_ptr())
    var source = view.addr
    var shape = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(view.shape.unsafe_ptr()))
    var strides = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(view.strides.unsafe_ptr()))
    var tasks = host_predict_task_count_work(count, count * 4)
    var chunk = host_predict_chunk(count, tasks)
    def _gather(task: Int) {imm count, imm chunk, imm rank, imm shape, imm strides, imm source, imm destination}:
        var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=destination)
        var first = task * chunk
        var last = min(count, first + chunk)
        if first >= last:
            return
        var coordinates = List[Int](length=rank, fill=0)
        var remaining = first
        var offset = 0
        for axis in range(rank - 1, -1, -1):  # driver-loop: initialize one worker's tensor coordinates
            coordinates[axis] = remaining % shape[axis]
            remaining //= shape[axis]
            offset += coordinates[axis] * strides[axis]
        var cursor = first
        while cursor < last:  # parallel-work: disjoint output spans per host worker
            var run = min(last - cursor, shape[rank - 1] - coordinates[rank - 1]) if rank else 1
            var step = strides[rank - 1] if rank else 0
            if step == 4:
                # Copy contiguous inner spans directly, including unaligned
                # exports. General signed strides retain exact raw-word loads.
                memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination + cursor * 4),
                    src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + offset), count=run * 4)
            else:
                for i in range(run):  # parallel-work: this worker's strided inner span
                    dst[cursor + i] = UInt32(pd_raw_at(source, offset + i * step, 4, True))
            cursor += run
            if cursor == last:
                break
            offset += run * step
            coordinates[rank - 1] += run
            # Carry once per inner span, not a division/modulo walk per value.
            for axis in range(rank - 1, -1, -1):  # driver-loop: advance tensor coordinates
                if coordinates[axis] < shape[axis]:
                    break
                offset -= coordinates[axis] * strides[axis]
                coordinates[axis] = 0
                if axis:
                    coordinates[axis - 1] += 1
                    offset += strides[axis - 1]
    if count > 0:
        if tasks <= 1:
            _gather(0)
        else:
            host_parallelize(_gather, tasks)
    _ = view
    keep.append(store^)
    return destination
