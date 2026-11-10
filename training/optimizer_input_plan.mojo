# SPDX-License-Identifier: Apache-2.0
"""Native optimizer buffer-layout and gradient-shape plans."""
from bindings.tensor_view import TensorView, base_format, shape_text

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
