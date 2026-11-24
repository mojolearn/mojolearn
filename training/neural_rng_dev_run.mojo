# SPDX-License-Identifier: Apache-2.0
"""RUN of `Generator.dropout` and `Generator.dropout_backward` on training
device arrays (lane vdf-tail-d, 2026-11-24): operation OPT_NATIVE_NEURAL_RNG_DEV
of the training image's optimizer native table (training/optimizer_native_api.mojo,
word layout in training/optimizer_native_extra.mojo).

The input (x for the forward, dy for the backward) is a TrainDeviceArray
handle of the training binding's pool and is read IN PLACE; the output is a
new handle the door allocated. Nothing is uploaded or downloaded: the finite
scan of the input runs on the device (`device_first_nonfinite`, one read of
the buffer, the host form's refusal words and index) and the draw is the one
`neural_rng_kernel` launch the host-input form makes (core/philox_neural.mojo
`neural_rng_host`), one thread per element, on the same words: the same bits
as the host-input path and the CPU column (`neural_rng_host_into`).

Typed Mojo only, no Python import. IDENTICAL GPU training images only (the
handles exist there: training/dev_tensors.mojo IDN_TRAIN_DEV_TENSORS); the
host image refuses the operation in the device-array words.
"""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_scan import device_first_nonfinite
from core.neural_context import neural_ctx
from core.philox_neural import (
    neural_rng_admit, neural_rng_kernel, NRNG_TPB, NRNG_DROPOUT_FORWARD, NRNG_DROPOUT_BACKWARD,
)
from core.train_dev_pool import train_dev_view

# The training binding's context (bindings/_mojolearn_training.mojo
# _NEURAL_CTX): the train_dev handles live on it (as residual_dropout_dev_run's).
comptime _NRNG_DEV_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"


def neural_rng_dev_run(n: Int, offset: Int, seed_lo: Int, seed_hi: Int, stream_id: Int, kind: Int,
                       output_handle: Int, input_handle: Int, p: Float32, scale: Float32) raises -> Int:
    """`n` dropout elements of stream `(seed, stream_id)` from element
    `offset`, read from handle `input_handle` and written to handle
    `output_handle` (n floats each, distinct handles). Returns `n`."""
    if kind != NRNG_DROPOUT_FORWARD and kind != NRNG_DROPOUT_BACKWARD:
        raise Error("mojolearn neural rng: device arrays are read by the dropout arms only, got kind " + String(kind))
    if output_handle == input_handle:
        raise Error("mojolearn neural rng: the dropout output must be disjoint from its input")
    _ = neural_rng_admit(n, offset, seed_lo, seed_hi, stream_id, kind, p, scale)
    var ctx = neural_ctx[_NRNG_DEV_CTX]()
    var dst = train_dev_view(ctx, output_handle, n, "Generator dropout output")
    var src = train_dev_view(ctx, input_handle, n, "Generator dropout input")
    var bad = device_first_nonfinite(ctx, src, n)
    if bad >= 0:
        raise Error("mojolearn neural rng: non-finite dropout input at " + String(bad))
    try:
        ctx.enqueue_function[neural_rng_kernel](
            dst.unsafe_ptr(),
            src.unsafe_ptr(),
            Int32(n),
            Int32(offset),
            UInt32(seed_lo).cast[DType.int32](),
            UInt32(seed_hi).cast[DType.int32](),
            UInt32(stream_id).cast[DType.int32](),
            Int32(kind),
            p,
            scale,
            grid_dim=((n + NRNG_TPB - 1) // NRNG_TPB, 1, 1),
            block_dim=(NRNG_TPB, 1, 1),
        )
        ctx.synchronize()
    except error:
        ctx.synchronize()
        raise error^
    # Both views are read by the queued launch above: keep them past the synchronize.
    _ = dst^
    _ = src^
    return n
