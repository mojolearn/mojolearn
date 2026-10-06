# SPDX-License-Identifier: Apache-2.0
"""Typed CNN training contract; no Python objects or callbacks.

The caller owns the resident model/data/tape described by this contract. The
native trainer owns epoch/minibatch sequencing, optimizer scratch and loss
summaries. Public model construction and prediction remain separate migration
work; this contract alone is not a claim of a Python-free public product.
"""
from bindings.hostptr import f64_ptr
from x_cnn.ops import adam_hyper_base


@fieldwise_init
struct TrainingSpec(Copyable, Movable):
    var blocks: List[List[Int]]
    var head: List[Int]
    var arrays: List[Int]
    var data: List[Int]
    var dims: List[Int]
    var weights: List[Int]
    var gradients: List[Int]
    var buffers: List[Int]
    var sizes: List[Int]
    var conv: List[List[List[Int32]]]
    var pool: List[List[List[Int32]]]
    var pooled: List[List[Bool]]


def training_steps(n: Int, batch: Int, epochs: Int, done: Int, first_epoch: Int) raises -> Int:
    if n <= 0 or n >= 2147483647 or batch <= 0 or epochs < 0 or done < 0 or first_epoch < 0:
        raise Error("x_cnn fit: positive rows/batch, nonnegative epochs/step required")
    return (n - 1) // batch + 1


def validate_spec(spec: TrainingSpec) raises:
    """Validate typed metadata before dereferencing resident handles."""
    if len(spec.head) != 4 or len(spec.arrays) != 6 or len(spec.data) != 3 or len(spec.dims) != 2:
        raise Error("x_cnn fit: invalid head/activation/data/dimension descriptor")
    if spec.dims[0] <= 0 or spec.dims[1] <= 0 or spec.data[2] <= 0:
        raise Error("x_cnn fit: positive feature/class/row dimensions required")
    var count = len(spec.weights)
    if len(spec.gradients) != count or len(spec.buffers) != count or len(spec.sizes) != count:
        raise Error("x_cnn fit: optimizer descriptors must have equal lengths")
    for i in range(count):
        if spec.weights[i] == 0 or spec.gradients[i] == 0 or spec.buffers[i] == 0 or spec.sizes[i] <= 0:
            raise Error("x_cnn fit: nonnull optimizer handles and positive sizes required")
    if len(spec.conv) != 2 or len(spec.pool) != 2 or len(spec.pooled) != 2:
        raise Error("x_cnn fit: full and final-batch plans required")
    var blocks = len(spec.blocks)
    for q in range(2):
        if len(spec.conv[q]) != blocks or len(spec.pool[q]) != blocks or len(spec.pooled[q]) != blocks:
            raise Error("x_cnn fit: one full/final plan per convolution block required")
    for j in range(blocks):
        if len(spec.blocks[j]) != 9:
            raise Error("x_cnn fit: nine resident handles per convolution block required")


def optimizer_base(fparams: List[Float64], adam: Bool) raises -> List[Float32]:
    if len(fparams) != 6:
        raise Error("x_cnn fit: optimizer requires exactly six scalar parameters")
    if adam:
        return adam_hyper_base(fparams[0], fparams[1], fparams[2], fparams[3], fparams[4], fparams[5])
    var base = List[Float32]()
    # SGD uses exactly two rows: first-ever step and all subsequent steps.
    # No table proportional to training rows/steps is constructed by Python.
    for r in range(2):
        for e in range(5):
            base.append(Float32(fparams[e]))
        base.append(Float32(1) if r == 0 else Float32(0))
    return base^


def epoch_loss_mean(losses_addr: Int, count: Int) raises -> Float64:
    """Same ordered CPython-3.12 Neumaier fold as the old _pm.nsum/count.

    Losses already crossed the binding's declared synchronization/readback
    boundary. This is compiled host Mojo scalar bookkeeping, never Python.
    """
    var losses = f64_ptr(losses_addr)
    var total = Float64(0) + losses[0]
    var correction = Float64(0)
    for i in range(1, count):
        var x = losses[i]
        var t = total + x
        if abs(total) >= abs(x):
            correction += (total - t) + x
        else:
            correction += (x - t) + total
        total = t
    if correction != 0.0 and correction - correction == 0.0:
        total += correction
    return total / Float64(count)
