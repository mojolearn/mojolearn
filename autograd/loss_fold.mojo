# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LOSS FOLD: numerator / denominator reduction of a loss over several parts (microbatches, logical shards,
devices), one block for every trainer that combines parts into one update (lane rnn-parity-c, Andrew 2026-11-25:
"sum of numerators / sum of denominators, never an average of per-microbatch means").

A part's loss is a pair (num, den): num the sum of its weighted per-element losses and den the sum of its weights
times its valid mask (the denominator; a plain unweighted, unmasked mean has den = its row count). The combined
loss is (num_1 + ... + num_k) / (den_1 + ... + den_k), and the combined gradient is the gradient of that, so a
microbatch with 2 valid tokens counts 2/22 against one with 20, never 1/2.

A part's gradient arrives in one of two forms, and the block gives the factor each takes before the fold:
- of the part's MEAN num_i / den_i (the trainers' built-in losses, mse ... cross_entropy): weight den_i / D, D the
  total denominator, known before the parts run (the minibatch's rows). With den_i = rows this is the rows / count
  word the distributed and partitioned trainers multiplied by before (the same bits).
- of the part's NUMERATOR num_i (a captured program with two outputs, (loss sum, denominator), loss="output"):
  weight 1 / D when D is known before the fold, else weight 1 and one scale of the folded total by 1 / D after the
  last part (`final_scale`); the distributed trainer folds wave by wave and learns D only at the end.

Users: autograd/distributed_program.mojo DistributedProgram.fit (logical shards over devices) and
autograd/partition_runtime.mojo PartitionedProgram.fit (pipeline microbatches). The generic Sequential loop
(autograd/model.mojo) and the recurrent estimators (sequence/recurrent.mojo rnn_fit) take one update per minibatch
(no accumulation): each update is its own minibatch's mean, and their epoch curves already fold num / den (the
batch losses times their rows, divided by the rows). Host scalars only: the factors are words the executors'
existing scale launches read; nothing here launches.

Fault injection (verify's sabotage arms, never in a product build): `MOJOLEARN_RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_
DEVICE` / `_HOST` make the factors of the device executor's trainers, or the host executor's, the average of the
parts' means (weight 1 / k of each mean). The verify lane ops-loss-fold-num-den (tools/identity_lanes/
nn_programs.py) must FAIL under either, on its side.
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from checks.numerics import ftz, identical_div, identical_mul_add

comptime RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_DEVICE = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_DEVICE"]()
comptime RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_HOST = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_HOST"]()


def loss_fold_sabotaged[E: Exec]() -> Bool:
    """True when this executor's side carries the mean-of-means fault (a verify build only)."""
    return (RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_DEVICE and E.IS_DEVICE) or (
        RNN_PARITY_SABOTAGE_MEAN_OF_MEANS_HOST and not E.IS_DEVICE
    )


struct LossFold(ImplicitlyCopyable, Movable):
    """The numerator / denominator fold of one update's parts (or of an epoch's updates, for the loss curve)."""
    #: True: each part's gradient is of its mean num_i / den_i; False: of its numerator num_i
    var of_mean: Bool
    #: the number of parts the update folds (the sabotage arms' 1 / k)
    var parts: Int
    #: the total denominator when known before the fold, else 0 (then `final_scale` carries 1 / D)
    var total: Float32
    #: the numerators and denominators folded so far, in part order
    var num: Float32
    var den: Float32

    def __init__(out self, of_mean: Bool, parts: Int, total: Float32):
        self.of_mean = of_mean
        self.parts = parts
        self.total = total
        self.num = Float32(0)
        self.den = Float32(0)

    def weight[E: Exec](self, den_i: Float32) raises -> Float32:
        """The factor of part i's gradient (den_i its denominator) before it joins the fold."""
        if loss_fold_sabotaged[E]():
            var k = Float32(self.parts)
            if self.of_mean:
                return ftz(identical_div(Float32(1), k))
            if self.total > Float32(0):
                return ftz(identical_div(ftz(identical_div(Float32(1), k)), den_i))
            return ftz(identical_div(Float32(1), den_i))
        if self.of_mean:
            if not (self.total > Float32(0)):
                raise Error("ValueError: loss fold: the total denominator must be positive")
            return ftz(identical_div(den_i, self.total))
        if self.total > Float32(0):
            return ftz(identical_div(Float32(1), self.total))
        return Float32(1)

    def final_scale[E: Exec](self) raises -> Float32:
        """The factor of the folded gradient after the last part: 1 / D when D was not known before the fold (the
        numerator form folded with weight 1), else 1 (no scale to launch)."""
        if self.of_mean or self.total > Float32(0):
            return Float32(1)
        if loss_fold_sabotaged[E]():
            return ftz(identical_div(Float32(1), Float32(self.parts)))
        if not (self.den > Float32(0)):
            raise Error("ValueError: loss fold: the parts' denominators sum to zero (no valid weighted element)")
        return ftz(identical_div(Float32(1), self.den))

    def add_mean(mut self, mean_i: Float32, den_i: Float32):
        """Fold a part given as its mean: num += den_i mean_i (one fused multiply-add), den += den_i."""
        self.num = ftz(identical_mul_add(den_i, mean_i, self.num))
        self.den = ftz(self.den + den_i)

    def add_sum(mut self, num_i: Float32, den_i: Float32):
        """Fold a part given as its numerator and denominator."""
        self.num = ftz(self.num + num_i)
        self.den = ftz(self.den + den_i)

    def mean(self) raises -> Float32:
        """The folded loss, sum num / sum den."""
        if not (self.den > Float32(0)):
            raise Error("ValueError: loss fold: the parts' denominators sum to zero (no valid weighted element)")
        return ftz(identical_div(self.num, self.den))
