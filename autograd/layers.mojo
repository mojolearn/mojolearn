# SPDX-License-Identifier: Apache-2.0
from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from sequence.layernorm import LN_FOLD_BLOCK, ln_fold_rows
from sequence.layernorm_core import layer_norm_fwd_core, layer_norm_bwd_core
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime LAYER_NORM = 1


@fieldwise_init
struct NeuralRules(BackwardRules):
    """Native backward adapters for existing neural kernels."""
    def supports(self, tag: Int) -> Bool:
        return tag == LAYER_NORM

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return True

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag != LAYER_NORM:
            raise Error("unknown neural backward rule")
        layer_norm_bwd_core(
            ex, inputs[0], inputs[1], upstream, gradients[0], gradients[1], gradients[2],
            saved.p4, saved.p5, rows[0], cols[0], True, saved.p6, saved.p7,
        )


def layer_norm[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, weight: Tensor, bias: Tensor, eps: Float32 = 1.0e-5,
) raises -> Tensor:
    """Normalize each row using native LayerNorm and record its backward rule."""
    var m = tape.rows(x)
    var d = tape.cols(x)
    if tape.rows(weight) != 1 or tape.cols(weight) != d or tape.rows(bias) != 1 or tape.cols(bias) != d:
        raise Error("LayerNorm weight and bias must have shape (1, features)")
    if not (eps > 0):
        raise Error("LayerNorm epsilon must be positive")
    var output = tape.reserve(ex, m * d)
    var mean = tape.reserve(ex, m)
    var rstd = tape.reserve(ex, m)
    layer_norm_fwd_core(ex, tape.value(x), tape.value(weight), tape.value(bias), output, mean, rstd, m, d, True, True, eps)
    var splits = min(max(8192 // d, 1), m // 1024) if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL else 0
    var split_rows = (m + splits - 1) // splits if splits > 1 else m
    comptime if LN_FOLD_BLOCK:
        split_rows = ln_fold_rows(m)
    splits = (m + split_rows - 1) // split_rows
    var saved = Args()
    # Retain epsilon for the recorded VJP used by higher-order autograd.
    saved.f0 = eps
    saved.p4 = mean
    saved.p5 = rstd
    saved.p6 = tape.reserve(ex, splits * d)
    saved.p7 = tape.reserve(ex, splits * d)
    # lane ag-rankn-core: the output keeps x's shape (metadata; rows x cols is m x d)
    return tape.custom_nd(ex, [x, weight, bias], output, tape.shape_of(x), LAYER_NORM, saved)
