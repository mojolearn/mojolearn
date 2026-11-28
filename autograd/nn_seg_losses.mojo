# SPDX-License-Identifier: Apache-2.0
"""The segmentation losses of `Sequential.compile(loss="dice")` and `loss="dice_ce"` (lane nn-unet-1, 2026-11-27).

The model's output sample is (C, *spatial): C class channels over S = prod(spatial) pixels (a U-Net's last 1 x 1
convolution). The target has the output's shape: one-hot (or soft) masks, one channel a class.

dice (MONAI's `DiceLoss(softmax=True)` for C >= 2, `sigmoid=True` for C == 1; `include_background=True`,
`squared_pred=False`, `batch=False`, `smooth_nr = smooth_dr = 1e-5`):

    p = softmax over the channels at every pixel (sigmoid for one channel)
    I[b, c] = sum_s p * y        P[b, c] = sum_s p        Y[b, c] = sum_s y
    term[b, c] = 1 - (2 I + 1e-5) / (Y + P + 1e-5)
    loss = mean (reduction 'mean') or sum ('sum') of the B * C terms

dice_ce (MONAI's `DiceCELoss`, lambda_dice = lambda_ce = 1): dice + the cross-entropy over the channels at every
pixel, - sum_c y log softmax(x), averaged over the B * S pixels ('mean') or summed ('sum'); one channel takes the
binary cross-entropy with logits of every pixel instead (tape.loss_reduced, LS_BCE_LOGITS, the bce_with_logits
loss's own body).

One body, no new kernel: every step is an existing tape operation with its existing backward (rankn_softmax: the
permute -> row softmax -> permute back of autograd/rankn_fold.mojo; rankn_binary on same shapes: the tape's binary
op; rankn_sum over the last axis: the tape's blocked row fold, fixed 64-term leaves; tape.add / scale / divide /
mean / sum). So the host column and every device run the same words in the same fold order: the pixel sums fold
each (b, c) row in AG_FOLD_LEAF leaves, then the leaves in order; the batch mean folds the B * C terms the same
way; dice_ce adds the dice word and the cross-entropy word, in that order. Nothing is launched per sample, class
or pixel: the sums are parallel over rows and leaves.

What differs from MONAI: the fold order of the sums (theirs is torch.sum's, ours the fixed blocked fold, so the
last bits of a loss can differ; the same on every vendor here); targets are masks of the output's shape (MONAI also
takes label maps with to_onehot_y: pass np.eye(C)[labels] moved to the channel axis); smooth and the lambdas are
the MONAI defaults, not options.
"""
from std.collections import Array
from sequence.exec_trait import Exec
from autograd.tape import Tape, Tensor
from autograd.ops import MULTIPLY
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_fold import rankn_sum, rankn_softmax, rankn_binary
from autograd.rankn_api import rn_as_shape
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM
from autograd.loss_ops import LS_BCE_LOGITS
from checks.numerics import identical_div

#: MONAI's smooth_nr and smooth_dr
comptime SEG_DICE_SMOOTH: Float32 = 1.0e-5

# Sabotage switches of the fault-injection arms (tools/identity_lanes/coverage_sabotage.tsv, the nn-unet arms): the
# patches under autograd/checks/sabotage/coverage_nn_seg_dice_* turn one of these on for one column.
comptime SEG_SABOTAGE_DICE_NO_DOUBLE = False


def _shape3(B: Int, C: Int, S: Int) -> ShapeN:
    var d = Array[Int, RN_MAX_RANK](fill=1)
    d[0] = B
    d[1] = C
    d[2] = S
    return ShapeN(3, d^)


def _seg_batch(mut tape: Tape, logits: Tensor, target: Tensor, C: Int, S: Int, what: String) raises -> Int:
    """The batch B of (B, C * S) logits and a target of the same element count; refuses anything else."""
    if C < 1 or S < 1:
        raise Error(what + ": the output needs at least one channel and one pixel")
    var words = tape.rows(logits) * tape.cols(logits)
    if words % (C * S) != 0:
        raise Error(what + ": the output is not a whole number of (" + String(C) + ", " + String(S) + ") samples")
    if tape.rows(target) * tape.cols(target) != words:
        raise Error(what + ": the target must have the output's shape (one mask channel a class)")
    return words // (C * S)


def seg_dice_loss[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, target: Tensor, C: Int, S: Int, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """The soft Dice loss of logits (B, C * S) (any view whose elements are the B samples' (C, S) channels in
    row-major order) against masks of the same shape. Only the logits take a gradient.

    Example::

        var loss = seg_dice_loss(tape, ex, logits, masks, 3, 64 * 64)
        tape.backward_scalar_with_rules(ex, loss, ModelRules())
    """
    if reduction != REDUCTION_MEAN and reduction != REDUCTION_SUM:
        raise Error("dice: reduction must be 'mean' or 'sum'")
    var B = _seg_batch(tape, logits, target, C, S, "dice")
    var s3 = _shape3(B, C, S)
    var p: Tensor
    if C == 1:
        p = rn_as_shape(tape, ex, tape.sigmoid(ex, logits), s3)
    else:
        p = rankn_softmax(tape, ex, rn_as_shape(tape, ex, logits, s3), 1)
    var y = rn_as_shape(tape, ex, target, s3)
    var pixels: List[Int] = [2]
    var rows = ShapeN.matrix(B * C, 1)
    var inter = rn_as_shape(tape, ex, rankn_sum(tape, ex, rankn_binary(tape, ex, p, y, MULTIPLY), pixels), rows)
    var psum = rn_as_shape(tape, ex, rankn_sum(tape, ex, p, pixels), rows)
    var ysum = rn_as_shape(tape, ex, rankn_sum(tape, ex, y, pixels), rows)
    var smooth = tape.scalar(ex, SEG_DICE_SMOOTH)
    var twice = Float32(2)
    comptime if SEG_SABOTAGE_DICE_NO_DOUBLE:
        twice = Float32(1)  # SABOTAGE: the numerator loses its factor 2
    var num = tape.add(ex, tape.scale(ex, inter, twice), smooth)
    # MONAI's denominator: ground_o + pred_o, then smooth_dr
    var den = tape.add(ex, tape.add(ex, ysum, psum), smooth)
    var ratio = tape.divide(ex, num, den)
    # 1 - ratio: an exact negation, then + 1
    var terms = tape.add(ex, tape.scale(ex, ratio, Float32(-1)), tape.scalar(ex, Float32(1)))
    if reduction == REDUCTION_SUM:
        return tape.sum(ex, terms)
    return tape.mean(ex, terms)


def seg_channel_cross_entropy[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, target: Tensor, C: Int, S: Int, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """The cross-entropy over the channels at every pixel of logits (B, C * S) against masks of the same shape:
    - sum_c y log softmax(x) averaged over the B * S pixels ('mean') or summed ('sum'). One channel: the binary
    cross-entropy with logits of every element (the bce_with_logits loss's body)."""
    if reduction != REDUCTION_MEAN and reduction != REDUCTION_SUM:
        raise Error("dice_ce: reduction must be 'mean' or 'sum'")
    var B = _seg_batch(tape, logits, target, C, S, "dice_ce")
    if C == 1:
        return tape.loss_reduced(ex, logits, target, LS_BCE_LOGITS, Float32(1), reduction != REDUCTION_SUM, False)
    var s3 = _shape3(B, C, S)
    var lp = rankn_softmax(tape, ex, rn_as_shape(tape, ex, logits, s3), 1, True)
    var y = rn_as_shape(tape, ex, target, s3)
    var every = List[Int]()
    var total = rankn_sum(tape, ex, rankn_binary(tape, ex, lp, y, MULTIPLY), every)
    var factor = Float32(-1)
    if reduction != REDUCTION_SUM:
        factor = -identical_div(Float32(1), Float32(B * S))
    return tape.scale(ex, rn_as_shape(tape, ex, total, ShapeN.matrix(1, 1)), factor)


def seg_dice_ce_loss[E: Exec](
    mut tape: Tape, mut ex: E, logits: Tensor, target: Tensor, C: Int, S: Int, reduction: Int = REDUCTION_MEAN,
) raises -> Tensor:
    """dice + cross-entropy (MONAI's DiceCELoss with both lambdas 1): the dice word plus the cross-entropy word."""
    var dice = seg_dice_loss(tape, ex, logits, target, C, S, reduction)
    var ce = seg_channel_cross_entropy(tape, ex, logits, target, C, S, reduction)
    return tape.add(ex, rn_as_shape(tape, ex, dice, ShapeN.matrix(1, 1)), ce)
