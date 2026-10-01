# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cross-entropy oracle's bits, rows over host tasks (lane neural-pass6).

`ce_forward_oracle` and `ce_backward_oracle` (`training/checks/loss_oracle.mojo`)
are THE normative answer and stay scalar, single threaded, appending every
stage to a list. The byte LM host training step ran them at [512, 8192], and
that was 90 ms of the step's wall on one core while 27 others waited.

This file computes the same bits with the same statements, in a different
SCHEDULE only:

  - a row's chain (its max, its shifts, its exponentials, its log-denominator,
    its target log-probability, its gradient cells) is a pure function of that
    row's logits and target, so rows run on host tasks through
    `host_parallelize` (the calling thread's floating-point environment on
    every task, DEVIATION 5900);
  - the ONE cross-cell fold inside a row, the denominator, is `ce_fold`'s:
    gemm v1's leaf-and-tree chain over the vocabulary. `ce_fold` makes one
    `gemm_host_rows` call per row at `(1, 1, V)`; this file makes one call at
    `(N, 1, V)`, and a gemm v1 cell is a pure function of its row, its column
    and `k` (DEVIATION 807), so cell `(i, 0)` here is `ce_fold`'s cell for row
    `i`. No task starts a parallel region: the denominators are folded between
    the two row passes, on the calling thread;
  - the cross-row fold, the total, is `ce_fold` over the row losses, the
    oracle's own call;
  - the stage lists the oracle appends (`shift`, `expo`, `weights`, ...) are not
    materialized except `expo`, which the backward reads; `shift[base + y]` is
    recomputed by the statement that produced it;
  - (lane neural-pass8) inside a row, `HOST_FW` cells advance together through
    the lane twins of the scalar seams (`expf_lanes` for `identical_exp`,
    `div_lanes` for `identical_div`, `ftz_lanes` for `ftz`), each equal to its
    scalar on every bit pattern; the row's tail cells take the scalar
    statements.

Label smoothing (`cfg.eps != 0`) and `REDUCTION_NONE` are refused here; the
caller runs the oracle for those. `tools/byte_lm_cpu_train_gate.py cpu` is the
gate: the step's loss bits and gradient against the three-vendor capture.
"""

from std.math import min

from checks.numerics import ftz, identical_div, identical_exp, identical_log
from core.host_lanes import (
    F32V,
    HOST_FW,
    div_lanes,
    expf_lanes,
    ftz_lanes,
    host_f32_uninit,
    host_row_tasks,
    lanes_are_identical,
)
from core.host_parallel import host_parallelize
from gemm.checks.gemm_oracle import OP_NN
from gemm.host.gemm_host_rows import gemm_host_rows
from training.checks.loss_oracle import (
    CeConfig,
    REDUCTION_NONE,
    _row_max,
    ce_count,
    ce_divisor,
    ce_fold,
    ce_ones,
    ce_refuse_inputs,
    ce_smoothing_targets,
    neg_by_bits,
)


def ce_host_rows(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> Tuple[Float32, List[Float32]]:
    """`(loss, dlogits)`: `ce_forward_oracle`'s `loss[0]` and
    `ce_backward_oracle`'s `dlogits`, rows over host tasks (module note)."""
    if cfg.smoothing_is_spelled():
        raise Error("ce_host_rows: label smoothing takes the oracle path")
    if cfg.reduction == REDUCTION_NONE:
        raise Error("ce_host_rows: REDUCTION_NONE has no backward")
    var n = ce_refuse_inputs(logits, targets, cfg)
    var v = cfg.vocab
    var ignore = cfg.ignore_index
    var tv = ce_smoothing_targets(cfg.eps, v)
    var t_target = tv[0]
    var t_other = tv[1]

    # ---- L1-L3 per row: the max, the shift, the exponential ---------------
    var max_v = List[Float32](length=n, fill=Float32(0.0))
    var expo = host_f32_uninit(n * v)
    var tasks = host_row_tasks(n, 4 * v)
    var chunk = (n + tasks - 1) // tasks
    var lp = logits.unsafe_ptr()
    var ep = expo.unsafe_ptr()
    # LANES (lane neural-pass8): `expf_lanes` equals `identical_exp` on every
    # bit pattern (byte_lm_host_exp_check), `ftz_lanes` equals `ftz`; the
    # row's tail cells take the scalar statements.
    def _expo_rows(t: Int) {imm logits, imm lp, imm ep, mut max_v, imm n, imm v, imm chunk}:
        for i in range(t * chunk, min((t + 1) * chunk, n)):
            var base = i * v
            var m = _row_max(logits, base, v)
            max_v[i] = m
            var vv = 0
            comptime if lanes_are_identical:
                var mv = ftz_lanes(F32V(m))
                while vv + HOST_FW <= v:
                    ep.unsafe_store(base + vv, expf_lanes(ftz_lanes(ftz_lanes(lp.unsafe_load[width=HOST_FW](base + vv)) - mv)))
                    vv += HOST_FW
            while vv < v:
                var s = ftz(ftz(lp.unsafe_load(base + vv)) - ftz(m))
                ep.unsafe_store(base + vv, identical_exp(s))
                vv += 1
    if tasks <= 1:
        _expo_rows(0)
    else:
        host_parallelize(_expo_rows, tasks)

    # ---- L4, the denominators: every row's `ce_fold` in one gemm call ------
    var ones_v = ce_ones(v)
    var denom = gemm_host_rows(expo, ones_v, OP_NN, n, 1, v)

    # ---- L5-L7, L11 per row: log-denominator, target log-prob, nll ---------
    var row = List[Float32](length=n, fill=Float32(0.0))
    def _loss_rows(t: Int) {imm logits, imm targets, imm max_v, imm denom, mut row, imm n, imm v, imm chunk, imm ignore}:
        for i in range(t * chunk, min((t + 1) * chunk, n)):
            var base = i * v
            var y = Int(targets[i])
            var ignored = y == ignore
            var logdenom = ftz(identical_log(ftz(denom[i])))
            var ty = y
            if ignored:
                ty = 0
            var shift_y = ftz(ftz(logits[base + ty]) - ftz(max_v[i]))
            var lp_y = ftz(ftz(shift_y) - ftz(logdenom))
            var nll = neg_by_bits(lp_y)
            if ignored:
                nll = Float32(0.0)
            row[i] = nll
    if tasks <= 1:
        _loss_rows(0)
    else:
        host_parallelize(_loss_rows, tasks)

    # ---- L12, L13: the total (the oracle's own fold) and the loss ----------
    var count = ce_count(targets, ignore)
    var wide = v
    if n > wide:
        wide = n
    var ones = ce_ones(wide)
    var total = ce_fold(row, 0, n, ones)
    var divisor = ce_divisor(cfg.reduction, count, cfg.num_items)
    var loss = ftz(identical_div(ftz(total), divisor))

    # ---- L14, L16 per row: the weights and the gradient --------------------
    var dlogits = host_f32_uninit(n * v)
    var dp = dlogits.unsafe_ptr()
    # LANES (lane neural-pass8): `div_lanes` is `portable_divf` on every lane;
    # every cell of a row takes `t_other` on lanes, then the target's cell is
    # rewritten by the scalar statement with `t_target`. An ignored row is zeros.
    def _grad_rows(t: Int) {imm targets, imm ep, imm denom, imm dp, imm n, imm v, imm chunk, imm ignore, imm divisor, imm t_target, imm t_other}:
        for i in range(t * chunk, min((t + 1) * chunk, n)):
            var base = i * v
            var y = Int(targets[i])
            var ignored = y == ignore
            var dn = denom[i]
            if ignored:
                for vv in range(v):
                    dp.unsafe_store(base + vv, Float32(0.0))
                continue
            var vv = 0
            comptime if lanes_are_identical:
                var dnv = ftz_lanes(F32V(dn))
                var tov = ftz_lanes(F32V(t_other))
                var dv = F32V(divisor)
                while vv + HOST_FW <= v:
                    var w = ftz_lanes(div_lanes(ftz_lanes(ep.unsafe_load[width=HOST_FW](base + vv)), dnv))
                    dp.unsafe_store(base + vv, ftz_lanes(div_lanes(ftz_lanes(ftz_lanes(w) - tov), dv)))
                    vv += HOST_FW
            while vv < v:
                var w = ftz(identical_div(ftz(ep.unsafe_load(base + vv)), ftz(dn)))
                dp.unsafe_store(base + vv, ftz(identical_div(ftz(ftz(w) - ftz(t_other)), divisor)))
                vv += 1
            var wy = ftz(identical_div(ftz(ep.unsafe_load(base + y)), ftz(dn)))
            dp.unsafe_store(base + y, ftz(identical_div(ftz(ftz(wy) - ftz(t_target)), divisor)))
    if tasks <= 1:
        _grad_rows(0)
    else:
        host_parallelize(_grad_rows, tasks)
    # KEEP-ALIVE: `expo` is reached by `ep` above after its last use by name
    # (the denominators' gemm call), and Mojo destroys a local at that last
    # use; without this line the gradient rows read a freed block (which the
    # allocator had handed back as `dlogits`, so the bytes were right by
    # chance; lane neural-pass9).
    _ = expo^
    return (loss, dlogits^)
