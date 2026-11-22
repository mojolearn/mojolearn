# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The transformer decoder-layer backward at a shape whose products run on
the packed GEMM plan: device words against host words.

`transformer/checks/transformer_backward_check.mojo` compares the 37 stages
at fixtures of d_model 32 to 48, far below the 128 x 128 packed tile. This
check runs one layer at 2,048 tokens and widths in the hundreds, where the
weight gradients and the input gradients of the projections take the packed
body, so the packed launches are what computes the compared words. The host
side is the oracle every host caller runs
(`transformer_block_backward_oracle`). (The concatenated fan-in arm
MOJOLEARN_GEMM_FAST_DX_CONCAT, the work-list arm MOJOLEARN_GEMM_FAST_WORKLIST
and the transformer-backward use of the store epilogue arm
MOJOLEARN_GEMM_FAST_EPI were deleted 2026-11-19, docs/TOMBSTONES.md.)

Cases (each: one host forward and backward, two device runs):
  0 q, k and v of one width (n_kv_heads = n_heads).
  1 grouped-query attention (n_kv_heads = n_heads / 2): fan-in members of
    unequal width.
Device run A is the production form (no trace, nothing materialized) and
compares what a training step reads: the nine weight gradients and d(x).
Device run B materializes the stages and also compares the two fan-in joins,
d(norm2.out) (gate + up) and d(norm1.out) (q, k, v).

The shape is chosen at run time: the first candidate whose main products
take the 128 x 128 plan on the device in front of it (else the first).

What a sabotage changes here: a fan-in joined in the wrong operand order or
twice (d(norm2.out), d(norm1.out), and everything downstream to d(x)); any
host/device difference in leaves or fold tree at these contraction lengths.

Every host-device transfer of this file goes through the kit.
"""
from std.testing import TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import (
    PLAN_TUNED_128_8X8,
    choose_gemm_plan,
)
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    ROPE_THETA,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    fixture_case_seed,
    fixture_tensor,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import (
    backward_oracle_dump,
    transformer_block_backward_oracle,
)
from transformer.impl.llama.modeling_llama import (
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    llama_decoder_layer_forward_planted,
)
# TOMBSTONE: MOJOLEARN_GEMM_FAST_DX_CONCAT (NOISE, A/B round 1 NVIDIA L40S: lm-train-step 1.00, samba-train-step 0.99) its decline tests here deleted 2026-11-19 by lane/fx-verdicts3; code recoverable at 913fc44cfa.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_DX_CONCAT.patch
# TOMBSTONE: MOJOLEARN_GEMM_FAST_WORKLIST (NOISE, A/B round 1 NVIDIA L40S, job nvc-ab-20261119T125554Z-6f7afc: lm-train-step 1.00, transformer-decode 1.00, lm-forward 0.93 once = another base run of the row) its not-vacuous shape test (gemm_worklist_mode per product) deleted 2026-11-19 by lane/fx-arm-cleanup; code recoverable at a7c89e7558.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_WORKLIST.patch
# TOMBSTONE: MOJOLEARN_GEMM_FAST_EPI, its transformer-backward use (NOISE, A/B round 1 NVIDIA L40S, job nvc-ab-20261119T125554Z-6f7afc: lm-train-step 0.98 = the row's other bases, samba-train-step 1.05) its not-vacuous shape test (_arm_reach, _ARM_GUARDED) deleted 2026-11-19 by lane/fx-arm-cleanup; code recoverable at a7c89e7558.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_EPI-transformer-backward.patch
from autograd.kernels.transformer import LlamaBackwardStages
from autograd.rules_transformer import llama_decoder_layer_backward

from checks.native_tests.kit import download, same_words, upload
from checks.native_tests.testsuite_cases import require


comptime TBP_CASES = 2
comptime TBP_CANDIDATES = 4
comptime TBP_L = 256
# Tensor ids of this file's fixture (transformer_fixture's are 1 to 11, the
# backward check's d_out is 21).
comptime _TID_X = 31
comptime _TID_D_OUT = 32
comptime _TID_W = 33

# The card indices compared (transformer_backward_oracle.backward_stage_tag).
comptime _S_DW_DOWN = 3
comptime _S_DW_GATE = 7
comptime _S_DW_UP = 8
comptime _S_D_NORM2_OUT = 9
comptime _S_DW_NORM2 = 11
comptime _S_DW_O = 16
comptime _S_DW_Q = 29
comptime _S_DW_K = 30
comptime _S_DW_V = 31
comptime _S_D_NORM1_OUT = 32
comptime _S_DW_NORM1 = 34
comptime _S_D_X = 36


def _cand_batch(which: Int) -> Int:
    """B of candidate `which` (L is TBP_L for every candidate)."""
    return 16 if which >= 2 else 8


def _cand_dims(which: Int, gqa: Bool) raises -> TransformerDims:
    """The layer shape of candidate `which`: token counts (B x TBP_L) and
    widths around a small language model's training step. Several, because
    which products a column runs on the packed body is the plan chooser's
    answer for the device in front of it, not something this file may
    assume."""
    var dm = 384
    var heads = 6
    var inter = 768
    if which == 1:
        dm = 512
        heads = 8
        inter = 1024
    elif which == 2:
        inter = 1536
    elif which == 3:
        dm = 768
        heads = 12
        inter = 1536
    var kv = heads
    if gqa:
        kv = heads // 2
    var dims = TransformerDims(dm, heads, kv, dm // heads, inter, TBP_L)
    dims.validate()
    return dims^


def _packed(m: Int, dims: TransformerDims) -> Bool:
    """Whether the default dispatch gives the main products the 128 x 128 plan."""
    var dm = dims.d_model
    var inter = dims.intermediate
    return (choose_gemm_plan(m, dm, inter) == PLAN_TUNED_128_8X8
            and choose_gemm_plan(inter, dm, m) == PLAN_TUNED_128_8X8)


def _pick(gqa: Bool) raises -> Int:
    """The candidate this build runs (module docstring): the first whose
    main products take the 128 x 128 plan, else the first."""
    for which in range(TBP_CANDIDATES):  # small-loop(4 candidate shapes): plan questions only
        var dims = _cand_dims(which, gqa)
        var m = _cand_batch(which) * TBP_L
        if _packed(m, dims):
            return which
    return 0


def _weights(dims: TransformerDims, seed: UInt64) -> TransformerWeights:
    """Projections at about 1 / sqrt(fan-in), so the attention rows stay
    soft (a saturated softmax sends zeros back through q and k and the
    fan-in members would not be told apart)."""
    var w = TransformerWeights(dims)
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    w.norm1_w = fixture_tensor(seed, _TID_W, dm, 0.5, 1.5)
    w.norm2_w = fixture_tensor(seed, _TID_W + 1, dm, 0.5, 1.5)
    w.w_q = fixture_tensor(seed, _TID_W + 2, qw * dm, -0.05, 0.05)
    w.w_k = fixture_tensor(seed, _TID_W + 3, kw * dm, -0.05, 0.05)
    w.w_v = fixture_tensor(seed, _TID_W + 4, kw * dm, -0.05, 0.05)
    w.w_o = fixture_tensor(seed, _TID_W + 5, dm * qw, -0.05, 0.05)
    w.w_gate = fixture_tensor(seed, _TID_W + 6, inter * dm, -0.05, 0.05)
    w.w_up = fixture_tensor(seed, _TID_W + 7, inter * dm, -0.05, 0.05)
    w.w_down = fixture_tensor(seed, _TID_W + 8, dm * inter, -0.05, 0.05)
    return w^


def _host_stages(
    w: TransformerWeights, dims: TransformerDims, x: List[Float32], d_out: List[Float32], b: Int, l: Int
) raises -> List[List[Float32]]:
    """The oracle's 37 stages, card order (the backward check's
    `run_host_backward`, full causal attention)."""
    var cache = TransformerKVCache(b, dims, l, 0)
    var rope = build_rope_table(dims)
    var plant = ScorePlant.none()
    var fwd = transformer_block_oracle(w, x, b, l, cache, rope, plant)
    var st = transformer_block_backward_oracle(w, fwd, d_out, b, l, 0, rope, 0)
    var stages = backward_oracle_dump(st)
    _ = st^
    _ = fwd^
    return stages^


def _compare(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], host: List[List[Float32]], stage: Int, what: String
) raises:
    var got = download[DType.float32](ctx, buf, len(host[stage]))
    same_words[DType.float32](got, host[stage], what)


def _device_run(
    ctx: DeviceContext, w: TransformerWeights, dims: TransformerDims, x: List[Float32], d_out: List[Float32],
    host: List[List[Float32]], b: Int, l: Int, materialize: Bool, what: String,
) raises:
    """One forward and one backward on the device from fresh buffers (the
    backward check's `run_device_backward`), then the comparisons."""
    var ldims = LlamaDims(dims.d_model, dims.n_heads, dims.n_kv_heads, dims.head_dim, dims.intermediate)
    var dw = LlamaDeviceWeights(
        ctx, ldims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v, w.w_o, w.w_gate, w.w_up, w.w_down,
    )
    var kv = LlamaKVCache(ctx, b, ldims, l, 0)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, b, l, l, ldims, 0)
    var dx = upload[DType.float32](ctx, x)
    var off = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, stages, kv, rope, dw, dx, b, l, 0, PLANT_AT_NONE, List[Int](), List[UInt32](), off, "fwd",
    )
    var bst = LlamaBackwardStages(ctx, b, l, l, ldims)
    var quiet = IdentityTrace.disabled()
    llama_decoder_layer_backward(
        ctx, bst, stages, dw, rope.cos, rope.sin, dx, d_out, b, l, 0, quiet, "tbp", materialize=materialize,
    )
    _compare(ctx, bst.dw_down, host, _S_DW_DOWN, what + " dW down")
    _compare(ctx, bst.dw_gate, host, _S_DW_GATE, what + " dW gate")
    _compare(ctx, bst.dw_up, host, _S_DW_UP, what + " dW up")
    _compare(ctx, bst.dw_norm2, host, _S_DW_NORM2, what + " dW norm2")
    _compare(ctx, bst.dw_o, host, _S_DW_O, what + " dW o")
    _compare(ctx, bst.dw_q, host, _S_DW_Q, what + " dW q")
    _compare(ctx, bst.dw_k, host, _S_DW_K, what + " dW k")
    _compare(ctx, bst.dw_v, host, _S_DW_V, what + " dW v")
    _compare(ctx, bst.dw_norm1, host, _S_DW_NORM1, what + " dW norm1")
    if materialize:
        _compare(ctx, bst.d_norm2_out, host, _S_D_NORM2_OUT, what + " d(norm2.out), the gate + up join")
        _compare(ctx, bst.d_norm1_out, host, _S_D_NORM1_OUT, what + " d(norm1.out), the q, k, v join")
    _compare(ctx, bst.d_x, host, _S_D_X, what + " d(x)")
    _ = bst^
    _ = stages^
    _ = dw^
    _ = kv^
    _ = rope^
    _ = dx^


def _case(ctx: DeviceContext, which: Int) raises:
    var gqa = which == 1
    var cand = _pick(gqa)
    var b = _cand_batch(cand)
    var l = TBP_L
    var dims = _cand_dims(cand, gqa)
    var seed = fixture_case_seed(100 + which)
    var n = b * l * dims.d_model
    var w = _weights(dims, seed)
    var x = fixture_tensor(seed, _TID_X, n, -2.0, 2.0)
    var d_out = fixture_tensor(seed, _TID_D_OUT, n, -1.0, 1.0)
    var host = _host_stages(w, dims, x, d_out, b, l)
    require(len(host) == 37, "the oracle dump is not the 37-stage card")
    var name = (String("case ") + String(which) + " (B " + String(b) + ", L " + String(l) + ", d_model "
                + String(dims.d_model) + ", kv heads " + String(dims.n_kv_heads) + ", intermediate "
                + String(dims.intermediate) + ")")
    _device_run(ctx, w, dims, x, d_out, host, b, l, False, name + " production run:")
    _device_run(ctx, w, dims, x, d_out, host, b, l, True, name + " materialized run:")


def test_cases() raises:
    var ctx = DeviceContext()
    var failed = String("")
    for which in range(TBP_CASES):  # driver-loop: over the check's two layer shapes, one device queue
        try:
            _case(ctx, which)
        except problem:
            failed += "\n      transformer_backward_packed row " + String(which) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("transformer_backward_packed: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
