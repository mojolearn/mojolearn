# SPDX-License-Identifier: Apache-2.0
"""The transformer decoder-layer backward at a shape whose products run on
the packed GEMM plan: device words against host words.

`transformer/checks/transformer_backward_check.mojo` compares the 37 stages
at fixtures of d_model 32 to 48, far below the 128 x 128 packed tile, so on
an arm build it exercises only the paths where the work-list arm
(`-D MOJOLEARN_GEMM_FAST_WORKLIST`) and the store epilogue arm
(`-D MOJOLEARN_GEMM_FAST_EPI=1`) DECLINE. This check runs one layer at
2,048 tokens and widths in the hundreds, where the weight gradients and the
input gradients of the projections take the packed body, so the pooled
launch and the fused store are what computes the compared words. The host
side is the oracle every host caller runs
(`transformer_block_backward_oracle`); WORKLIST and EPI claim unchanged
bits, so the oracle is the default one for them. (The concatenated fan-in
arm MOJOLEARN_GEMM_FAST_DX_CONCAT was deleted 2026-11-19, docs/TOMBSTONES.md.)

Cases (each: one host forward and backward, two device runs):
  0 q, k and v of one width (n_kv_heads = n_heads): the q, k, v fan-in is six
    listed products under WORKLIST.
  1 grouped-query attention (n_kv_heads = n_heads / 2): members of unequal
    width in one list.
Device run A is the production form (no trace, nothing materialized: the
products stay pending on the list until the member that completes a fan-in)
and compares what a training step reads: the nine weight gradients and d(x).
Device run B materializes the stages and also compares the two fan-in joins,
d(norm2.out) (gate + up) and d(norm1.out) (q, k, v).

NOT VACUOUS: the shape is chosen at run time, the first candidate for which
the build's own plan functions (`gemm_worklist_mode`,
`gemm_workspace_epi_takes`) say the arm's path is taken for the gate/up
fan-in and the q/k/v weight gradients. On an arm build with no such candidate the case FAILS as a fixture
error that lists every product's answer: it never passes on the declined
path. A default build has nothing to decline and runs the first candidate
whose products take the 128 x 128 plan (else the first).

What a sabotage changes here: a list launch that hands a block the wrong
product's operands, table row or first-block offset (any dW or dX word); a
grouped product folded from another product's node region; a fused store
that joins in the wrong operand order or joins twice (d(norm2.out),
d(norm1.out), and everything downstream to d(x)); a list never run for the
last fan-in member (run A's weight gradients stay unwritten); any host/device difference in leaves or
fold tree at these contraction lengths.

Every host-device transfer of this file goes through the kit.
"""
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import (
    GEMM_WORKLIST_NONE,
    PLAN_TUNED_128_8X8,
    choose_gemm_plan,
    gemm_worklist_mode,
)
from gemm.checks.gemm_epilogue import GEMM_EPI_FUSES
from gemm.neural_dispatch import NEURAL_GEMM_WORKLIST, gemm_workspace_epi_takes
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
from autograd.kernels.transformer import BWD_GEMM_EPI, LlamaBackwardStages
from autograd.rules_transformer import llama_decoder_layer_backward

from checks.native_tests.kit import download, same_words, upload
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime TBP_CASES = 2
comptime TBP_CANDIDATES = 4
comptime TBP_L = 256
#: Whether this build compiles an arm whose path a shape can decline (the
#: work list and the fused store are NVIDIA's packed body only; elsewhere the
#: defines compile to the default launches and there is nothing to decline).
comptime _ARM_GUARDED = NEURAL_GEMM_WORKLIST or (BWD_GEMM_EPI and GEMM_EPI_FUSES)

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


def _arm_reach(m: Int, dims: TransformerDims) raises -> String:
    """Empty when every arm of this build takes its path at `(m, dims)`;
    otherwise the products that decline. A forward projection `(m, n, k)`
    has its dX at GEMM `(m, k, n)` and its dW at GEMM `(n, k, m)`
    (gemm/checks/gemm_backward.mojo)."""
    var dm = dims.d_model
    var inter = dims.intermediate
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var miss = String("")
    comptime if NEURAL_GEMM_WORKLIST:
        if gemm_worklist_mode(inter, dm, m) == GEMM_WORKLIST_NONE:
            miss += " [work list: dW gate/up (" + String(inter) + ", " + String(dm) + ", " + String(m) + ") keeps its own launch]"
        if gemm_worklist_mode(qw, dm, m) == GEMM_WORKLIST_NONE:
            miss += " [work list: dW q (" + String(qw) + ", " + String(dm) + ", " + String(m) + ") keeps its own launch]"
        comptime if not BWD_GEMM_EPI:
            if gemm_worklist_mode(m, dm, inter) == GEMM_WORKLIST_NONE:
                miss += " [work list: dX gate/up (" + String(m) + ", " + String(dm) + ", " + String(inter) + ") keeps its own launch]"
    comptime if BWD_GEMM_EPI and GEMM_EPI_FUSES:
        if not gemm_workspace_epi_takes(m, dm, inter):
            miss += " [epilogue: dX gate (" + String(m) + ", " + String(dm) + ", " + String(inter) + ") declines]"
        if not gemm_workspace_epi_takes(m, dm, kw):
            miss += " [epilogue: dX k, v (" + String(m) + ", " + String(dm) + ", " + String(kw) + ") declines]"
    return miss


def _packed(m: Int, dims: TransformerDims) -> Bool:
    """Whether the default dispatch gives the main products the 128 x 128 plan."""
    var dm = dims.d_model
    var inter = dims.intermediate
    return (choose_gemm_plan(m, dm, inter) == PLAN_TUNED_128_8X8
            and choose_gemm_plan(inter, dm, m) == PLAN_TUNED_128_8X8)


def _pick(gqa: Bool) raises -> Int:
    """The candidate this build runs (module docstring, NOT VACUOUS)."""
    var said = String("")
    for which in range(TBP_CANDIDATES):  # small-loop(4 candidate shapes): plan questions only
        var dims = _cand_dims(which, gqa)
        var m = _cand_batch(which) * TBP_L
        var miss = _arm_reach(m, dims)
        if miss == "" and (_ARM_GUARDED or _packed(m, dims)):
            return which
        said += " candidate " + String(which) + ":" + miss
    if _ARM_GUARDED:
        raise Error("transformer_backward_packed: FIXTURE ERROR, no candidate shape reaches this build's arm paths;"
                    + " the check would compare only the declined path." + said)
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


def _run_gpu(mut report: Report) raises:
    var ctx = DeviceContext()
    var cases = report.cases(TBP_CASES)
    for which in range(TBP_CASES):  # driver-loop: over the check's two layer shapes, one device queue
        try:
            _case(ctx, which)
        except error:
            cases.fail(which, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("transformer_backward_packed", cpu=False, gpu=True, serial="", run=run_cases)
