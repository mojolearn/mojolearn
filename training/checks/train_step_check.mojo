# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of `mojolearn.identical.train.step.fp32.v1`.

**THIS BANNER WAS FALSE AND IS CORRECTED. THIS GATE HAS RUN: ALL CLAUSES
GREEN ON ONE DEVICE.** Until 2026-08-31 this header read "NOTHING IN THIS FILE
HAS EVER BEEN COMPILED OR EXECUTED", added that no compiler had read it, no
device had run it and no bit produced by it had been observed, said the same of
`train_loop.mojo` and `training/TRAINING_LOOP_PLAN.md`, and closed by calling
every "passes" and every "fires" below a PREDICTION. **Commit `5ce6eb17` is
the first execution and it falsified all of that.** Thirteen stages compared
device against host oracle BITWISE, four negative controls all firing,
parameters measurably moved everywhere, a checkpoint at step 4 resumed through
step 8 reproducing eight continuous steps exactly, and the same digest twice.
Written 2026-08-25 by the training-loop lane, DEVIATIONS 1550 through 1589.

**READ THE CLAUSES BELOW AS MEASURED ON ONE DEVICE AND UNRUN ON THE OTHER
TWO.** The run's own summary is the honest one: "ALL CLAUSES GREEN on ONE
device. That makes the cross-vendor run worth paying for; it does not replace
it, and it does not make the two ungated stages correct on any other vendor."
There is still no ledger of results in this file, because a fabricated ledger
reads exactly like evidence; the ledger is the commit.

WHY THIS FILE IS SHAPED THE WAY IT IS
--------------------------------------
Running gates for the first time on 2026-08-24 and 2026-08-25 found, across
five lanes, four fixtures that could not separate anything, two controls that
could not fire, one inverted claim, three contract clauses with no
implementation, and two real kernel defects where a promised refusal existed
only in the oracle. **A training loop is the single easiest place in this
repository to produce a meaningless green**, because parameters that do not
move, a digest over the wrong buffer, a loop that silently runs zero steps,
and a seed that is the same because it was never used ALL yield three
matching checkpoint hashes across three vendors.

So this file is built to say so. Plan section 4.1 enumerates twelve ways to
get a vacuous green and names the assertion for each; the seven clauses below
are those assertions. **A run that produced no divergence where one was
required reports VACUOUS and not PASS.**

THE ORDER THIS MUST BE RUN IN
------------------------------
Clause (a) -- one step against the composed host oracle, on ONE device --
comes before any GPU is rented. Two of the twelve stages of a step have no
gate of their own (`transformer/checks/transformer_backward.mojo` and
`embedding/checks/embedding_identical.mojo`), so a cross-vendor agreement
without clause (a) would show only that three machines compute the same
thing, which is compatible with all three being wrong.

THE ENVIRONMENT
----------------
    MOJOLEARN_TRAIN_STEPS       N for the multi-step clauses (default 8)
    MOJOLEARN_TRAIN_SEED        the one integer the data comes from
    MOJOLEARN_TRAIN_EXPECT_ARM  the misspelled-selector guard
    MOJOLEARN_IDENTITY_TRACE    where clause (e) writes and re-reads a card

WHAT THIS FILE IS LEAST CONFIDENT COMPILES
-------------------------------------------
  1. `transformer_block_backward_oracle`'s and `transformer_block_oracle`'s
     `TransformerWeights` / `TransformerDims`, which are a DIFFERENT PAIR of
     structs from the device path's `LlamaDims` / `LlamaDeviceWeights`.
     `assert_shape_structs_agree` exists because two shape structs describing
     one model is a place to put a different number in each (DEVIATION 1552).
  2. Reading a trace file back and counting its records (clause (e)). The
     file format is five TAB-separated fields with `#` comments, per
     `core/identity_trace.mojo`, and this file parses it by walking bytes
     because `s[:n]` is refused in this toolchain.
  3. `optimizer_step_oracle` takes FIVE `mut` arguments and clips `grad` in
     place. The locals below are transferred rather than assigned.
"""

from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import FNV_OFFSET, IdentityTrace, fnv1a64_bytes

from gemm.checks.gemm_oracle import OP_NT, gemm_oracle
from gemm.checks.gemm_backward import (
    BWD_DC_LEFT,
    gemm_backward_a_call,
    gemm_backward_b_call,
    gemm_backward_sabotage_name,
)
from gemm.checks.gemm_identical import gemm_sabotage_name

from embedding.checks.embedding_oracle import (
    EmbConfig,
    emb_backward_oracle,
    emb_forward_oracle,
)

from training.checks.loss import loss_sabotage_name
from training.checks.loss_oracle import (
    CeConfig,
    ce_backward_oracle,
    ce_forward_oracle,
)
from training.checks.optimizer import optimizer_sabotage_name
from training.checks.optimizer_oracle import optimizer_step_oracle

from transformer.checks.transformer_fixture import (
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    fixture_splitmix64,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import (
    transformer_block_backward_oracle,
)
from transformer.impl.transformers.models.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaRopeTable,
    llama_block_sabotage_name,
)
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    llama_backward_sabotage_name,
)

from training.checks.train_loop import (
    ARM_DATA_REVERSE,
    ARM_NONE,
    ARM_SEED_PLUS_ONE,
    ARM_ULP,
    ARM_ZERO_LR,
    PID_EMBED,
    PID_LM_HEAD,
    PID_NORM1,
    PID_NORM2,
    PID_W_DOWN,
    PID_W_GATE,
    PID_W_K,
    PID_W_O,
    PID_W_Q,
    PID_W_UP,
    PID_W_V,
    SEED_BASE,
    TRAIN_B,
    TRAIN_D_MODEL,
    TRAIN_HEAD_DIM,
    TRAIN_INTERMEDIATE,
    TRAIN_J,
    TRAIN_L,
    TRAIN_N_HEADS,
    TRAIN_N_KV,
    TRAIN_RMS_EPS,
    TRAIN_ROPE_POSITIONS,
    TRAIN_ROPE_THETA,
    TRAIN_VOCAB,
    TrainBuffers,
    TrainConfig,
    arm_name,
    batch_inputs,
    batch_targets,
    download_f32,
    env_int,
    env_str,
    env_u64,
    hex16,
    mode_banner,
    param_id_count,
    param_id_name,
    resume_training,
    run_training,
    train_batch_ids,
    train_dims,
    train_init_params,
    train_init_tensor,
    train_n_total,
    train_offsets,
    train_splitmix64,
    train_step,
    unpack_params,
)


comptime M_ROWS = TRAIN_B * TRAIN_L
comptime DEFAULT_TRACE = "/tmp/mojolearn_train_step.trace"

def card_path() -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else this check's
    own scratch path.

    DEVIATION 1939, 2026-08-28. Same precedence as
    `isolation_forest/checks/if_check.mojo::card_path`. This lane built a
    complete card and wrote it to a hardcoded path no harness collects, so it
    reported `NO CARD written` in every round while its own gate went green.

    ONLY THE PRIMARY CARD MOVES. The second path in this check is the
    run-to-run CONTROL, and pointing it at the harness too would overwrite
    the card with it.
    """
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(DEFAULT_TRACE)




# ===========================================================================
# BITWISE COMPARISON
# ===========================================================================
# **BY BITS AND NEVER BY A TOLERANCE.** A tolerance turns an identity gate
# into an accuracy gate, and the whole claim of this repository is the
# difference between the two.


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def f32_report(v: Float32) -> String:
    """`<decimal>/<hex bits>`. `[[mojo-string-float-roundtrip]]`: the decimal
    does not round trip in this toolchain, so the HEX is the value and the
    decimal is for the reader."""
    return String(v) + "/" + hex16(UInt64(bits_of(v)))


struct StageVerdict(Copyable, Movable):
    var name: String
    var cells: Int
    var moved: Int
    var first: Int

    def __init__(out self, name: String):
        self.name = name
        self.cells = 0
        self.moved = 0
        self.first = -1


def compare_bits(
    name: String, host: List[Float32], dev: List[Float32]
) raises -> StageVerdict:
    """Cell for cell, by bit pattern. A length mismatch is a FAILURE and not
    a truncated comparison -- comparing the shorter prefix of two different
    shapes is how a gate reports agreement about a stage that does not
    exist."""
    var v = StageVerdict(name)
    if len(host) != len(dev):
        raise Error(
            String("train_step_check: stage '")
            + name
            + "' host has "
            + String(len(host))
            + " cells and device has "
            + String(len(dev))
            + ". A length mismatch is a defect, not a shorter comparison."
        )
    v.cells = len(host)
    for i in range(len(host)):
        if bits_of(host[i]) != bits_of(dev[i]):
            v.moved += 1
            if v.first < 0:
                v.first = i
    return v^


def report_stage(v: StageVerdict, host: List[Float32], dev: List[Float32]):
    if v.moved == 0:
        print("    " + v.name + "  " + String(v.cells) + " cells  IDENTICAL")
        return
    print(
        "    "
        + v.name
        + "  "
        + String(v.cells)
        + " cells  **"
        + String(v.moved)
        + " MOVED**, first at "
        + String(v.first)
        + "  host "
        + f32_report(host[v.first])
        + "  device "
        + f32_report(dev[v.first])
    )


def slice_of(xs: List[Float32], begin: Int, count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(xs[begin + i])
    return out^


# ===========================================================================
# THE TWO SHAPE STRUCTS (DEVIATION 1552)
# ===========================================================================


def oracle_dims() -> TransformerDims:
    """The oracle half's shape struct, built from the SAME integers."""
    return TransformerDims(
        TRAIN_D_MODEL,
        TRAIN_N_HEADS,
        TRAIN_N_KV,
        TRAIN_HEAD_DIM,
        TRAIN_INTERMEDIATE,
        TRAIN_ROPE_POSITIONS,
    )


def assert_shape_structs_agree() raises:
    """`LlamaDims` and `TransformerDims` describe ONE model in TWO structs.

    The device path uses `LlamaDims` (in `transformer/impl/`) and the
    oracle path uses `TransformerDims` (in `transformer/checks/`), because
    the two halves of that lane were written by different agents and
    deliberately do not import each other. **Two shape structs describing one
    model is a place to put a different number in each**, and the resulting
    defect is a card of plausible, in-bounds, wrong numbers -- the `OP_NT`
    trap's class. Asserted field for field before anything runs.
    """
    var a = train_dims()
    var b = oracle_dims()
    a.validate()
    b.validate()
    if a.d_model != b.d_model:
        raise Error("train_step_check: d_model disagrees between the two shape structs")
    if a.n_heads != b.n_heads:
        raise Error("train_step_check: n_heads disagrees between the two shape structs")
    if a.n_kv != b.n_kv_heads:
        raise Error("train_step_check: n_kv disagrees between the two shape structs")
    if a.head_dim != b.head_dim:
        raise Error("train_step_check: head_dim disagrees between the two shape structs")
    if a.intermediate != b.intermediate:
        raise Error(
            "train_step_check: intermediate disagrees between the two shape structs"
        )
    if a.q_width() != b.q_width() or a.kv_width() != b.kv_width():
        raise Error(
            "train_step_check: the derived widths disagree between the two"
            " shape structs"
        )


def oracle_weights(seed: UInt64) raises -> TransformerWeights:
    """The block's nine parameters, from the SAME initializer the device
    path uses. Not a second generator."""
    var w = TransformerWeights(oracle_dims())
    w.norm1_w = train_init_tensor(seed, PID_NORM1)
    w.norm2_w = train_init_tensor(seed, PID_NORM2)
    w.w_q = train_init_tensor(seed, PID_W_Q)
    w.w_k = train_init_tensor(seed, PID_W_K)
    w.w_v = train_init_tensor(seed, PID_W_V)
    w.w_o = train_init_tensor(seed, PID_W_O)
    w.w_gate = train_init_tensor(seed, PID_W_GATE)
    w.w_up = train_init_tensor(seed, PID_W_UP)
    w.w_down = train_init_tensor(seed, PID_W_DOWN)
    return w^


def device_weights(
    ctx: DeviceContext, seed: UInt64
) raises -> LlamaDeviceWeights:
    return LlamaDeviceWeights(
        ctx,
        train_dims(),
        TRAIN_RMS_EPS,
        train_init_tensor(seed, PID_NORM1),
        train_init_tensor(seed, PID_NORM2),
        train_init_tensor(seed, PID_W_Q),
        train_init_tensor(seed, PID_W_K),
        train_init_tensor(seed, PID_W_V),
        train_init_tensor(seed, PID_W_O),
        train_init_tensor(seed, PID_W_GATE),
        train_init_tensor(seed, PID_W_UP),
        train_init_tensor(seed, PID_W_DOWN),
    )


# ===========================================================================
# THE COMPOSED HOST ORACLE
# ===========================================================================


struct HostStep(Movable):
    """Every intermediate of one host step, in the step's order.

    **EVERY ONE IS COMPARED, not only `param_out`.** A disagreement in the
    final parameters tells you a step is wrong and nothing about where, and
    localization is the entire reason `core/identity_trace.mojo` exists.
    """

    var x: List[Float32]  # 2. embed
    var h: List[Float32]  # 3. block forward, residual2
    var logits: List[Float32]  # 4. lm_head
    var loss: List[Float32]  # 5. loss, [1]
    var dlogits: List[Float32]  # 6. loss backward
    var d_h: List[Float32]  # 7. lm_head backward, dA
    var dw_lm: List[Float32]  # 7. lm_head backward, dB
    var d_x: List[Float32]  # 8. block backward
    var dw_emb: List[Float32]  # 9. embedding backward
    var grad: List[Float32]  # 10. packed flat gradient
    var param: List[Float32]  # 11. optimizer, param out
    var m_state: List[Float32]
    var v_state: List[Float32]

    def __init__(out self):
        self.x = List[Float32]()
        self.h = List[Float32]()
        self.logits = List[Float32]()
        self.loss = List[Float32]()
        self.dlogits = List[Float32]()
        self.d_h = List[Float32]()
        self.dw_lm = List[Float32]()
        self.d_x = List[Float32]()
        self.dw_emb = List[Float32]()
        self.grad = List[Float32]()
        self.param = List[Float32]()
        self.m_state = List[Float32]()
        self.v_state = List[Float32]()


def _oracle_gemm_backward_a(
    dc: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """`dA`, through `gemm_backward_a_call`'s OWN table.

    **THE SHAPES ARE NOT DERIVED HERE** (DEVIATION 1574). The device
    launcher asks `gemm_backward_a_call` which op, which `(m, n, k)` and
    which operand side; this asks the SAME function and feeds the answer to
    `gemm_oracle`. A hand-derived transpose here would be a second spelling
    of the routing, and the routing is exactly what `SAB_BWD_UNTRANSPOSED`
    and `SAB_BWD_OPERAND_ORDER` exist to falsify. Two spellings means the
    gate could agree with itself while both are wrong.
    """
    var call = gemm_backward_a_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return gemm_oracle(dc, b, call[0], call[1], call[2], call[3])
    return gemm_oracle(b, dc, call[0], call[1], call[2], call[3])


def _oracle_gemm_backward_b(
    dc: List[Float32],
    a: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """`dB`, through `gemm_backward_b_call`'s own table. Same argument."""
    var call = gemm_backward_b_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return gemm_oracle(dc, a, call[0], call[1], call[2], call[3])
    return gemm_oracle(a, dc, call[0], call[1], call[2], call[3])


def host_step(
    seed: UInt64,
    cfg: TrainConfig,
    ids: List[Int32],
    t: Int,
    param_in: List[Float32],
    m_in: List[Float32],
    v_in: List[Float32],
) raises -> HostStep:
    """ONE step, entirely on the host, composed of the EXISTING oracles.

    No arithmetic is written here. Every line delegates to a function that
    is already the normative answer of some profile, which is the only way
    this file can be evidence about the device rather than about itself.
    """
    comptime DM = TRAIN_D_MODEL
    comptime V = TRAIN_VOCAB
    var offs = train_offsets()
    var out = HostStep()

    var emb_cfg = EmbConfig.llama(V, DM)
    var ce_cfg = CeConfig.causal_lm(V)
    var inputs = batch_inputs(ids)
    var targets = batch_targets(ids)

    # ---- 2. embed --------------------------------------------------------
    var emb_w = slice_of(param_in, offs[PID_EMBED], param_id_count(PID_EMBED))
    out.x = emb_forward_oracle(emb_w, inputs, emb_cfg)

    # ---- 3. block forward -------------------------------------------------
    var tdims = oracle_dims()
    var tw = TransformerWeights(tdims)
    tw.norm1_w = slice_of(param_in, offs[PID_NORM1], param_id_count(PID_NORM1))
    tw.norm2_w = slice_of(param_in, offs[PID_NORM2], param_id_count(PID_NORM2))
    tw.w_q = slice_of(param_in, offs[PID_W_Q], param_id_count(PID_W_Q))
    tw.w_k = slice_of(param_in, offs[PID_W_K], param_id_count(PID_W_K))
    tw.w_v = slice_of(param_in, offs[PID_W_V], param_id_count(PID_W_V))
    tw.w_o = slice_of(param_in, offs[PID_W_O], param_id_count(PID_W_O))
    tw.w_gate = slice_of(param_in, offs[PID_W_GATE], param_id_count(PID_W_GATE))
    tw.w_up = slice_of(param_in, offs[PID_W_UP], param_id_count(PID_W_UP))
    tw.w_down = slice_of(param_in, offs[PID_W_DOWN], param_id_count(PID_W_DOWN))

    var cache = TransformerKVCache(TRAIN_B, tdims, TRAIN_L)
    var rope = build_rope_table(tdims)
    var fwd = transformer_block_oracle(
        tw, out.x, TRAIN_B, TRAIN_L, cache, rope, ScorePlant.none()
    )
    out.h = fwd.residual2_out.copy()

    # ---- 4. lm_head -------------------------------------------------------
    var lm_w = slice_of(
        param_in, offs[PID_LM_HEAD], param_id_count(PID_LM_HEAD)
    )
    out.logits = gemm_oracle(out.h, lm_w, OP_NT, M_ROWS, V, DM)

    # ---- 5 and 6. loss ----------------------------------------------------
    var st = ce_forward_oracle(out.logits, targets, ce_cfg)
    ce_backward_oracle(st, targets, ce_cfg)
    out.loss = st.loss.copy()
    out.dlogits = st.dlogits.copy()

    # ---- 7. lm_head backward ----------------------------------------------
    out.d_h = _oracle_gemm_backward_a(
        out.dlogits, lm_w, OP_NT, M_ROWS, V, DM
    )
    out.dw_lm = _oracle_gemm_backward_b(
        out.dlogits, out.h, OP_NT, M_ROWS, V, DM
    )

    # ---- 8. block backward -------------------------------------------------
    var bwd = transformer_block_backward_oracle(
        tw, fwd, out.d_h, TRAIN_B, TRAIN_L, 0, rope
    )
    out.d_x = bwd.d_x.copy()

    # ---- 9. embedding backward ---------------------------------------------
    # `accumulate` is FALSE, so `dw_prev` is unread and an empty list is the
    # honest argument. `emb_backward_seed` is what decides that and it is the
    # oracle's own function.
    out.dw_emb = emb_backward_oracle(
        out.d_x, inputs, emb_cfg, List[Float32]()
    )

    # ---- 10. pack ----------------------------------------------------------
    # **A THIRD SPELLING OF THE LAYOUT.** `train_offsets`, `pack_grads` and
    # this loop must agree, and clause (f) is what checks that they do. Owed
    # item 11 in the plan is a single table all three consume.
    var g = List[Float32]()
    for i in range(len(out.dw_emb)):
        g.append(out.dw_emb[i])
    for i in range(len(bwd.dw_norm1)):
        g.append(bwd.dw_norm1[i])
    for i in range(len(bwd.dw_q)):
        g.append(bwd.dw_q[i])
    for i in range(len(bwd.dw_k)):
        g.append(bwd.dw_k[i])
    for i in range(len(bwd.dw_v)):
        g.append(bwd.dw_v[i])
    for i in range(len(bwd.dw_o)):
        g.append(bwd.dw_o[i])
    for i in range(len(bwd.dw_norm2)):
        g.append(bwd.dw_norm2[i])
    for i in range(len(bwd.dw_gate)):
        g.append(bwd.dw_gate[i])
    for i in range(len(bwd.dw_up)):
        g.append(bwd.dw_up[i])
    for i in range(len(bwd.dw_down)):
        g.append(bwd.dw_down[i])
    for i in range(len(out.dw_lm)):
        g.append(out.dw_lm[i])

    if len(g) != train_n_total():
        raise Error(
            String("train_step_check: the host pack produced ")
            + String(len(g))
            + " gradient cells and the layout wants "
            + String(train_n_total())
            + ". The three spellings of the parameter layout disagree."
        )

    # ---- 11. optimizer ------------------------------------------------------
    var p = param_in.copy()
    var mm = m_in.copy()
    var vv = v_in.copy()
    var flags = List[Bool]()
    for _ in range(TRAIN_J):
        flags.append(False)
    var ostages = optimizer_step_oracle(
        p, g, mm, vv, flags, offs, cfg.optimizer(), t
    )
    _ = ostages^
    # `optimizer_step_oracle` CLIPS `grad` IN PLACE, so `g` after the call is
    # the CLIPPED gradient and that is what the device's flat `grad` holds
    # after `identical_optimizer_step` too. Comparing the pre-clip value
    # against the post-clip buffer would report a difference that is not one.
    out.grad = g^
    out.param = p^
    out.m_state = mm^
    out.v_state = vv^
    return out^


# ===========================================================================
# CLAUSE (a): ONE STEP, DEVICE vs COMPOSED HOST ORACLE, BITWISE
# ===========================================================================


def clause_a(ctx: DeviceContext, seed: UInt64) raises -> Int:
    """Returns the number of stages that MOVED. Zero is the required answer.

    **THIS IS THE ONLY CLAUSE THAT ESTABLISHES CORRECTNESS.** Everything
    else in this file establishes that the harness is not lying about
    identity, which is a different and weaker thing.
    """
    print("clause (a): one step, every intermediate, device vs host oracle, BITWISE")

    var n = train_n_total()
    var cfg = TrainConfig.for_arm(1, seed, ARM_NONE)
    var ids = train_batch_ids(seed, 1)

    var tb = TrainBuffers(ctx, seed)
    var dims = train_dims()
    var w = device_weights(ctx, seed)
    var rope = LlamaRopeTable(
        ctx, dims, TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS
    )
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, dims)
    var off_trace = IdentityTrace.disabled()

    unpack_params(ctx, tb, w)
    var loss = train_step(
        ctx, tb, w, rope, stages, bst, off_trace, cfg, ids, 1
    )

    comptime DM = TRAIN_D_MODEL
    comptime V = TRAIN_VOCAB
    var d_x = download_f32(ctx, tb.x, M_ROWS * DM)
    var d_h_out = download_f32(ctx, stages.residual2, M_ROWS * DM)
    var d_logits = download_f32(ctx, tb.logits, M_ROWS * V)
    var d_loss = download_f32(ctx, tb.ce_loss, 1)
    var d_dlogits = download_f32(ctx, tb.ce_dlogits, M_ROWS * V)
    var d_dh = download_f32(ctx, tb.d_h, M_ROWS * DM)
    var d_dwlm = download_f32(ctx, tb.dw_lm, V * DM)
    var d_dx = download_f32(ctx, bst.d_x, M_ROWS * DM)
    var d_dwemb = download_f32(ctx, tb.dw_emb, V * DM)
    var d_grad = download_f32(ctx, tb.grad, n)
    var d_param = download_f32(ctx, tb.param, n)
    var d_m = download_f32(ctx, tb.m_state, n)
    var d_v = download_f32(ctx, tb.v_state, n)

    var zeros = List[Float32]()
    for _ in range(n):
        zeros.append(Float32(0.0))
    var hs = host_step(seed, cfg, ids, 1, train_init_params(seed), zeros, zeros)

    var moved_stages = 0

    var v1 = compare_bits("2. embed.out", hs.x, d_x)
    report_stage(v1, hs.x, d_x)
    if v1.moved > 0:
        moved_stages += 1
    var v2 = compare_bits("3. block.residual2", hs.h, d_h_out)
    report_stage(v2, hs.h, d_h_out)
    if v2.moved > 0:
        moved_stages += 1
    var v3 = compare_bits("4. lm_head.logits", hs.logits, d_logits)
    report_stage(v3, hs.logits, d_logits)
    if v3.moved > 0:
        moved_stages += 1
    var v4 = compare_bits("5. ce.loss", hs.loss, d_loss)
    report_stage(v4, hs.loss, d_loss)
    if v4.moved > 0:
        moved_stages += 1
    var v5 = compare_bits("6. ce.dlogits", hs.dlogits, d_dlogits)
    report_stage(v5, hs.dlogits, d_dlogits)
    if v5.moved > 0:
        moved_stages += 1
    var v6 = compare_bits("7. head.d_h", hs.d_h, d_dh)
    report_stage(v6, hs.d_h, d_dh)
    if v6.moved > 0:
        moved_stages += 1
    var v7 = compare_bits("7. head.dw_lm", hs.dw_lm, d_dwlm)
    report_stage(v7, hs.dw_lm, d_dwlm)
    if v7.moved > 0:
        moved_stages += 1
    var v8 = compare_bits("8. block.d_x", hs.d_x, d_dx)
    report_stage(v8, hs.d_x, d_dx)
    if v8.moved > 0:
        moved_stages += 1
    var v9 = compare_bits("9. embed.dw", hs.dw_emb, d_dwemb)
    report_stage(v9, hs.dw_emb, d_dwemb)
    if v9.moved > 0:
        moved_stages += 1
    var v10 = compare_bits("10. grad.flat", hs.grad, d_grad)
    report_stage(v10, hs.grad, d_grad)
    if v10.moved > 0:
        moved_stages += 1
    var v11 = compare_bits("11. param.out", hs.param, d_param)
    report_stage(v11, hs.param, d_param)
    if v11.moved > 0:
        moved_stages += 1
    var v12 = compare_bits("11. adam.m", hs.m_state, d_m)
    report_stage(v12, hs.m_state, d_m)
    if v12.moved > 0:
        moved_stages += 1
    var v13 = compare_bits("11. adam.v", hs.v_state, d_v)
    report_stage(v13, hs.v_state, d_v)
    if v13.moved > 0:
        moved_stages += 1

    print(
        "  device loss "
        + f32_report(loss)
        + "   host loss "
        + f32_report(hs.loss[0])
    )
    if moved_stages == 0:
        print("  clause (a) GREEN: thirteen stages identical")
    else:
        print(
            "  clause (a) **FAILED**: "
            + String(moved_stages)
            + " stages moved. The FIRST one in the list above is the"
            + " address; everything after it is downstream."
        )
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    return moved_stages


# ===========================================================================
# CLAUSE (b): THE NEGATIVE CONTROLS
# ===========================================================================


def clause_b(ctx: DeviceContext, seed: UInt64, steps: Int) raises -> Int:
    """N1 through N4. Returns the number of FAILED requirements.

    **A run that produced no divergence where one was required is VACUOUS
    and not a pass.** Plan section 4.2.
    """
    print(
        "clause (b): the negative controls, N = " + String(steps) + " steps"
    )
    var failures = 0
    var off1 = IdentityTrace.disabled()
    var off2 = IdentityTrace.disabled()

    var clean = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), off1, off2
    )
    print("  clean      h_all=" + hex16(clean.final.h_all))

    # ---- N1 -------------------------------------------------------------
    var n1 = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_SEED_PLUS_ONE), off1, off2
    )
    print("  N1 seed    h_all=" + hex16(n1.final.h_all))
    if n1.final.h_all == clean.final.h_all:
        failures += 1
        print(
            "    **FAILED**: the seed does not reach the data. Every"
            " machine would generate the same constant batch, and three"
            " matching digests would mean nothing (plan V4)."
        )

    # ---- N2 -------------------------------------------------------------
    # REFUSED rather than reported at N == 1: reversing a one-element
    # sequence is the identity, so the arm cannot fire (DEVIATION 1560).
    if steps < 2:
        print(
            "  N2 order   SKIPPED and NOT PASSED: at N = 1 reversing the"
            " step order is the identity, so this control CANNOT FIRE. A"
            " control that cannot fire is not a control that passed. Run"
            " at N >= 2."
        )
        failures += 1
    else:
        var n2 = run_training(
            ctx, TrainConfig.for_arm(steps, seed, ARM_DATA_REVERSE), off1, off2
        )
        print("  N2 order   h_all=" + hex16(n2.final.h_all))
        if n2.final.h_all == clean.final.h_all:
            failures += 1
            print(
                "    **FAILED**: the step index does not reach the data"
                " generator, or the optimizer's update is order free. Every"
                " step trains on the same batch (plan V5)."
            )

    # ---- N3 -------------------------------------------------------------
    var n3 = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_ULP), off1, off2
    )
    print("  N3 ulp     h_all=" + hex16(n3.final.h_all))
    if n3.final.h_all == clean.final.h_all:
        failures += 1
        print(
            "    **FAILED**: a one-ULP change to the last float of w_v did"
            " not reach the checkpoint. The digest reads a buffer the model"
            " does not compute with (plan V3)."
        )
    # DEVIATION 1561. THE ARM MUST ALSO MOVE THE LOSS. If the digest moved
    # but the loss did not, the perturbation reached the checkpoint by being
    # COPIED there rather than by being computed with, and the arm has
    # demonstrated the opposite of what it was armed for
    # (`[[reached-but-inert]]` pointed at a control).
    # DEVIATION 1580: THIS HALF IS REPORTED, NOT ASSERTED, AND A SWEEP IS
    # WHY. N3 asserted two different things at once -- that the DIGEST reads
    # live memory, and that the perturbed weight REACHES THE LOSS. The first
    # is true at every magnitude (`h_all` moved on all three runs below). The
    # second was settled by sweeping `MOJOLEARN_TRAIN_ULPS` on 2026-08-25:
    #
    #       1 ULP           step 1 loss bit-identical
    #       1,000 ULPs      step 1 loss bit-identical
    #       100,000,000     step 1 loss MOVED
    #
    # So the element IS read and is NOT dead code. Its LEVERAGE is below FP32
    # resolution: the last float of `w_v` feeds one element of one value
    # vector for one KV head, and after o_proj, the residual, the norm, the
    # MLP, the head and a softmax, a 1.2e-4 relative change on it does not
    # survive to a scalar loss.
    #
    # ASSERTING IT WOULD FAIL A CORRECT KERNEL, which is the same defect the
    # transformer backward plan's clause (c) list carried. The digest half
    # above stays ASSERTED and is what N3 exists for.
    #
    # OWED: a leverage-aware target. Perturb a weight whose influence on the
    # loss is O(1) -- a `norm1` scale or an `lm_head` row -- or assert on an
    # intermediate stage rather than on the scalar loss.
    if len(n3.losses) > 0 and len(clean.losses) > 0:
        if bits_of(n3.losses[0]) == bits_of(clean.losses[0]):
            print(
                "    REPORTED, NOT ASSERTED (DEVIATION 1580): the digest"
                " moved and step 1's loss did"
                " NOT. clean "
                + f32_report(clean.losses[0])
                + " vs ulp "
                + f32_report(n3.losses[0])
                + ". The perturbation was carried into the checkpoint"
                + " without ever being computed with."
            )
        else:
            print(
                "    loss moved as required: "
                + f32_report(clean.losses[0])
                + " -> "
                + f32_report(n3.losses[0])
            )

    # ---- N4 -------------------------------------------------------------
    var n4 = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_ZERO_LR), off1, off2
    )
    print(
        "  N4 zerolr  param="
        + hex16(n4.final.h_param)
        + " m="
        + hex16(n4.final.h_m)
        + " v="
        + hex16(n4.final.h_v)
    )
    if n4.final.h_param != n4.initial.h_param:
        failures += 1
        print(
            "    **FAILED**: at lr = 0 with weight_decay = 0 the parameters"
            " must be UNCHANGED bit for bit (decay_mul is exactly 1 and"
            " step_size is exactly +0.0). Something outside the optimizer"
            " is writing parameters, or lr did not reach the kernel."
        )
    if n4.final.h_m == n4.initial.h_m:
        failures += 1
        print(
            "    **FAILED**: `m` did not change at lr = 0. The moment"
            " buffers are not being written at all, and N6 leans on them."
        )
    if n4.final.h_v == n4.initial.h_v:
        failures += 1
        print("    **FAILED**: `v` did not change at lr = 0. Same reason.")

    if failures == 0:
        print("  clause (b) GREEN: four controls, every requirement met")
    return failures


# ===========================================================================
# CLAUSE (c): MOVEMENT -- THE VACUITY CONTROL
# ===========================================================================


def clause_c(ctx: DeviceContext, seed: UInt64, steps: Int) raises -> Int:
    """N5. **This is the clause that stops three machines agreeing that
    nothing happened.**"""
    print("clause (c): movement, N = " + String(steps))
    var failures = 0
    var off1 = IdentityTrace.disabled()
    var off2 = IdentityTrace.disabled()
    var run = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), off1, off2
    )

    if run.steps_run != steps:
        print(
            "  **FAILED**: steps_run = "
            + String(run.steps_run)
            + " and N = "
            + String(steps)
            + " (plan V1)"
        )
        return failures + 1

    var n = len(run.param0)
    var changed = 0
    for i in range(n):
        if bits_of(run.param0[i]) != bits_of(run.paramN[i]):
            changed += 1
    print(
        "  moved " + String(changed) + "/" + String(n) + " parameter cells"
    )
    # 1. the global floor.
    if changed * 100 < n * 99:
        failures += 1
        print(
            "    **FAILED**: fewer than 99 percent of parameters changed"
            " bits. A run whose parameters barely move produces the same"
            " digest on any hardware (plan V2)."
        )

    # 2 and 3. PER TENSOR. A whole-model check passes with ten of eleven
    # tensors frozen, and a frozen tensor is what a structurally zero
    # gradient looks like.
    var offs = train_offsets()
    for j in range(TRAIN_J):
        var lo = offs[j]
        var hi = offs[j + 1]
        var moved = 0
        var worst = Float32(0.0)
        for i in range(lo, hi):
            if bits_of(run.param0[i]) != bits_of(run.paramN[i]):
                moved += 1
            var dv = run.paramN[i] - run.param0[i]
            if dv < Float32(0.0):
                dv = -dv
            if dv > worst:
                worst = dv
        print(
            "    p"
            + String(j)
            + " "
            + param_id_name(j)
            + " moved "
            + String(moved)
            + "/"
            + String(hi - lo)
            + " max|dp| "
            + f32_report(worst)
        )
        if moved == 0:
            failures += 1
            print(
                "      **FAILED**: tensor "
                + param_id_name(j)
                + " is completely frozen. Its gradient is structurally"
                + " zero and every digest that includes it is carrying a"
                + " constant."
            )
        if worst < Float32(1e-4):
            failures += 1
            print(
                "      **FAILED**: max|dp| below the 1e-4 floor. Plan"
                " section 5.3 predicts about 8e-3 after 8 AdamW steps at"
                " lr = 1e-3; a value near the floor means the paper"
                " arithmetic was wrong and the fixture is weak."
            )

    # 4. THE EMBEDDING SEPARATION (DEVIATION 1562). Weight decay moves every
    # row, present or absent, so "the embedding moved" is satisfied by decay
    # alone and proves nothing about the embedding GRADIENT.
    var ids1 = batch_inputs(train_batch_ids(seed, 1))
    var present = List[Bool]()
    for _ in range(TRAIN_VOCAB):
        present.append(False)
    for i in range(len(ids1)):
        present[Int(ids1[i])] = True
    var row_present = -1
    var row_absent = -1
    for vtok in range(TRAIN_VOCAB):
        if present[vtok] and row_present < 0:
            row_present = vtok
        if not present[vtok] and row_absent < 0:
            row_absent = vtok
    if row_present < 0 or row_absent < 0:
        failures += 1
        print(
            "    **FAILED**: step 1's batch has no absent token (or no"
            " present one) at V = "
            + String(TRAIN_VOCAB)
            + ", so the embedding separation cannot be evaluated. That is"
            + " NOT the same as it passing."
        )
    else:
        comptime DM = TRAIN_D_MODEL
        var base = offs[PID_EMBED]
        var dp_present = Float32(0.0)
        var dp_absent = Float32(0.0)
        for jcol in range(DM):
            var ip = base + row_present * DM + jcol
            var ia = base + row_absent * DM + jcol
            var a = run.paramN[ip] - run.param0[ip]
            if a < Float32(0.0):
                a = -a
            if a > dp_present:
                dp_present = a
            var b = run.paramN[ia] - run.param0[ia]
            if b < Float32(0.0):
                b = -b
            if b > dp_absent:
                dp_absent = b
        print(
            "    embed row "
            + String(row_present)
            + " (present) max|dp| "
            + f32_report(dp_present)
            + "   row "
            + String(row_absent)
            + " (absent) max|dp| "
            + f32_report(dp_absent)
        )
        if bits_of(dp_present) == bits_of(dp_absent):
            failures += 1
            print(
                "      **FAILED**: a row the batch used and a row it never"
                " touched moved by the SAME amount, so the motion is weight"
                " decay alone and the embedding gradient contributed"
                " nothing. Stage 9 is inert while the digest changes every"
                " step."
            )

    if failures == 0:
        print("  clause (c) GREEN: the parameters moved, everywhere, measurably")
    return failures


# ===========================================================================
# CLAUSE (d): STEP-COUNT COMPOSITION
# ===========================================================================


def clause_d(ctx: DeviceContext, seed: UInt64, steps: Int) raises -> Int:
    """N6. `first + rest == N` must equal one uninterrupted run of `N`.

    **THIS IS A RESUME WITHIN ONE PROCESS AND NOT FROM A FILE.** There is no
    checkpoint serialization in v1, so `OPT_SAB_RESUME_REINIT`'s real form is
    not reachable here and this clause does not test it.
    """
    print("clause (d): step-count composition")
    if steps < 2:
        print(
            "  SKIPPED and NOT PASSED: a split needs N >= 2. At N = 1 there"
            " is nothing to compose and the clause cannot fire."
        )
        return 1
    var first = steps // 2
    if first < 1:
        first = 1
    var off1 = IdentityTrace.disabled()
    var off2 = IdentityTrace.disabled()
    var whole = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), off1, off2
    )
    var split = resume_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), first, off1, off2
    )
    print(
        "  whole ("
        + String(steps)
        + ") h_all="
        + hex16(whole.final.h_all)
        + "   split ("
        + String(first)
        + "+"
        + String(steps - first)
        + ") h_all="
        + hex16(split.h_all)
    )
    if whole.final.h_all != split.h_all:
        print(
            "  **FAILED**: `t` did not continue across the split, or the"
            " momentum flags reset, or the data generator restarted its"
            " stream (plan V6)."
        )
        return 1
    print("  clause (d) GREEN")
    return 0


# ===========================================================================
# CLAUSE (e): HYGIENE
# ===========================================================================


def count_trace_records(path: String) raises -> Int:
    """Non-comment lines in a trace file.

    Parsed by walking bytes: `s[:n]` is refused in this toolchain and
    `len(String)` is unsupported, so `len(s.as_bytes())` and a manual scan
    are the spellings that exist.
    """
    var text = String("")
    with open(path, "r") as fh:
        text = fh.read()
    var b = text.as_bytes()
    var n = len(b)
    var count = 0
    var at_line_start = True
    var is_comment = False
    var saw_content = False
    for i in range(n):
        var c = Int(b[i])
        if at_line_start:
            is_comment = c == 35  # '#'
            saw_content = False
            at_line_start = False
        if c == 10:
            if not is_comment and saw_content:
                count += 1
            at_line_start = True
            continue
        if c != 13:
            saw_content = True
    if not at_line_start and not is_comment and saw_content:
        count += 1
    return count


def clause_e(ctx: DeviceContext, seed: UInt64, steps: Int) raises -> Int:
    """Record count, empty trace, hash agreement, splitmix agreement.

    **A trace with zero records compared against a trace with zero records
    is three machines agreeing about nothing**, and that failure is
    otherwise completely silent (plan V9).
    """
    print("clause (e): hygiene")
    var failures = 0

    # ---- the two copies of splitmix64 must agree ------------------------
    # `train_loop.mojo` COPIES the four lines rather than importing them, for
    # DEVIATION 1000's reason. The copy's only possible failure is that the
    # two are edited apart, and this is the cheap assertion that catches it.
    for i in range(8):
        var z = UInt64(i) * UInt64(0x0123456789ABCDEF)
        if train_splitmix64(z) != fixture_splitmix64(z):
            failures += 1
            print(
                "  **FAILED**: train_splitmix64 and fixture_splitmix64"
                " disagree at input "
                + String(i)
                + ". The two copies have been edited apart."
            )

    # ---- `hex16` must agree with the trace's own formatter ---------------
    if hex16(FNV_OFFSET) != "cbf29ce484222325":
        failures += 1
        print(
            "  **FAILED**: hex16 does not render the FNV offset basis"
            " correctly, so every printed digest is misread even when the"
            " underlying value is right."
        )

    # ---- the digest must be a pure function of the bits -------------------
    # Two identical host lists must hash identically, and one changed cell
    # must change the hash. Trivial, and it is the assertion that fails if
    # `fnv1a64_bytes` is ever given the wrong byte count.
    var a = List[Float32]()
    for i in range(64):
        a.append(Float32(i))
    var b2 = a.copy()
    var pa = a.unsafe_ptr().bitcast[UInt8]()
    var pb = b2.unsafe_ptr().bitcast[UInt8]()
    if fnv1a64_bytes(FNV_OFFSET, pa, 256) != fnv1a64_bytes(
        FNV_OFFSET, pb, 256
    ):
        failures += 1
        print("  **FAILED**: the digest is not a function of the bytes alone")
    b2[63] = Float32(999.0)
    var pb2 = b2.unsafe_ptr().bitcast[UInt8]()
    if fnv1a64_bytes(FNV_OFFSET, pa, 256) == fnv1a64_bytes(
        FNV_OFFSET, pb2, 256
    ):
        failures += 1
        print("  **FAILED**: the digest did not see a changed cell")
    _ = a
    _ = b2

    # ---- the record count -------------------------------------------------
    var path = card_path()
    var trace = IdentityTrace.to_path(path)
    var off2 = IdentityTrace.disabled()
    var run = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), trace, off2
    )
    _ = trace^
    var got = count_trace_records(path)
    var want = 2 + 4 * steps
    print(
        "  trace records "
        + String(got)
        + ", required "
        + String(want)
        + " (2 at step 0, then data.ids + loss + ckpt.tensors +"
        " ckpt.digest per step)"
    )
    if got == 0:
        failures += 1
        print(
            "  **FAILED**: the trace is EMPTY. A comparison of two empty"
            " traces reports agreement and means nothing (plan V9)."
        )
    elif got != want:
        failures += 1
        print(
            "  **FAILED**: the record count is wrong, so two traces of"
            " different lengths would be ALIGNED WRONGLY by"
            " tools/identity_trace_diff.py and it would pair one run's"
            " stage against another run's different stage."
        )

    if run.steps_run != steps:
        failures += 1
        print(
            "  **FAILED**: steps_run = "
            + String(run.steps_run)
            + ", asked for "
            + String(steps)
            + " (plan V1)"
        )

    if failures == 0:
        print("  clause (e) GREEN")
    return failures


# ===========================================================================
# CLAUSE (f): THE OFFSET TABLE
# ===========================================================================


def _check_slice(
    ctx: DeviceContext,
    planted: List[Float32],
    offs: List[Int],
    j: Int,
    mut dst: DeviceBuffer[DType.float32],
) raises -> Int:
    """One destination buffer against the slice the offset table names.

    Returns 1 on a mismatch and 0 otherwise, so the caller can count.
    """
    var count = param_id_count(j)
    var want = slice_of(planted, offs[j], count)
    var got = download_f32(ctx, dst, count)
    var v = compare_bits(param_id_name(j), want, got)
    if v.moved == 0:
        return 0
    print(
        "  **FAILED**: unpack put the wrong data in "
        + param_id_name(j)
        + ", "
        + String(v.moved)
        + "/"
        + String(count)
        + " cells, first at "
        + String(v.first)
        + "  want "
        + f32_report(want[v.first])
        + "  got "
        + f32_report(got[v.first])
    )
    return 1


def clause_f(ctx: DeviceContext) raises -> Int:
    """The layout, three ways, must be one layout.

    A wrong offset gives plausible, in-bounds, wrong numbers that are
    IDENTICAL on all three vendors, so **no cross-vendor comparison can see
    it** (plan V7). This is the only clause that can.
    """
    print("clause (f): the parameter layout")
    var failures = 0
    var offs = train_offsets()
    var n = train_n_total()

    if len(offs) != TRAIN_J + 1:
        failures += 1
        print(
            "  **FAILED**: offsets has "
            + String(len(offs))
            + " entries and J + 1 is "
            + String(TRAIN_J + 1)
        )
        return failures
    if offs[0] != 0 or offs[TRAIN_J] != n:
        failures += 1
        print("  **FAILED**: offsets does not span [0, n_total)")
    for j in range(TRAIN_J):
        if offs[j + 1] - offs[j] != param_id_count(j):
            failures += 1
            print(
                "  **FAILED**: offsets disagrees with param_id_count at j = "
                + String(j)
                + " ("
                + param_id_name(j)
                + ")"
            )
        if offs[j + 1] <= offs[j]:
            failures += 1
            print(
                "  **FAILED**: offsets is not strictly increasing at j = "
                + String(j)
            )

    # ---- the DESTINATION buffers must be the size the table says ----------
    # This is the strongest cheap assertion available: it checks the offset
    # table against the allocation `LlamaDeviceWeights` made from `LlamaDims`,
    # which is an INDEPENDENT computation of the same eleven sizes.
    var w = device_weights(ctx, SEED_BASE)
    if len(w.norm1_w) != param_id_count(PID_NORM1):
        failures += 1
        print("  **FAILED**: norm1_w allocation disagrees with the layout")
    if len(w.norm2_w) != param_id_count(PID_NORM2):
        failures += 1
        print("  **FAILED**: norm2_w allocation disagrees with the layout")
    if len(w.w_q) != param_id_count(PID_W_Q):
        failures += 1
        print("  **FAILED**: w_q allocation disagrees with the layout")
    if len(w.w_k) != param_id_count(PID_W_K):
        failures += 1
        print("  **FAILED**: w_k allocation disagrees with the layout")
    if len(w.w_v) != param_id_count(PID_W_V):
        failures += 1
        print("  **FAILED**: w_v allocation disagrees with the layout")
    if len(w.w_o) != param_id_count(PID_W_O):
        failures += 1
        print("  **FAILED**: w_o allocation disagrees with the layout")
    if len(w.w_gate) != param_id_count(PID_W_GATE):
        failures += 1
        print("  **FAILED**: w_gate allocation disagrees with the layout")
    if len(w.w_up) != param_id_count(PID_W_UP):
        failures += 1
        print("  **FAILED**: w_up allocation disagrees with the layout")
    if len(w.w_down) != param_id_count(PID_W_DOWN):
        failures += 1
        print("  **FAILED**: w_down allocation disagrees with the layout")

    # ---- the round trip, on DISTINCT values -------------------------------
    # **A UNIFORM FILL WOULD PASS THIS WITH EVERY OFFSET WRONG.**
    # `[[uniform-test-data-hides-permutation]]` applied to a copy. The fill
    # below is `f32_from_bits(0x3F800000 + i)`, which is distinct per cell
    # and normal at every index this shape reaches.
    var tb = TrainBuffers(ctx, SEED_BASE)
    var planted = List[Float32]()
    for i in range(n):
        planted.append(bitcast[DType.float32](UInt32(0x3F800000) + UInt32(i)))
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        hbuf.unsafe_ptr().unsafe_store(i, planted[i])
    ctx.enqueue_copy(dst_buf=tb.param, src_ptr=hbuf.unsafe_ptr())
    ctx.synchronize()
    _ = hbuf

    unpack_params(ctx, tb, w)

    # Read the eleven destinations back and check each against the slice the
    # table says it should hold. This is a THIRD spelling of the layout and
    # the plan's owed item 11 is a single shared table; until then, the
    # spellings are checked against each other rather than trusted.
    # Eleven explicit comparisons rather than a `List[List[Float32]]`.
    # Nothing copies implicitly in this toolchain and a list of lists is one
    # more thing this lane cannot check without a compiler.
    var bad = 0
    bad += _check_slice(
        ctx, planted, offs, PID_EMBED, tb.emb_w
    )
    bad += _check_slice(ctx, planted, offs, PID_NORM1, w.norm1_w)
    bad += _check_slice(ctx, planted, offs, PID_W_Q, w.w_q)
    bad += _check_slice(ctx, planted, offs, PID_W_K, w.w_k)
    bad += _check_slice(ctx, planted, offs, PID_W_V, w.w_v)
    bad += _check_slice(ctx, planted, offs, PID_W_O, w.w_o)
    bad += _check_slice(ctx, planted, offs, PID_NORM2, w.norm2_w)
    bad += _check_slice(ctx, planted, offs, PID_W_GATE, w.w_gate)
    bad += _check_slice(ctx, planted, offs, PID_W_UP, w.w_up)
    bad += _check_slice(ctx, planted, offs, PID_W_DOWN, w.w_down)
    bad += _check_slice(ctx, planted, offs, PID_LM_HEAD, tb.lm_w)
    if bad > 0:
        failures += bad
    else:
        print("  unpack round trip clean on distinct planted values")

    _ = tb^
    _ = w^
    if failures == 0:
        print("  clause (f) GREEN")
    return failures


# ===========================================================================
# CLAUSE (g): SAME-DEVICE DETERMINISM
# ===========================================================================


def clause_g(ctx: DeviceContext, seed: UInt64, steps: Int) raises -> Int:
    """The same run twice, in one process, must produce the same digests.

    **THIS IS THE ONE FAILURE A CROSS-VENDOR COMPARISON CANNOT SEE.**
    Uninitialized memory folded into a digest differs on ONE machine, so it
    differs on all three and looks like a divergence rather than like the
    instrument being broken. Plan V11, and it is why every count in the
    digest is `n_total` and never `len(buf)`.
    """
    print("clause (g): same-device determinism")
    var off1 = IdentityTrace.disabled()
    var off2 = IdentityTrace.disabled()
    var a = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), off1, off2
    )
    var b = run_training(
        ctx, TrainConfig.for_arm(steps, seed, ARM_NONE), off1, off2
    )
    var failures = 0
    if a.final.h_param != b.final.h_param:
        failures += 1
        print("  **FAILED**: h_param differs between two runs on ONE device")
    if a.final.h_m != b.final.h_m:
        failures += 1
        print("  **FAILED**: h_m differs between two runs on ONE device")
    if a.final.h_v != b.final.h_v:
        failures += 1
        print("  **FAILED**: h_v differs between two runs on ONE device")
    if a.final.h_all != b.final.h_all:
        failures += 1
        print(
            "  **FAILED**: h_all differs between two runs on ONE device."
            " Something outside the parameter bits is reaching the digest"
            " -- uninitialized tail memory, an allocation size, or a"
            " pointer. No cross-vendor comparison could have told you this."
        )
    if failures == 0:
        print("  clause (g) GREEN: " + hex16(a.final.h_all) + " twice")
    return failures


# ===========================================================================
# MAIN
# ===========================================================================


def main() raises:
    var steps = env_int("MOJOLEARN_TRAIN_STEPS", 8)
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)

    print(
        "=== training-step identity gate, profile"
        " mojolearn.identical.train.step.fp32.v1"
    )
    print(
        "=== NOTHING IN THIS FILE, IN train_loop.mojo OR IN"
        " TRAINING_LOOP_PLAN.md HAD EVER BEEN COMPILED OR RUN BEFORE THIS"
        " PROCESS. Every prediction in their headers is unfalsified."
    )
    print(
        "=== TWO OF THE TWELVE STAGES OF A STEP HAVE NO GATE OF THEIR OWN:"
        " transformer/checks/transformer_backward.mojo and"
        " embedding/checks/embedding_identical.mojo. Clause (a) below is"
        " the ONLY thing in this repository that has ever compared either"
        " of them to its oracle."
    )
    print(
        "mode "
        + mode_banner()
        + "   gemm: "
        + gemm_sabotage_name()
        + "   gemm_backward: "
        + gemm_backward_sabotage_name()
        + "   llama_fwd: "
        + llama_block_sabotage_name()
        + "   llama_bwd: "
        + llama_backward_sabotage_name()
        + "   loss: "
        + loss_sabotage_name()
        + "   optimizer: "
        + optimizer_sabotage_name()
    )
    print(
        "steps N = "
        + String(steps)
        + "   seed = "
        + String(seed)
        + "   n_total = "
        + String(train_n_total())
        + "   J = "
        + String(TRAIN_J)
    )

    # V10: a misspelled selector must never read as a clean run.
    var expect = env_str("MOJOLEARN_TRAIN_EXPECT_ARM")
    if expect != "" and expect != arm_name(ARM_NONE):
        raise Error(
            String("train_step_check: MOJOLEARN_TRAIN_EXPECT_ARM is '")
            + expect
            + "' but this gate drives the arms itself and must run from a"
            + " CLEAN binary. An arm compiled or selected underneath the"
            + " gate would make every control's verdict meaningless."
        )

    if steps < 2:
        print(
            "WARNING: at N < 2 the data-order control (b/N2) and the"
            " composition clause (d) CANNOT FIRE, and both are reported as"
            " FAILURES rather than as passes. N = 1 is for clause (a)"
            " alone. Plan DEVIATION 1560."
        )

    assert_shape_structs_agree()
    print("shape structs agree (LlamaDims vs TransformerDims)")

    var ctx = DeviceContext()

    var fa = clause_a(ctx, seed)
    var ff = clause_f(ctx)
    var fb = clause_b(ctx, seed, steps)
    var fc = clause_c(ctx, seed, steps)
    var fd = clause_d(ctx, seed, steps)
    var fe = clause_e(ctx, seed, steps)
    var fg = clause_g(ctx, seed, steps)

    var total = fa + fb + fc + fd + fe + ff + fg
    print("")
    print("=== SUMMARY")
    print("  (a) oracle       " + String(fa) + " stages moved")
    print("  (b) controls     " + String(fb) + " requirements failed")
    print("  (c) movement     " + String(fc) + " failed")
    print("  (d) composition  " + String(fd) + " failed")
    print("  (e) hygiene      " + String(fe) + " failed")
    print("  (f) layout       " + String(ff) + " failed")
    print("  (g) determinism  " + String(fg) + " failed")

    if total != 0:
        raise Error(
            String("train_step_check: ")
            + String(total)
            + " failures. See the clause reports above."
        )
    if steps < 2:
        print(
            "VACUOUS: every clause that ran is green, but at N = 1 the"
            " data-order control and the composition clause could not fire."
            " This is NOT a pass and must not be recorded as one."
        )
        raise Error(
            "train_step_check: refusing to report a pass at N < 2"
            " (DEVIATION 1560)"
        )
    print(
        "ALL CLAUSES GREEN on ONE device. That makes the cross-vendor run"
        " worth paying for; it does not replace it, and it does not make"
        " the two ungated stages correct on any other vendor."
    )
