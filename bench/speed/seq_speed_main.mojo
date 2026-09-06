# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOW FAST OUR SEQUENCE-MODEL PATH IS: the FAST arm of `transformer/` and
`mamba/`, one lane per process, against `tools/speed_torch_seq.py`.

    MOJOLEARN_SPEED_LANE=transformer \\
        pixi run mojo run -I . bench/speed/seq_speed_main.mojo
    MOJOLEARN_SPEED_LANE=mamba MOJOLEARN_SPEED_SIZE=smoke \\
        pixi run mojo run -I . bench/speed/seq_speed_main.mojo

**NOTHING IN THIS FILE HAS BEEN COMPILED OR RUN.** It was written on
2026-08-25 by an agent that was forbidden to run anything, against the
signatures it read in `transformer/impl/.../modeling_llama.mojo`,
`mamba/impl/.../modeling_mamba.mojo` and
`mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`. Every sentence
below that says what a number WILL be is a prediction. The first run of this
file is a BUILD, not a benchmark, and the mamba lane's FAST arm has per the
lane notes never been built on some vendors at all.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
========================================
The FAST path -- the DEFAULT build, NOT `-D MOJOLEARN_NUMERIC_IDENTICAL=1`.
That is the non-deterministic, non-bitwise-identical arm, and it is the arm
an ordinary user of this library gets. The opponent is what an NVIDIA user
would actually run: torch on CUDA, and for mamba the fused
`selective_scan_cuda` out of `mamba-ssm` when it is installable.

**IT IS NOT A MODEL AND IT IS NOT A TOKEN.** One decoder block, or one
Mamba-1 block, or one submodule of either. A served model is 32 of these
plus an embedding, a head, a sampler and a scheduler, and nothing here
measures any of that.

THREE THINGS THAT WILL DOMINATE, STATED BEFORE THE FIRST NUMBER
===============================================================
**(1) `llama_refuse_bad_inputs` and `mamba_refuse_bad_inputs` DOWNLOAD EVERY
WEIGHT TO THE HOST ON EVERY BLOCK CALL** and scan it for NaN and infinity
(`modeling_llama.mojo:2121`, `modeling_mamba.mojo:1053`; contract section 8,
DEVIATION 1027). At the Llama-8B row that is about 872 MB copied device to
host per call, plus a host loop over 218 million floats. It is inside the
block entry, so it is inside the timed region, so at the big shapes the
`transformer` lane number is a PCIe measurement with a GEMM attached. That
is an honest report of what our block entry costs today; it is not a report
of what our kernels cost. The `attention`, `mlp`, `rmsnorm` and
`selective_scan` sub-lanes call submodules that do NOT refuse, which is why
they exist, and the refusal is ALSO timed on its own and printed as an
`FSPEED-NOTE` so a reader can subtract it rather than guess at it.

**(2) OUR GEMM IS `identical_gemm` IN BOTH MODES.** `modeling_llama.mojo`
imports `gemm.checks.gemm_identical.identical_gemm` and calls it for every
projection; FAST does not swap in `linalg.matmul`, it compiles the same
pinned balanced-tree kernel with its pins removed. So the ratio this file
prints for `attention` and `mlp` is mostly the ratio between that kernel and
cuBLAS, and `bench/gemm_price_main.mojo` plus `tools/vendor_gemm_price.py`
already price exactly that. Do not read a transformer-shaped number here as
new information about the GEMM.

**(3) `eager_attention_forward` SYNCHRONIZES INSIDE A `B * n_heads` LOOP**
(`modeling_llama.mojo:2254`, and twice more per head), and materializes
seven separate score-sized buffers because contract section 6 says a lane
whose instrument is the per-stage card may not begin by fusing the stages
away. At `n_heads = 32` that is at least 32 round trips per call before any
arithmetic is counted. It is a deliberate design decision of the identity
lane and it is a real cost of the port. Say so beside the number.

THE HOUSE SHAPE THIS FILE FOLLOWS
==================================
`bench/lanes_price_main.mojo`: one lane per process chosen by environment, a
comptime mode witness, an untimed warm-up printed but never in the table, an
FNV-1a64 of the output bytes every round, and an in-process hash-stability
finding that is a REPORT under FAST and never a failure. Read that file
first; this one is its sequence-model sibling with a different line format,
because the line format here has to be shared with a Python opponent.

THE SHAPE TABLE IS THE SINGLE SOURCE OF TRUTH AND IT IS PARSED, NOT COPIED.
`tools/speed_torch_seq.py` reads the `seq_shape_*` if-ladders below out of
THIS FILE, exactly as `tools/vendor_gemm_price.py` reads `gemm_shape_*` out
of `bench/gemm_shapes.mojo`. A second hand-written copy of eleven
configurations is a table that drifts, and a drifted opponent still prints
numbers. If you add a row, add it here and nowhere else, and keep the
ladders to the plain `if i == <n>:` / `return <literal>` form the parser
recognizes -- it REFUSES a line it does not understand rather than guessing.

THE WEIGHTS ARE THE SAME ON BOTH SIDES, BY CONSTRUCTION AND BY WITNESS
======================================================================
Both sides build every tensor from the SAME hashed generator: the spec
`value = f32(lo + (hi-lo) * top24(splitmix64(key + i)) * 2^-24)` with
`key = splitmix64(seed ^ (tensor_id << 32))`, which is
`mojolearn.mamba.corpus.hash.v1` and, with different ids and ranges,
`mojolearn.transformer.fixture.hash.v1`. This file calls the LANE's own
generator (`corpus_tensor` and `fixture_tensor`); the Python side calls the
corpus generators' `hashed_unit`. Nothing is re-spelled.

That is construction, not proof, so every tensor also gets a WITNESS HASH:
FNV-1a64 over a fixed strided sample of the tensor's bytes plus its length,
printed by BOTH sides as `FSPEED-WEIGHTS`. Two sides that agree on the
witness for all ten or eleven tensors agree on the generator; a side that
drifts prints a different sixteen hex digits and the run is void. The sample
rather than the whole tensor is DEVIATION 1850 and its reason is arithmetic:
the Llama-8B row's `down_proj` alone is 58.7 million floats and FNV-1a64 is
byte-at-a-time by construction (`core/identity_trace.mojo` says why), so a
whole-tensor witness would cost more on the Python side than the benchmark
does. The sample walks the tensor at a fixed stride so a permutation, a
transposition or a wrong range still moves it.

RANGES. The transformer weights use the CORPUS's ranges
(`transformer/corpus/gen_corpus.py::BASE_RANGES` and `ranges_for`), not
`transformer_fixture.mojo::fixture_weights`'s, and the choice is not a
preference. `fixture_weights` pins `o_proj` and `down_proj` at fixed dyadic
ranges tuned for `d_model = 32`; the corpus scales those two by
`fan_in_scale`, which is what keeps the block finite at `d_model = 4096`
and `intermediate = 14336`. The seed base is this lane's own
(`SEQ_SEED_BASE`), NOT the corpus's, so no row here can be mistaken for a
corpus case: these are corpus SHAPES with speed-lane bits, and no reference
value from `transformer/corpus/` or `mamba/corpus/` applies to them.

**THE BIG SHAPES HAVE A DEGENERATE SOFTMAX AND THAT IS FINE FOR TIMING AND
NOT FINE FOR ANYTHING ELSE.** At `d_model = 4096` with `q_proj` uniform on
[-0.5, 0.5] the pre-softmax scores run to a few hundred in magnitude, so
after the max subtraction most weights underflow to exactly zero and the row
is nearly one-hot. Every cell still costs the same, so the TIMING is
unaffected, and both sides underflow identically, so `FSPEED-AGREE` is
unaffected. But nobody should read an accuracy claim out of these rows.

WHAT ONE ROUND IS
=================
One call of the lane's entry, with a `ctx.synchronize()` after it, timed on
the host with `perf_counter_ns`. Setup, allocation, weight generation,
weight upload and the input upload happen ONCE, before the loop. Anything
the entry MUTATES that it also READS is restored before each round and
OUTSIDE the timing: the Llama KV cache (contents and used length) and the
Mamba recurrent state. A round that continued the previous round's state
would be a different call every time and its milliseconds would not be
comparable with each other.

Environment:
    MOJOLEARN_SPEED_LANE    transformer | attention | mlp | rmsnorm
                            | mamba | selective_scan          (required)
    MOJOLEARN_SPEED_ROUNDS  timed rounds per shape (default 10)
    MOJOLEARN_SPEED_SIZE    shipped (default) | smoke
    MOJOLEARN_SPEED_ROW     one row of the table; default every row of the
                            lane's family
    MOJOLEARN_SPEED_DEVICE  the device name for the header line. There is no
                            device-name accessor used anywhere in this
                            repository, so this is passed in rather than
                            guessed; the Python side prints torch's own
                            `get_device_name(0)` and that is the authority.
    MOJOLEARN_SPEED_DUMP_DIR  where to write the raw float32 output of the
                            LAST round, so `tools/speed_torch_seq.py` can
                            emit `FSPEED-AGREE`. Unset means no dump and no
                            agreement line, which is a WEAKER run.

Deviations recorded by this lane: 1850 (the sampled weight witness), 1851
(the speed lane's own seed base), 1853 (the KV-cache restore between
rounds), 1854 (the mamba decode row is the mamba prefill row at L=1).
No pixi task; `bench/speed/README.md` has the procedure.
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import FNV_OFFSET, IdentityTrace, _hex16, fnv1a64_bytes
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

# ---- transformer -----------------------------------------------------------
from transformer.checks.transformer_fixture import RMS_EPS as LLAMA_RMS_EPS
from transformer.checks.transformer_fixture import ROPE_THETA as LLAMA_ROPE_THETA
from transformer.checks.transformer_fixture import TID_NORM1_W as T_TID_NORM1_W
from transformer.checks.transformer_fixture import TID_NORM2_W as T_TID_NORM2_W
from transformer.checks.transformer_fixture import TID_W_DOWN as T_TID_W_DOWN
from transformer.checks.transformer_fixture import TID_W_GATE as T_TID_W_GATE
from transformer.checks.transformer_fixture import TID_W_K as T_TID_W_K
from transformer.checks.transformer_fixture import TID_W_O as T_TID_W_O
from transformer.checks.transformer_fixture import TID_W_Q as T_TID_W_Q
from transformer.checks.transformer_fixture import TID_W_UP as T_TID_W_UP
from transformer.checks.transformer_fixture import TID_W_V as T_TID_W_V
from transformer.checks.transformer_fixture import TID_X as T_TID_X
from transformer.checks.transformer_fixture import fixture_tensor
from transformer.impl.transformers.models.llama.modeling_llama import (
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _upload,
    llama_attention_forward,
    llama_decoder_layer_forward,
    llama_mlp_forward,
    llama_refuse_bad_inputs,
    llama_rms_norm,
)

# ---- mamba -----------------------------------------------------------------
from mamba.checks.mamba_fixture import (
    D_STATE,
    MambaDims,
    MambaWeights,
    corpus_weights,
    corpus_x,
    fan_in_scale,
)
from mamba.impl.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_block_forward,
    mamba_refuse_bad_inputs,
    mamba_upload,
)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _env_str(name: String, default: String) -> String:
    var s = String(getenv(name))
    if s == "":
        return default
    return s^


# ===========================================================================
# THE SHAPE TABLE. PARSED by `tools/speed_torch_seq.py`; keep the ladders in
# the plain `if i == <n>:` / `return <literal>` form or the parser refuses.
#
# EVERY ROW CARRIES ITS PROVENANCE in `seq_shape_provenance`, for the reason
# `bench/gemm_shapes.mojo` gives: a benchmark shape with no caller behind it
# measures a number nobody will ever see.
#
# THE LADDER IS SWEPT WHOLE AND NO ROW MAY BE DROPPED BECAUSE IT FLATTERS
# NOBODY. `[[no-dataset-cherry-picking]]`. The token counts 1, 8, 128 and
# 512 are there because the ratio is known to move enormously with
# arithmetic intensity -- a pinned GEMM cost 1.02x at 2 rows and 2.33x at
# 512 rows on one H100 -- and because prefill and single-token decode are
# different regimes that a single number cannot describe.
# ===========================================================================

comptime SEQ_SHAPE_COUNT = 11

comptime FAM_LLAMA = 0
comptime FAM_MAMBA = 1


def seq_shape_family(i: Int) -> Int:
    if i == 0:
        return 0
    if i == 1:
        return 0
    if i == 2:
        return 0
    if i == 3:
        return 0
    if i == 4:
        return 0
    if i == 5:
        return 0
    return 1


def seq_shape_name(i: Int) -> String:
    if i == 0:
        return String("lane.b2_l4_d32_kv2")
    if i == 1:
        return String("llama8b.prefill.t1")
    if i == 2:
        return String("llama8b.prefill.t8")
    if i == 3:
        return String("llama8b.prefill.t128")
    if i == 4:
        return String("llama8b.prefill.t512")
    if i == 5:
        return String("llama8b.decode.t1.ctx512")
    if i == 6:
        return String("lane.b2_l4_d8")
    if i == 7:
        return String("mamba130m.prefill.t1")
    if i == 8:
        return String("mamba130m.prefill.t8")
    if i == 9:
        return String("mamba130m.prefill.t128")
    return String("mamba130m.prefill.t512")


def seq_shape_b(i: Int) -> Int:
    if i == 0:
        return 2
    if i == 6:
        return 2
    return 1


def seq_shape_l(i: Int) -> Int:
    if i == 0:
        return 4
    if i == 1:
        return 1
    if i == 2:
        return 8
    if i == 3:
        return 128
    if i == 4:
        return 512
    if i == 5:
        return 1
    if i == 6:
        return 4
    if i == 7:
        return 1
    if i == 8:
        return 8
    if i == 9:
        return 128
    return 512


def seq_shape_ctx(i: Int) -> Int:
    """How many tokens are ALREADY in the KV cache when the timed call runs.

    Nonzero on exactly one row, and that row is the whole point of the
    decode regime: at `ctx = 512` the attention arm folds over 513 keys for
    ONE query, which is a bandwidth problem, where the prefill rows are
    compute problems. Zero everywhere else, including every mamba row --
    Mamba's recurrence is O(1) per token whatever the history, so its decode
    cost does not depend on `ctx` and row 7 (L=1) already IS the decode
    shape. DEVIATION 1854, stated rather than hidden by a duplicate row."""
    if i == 5:
        return 512
    return 0


def seq_shape_d_model(i: Int) -> Int:
    if i == 0:
        return 32
    if i == 1:
        return 4096
    if i == 2:
        return 4096
    if i == 3:
        return 4096
    if i == 4:
        return 4096
    if i == 5:
        return 4096
    if i == 6:
        return 8
    return 768


def seq_shape_n_heads(i: Int) -> Int:
    """Zero on the mamba rows, where it has no meaning."""
    if i == 0:
        return 2
    if i == 1:
        return 32
    if i == 2:
        return 32
    if i == 3:
        return 32
    if i == 4:
        return 32
    if i == 5:
        return 32
    return 0


def seq_shape_n_kv(i: Int) -> Int:
    if i == 0:
        return 2
    if i == 1:
        return 8
    if i == 2:
        return 8
    if i == 3:
        return 8
    if i == 4:
        return 8
    if i == 5:
        return 8
    return 0


def seq_shape_head_dim(i: Int) -> Int:
    if i == 0:
        return 16
    if i == 1:
        return 128
    if i == 2:
        return 128
    if i == 3:
        return 128
    if i == 4:
        return 128
    if i == 5:
        return 128
    return 0


def seq_shape_intermediate(i: Int) -> Int:
    if i == 0:
        return 64
    if i == 1:
        return 14336
    if i == 2:
        return 14336
    if i == 3:
        return 14336
    if i == 4:
        return 14336
    if i == 5:
        return 14336
    return 0


def seq_shape_smoke(i: Int) -> Int:
    """1 if `MOJOLEARN_SPEED_SIZE=smoke` runs this row.

    The smoke set is the two rows the correctness lanes already run, and it
    exists to prove the BUILD, the witness and the hash on a laptop-sized
    allocation. A number taken at `size=smoke` is not a measurement of
    anything and every line carries `size=` so it cannot be mistaken for
    one."""
    if i == 0:
        return 1
    if i == 6:
        return 1
    return 0


def seq_shape_provenance(i: Int) -> String:
    if i == 0:
        return String(
            "transformer/corpus/gen_corpus.py CASES base_b2_l4_d32_kv2 --"
            " the shape transformer_check.mojo's clause (a) runs. SPEED-LANE"
            " BITS, not corpus bits (DEVIATION 1851)."
        )
    if i == 1:
        return String(
            "Llama-3-8B config at ONE token, the same d_model 4096 /"
            " n_heads 32 / n_kv 8 / head_dim 128 / intermediate 14336 that"
            " bench/gemm_shapes.mojo rows 8-19 are named after. Prefill of"
            " length 1, empty cache."
        )
    if i == 2:
        return String("Llama-3-8B at 8 tokens; gemm_shapes.mojo's t8 rows.")
    if i == 3:
        return String(
            "Llama-3-8B at 128 tokens. Between gemm_shapes.mojo's t8 and"
            " t512, and the first row where the score matrix is bigger than"
            " the projections."
        )
    if i == 4:
        return String(
            "Llama-3-8B at 512 tokens, a prefill chunk; gemm_shapes.mojo's"
            " t512 rows."
        )
    if i == 5:
        return String(
            "Llama-3-8B decode: ONE query token against a 512-token cache."
            " The regime every served token after the first is in, and the"
            " only row where the attention arm is bandwidth bound."
        )
    if i == 6:
        return String(
            "mamba/corpus CASES base_b2_l4_d8 -- the shape mamba_check.mojo"
            " runs, and the smallest thing that exercises the conv window"
            " exactly (L == d_conv). SPEED-LANE BITS."
        )
    if i == 7:
        return String(
            "state-spaces/mamba-130m config (d_model 768, so d_inner 1536"
            " and dt_rank 48) at ONE token. For Mamba this IS the decode"
            " step: the recurrence is O(1) per token."
        )
    if i == 8:
        return String("mamba-130m at 8 tokens.")
    if i == 9:
        return String("mamba-130m at 128 tokens.")
    return String(
        "mamba-130m at 512 tokens, the prefill chunk, and the row where a"
        " fused selective_scan_cuda has the most to win over a sequential"
        " torch scan."
    )


comptime SEQ_SEED_BASE: UInt64 = 0x53657153706564FF
"""DEVIATION 1851. This lane's OWN seed base, distinct from
`transformer/corpus/gen_corpus.py::SEED_BASE` (0x58666D72436F7270) and from
`mamba/corpus/gen_corpus.py::SEED_BASE` (0x4D616D6261436F72), so that no
tensor generated here can collide with a corpus case and no reference value
recorded for a corpus case can be mistaken for a reference for a row of this
table. These are corpus SHAPES with speed-lane BITS."""


def seq_seed(row: Int) -> UInt64:
    """Row `k`'s seed, the corpus rule with this lane's base."""
    return SEQ_SEED_BASE + UInt64(0x1000) * UInt64(row)


comptime TID_CTX_X = 20
"""The tensor id of the PRIOR CONTEXT's hidden states on the decode row.

`transformer_fixture.mojo` uses ids 1 through 11, so 20 is free. It has to
be its OWN id rather than `TID_X`: the generator's values depend only on
(key, index), so a context of `B*ctx*d_model` values drawn from `TID_X`
would have this call's `x` as its exact prefix, and the decode token would
be a repeat of context token 0. That is not wrong, it is just a fixture that
quietly says less than it looks like it says."""


# ===========================================================================
# THE HASH AND THE WITNESS
# ===========================================================================


def _hash_device_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> UInt64:
    """FNV-1a64 over the first `n` floats' BYTES, through
    `core/identity_trace.mojo::fnv1a64_bytes` itself rather than a second
    spelling of it. Little endian, byte at a time; that file explains at
    length why a word-at-a-time variant is a different function."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var h = fnv1a64_bytes(FNV_OFFSET, host.unsafe_ptr().bitcast[UInt8](), n * 4)
    # `[[mojo-buffer-freed-at-last-use]]`: keep the host buffer alive past
    # the read, or the hash runs over freed memory.
    _ = host^
    return h


comptime WITNESS_SAMPLES = 4096
"""How many elements of a tensor the weight witness folds. DEVIATION 1850.

The whole tensor would be better and is not affordable: `down_proj` at the
Llama-8B row is 58.7 million floats, FNV-1a64 is byte-at-a-time, and the
Python opponent would spend longer computing the witness than running the
benchmark. A fixed stride is chosen rather than a prefix precisely so that a
transposed, permuted or wrongly-ranged tensor still moves the witness: a
prefix witness agrees with any tensor that happens to share a first page."""


def _witness_hash(values: List[Float32]) -> UInt64:
    """FNV-1a64 over `len(values)` and then over a fixed strided sample.

    THE LENGTH IS FOLDED FIRST so two tensors that agree on every sampled
    element but not on their shape still disagree. The stride rule is
    `max(1, n // WITNESS_SAMPLES)` and `tools/speed_torch_seq.py` spells the
    same rule; if the two ever disagree, every witness disagrees at once,
    which is the loud failure and not the quiet one."""
    var n = len(values)
    var h = FNV_OFFSET
    var nb = UInt64(n)
    for lenbyte in range(8):
        h = (
            (h ^ ((nb >> UInt64(8 * lenbyte)) & UInt64(0xFF)))
            * UInt64(0x100000001B3)
        )
    var stride = n // WITNESS_SAMPLES
    if stride < 1:
        stride = 1
    var i = 0
    while i < n:
        # `[[mojo-int-widening-sign-extends]]`: mask AFTER the widen. The
        # source is a UInt32 so this should already be a zero extend, and
        # the mask costs nothing and removes the question.
        var bits = UInt64(bitcast[DType.uint32](values[i])) & UInt64(0xFFFFFFFF)
        for valbyte in range(4):
            var b = (bits >> UInt64(8 * valbyte)) & UInt64(0xFF)
            h = (h ^ b) * UInt64(0x100000001B3)
        i += stride
    return h


def _emit_witness(lane: String, tag: String, name: String, values: List[Float32]):
    print(
        "FSPEED-WEIGHTS lane=" + lane + " shape=" + tag + " tensor=" + name
        + " n=" + String(len(values)) + " hash=" + _hex16(_witness_hash(values))
    )


# ===========================================================================
# THE OUTPUT DUMP, so the Python side can print FSPEED-AGREE
# ===========================================================================


def _dump_f32(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    path: String,
) raises:
    """Write the first `n` floats as raw little-endian float32 bytes.

    The pattern is `core/identity_trace.mojo::_emit`'s `.bin` dump, copied
    rather than reached for because that one is reached only through a
    matching trace tag and this lane wants the buffer by name. One
    allocation of `4n` bytes on the host; at the largest row that is 8.4 MB
    and it happens ONCE, after the last timed round."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var raw = host.unsafe_ptr().bitcast[UInt8]()
    var bytes = List[UInt8]()
    for i in range(n * 4):
        bytes.append(raw.unsafe_load(i))
    with open(path, "w") as fh:
        fh.write_bytes(Span(bytes))
    _ = host^


def _dump_path(lane: String, tag: String) -> String:
    var d = String(getenv("MOJOLEARN_SPEED_DUMP_DIR"))
    if d == "":
        return String("")
    return d + "/seq." + lane + "." + tag + ".f32.bin"


# ===========================================================================
# THE LINES
# ===========================================================================


struct Ledger(Movable):
    """One shape's rounds, so the hash-stability finding is printed once with
    the rounds that disagreed named.

    Under FAST a moved hash is a REPORT of a non-deterministic arm and this
    struct does not raise on it. That is the instruction, and it is also the
    right behavior: an arm that is allowed to be non-deterministic and turns
    out to be non-deterministic has told us something, and killing the
    process would throw away every remaining shape on a rented box."""

    var lane: String
    var arm: String
    var tag: String
    var hashes: List[UInt64]
    var labels: List[String]

    def __init__(out self, lane: String, arm: String, tag: String):
        self.lane = lane
        self.arm = arm
        self.tag = tag
        self.hashes = List[UInt64]()
        self.labels = List[String]()

    def warmup(self, ns: Int):
        print(
            "FSPEED-WARMUP lane=" + self.lane + " arm=" + self.arm + " shape="
            + self.tag + " ms=" + String(Float64(ns) / 1.0e6)
        )

    def round(mut self, i: Int, ns: Int, h: UInt64):
        print(
            "FSPEED lane=" + self.lane + " arm=" + self.arm + " shape="
            + self.tag + " round=" + String(i) + " ms="
            + String(Float64(ns) / 1.0e6) + " hash=" + _hex16(h)
        )
        self.hashes.append(h)
        self.labels.append(String(i))

    def verdict(self):
        if len(self.hashes) == 0:
            return
        var first = self.hashes[0]
        for i in range(1, len(self.hashes)):
            if self.hashes[i] != first:
                print(
                    "FSPEED-NOTE lane=" + self.lane + " arm=" + self.arm
                    + " hash moved across rounds: " + _hex16(first) + " "
                    + _hex16(self.hashes[i])
                )
                return


def _refused(lane: String, arm: String, reason: String):
    print("FSPEED-REFUSED lane=" + lane + " arm=" + arm + " reason=" + reason)


def _note(lane: String, arm: String, text: String):
    print("FSPEED-NOTE lane=" + lane + " arm=" + arm + " " + text)


# ===========================================================================
# THE TRANSFORMER WEIGHTS
#
# TRANSCRIBED from `transformer/corpus/gen_corpus.py::BASE_RANGES` (:648) and
# `ranges_for` (:874), for the reason the file docstring gives. The tensor
# ids are IMPORTED from `transformer_fixture.mojo` and are the same ten ids
# the corpus uses (`TENSOR_IDS`, :183), which is checkable by eye and is
# checked by the witness in any case.
# ===========================================================================


struct LlamaHostWeights(Movable):
    var norm1_w: List[Float32]
    var norm2_w: List[Float32]
    var w_q: List[Float32]
    var w_k: List[Float32]
    var w_v: List[Float32]
    var w_o: List[Float32]
    var w_gate: List[Float32]
    var w_up: List[Float32]
    var w_down: List[Float32]

    def __init__(out self, seed: UInt64, dims: LlamaDims) raises:
        var dm = dims.d_model
        var qw = dims.q_width()
        var kw = dims.kv_width()
        var it = dims.intermediate
        var s_o = fan_in_scale(qw)
        var s_d = fan_in_scale(it)
        self.norm1_w = fixture_tensor(seed, T_TID_NORM1_W, dm, 0.5, 1.5)
        self.norm2_w = fixture_tensor(seed, T_TID_NORM2_W, dm, 0.5, 1.5)
        self.w_q = fixture_tensor(seed, T_TID_W_Q, qw * dm, -0.5, 0.5)
        self.w_k = fixture_tensor(seed, T_TID_W_K, kw * dm, -0.5, 0.5)
        self.w_v = fixture_tensor(seed, T_TID_W_V, kw * dm, -0.5, 0.5)
        self.w_o = fixture_tensor(seed, T_TID_W_O, dm * qw, -s_o, s_o)
        self.w_gate = fixture_tensor(seed, T_TID_W_GATE, it * dm, -0.25, 0.25)
        self.w_up = fixture_tensor(seed, T_TID_W_UP, it * dm, -0.25, 0.25)
        self.w_down = fixture_tensor(seed, T_TID_W_DOWN, dm * it, -s_d, s_d)

    def emit_witness(self, lane: String, tag: String):
        _emit_witness(lane, tag, "norm1.weight", self.norm1_w)
        _emit_witness(lane, tag, "norm2.weight", self.norm2_w)
        _emit_witness(lane, tag, "q_proj.weight", self.w_q)
        _emit_witness(lane, tag, "k_proj.weight", self.w_k)
        _emit_witness(lane, tag, "v_proj.weight", self.w_v)
        _emit_witness(lane, tag, "o_proj.weight", self.w_o)
        _emit_witness(lane, tag, "gate_proj.weight", self.w_gate)
        _emit_witness(lane, tag, "up_proj.weight", self.w_up)
        _emit_witness(lane, tag, "down_proj.weight", self.w_down)


# ===========================================================================
# THE LLAMA LANES
# ===========================================================================


def run_llama(
    ctx: DeviceContext, lane: String, row: Int, rounds: Int, size: String
) raises:
    """One row of the table through one of the four llama entries.

    `transformer` calls `llama_decoder_layer_forward`, which is the whole
    block INCLUDING `llama_refuse_bad_inputs`. `attention`, `mlp` and
    `rmsnorm` call `llama_attention_forward`, `llama_mlp_forward` and
    `llama_rms_norm`, which do not refuse and which read stage buffers this
    function fills with one untimed whole-block call before the loop. A
    submodule timed on ZERO-FILLED inputs would be timing denormals and a
    softmax over a constant, which is a different arithmetic on some
    hardware; the setup call is what makes the inputs realistic."""
    var b = seq_shape_b(row)
    var l = seq_shape_l(row)
    var ctxlen = seq_shape_ctx(row)
    var tag = seq_shape_name(row)
    var dims = LlamaDims(
        seq_shape_d_model(row),
        seq_shape_n_heads(row),
        seq_shape_n_kv(row),
        seq_shape_head_dim(row),
        seq_shape_intermediate(row),
    )
    dims.validate()

    var s_max = ctxlen + l
    var p_max = s_max
    if p_max < 512:
        p_max = 512
    var m = b * l
    var dm = dims.d_model

    var seed = seq_seed(row)
    var hw = LlamaHostWeights(seed, dims)
    hw.emit_witness(lane, tag)
    var hx = fixture_tensor(seed, T_TID_X, m * dm, -2.0, 2.0)
    _emit_witness(lane, tag, "x", hx)

    var dw = LlamaDeviceWeights(
        ctx,
        dims,
        LLAMA_RMS_EPS,
        hw.norm1_w,
        hw.norm2_w,
        hw.w_q,
        hw.w_k,
        hw.w_v,
        hw.w_o,
        hw.w_gate,
        hw.w_up,
        hw.w_down,
    )
    # FREE THE HOST COPY AS SOON AS THE DEVICE HAS IT. At row 4 these nine
    # lists are about 872 MB, `LlamaDeviceWeights` has already copied every
    # one of them through a host buffer, and a rented box with 32 GB of RAM
    # holding two copies plus the generator's temporaries is how a lease ends
    # in the OOM killer instead of in a number.
    _ = hw^
    var kv = LlamaKVCache(ctx, b, dims, s_max)
    var rope = LlamaRopeTable(ctx, dims, LLAMA_ROPE_THETA, p_max)
    var stages = LlamaDeviceStages(ctx, b, l, s_max, dims)
    var dx = _upload(ctx, hx)
    _ = hx^
    var off = IdentityTrace.disabled()

    # ---- THE PRIOR CONTEXT, for the decode row only. Built by running the
    #      block on `ctxlen` tokens with its OWN stage buffers, which is the
    #      only honest way to get a cache holding values a block computed
    #      rather than values a harness invented. It costs one extra
    #      allocation of the prefill's stages and it is thrown away.
    if ctxlen > 0:
        var hx_ctx = fixture_tensor(
            seed, TID_CTX_X, b * ctxlen * dm, -2.0, 2.0
        )
        _emit_witness(lane, tag, "ctx.x", hx_ctx)
        var dx_ctx = _upload(ctx, hx_ctx)
        var st_ctx = LlamaDeviceStages(ctx, b, ctxlen, s_max, dims)
        llama_decoder_layer_forward(
            ctx, st_ctx, kv, rope, dw, dx_ctx, b, ctxlen, 0, off, "ctx"
        )
        _ = hx_ctx^
        _ = dx_ctx^
        _ = st_ctx^

    # ---- DEVIATION 1853. The block APPENDS to the KV cache and advances
    #      `kv.s`, so a second round is a different call unless the cache is
    #      put back. Both buffers and the used length are snapshotted here,
    #      restored before every round, and the restore is OUTSIDE the
    #      clock. Without this the round-2 hash would move for a reason that
    #      has nothing to do with FAST being non-deterministic, which is
    #      exactly the confusion `FSPEED-NOTE` exists to avoid.
    var snap_k = ctx.enqueue_create_buffer[DType.float32](len(kv.k))
    var snap_v = ctx.enqueue_create_buffer[DType.float32](len(kv.v))
    ctx.enqueue_copy(dst_buf=snap_k, src_buf=kv.k)
    ctx.enqueue_copy(dst_buf=snap_v, src_buf=kv.v)
    ctx.synchronize()
    var s0 = kv.s

    # ---- The setup call. Fills every stage buffer with values a real block
    #      call produced, which is what the submodule lanes read.
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, dw, dx, b, l, s0, off, "setup"
    )
    ctx.enqueue_copy(dst_buf=kv.k, src_buf=snap_k)
    ctx.enqueue_copy(dst_buf=kv.v, src_buf=snap_v)
    ctx.synchronize()
    kv.s = s0

    # ---- What the refusal costs on its own, so a reader can subtract it
    #      from the `transformer` lane rather than guess. NOT part of the
    #      table and NOT an arm.
    var tr0 = perf_counter_ns()
    llama_refuse_bad_inputs(ctx, dw, rope, dx, kv, b, l)
    var tr1 = perf_counter_ns()
    _note(
        lane,
        "ours",
        "llama_refuse_bad_inputs alone shape=" + tag + " ms="
        + String(Float64(tr1 - tr0) / 1.0e6)
        + " (inside the transformer lane's round, not inside attention/mlp/"
        "rmsnorm)",
    )

    var ledger = Ledger(lane, "ours", tag)
    var out_n = m * dm
    for r in range(rounds + 1):
        ctx.enqueue_copy(dst_buf=kv.k, src_buf=snap_k)
        ctx.enqueue_copy(dst_buf=kv.v, src_buf=snap_v)
        ctx.synchronize()
        kv.s = s0
        var t0 = perf_counter_ns()
        if lane == "transformer":
            llama_decoder_layer_forward(
                ctx, stages, kv, rope, dw, dx, b, l, s0, off, "speed"
            )
        elif lane == "attention":
            llama_attention_forward(
                ctx,
                stages,
                kv,
                rope,
                dw,
                b,
                l,
                s0,
                PLANT_AT_NONE,
                List[Int](),
                List[UInt32](),
                off,
                "speed",
            )
        elif lane == "mlp":
            llama_mlp_forward(ctx, stages, dw, m, off, "speed")
        else:
            llama_rms_norm(
                ctx,
                stages.norm1_sumsq,
                stages.norm1_out,
                dx,
                dw.norm1_w,
                m,
                dm,
                dw.eps,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if r == 0:
            ledger.warmup(t1 - t0)
        else:
            var h: UInt64
            if lane == "transformer":
                h = _hash_device_f32(ctx, stages.residual2, out_n)
            elif lane == "attention":
                h = _hash_device_f32(ctx, stages.o_proj, out_n)
            elif lane == "mlp":
                h = _hash_device_f32(ctx, stages.down_proj, out_n)
            else:
                h = _hash_device_f32(ctx, stages.norm1_out, out_n)
            ledger.round(r, t1 - t0, h)
    ledger.verdict()

    var path = _dump_path(lane, tag)
    if path == "":
        _refused(
            lane,
            "agree",
            "MOJOLEARN_SPEED_DUMP_DIR unset, so no output was dumped and no"
            " FSPEED-AGREE can be computed for shape " + tag,
        )
    else:
        if lane == "transformer":
            _dump_f32(ctx, stages.residual2, out_n, path)
        elif lane == "attention":
            _dump_f32(ctx, stages.o_proj, out_n, path)
        elif lane == "mlp":
            _dump_f32(ctx, stages.down_proj, out_n, path)
        else:
            _dump_f32(ctx, stages.norm1_out, out_n, path)
        _note(
            lane, "ours", "dump shape=" + tag + " n=" + String(out_n)
            + " path=" + path
        )

    _ = snap_k^
    _ = snap_v^
    _ = dw^
    _ = kv^
    _ = rope^
    _ = stages^
    _ = dx^


# ===========================================================================
# THE MAMBA LANES
# ===========================================================================


def run_mamba(
    ctx: DeviceContext, lane: String, row: Int, rounds: Int, size: String
) raises:
    """One row through `mamba_block_forward` or through `selective_scan_fn`.

    `selective_scan` is timed on the buffers a REAL block call produced --
    `stages.silu_out` as `u`, `stages.softplus_out` as `delta`,
    `stages.a_out`, `stages.b_mat`, `stages.c_mat` and `w.d_skip` -- because
    the scan's cost per element does not depend on the values but the
    `expf` in `exp(A * dt)` can, and because a scan fed zeros is a scan over
    a state that never grows. `selective_scan_fn` REFUSES `z`,
    `delta_bias` and `delta_softplus` (DEVIATION 723): those three seams
    belong to the block, so `delta` arrives already softplused, which is
    what `stages.softplus_out` holds. The torch opponent must be called the
    same way and `tools/speed_torch_seq.py` does."""
    var b = seq_shape_b(row)
    var l = seq_shape_l(row)
    var tag = seq_shape_name(row)
    var dims = MambaDims.of(seq_shape_d_model(row))
    var m = b * l
    var dm = dims.d_model
    var di = dims.d_inner

    var seed = seq_seed(row)
    var w = corpus_weights(seed, dims)
    var hx = corpus_x(seed, b, l, dm)
    _emit_witness(lane, tag, "norm.weight", w.norm_w)
    _emit_witness(lane, tag, "in_proj.weight", w.w_in)
    _emit_witness(lane, tag, "conv1d.weight", w.conv_w)
    _emit_witness(lane, tag, "conv1d.bias", w.conv_b)
    _emit_witness(lane, tag, "x_proj.weight", w.w_x)
    _emit_witness(lane, tag, "dt_proj.weight", w.w_dt)
    _emit_witness(lane, tag, "dt_proj.bias", w.b_dt)
    _emit_witness(lane, tag, "A_log", w.a_log)
    _emit_witness(lane, tag, "D", w.d_skip)
    _emit_witness(lane, tag, "out_proj.weight", w.w_out)
    _emit_witness(lane, tag, "x", hx)

    var dw = MambaDeviceWeights(ctx, w)
    var state = MambaDeviceState(ctx, b, dims)
    var stages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, hx)
    var off = IdentityTrace.disabled()

    # The setup call, for the same reason the llama lanes have one: the
    # scan lane reads stage buffers, and a scan over the `_zeros` fill is a
    # scan whose `exp` argument is exactly zero everywhere.
    mamba_block_forward(ctx, stages, state, dw, dx, b, l, off, "setup")

    var tr0 = perf_counter_ns()
    mamba_refuse_bad_inputs(ctx, dw, dx, state, b, l)
    var tr1 = perf_counter_ns()
    _note(
        lane,
        "ours",
        "mamba_refuse_bad_inputs alone shape=" + tag + " ms="
        + String(Float64(tr1 - tr0) / 1.0e6)
        + " (inside the mamba lane's round, not inside selective_scan)",
    )

    var ledger = Ledger(lane, "ours", tag)
    var out_n = m * dm
    if lane == "selective_scan":
        out_n = m * di
    for r in range(rounds + 1):
        # The recurrent state is IN AND OUT on both entries, so it is zeroed
        # before every round and the zeroing is outside the clock. For the
        # block that means the conv window as well as the SSM state.
        ctx.enqueue_memset(state.h, Float32(0.0))
        if lane == "mamba":
            ctx.enqueue_memset(state.conv_win, Float32(0.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        if lane == "mamba":
            mamba_block_forward(ctx, stages, state, dw, dx, b, l, off, "speed")
        else:
            selective_scan_fn(
                ctx,
                stages.skip_out,
                stages.scan_y,
                state.h,
                stages.silu_out,
                stages.softplus_out,
                stages.a_out,
                stages.b_mat,
                stages.c_mat,
                dw.d_skip,
                b,
                l,
                di,
                False,
                False,
                False,
                True,
                off,
                "speed",
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if r == 0:
            ledger.warmup(t1 - t0)
        else:
            var h: UInt64
            if lane == "mamba":
                h = _hash_device_f32(ctx, stages.residual_out, out_n)
            else:
                h = _hash_device_f32(ctx, stages.skip_out, out_n)
            ledger.round(r, t1 - t0, h)
    ledger.verdict()

    var path = _dump_path(lane, tag)
    if path == "":
        _refused(
            lane,
            "agree",
            "MOJOLEARN_SPEED_DUMP_DIR unset, so no output was dumped and no"
            " FSPEED-AGREE can be computed for shape " + tag,
        )
    else:
        if lane == "mamba":
            _dump_f32(ctx, stages.residual_out, out_n, path)
        else:
            _dump_f32(ctx, stages.skip_out, out_n, path)
        _note(
            lane, "ours", "dump shape=" + tag + " n=" + String(out_n)
            + " path=" + path
        )

    _ = dw^
    _ = state^
    _ = stages^
    _ = dx^
    _ = w^
    _ = hx^


# ===========================================================================
# main
# ===========================================================================


def _family_of_lane(lane: String) raises -> Int:
    if lane == "transformer" or lane == "attention" or lane == "mlp":
        return FAM_LLAMA
    if lane == "rmsnorm":
        return FAM_LLAMA
    if lane == "mamba" or lane == "selective_scan":
        return FAM_MAMBA
    raise Error(
        "seq_speed: MOJOLEARN_SPEED_LANE must be one of transformer attention"
        " mlp rmsnorm mamba selective_scan; got '" + lane + "'"
    )


def main() raises:
    var lane = String(getenv("MOJOLEARN_SPEED_LANE"))
    var fam = _family_of_lane(lane)
    var rounds = _env_int("MOJOLEARN_SPEED_ROUNDS", 10)
    if rounds < 1:
        raise Error("seq_speed: MOJOLEARN_SPEED_ROUNDS must be >= 1")
    var size = _env_str("MOJOLEARN_SPEED_SIZE", "shipped")
    if size != "shipped" and size != "smoke":
        raise Error(
            "seq_speed: MOJOLEARN_SPEED_SIZE must be 'shipped' or 'smoke',"
            " got '" + size + "'"
        )
    var device = _env_str("MOJOLEARN_SPEED_DEVICE", "unset")
    var only = _env_int("MOJOLEARN_SPEED_ROW", -1)

    print(
        "FSPEED-HEADER family=seq lane=" + lane + " arm=ours mode="
        + _mode_name() + " device=" + device + " rounds=" + String(rounds)
        + " size=" + size
    )
    print(
        "FSPEED-NOTE lane=" + lane + " arm=ours entry costs the port pays and"
        " the opponent does not: llama_refuse_bad_inputs /"
        " mamba_refuse_bad_inputs download every weight per call, and"
        " eager_attention_forward synchronizes per (batch, head). See"
        " bench/speed/README.md."
    )
    # DEVIATION 1885 -- THE PRECISION CUT TRAVELS WITH THE NUMBER.
    #
    # Under FAST every GEMM in these lanes goes through
    # `_fast_vendor_gemm`, which on an NVIDIA target is MAX's tensor-core
    # `matmul`: measured at 200 TFLOP/s on an H100 against `cublas-fp32`'s
    # 44.4 and `cublas-tf32`'s 207.5, i.e. on the TF32 line, i.e. TEN
    # explicit mantissa bits rather than twenty-three.
    #
    # It made the Llama block 92x to 536x faster and it made the agreement
    # against a torch FP32 reference 47x to 239x worse, while `rmsnorm` --
    # which routes no GEMM -- did not move by one bit. A FAST path may be
    # non-deterministic and may take the vendor's fastest kernel. It may
    # NOT take a precision cut silently, and a speedup printed beside an
    # fp32 agreement column reads as "faster AND as accurate", which is
    # false.
    #
    # Emitted on EVERY lane, including rmsnorm and the mamba pair, because
    # "this lane is unaffected" is itself the information -- rmsnorm is the
    # control that made the cut visible at all.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        print(
            "FSPEED-NOTE lane=" + lane + " arm=ours PRECISION: FAST routes"
            " every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE"
            " (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s"
            " against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100."
            " The fair vendor opponent for a lane that routes a GEMM is"
            " therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an"
            " FSPEED-AGREE line computed against an fp32 reference is"
            " measuring THIS CUT and not a defect. Lanes routing no GEMM"
            " (rmsnorm) are unaffected and are the control. DEVIATION 1885."
        )

    var ctx = DeviceContext()
    for row in range(SEQ_SHAPE_COUNT):
        if seq_shape_family(row) != fam:
            continue
        if only >= 0 and row != only:
            continue
        if size == "smoke" and seq_shape_smoke(row) != 1:
            continue
        # ONE ROW MAY NOT TAKE THE RUN DOWN. The box is rented and the lease
        # is an hour; a shape that will not allocate or a submodule that
        # refuses must cost its own row and no more.
        try:
            if fam == FAM_LLAMA:
                run_llama(ctx, lane, row, rounds, size)
            else:
                run_mamba(ctx, lane, row, rounds, size)
        except e:
            _refused(
                lane,
                "ours",
                "row " + String(row) + " " + seq_shape_name(row) + ": "
                + String(e),
            )
    print("FSPEED-DONE lane=" + lane + " mode=" + _mode_name())
