# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused attention at the LM target shape: bit equality of every arm
against the shipped kernels (and the shipped kernels against the eager
oracle), reach of every candidate arm by sabotage, and the price of each
arm, forward alone and forward plus backward, arms alternated inside one
process. DEVIATIONS 2525 to 2528, brief
`docs/lanes/BRIEF_attention_step_2026-09-11.md` (2528 is section 12).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . \\
        bench/attention_step_price_main.mojo -o /tmp/attn_price
    MOJOLEARN_ATTN_ARM=bwd_stash /tmp/attn_price

THE SHAPE is the byte LM target (`tools/lm_step_memory_probe.py`
TARGET_SHAPE): batch 1, L 2048, 12 heads, 12 kv heads, head_dim 64, causal
(window 0, pos0 0, so S = L). Every knob below moves it, and a shape that is
not the target is labeled by its own numbers in every line printed.

INPUT KINDS (MOJOLEARN_ATTN_KINDS, comma separated):
  file:<dir>  REAL ACTIVATIONS, the kind a timing or promotion claim quotes
              (ENGINEERING_RULES section 9, Andrew 2026-09-11: the two
              kinds are two ORDINARY corpora that differ in what they are,
              never an adversarial fixture). `<dir>` holds q.bin, k.bin,
              v.bin, dctx.bin and meta.txt as the backward launcher dumps
              them from one real lean step under
              `-D MOJOLEARN_ATTN_OPERAND_DUMP=1` (the last layer's operands
              of the first step, on the corpus the step was fed); the
              shape is the dump's. The leg feeds two such directories, one
              per corpus (English text, `training/corpus/enwik8`; source
              code, `training/corpus/pile_github`; ENGINEERING_RULES 9).
  hashed      a cheap bit-equality SMOKE only: the k-NN gate's
              `hashed_block` profile in Mojo (log-uniform magnitudes over
              2.5 e-folds, per-column octave scales, twelve cluster
              offsets), every cell a distinct 53-bit hash, never uniform
              (uniform data hides permutation bugs); q and k scaled by 1/8.
              Its PRICE lines are not a claim.
  heavytail   an adversarial profile (six e-folds, one cell in 64 boosted
              eight times, a diagonal content term) for the CORRECTNESS
              sections only; never a timing input (set
              MOJOLEARN_ATTN_TIMING=0 when running it, as the leg does).

WHAT IT ASSERTS, per kind:
  1. the shipped kernels (arm `baseline`) report FUSED_RAN and, with
     MOJOLEARN_ATTN_ORACLE=1 (default), equal the eager stage kernels bit
     for bit on ctx, amax, denom, zdot, dq, dk, dv (the fused check does
     this at small shapes; this is the target shape);
  2. every candidate arm reports FUSED_RAN and equals the baseline bit for
     bit on the same seven buffers;
  3. REACH: every candidate arm's sabotage instantiation moves at least
     one cell of some buffer, and the clean arm restores the baseline bits
     afterwards. On a build without -D MOJOLEARN_ATTN_ARM_TRIAL=1 every
     arm value runs the shipped kernels: the sabotage moves nothing and
     the harness FAILS, saying so. A candidate with a second-round bit
     (DEVIATION 2528 `_ztiled`, 2531 `_fgrid`, 2530 `_qres`, 2533 `_pf`)
     proves reach with ATTN_ARM_SABOTAGE_NEW, which flips only the new
     kernels, so a proof on top of stash_tiled names the new kernels. Per
     branch: an arm with a second-round forward kernel must move the
     forward and one without must move no forward cell; an arm with a
     second-round backward kernel must move the backward, and one without
     must move no backward cell whenever the sabotaged forward left amax
     and denom (the backward's forward inputs) alone. Attribution (brief
     section 14.5): the 2530 flip must move amax or denom, the 2531 and
     2533 forward flips must hold them and the 2533 forward flip must hold
     ctx columns 16 and up; the 2533 backward flip must move zdot and hold
     dv when 2528 is not in the arm.
  4. PRICE: MOJOLEARN_ATTN_WARMUPS (2) untimed calls, then
     MOJOLEARN_ATTN_ROUNDS (7) timed rounds, the two arms alternated
     inside each round (A B, then B A), each sample one `PRICE` line;
     medians and achieved TFLOP/s at the end. Achieved TFLOP/s is the
     HANDOFF_speed_gemm definition applied to attention: useful flops over
     the VISIBLE cells only, `4 * cells * head_dim` forward (two
     contractions) and `10 * cells * head_dim` backward (five: the score
     recompute, dP, dV, dQ, dK), divided by the median milliseconds times
     1e9. The shipped kernels EXECUTE more than that (three score dots in
     the forward, six dots in the backward), and the table prints that
     multiplier beside the useful figure so a reader can see which arm
     removed which recompute.

The timed region is the launcher call: the regime scans, the corner flag
read and (for the stash arms) the scratch allocation are inside it, as
they are inside `attn.core` and `bwd.attention` in the LM step. A build
with -D MOJOLEARN_ATTN_PHASE_TIMERS=1 run with
MOJOLEARN_TRANSFORMER_TIMING=1 prints the per-kernel split; such a run is
serialized and its PRICE lines are not a price (the harness says so in
its header).

KNOBS (environment): MOJOLEARN_ATTN_ARM (candidate, default bwd_stash;
any name `fused_attention_arm_parse` reads, brief section 12.1, e.g.
stash_tiled_ztiled, stash_tiled_ztiled_r32, stash_tiled_ztiled_r64, and
section 14's stash_tiled_pf, stash_tiled_fgrid_r32, stash_tiled_fgrid_r64,
stash_tiled_fgrid_r32_qres, stash_tiled_fgrid_r32_qres_pf),
MOJOLEARN_ATTN_BASELINE (default baseline; stash_tiled is the baseline a
second-round arm is priced against). Either name may be `default`, the
column's shipped arm (kernel matrix `attn_default_arm_for`, DEVIATION 2534:
NVIDIA stash_tiled_fgrid_r32_qres_pf, AMD stash_tiled_fgrid_r32_qres_pf_kvgrid_r32
since brief section 18, every other column stash_tiled); the
harness prints the explicit name everywhere and a `DEFAULT` line beside the
request. The two `PATH` lines print each arm's resolved kernels
(`is_default`, `resolved_hd64`), and every correctness run prints a `RAN`
line naming the kernels that launched and fails at head_dim 64 when they are
not the arm's. MOJOLEARN_ATTN_KINDS
(default hashed; the leg passes file:<dir> per corpus), MOJOLEARN_ATTN_L, _NH, _NKV, _HD, _B, _WINDOW,
MOJOLEARN_ATTN_ROUNDS, _WARMUPS, MOJOLEARN_ATTN_ORACLE (1),
MOJOLEARN_ATTN_REACH (1), MOJOLEARN_ATTN_TIMING (1),
MOJOLEARN_ATTN_RESOURCES (1).

DEVIATIONS 2596 AND 2597 (brief section 16). A candidate carrying
`_kvrecompute`, `_kvgrid` or `_kvsplit` proves its dk/dv launch's reach with
a further run under ATTN_ARM_SABOTAGE_KV (a `REACH_KV` line): dk and dv must
move, zdot, dq and the forward must hold. MOJOLEARN_ATTN_RESOURCES=1 prints,
before the kinds, the compiled attributes of the backward kernels at head_dim
64 (`RESOURCES` lines: regs, local, shared, const, max_threads,
blocks_per_sm_256, beside each kernel's source counts), a readback brief 16.3
reads against the step timers; it launches nothing.

DEVIATION 2598 (brief section 17). A candidate carrying `_zdefer` or `_zlag`
(the zdot stash kernel's schedule, e.g. stash_tiled_fgrid_r32_qres_pf_zlag,
composable with the kv tokens after it) gets a `REACH_Z` run per kind: the
forward CLEAN and the backward under ATTN_ARM_SABOTAGE_NEW, so the backward
reads the clean amax and denom even when the arm's Q residency flip would move
them. zdot must move at odd flat rows and hold at even rows (2533's flip moves
every row), dv and the forward must hold. `PATH` prints `zsched=`, and the
RESOURCES readback covers the two schedule copies.
"""

from std.math import exp
from std.memory import bitcast, memcpy
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import Attribute, DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from transformer.checks.transformer_fixture import fixture_splitmix64
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    bwd_attention_eager_stages,
)
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_BASELINE,
    ATTN_ARM_BWD_KVSPLIT,
    ATTN_ARM_BWD_ZTILED,
    ATTN_ARM_DEFAULT,
    ATTN_ARM_FWD_QRES,
    ATTN_ARM_PREFLUSH,
    ATTN_ARM_SABOTAGE_KV,
    ATTN_ARM_SABOTAGE_NEW,
    ATTN_ARM_TRIAL,
    ATTN_PHASE_TIMERS,
    ATTN_STASH_HD,
    FUSED_RAN,
    fused_attention_arm_backward_resolved,
    fused_attention_arm_forward_resolved,
    fused_attention_arm_kv,
    fused_attention_arm_name,
    fused_attention_arm_new_backward,
    fused_attention_arm_new_forward,
    fused_attention_arm_parse,
    fused_attention_arm_reach_bit,
    fused_attention_arm_zsched,
    fused_attention_fwd_rows,
    fused_attention_kv_keys,
    fused_attention_zdot_rows,
    fused_attention_zsched_name,
    fused_backward_launch_arm,
    fused_backward_launch_ran,
    fused_bwd_dkdv_kernel,
    fused_bwd_dkdv_r2_kernel,
    fused_bwd_dkdv_tiled_kernel,
    fused_bwd_dkdv_tiled_pf_kernel,
    fused_bwd_dq_tiled_pf_kernel,
    fused_bwd_kvfold_r2_kernel,
    fused_bwd_zdot_sched_pf_kernel,
    fused_bwd_zdot_stash_pf_kernel,
    fused_forward_launch_arm,
    fused_forward_launch_ran,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDims,
    PLANT_AT_NONE,
    _download,
    _upload,
    attention_eager_core,
    llama_attention_scale,
    llama_key_lo,
    llama_key_span,
)


comptime SEED: UInt64 = 0x41746E5374657050
"""'AtnSteP', distinct from every fixture seed base and from the fused check's."""


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _env_str(name: String, default: String) -> String:
    var s = String(getenv(name))
    if s == "":
        return default
    return s


def _path_line(role: String, arm: Int) -> String:
    """ENGINEERING_RULES 8: the harness names the path beside the timing.
    `zdot_rows` is DEVIATION 2528's resolved geometry (the arm's `_r32` /
    `_r64`, else the column's kernel-matrix row; `-` when the arm does not
    run 2528, 0 when the page does not fit and the first-round kernels
    run); `fwd_rows` is the second-round forward's resolved geometry
    (DEVIATIONS 2531 and 2530, and 2533's forward half: `-` when the arm
    runs no second-round forward, 0 when its page does not fit and the
    first-round sstash kernel runs); `preflush` says whether DEVIATION 2533
    is in the arm; `reach_bit` is the sabotage this arm's reach proof uses.
    `is_default` says whether the arm is this column's shipped default
    (kernel matrix `attn_default_arm_for`, DEVIATION 2534) and
    `resolved_hd64` names the kernels the launchers run for it at head_dim
    64 on this build (geometry resolved)."""
    var rows = String("-")
    if (arm & ATTN_ARM_BWD_ZTILED) != 0:
        rows = String(fused_attention_zdot_rows(arm))
    var frows = String("-")
    if fused_attention_arm_new_forward(arm):
        frows = String(fused_attention_fwd_rows(arm))
    var reach = String("sabotage")
    if fused_attention_arm_reach_bit(arm) == ATTN_ARM_SABOTAGE_NEW:
        reach = String("sabotage_new")
    # DEVIATIONS 2596 and 2597: the dk/dv launch's keys per block (`-` when
    # the arm names neither; 4 for `_kvrecompute`), and whether it splits.
    var kv = String("-")
    if fused_attention_arm_kv(arm):
        kv = String(fused_attention_kv_keys(arm))
    var resolved = fused_attention_arm_forward_resolved(arm) | fused_attention_arm_backward_resolved(arm)
    return (
        "PATH " + role + " arm=" + fused_attention_arm_name(arm)
        + " is_default=" + String(arm == ATTN_ARM_DEFAULT)
        + " resolved_hd64=" + fused_attention_arm_name(resolved)
        + " zdot_rows=" + rows + " fwd_rows=" + frows + " preflush="
        + String((arm & ATTN_ARM_PREFLUSH) != 0) + " reach_bit=" + reach
        + " kv_keys=" + kv + " kv_split=" + String((arm & ATTN_ARM_BWD_KVSPLIT) != 0)
        + " zsched=" + fused_attention_zsched_name(arm)
    )


def _resolve_arm_name(raw: String) -> String:
    """`default` is an alias for this column's shipped default arm
    (DEVIATION 2534); the harness replaces it with the explicit name before
    anything is printed, so no PRICE, BITS, REACH or TABLE line ever says
    `default`."""
    if raw == "default":
        return fused_attention_arm_name(ATTN_ARM_DEFAULT)
    return raw


def _median(var xs: List[Float64]) -> Float64:
    var n = len(xs)
    for i in range(1, n):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _unit(h: UInt64) -> Float64:
    """`[0, 1)` from the top 53 bits."""
    return Float64(Int(h >> 11)) * 0.00000000000000011102230246251565  # 2^-53


def _pow2(k: Int) -> Float64:
    var v = 1.0
    if k >= 0:
        for _ in range(k):
            v *= 2.0
    else:
        for _ in range(-k):
            v *= 0.5
    return v


def _split_list(spec: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for cp in spec.codepoint_slices():
        var c = String(cp)
        if c == ",":
            if cur != "":
                out.append(cur)
            cur = String("")
        else:
            cur += c
    if cur != "":
        out.append(cur)
    return out^


# ---------------------------------------------------------------------------
# THE TWO INPUT KINDS. Every tensor is `[rows][n_heads][hd]` (q, dctx) or
# `[n_kv][rows][hd]` (k, v) flat; the generator sees (row, head, d) so the
# per-column scale and the cluster offset follow the head-dim axis.
# ---------------------------------------------------------------------------


def _hashed_cell(row: Int, col: Int, salt: UInt64) -> Float64:
    """`tools/knn_selection_gate.py` hashed_block, one cell."""
    var base = SEED * UInt64(0x2545F4914F6CDD1D) + salt
    var cell = UInt64(row) * UInt64(0x9E3779B97F4A7C15) + UInt64(col) * UInt64(0xD1B54A32D192ED03) + base
    var h1 = fixture_splitmix64(cell)
    var h2 = fixture_splitmix64(h1 ^ UInt64(0xA5A5A5A5A5A5A5A5))
    var u1 = _unit(h1)
    var u2 = _unit(h2)
    var sign = 1.0
    if (h2 & UInt64(1)) != UInt64(0):
        sign = -1.0
    var magnitude = exp(2.5 * (u1 - 0.5))
    var fs = Int(fixture_splitmix64(UInt64(col) + salt * UInt64(7)) % UInt64(7)) - 3
    var feature_scale = _pow2(fs)
    var cluster = Int(fixture_splitmix64(UInt64(row) + salt * UInt64(13)) % UInt64(12))
    var offset = _unit(fixture_splitmix64(UInt64(cluster * 4096 + col) + salt * UInt64(17))) * 8.0 - 4.0
    return sign * magnitude * feature_scale + offset + 0.37 * (u2 - 0.5)


def _heavytail_cell(row: Int, col: Int, salt: UInt64, content: Float64) -> Float64:
    """Six e-folds, one cell in 64 boosted eight times, plus the shared
    per-position content (zero for v and dctx)."""
    var base = SEED * UInt64(0x9E3779B97F4A7C15) + salt * UInt64(0x2545F4914F6CDD1D)
    var cell = UInt64(row) * UInt64(0xD1B54A32D192ED03) + UInt64(col) * UInt64(0x9E3779B97F4A7C15) + base
    var h1 = fixture_splitmix64(cell)
    var h2 = fixture_splitmix64(h1 ^ UInt64(0x5A5A5A5A5A5A5A5A))
    var u1 = _unit(h1)
    var sign = 1.0
    if (h2 & UInt64(1)) != UInt64(0):
        sign = -1.0
    var magnitude = exp(6.0 * (u1 - 0.5))
    if (h2 % UInt64(64)) == UInt64(0):
        magnitude *= 8.0
    return sign * magnitude + content


def _content(pos: Int, col: Int) -> Float64:
    """The shared per-position vector of the heavytail kind: q[t] and k[j]
    both carry `content(t, d)`, so the score peaks at j == t."""
    var h = fixture_splitmix64(UInt64(pos) * UInt64(0xBF58476D1CE4E5B9) + UInt64(col) + SEED)
    # |content| <= 14 before the 1/16 scale, so the diagonal dot is about
    # 64 * 65 / 256 = 16 before the attention scale (about 2 after), beside
    # noise whose typical scaled magnitude is below 1.
    return (_unit(h) - 0.5) * 28.0


def make_tensor(
    kind: String, which: Int, rows: Int, heads: Int, hd: Int, scale: Float64, with_content: Bool
) raises -> List[Float32]:
    """`which` salts the four tensors apart (1 q, 2 k, 3 v, 4 dctx)."""
    var out = List[Float32]()
    var salt = UInt64(which) * UInt64(0x100000001B3)
    for r in range(rows):
        for h in range(heads):
            for d in range(hd):
                var col = h * hd + d
                var v: Float64
                if kind == "hashed":
                    v = _hashed_cell(r, col, salt)
                elif kind == "heavytail":
                    var c = 0.0
                    if with_content:
                        c = _content(r, col)
                    v = _heavytail_cell(r, col, salt, c)
                else:
                    raise Error("input kind '" + kind + "' is not hashed or heavytail")
                out.append(Float32(v * scale))
    return out^


def to_kv_layout(x: List[Float32], rows: Int, heads: Int, hd: Int) -> List[Float32]:
    """`[rows][heads][hd]` to the packed `[heads][rows][hd]` the caches use."""
    var out = List[Float32](length=len(x), fill=Float32(0.0))
    for r in range(rows):
        for h in range(heads):
            for d in range(hd):
                out[(h * rows + r) * hd + d] = x[(r * heads + h) * hd + d]
    return out^


# ---------------------------------------------------------------------------
# OUTPUTS, DIGESTS, COMPARISONS
# ---------------------------------------------------------------------------


struct Outputs(Movable):
    var ctxv: List[Float32]
    var amax: List[Float32]
    var denom: List[Float32]
    var zdot: List[Float32]
    var dq: List[Float32]
    var dk: List[Float32]
    var dv: List[Float32]

    def __init__(out self):
        self.ctxv = List[Float32]()
        self.amax = List[Float32]()
        self.denom = List[Float32]()
        self.zdot = List[Float32]()
        self.dq = List[Float32]()
        self.dk = List[Float32]()
        self.dv = List[Float32]()


def fnv1a64(x: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(x)):
        var u = bitcast[DType.uint32](x[i])
        for k in range(4):
            h = (h ^ UInt64((u >> UInt32(8 * k)) & UInt32(0xFF))) * UInt64(0x100000001B3)
    return h


def hex64(h: UInt64) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((h >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def moved_cells(a: List[Float32], b: List[Float32]) raises -> Int:
    if len(a) != len(b):
        raise Error("compared buffers differ in length: " + String(len(a)) + " vs " + String(len(b)))
    var n = 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            n += 1
    return n


def moved_cells_from_column(a: List[Float32], b: List[Float32], hd: Int, lo: Int) raises -> Int:
    """Cells that differ by bits at head-dim columns `lo` and up of a
    `[B*L][nh*hd]` buffer (the flat index modulo `hd` is the column); the
    DEVIATION 2533 forward sabotage moves columns 0 to 15 only."""
    if len(a) != len(b):
        raise Error("compared buffers differ in length: " + String(len(a)) + " vs " + String(len(b)))
    var n = 0
    for i in range(len(a)):
        if i % hd >= lo:
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                n += 1
    return n


def moved_cells_at_parity(a: List[Float32], b: List[Float32], parity: Int) raises -> Int:
    """Cells that differ by bits at flat indices of the given parity (0 even,
    1 odd). On zdot the flat index is the row `(bb * nh + h) * L + t`; the
    DEVIATION 2598 sabotage flips odd rows only, the 2533 one every row."""
    if len(a) != len(b):
        raise Error("compared buffers differ in length: " + String(len(a)) + " vs " + String(len(b)))
    var n = 0
    for i in range(len(a)):
        if i % 2 == parity:
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                n += 1
    return n


def compare_outputs(kind: String, label: String, refout: Outputs, got: Outputs) raises -> Int:
    """Prints one `BITS` line per buffer; returns the moved-cell total."""
    var total = 0
    var names = List[String]()
    names.append("ctx")
    names.append("amax")
    names.append("denom")
    names.append("zdot")
    names.append("dq")
    names.append("dk")
    names.append("dv")
    for i in range(7):
        var m: Int
        var n: Int
        if i == 0:
            m = moved_cells(refout.ctxv, got.ctxv)
            n = len(refout.ctxv)
        elif i == 1:
            m = moved_cells(refout.amax, got.amax)
            n = len(refout.amax)
        elif i == 2:
            m = moved_cells(refout.denom, got.denom)
            n = len(refout.denom)
        elif i == 3:
            m = moved_cells(refout.zdot, got.zdot)
            n = len(refout.zdot)
        elif i == 4:
            m = moved_cells(refout.dq, got.dq)
            n = len(refout.dq)
        elif i == 5:
            m = moved_cells(refout.dk, got.dk)
            n = len(refout.dk)
        else:
            m = moved_cells(refout.dv, got.dv)
            n = len(refout.dv)
        var verdict = String("MATCH")
        if m > 0:
            verdict = String("MOVED")
        print("BITS " + kind + " " + label + " " + names[i] + " " + verdict + " " + String(m) + " of " + String(n))
        total += m
    return total


def digest_line(kind: String, label: String, o: Outputs):
    print(
        "DIGEST " + kind + " " + label + " ctx " + hex64(fnv1a64(o.ctxv))
        + " amax " + hex64(fnv1a64(o.amax)) + " denom " + hex64(fnv1a64(o.denom))
        + " zdot " + hex64(fnv1a64(o.zdot)) + " dq " + hex64(fnv1a64(o.dq))
        + " dk " + hex64(fnv1a64(o.dk)) + " dv " + hex64(fnv1a64(o.dv))
    )


# ---------------------------------------------------------------------------
# THE REAL-ACTIVATION KIND: `file:<dir>`, the operands a backward launcher
# dumped from the real training path (`-D MOJOLEARN_ATTN_OPERAND_DUMP=1`,
# `MOJOLEARN_ATTN_OPERAND_DUMP_DIR`; fused_attention.mojo `ATTN_OPERAND_DUMP`).
# `meta.txt` line 1: `b l nh nkv hd s pos0 key_lo window scale_bits`;
# q.bin / dctx.bin `[B*L][nh*hd]`, k.bin / v.bin `[B][nkv][S][hd]`, float32
# little-endian, exactly the launcher layouts.
# ---------------------------------------------------------------------------


def _read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def _read_f32(path: String, n: Int) raises -> List[Float32]:
    var bytes = _read_all(path)
    if len(bytes) != n * 4:
        raise Error(path + " holds " + String(len(bytes)) + " bytes, expected " + String(n * 4))
    var out = List[Float32](length=n, fill=Float32(0.0))
    memcpy(dest=out.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=n * 4)
    return out^


def _read_meta(dir: String) raises -> List[Int]:
    """The ten integers of `meta.txt`'s first line."""
    var f = open(dir + "/meta.txt", "r")
    var text = f.read()
    f.close()
    var out = List[Int]()
    var cur = String("")
    var done = False
    for cp in text.codepoint_slices():
        var c = String(cp)
        if c == "\n":
            if cur != "":
                out.append(Int(atol(cur)))
            done = True
            break
        if c == " ":
            if cur != "":
                out.append(Int(atol(cur)))
            cur = String("")
        else:
            cur += c
    if not done and cur != "":
        out.append(Int(atol(cur)))
    if len(out) != 10:
        raise Error(dir + "/meta.txt: expected 10 integers on line 1, read " + String(len(out)))
    return out^


def _kind_label(kind: String) -> String:
    """`file:/a/b/operands-shakespeare` prints as `file:operands-shakespeare`."""
    if not kind.startswith("file:"):
        return kind
    var last = String("")
    var cur = String("")
    for cp in kind.codepoint_slices():
        var c = String(cp)
        if c == "/":
            if cur != "":
                last = cur
            cur = String("")
        else:
            cur += c
    if cur != "":
        last = cur
    return "file:" + last


# ---------------------------------------------------------------------------
# ONE CASE: the device buffers and the launches
# ---------------------------------------------------------------------------


struct Case(Movable):
    var b: Int
    var l: Int
    var nh: Int
    var nkv: Int
    var hd: Int
    var window: Int
    var pos0: Int
    var key_lo: Int
    var s: Int
    var scale: Float32
    var qn: Int
    var kn: Int
    var rows_n: Int
    var q: DeviceBuffer[DType.float32]
    var k: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var dctx: DeviceBuffer[DType.float32]
    var ctxv: DeviceBuffer[DType.float32]
    var amax: DeviceBuffer[DType.float32]
    var denom: DeviceBuffer[DType.float32]
    var zdot: DeviceBuffer[DType.float32]
    var dq: DeviceBuffer[DType.float32]
    var dk: DeviceBuffer[DType.float32]
    var dv: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, kind: String, b: Int, l: Int, nh: Int,
        nkv: Int, hd: Int, window: Int,
    ) raises:
        if kind.startswith("file:"):
            # The real-activation kind: the shape is the dump's, not the
            # environment's.
            var dir = String(kind.removeprefix("file:"))
            var m = _read_meta(dir)
            self.b = m[0]
            self.l = m[1]
            self.nh = m[2]
            self.nkv = m[3]
            self.hd = m[4]
            self.s = m[5]
            self.pos0 = m[6]
            self.key_lo = m[7]
            self.window = m[8]
            self.scale = bitcast[DType.float32](UInt32(m[9]))
            self.qn = self.b * self.l * self.nh * self.hd
            self.kn = self.b * self.nkv * self.s * self.hd
            self.rows_n = self.b * self.nh * self.l
            self.q = _upload(ctx, _read_f32(dir + "/q.bin", self.qn))
            self.k = _upload(ctx, _read_f32(dir + "/k.bin", self.kn))
            self.v = _upload(ctx, _read_f32(dir + "/v.bin", self.kn))
            self.dctx = _upload(ctx, _read_f32(dir + "/dctx.bin", self.qn))
        else:
            self.b = b
            self.l = l
            self.nh = nh
            self.nkv = nkv
            self.hd = hd
            self.window = window
            self.pos0 = 0
            self.key_lo = llama_key_lo(0, window)
            self.s = llama_key_span(0, l, window)
            self.scale = llama_attention_scale(hd)
            self.qn = b * l * nh * hd
            self.kn = b * nkv * self.s * hd
            self.rows_n = b * nh * l
            var qk_scale = 0.125
            if kind == "heavytail":
                qk_scale = 0.0625
            # q and dctx are `[B*L][nh][hd]`; k and v are generated
            # `[B*S][nkv][hd]` and packed to `[B][nkv][S][hd]` (B == 1 here;
            # the general packing is per batch and this harness keeps B at
            # 1 for the caches).
            if b != 1:
                raise Error("attention_step_price: batch must be 1 (the caches are packed per batch)")
            var qh = make_tensor(kind, 1, l, nh, hd, qk_scale, True)
            var kh = to_kv_layout(make_tensor(kind, 2, self.s, nkv, hd, qk_scale, True), self.s, nkv, hd)
            var vh = to_kv_layout(make_tensor(kind, 3, self.s, nkv, hd, 1.0, False), self.s, nkv, hd)
            var dh = make_tensor(kind, 4, l, nh, hd, 1.0, False)
            self.q = _upload(ctx, qh)
            self.k = _upload(ctx, kh)
            self.v = _upload(ctx, vh)
            self.dctx = _upload(ctx, dh)
        self.ctxv = _upload(ctx, List[Float32](length=self.qn, fill=Float32(0.0)))
        self.amax = _upload(ctx, List[Float32](length=self.rows_n, fill=Float32(0.0)))
        self.denom = _upload(ctx, List[Float32](length=self.rows_n, fill=Float32(0.0)))
        self.zdot = _upload(ctx, List[Float32](length=self.rows_n, fill=Float32(0.0)))
        self.dq = _upload(ctx, List[Float32](length=self.qn, fill=Float32(0.0)))
        self.dk = _upload(ctx, List[Float32](length=self.kn, fill=Float32(0.0)))
        self.dv = _upload(ctx, List[Float32](length=self.kn, fill=Float32(0.0)))

    def visible_cells(self) -> Int:
        """Visible (query, key) cells per (batch, head): the mask solved
        row by row, so a window counts what the kernels compute."""
        var n = 0
        for t in range(self.l):
            var p_q = self.pos0 + t
            var hi = p_q - self.key_lo
            if hi > self.s - 1:
                hi = self.s - 1
            var lo = 0
            if self.window > 0:
                lo = p_q - self.window + 1 - self.key_lo
                if lo < 0:
                    lo = 0
            if hi >= lo:
                n += hi - lo + 1
        return n

    def forward(mut self, ctx: DeviceContext, arm: Int) raises -> Int:
        return fused_forward_launch_arm(
            ctx, self.ctxv, self.amax, self.denom, self.q, self.k, self.v,
            self.b, self.l, self.nh, self.nkv, self.hd, self.s, self.pos0,
            self.key_lo, self.window, self.scale, arm,
        )

    def backward(mut self, ctx: DeviceContext, arm: Int) raises -> Int:
        return fused_backward_launch_arm(
            ctx, self.zdot, self.dq, self.dk, self.dv, self.q, self.dctx,
            self.k, self.v, self.amax, self.denom, self.b, self.l, self.nh,
            self.nkv, self.hd, self.s, self.pos0, self.key_lo, self.window,
            self.scale, arm,
        )

    def run_both(mut self, ctx: DeviceContext, arm: Int, label: String) raises:
        """Forward then backward under `arm`, each FUSED_RAN, then a `RAN`
        line naming the kernels that launched (DEVIATION 2534). At head_dim
        64 they must be the arm's resolved kernels on this build, so a
        result line can never carry one arm's name over another's kernels."""
        var ran_f = -1
        var sf = fused_forward_launch_ran(
            ctx, self.ctxv, self.amax, self.denom, self.q, self.k, self.v,
            self.b, self.l, self.nh, self.nkv, self.hd, self.s, self.pos0,
            self.key_lo, self.window, self.scale, arm, ran_f,
        )
        if sf != FUSED_RAN:
            raise Error(label + ": the fused forward did not report FUSED_RAN (status " + String(sf) + ")")
        var ran_b = -1
        var sb = fused_backward_launch_ran(
            ctx, self.zdot, self.dq, self.dk, self.dv, self.q, self.dctx,
            self.k, self.v, self.amax, self.denom, self.b, self.l, self.nh,
            self.nkv, self.hd, self.s, self.pos0, self.key_lo, self.window,
            self.scale, arm, ran_b,
        )
        if sb != FUSED_RAN:
            raise Error(label + ": the fused backward did not report FUSED_RAN (status " + String(sb) + ")")
        print(
            "RAN " + label + " forward=" + fused_attention_arm_name(ran_f)
            + " backward=" + fused_attention_arm_name(ran_b)
        )
        if self.hd == ATTN_STASH_HD:
            var want_f = fused_attention_arm_forward_resolved(arm)
            var want_b = fused_attention_arm_backward_resolved(arm)
            if ran_f != want_f or ran_b != want_b:
                raise Error(
                    label + ": RAN forward " + fused_attention_arm_name(ran_f)
                    + " backward " + fused_attention_arm_name(ran_b)
                    + " and this build resolves " + fused_attention_arm_name(want_f)
                    + " / " + fused_attention_arm_name(want_b)
                )

    def run_pair(mut self, ctx: DeviceContext, fwd_arm: Int, bwd_arm: Int, label: String) raises:
        """`run_both` with the forward under `fwd_arm` and the backward
        under `bwd_arm` (DEVIATION 2598's REACH_Z: a clean forward, so the
        backward's sabotage is read against clean amax and denom). The `RAN`
        check is `run_both`'s, per direction."""
        var ran_f = -1
        var sf = fused_forward_launch_ran(
            ctx, self.ctxv, self.amax, self.denom, self.q, self.k, self.v,
            self.b, self.l, self.nh, self.nkv, self.hd, self.s, self.pos0,
            self.key_lo, self.window, self.scale, fwd_arm, ran_f,
        )
        if sf != FUSED_RAN:
            raise Error(label + ": the fused forward did not report FUSED_RAN (status " + String(sf) + ")")
        var ran_b = -1
        var sb = fused_backward_launch_ran(
            ctx, self.zdot, self.dq, self.dk, self.dv, self.q, self.dctx,
            self.k, self.v, self.amax, self.denom, self.b, self.l, self.nh,
            self.nkv, self.hd, self.s, self.pos0, self.key_lo, self.window,
            self.scale, bwd_arm, ran_b,
        )
        if sb != FUSED_RAN:
            raise Error(label + ": the fused backward did not report FUSED_RAN (status " + String(sb) + ")")
        print(
            "RAN " + label + " forward=" + fused_attention_arm_name(ran_f)
            + " backward=" + fused_attention_arm_name(ran_b)
        )
        if self.hd == ATTN_STASH_HD:
            var want_f = fused_attention_arm_forward_resolved(fwd_arm)
            var want_b = fused_attention_arm_backward_resolved(bwd_arm)
            if ran_f != want_f or ran_b != want_b:
                raise Error(
                    label + ": RAN forward " + fused_attention_arm_name(ran_f)
                    + " backward " + fused_attention_arm_name(ran_b)
                    + " and this build resolves " + fused_attention_arm_name(want_f)
                    + " / " + fused_attention_arm_name(want_b)
                )

    def download(mut self, ctx: DeviceContext) raises -> Outputs:
        var o = Outputs()
        o.ctxv = _download(ctx, self.ctxv, self.qn)
        o.amax = _download(ctx, self.amax, self.rows_n)
        o.denom = _download(ctx, self.denom, self.rows_n)
        o.zdot = _download(ctx, self.zdot, self.rows_n)
        o.dq = _download(ctx, self.dq, self.qn)
        o.dk = _download(ctx, self.dk, self.kn)
        o.dv = _download(ctx, self.dv, self.kn)
        return o^

    def clear_outputs(mut self, ctx: DeviceContext) raises:
        """Poison the outputs between arms so an arm that wrote nothing
        cannot pass by inheriting the previous arm's bits."""
        var nan = bitcast[DType.float32](UInt32(0x7FC00000))
        self.ctxv = _upload(ctx, List[Float32](length=self.qn, fill=nan))
        self.amax = _upload(ctx, List[Float32](length=self.rows_n, fill=nan))
        self.denom = _upload(ctx, List[Float32](length=self.rows_n, fill=nan))
        self.zdot = _upload(ctx, List[Float32](length=self.rows_n, fill=nan))
        self.dq = _upload(ctx, List[Float32](length=self.qn, fill=nan))
        self.dk = _upload(ctx, List[Float32](length=self.kn, fill=nan))
        self.dv = _upload(ctx, List[Float32](length=self.kn, fill=nan))


def eager_oracle(ctx: DeviceContext, mut c: Case) raises -> Outputs:
    """The eager stage kernels on the same inputs, as
    `transformer_fused_check` runs them: the profile's own spelling."""
    var dm = c.nh * c.hd
    var dims = LlamaDims(dm, c.nh, c.nkv, c.hd, 4 * dm)
    dims.validate()
    var stages = LlamaDeviceStages(ctx, c.b, c.l, c.pos0 + c.l, dims, c.window)
    stages.q_rope = _upload(ctx, _download(ctx, c.q, c.qn))
    stages.k_cache = _upload(ctx, _download(ctx, c.k, c.kn))
    stages.v_cache = _upload(ctx, _download(ctx, c.v, c.kn))
    var off = IdentityTrace.disabled()
    var empty_i = List[Int]()
    var empty_b = List[UInt32]()
    attention_eager_core(
        ctx, stages, c.b, c.l, c.s, c.pos0, c.key_lo, c.window, dims,
        PLANT_AT_NONE, empty_i, empty_b, off, String(""),
    )
    var o = Outputs()
    o.ctxv = _download(ctx, stages.ctxv, c.qn)
    o.amax = _download(ctx, stages.amax, c.rows_n)
    o.denom = _download(ctx, stages.denom, c.rows_n)
    var bst = LlamaBackwardStages(ctx, c.b, c.l, c.pos0 + c.l, dims)
    bst.d_attn_ctx = _upload(ctx, _download(ctx, c.dctx, c.qn))
    var offb = IdentityTrace.disabled()
    bwd_attention_eager_stages(
        ctx, bst, stages, c.b, c.l, c.s, c.pos0, c.key_lo, c.window, dims,
        c.scale, offb, String(""),
    )
    o.zdot = _download(ctx, bst.attn_zdot, c.rows_n)
    o.dq = _download(ctx, bst.d_q_rope, c.qn)
    o.dk = _download(ctx, bst.d_k_cache, c.kn)
    o.dv = _download(ctx, bst.d_v_cache, c.kn)
    _ = bst^
    _ = stages^
    return o^


# ---------------------------------------------------------------------------
# RESOURCES (DEVIATIONS 2596 and 2597, brief section 16.3): the compiled
# backward kernels' own attributes, the calls
# bench/gemm_step_resources_main.mojo reads. Launches nothing. Each kernel
# prints its source counts first (per thread, at head_dim 64), then one line
# per attribute, so a vendor that answers some attributes and raises on
# another keeps what it answered; each kernel is its own try.
# ---------------------------------------------------------------------------


def _res_begin(label: String, geometry: String, acc: Int, operands: Int, local_floats: Int, page: Int, allocs: Int):
    print(
        "RESOURCES_BEGIN label=" + label + " " + geometry
        + " accumulators_per_thread=" + String(acc)
        + " operand_registers_per_thread=" + String(operands)
        + " thread_local_floats=" + String(local_floats)
        + " page_bytes=" + String(page) + " allocations=" + String(allocs)
        + " threads_per_block=256"
    )


def _res_zdot_stash_pf(ctx: DeviceContext) raises:
    comptime kern = fused_bwd_zdot_stash_pf_kernel[ATTN_STASH_HD, 4, False]
    var label = String("zdot_stash_pf")
    _res_begin(label, "rows=4 keys_per_iteration=32", 2, 4, 64, 17696, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_zdot_sched_pf[LAG: Bool](ctx: DeviceContext, label: String) raises:
    """DEVIATION 2598's zdot schedule copy: the zdot stash copy's counts,
    plus one shared load and one global store per thread in the staging
    phase (and, under LAG, lane 0's z fold there)."""
    comptime kern = fused_bwd_zdot_sched_pf_kernel[ATTN_STASH_HD, 4, LAG, False]
    _res_begin(label, "rows=4 keys_per_iteration=32", 2, 4, 64, 17696, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_dq_tiled_pf(ctx: DeviceContext) raises:
    comptime kern = fused_bwd_dq_tiled_pf_kernel[ATTN_STASH_HD]
    var label = String("dq_tiled_pf")
    _res_begin(label, "rows=64 keys_per_tile=16", 16, 5, 0, 8448, 3)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_dkdv_recompute(ctx: DeviceContext) raises:
    """The `baseline` arm's dk/dv kernel, which DEVIATION 2596 launches."""
    comptime kern = fused_bwd_dkdv_kernel[ATTN_STASH_HD, 4]
    var label = String("dkdv_recompute")
    _res_begin(label, "keys=4 queries_per_iteration=32", 3, 4, 64, 17696, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_dkdv_tiled(ctx: DeviceContext) raises:
    comptime kern = fused_bwd_dkdv_tiled_kernel[ATTN_STASH_HD, False]
    var label = String("dkdv_tiled")
    _res_begin(label, "keys=64 queries_per_tile=16", 32, 10, 0, 16384, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_dkdv_tiled_pf(ctx: DeviceContext) raises:
    comptime kern = fused_bwd_dkdv_tiled_pf_kernel[ATTN_STASH_HD]
    var label = String("dkdv_tiled_pf")
    _res_begin(label, "keys=64 queries_per_tile=16", 32, 10, 0, 16384, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_dkdv_r2[BJ: Int](ctx: DeviceContext, label: String) raises:
    comptime kern = fused_bwd_dkdv_r2_kernel[ATTN_STASH_HD, BJ, False]
    _res_begin(label, "keys=" + String(BJ) + " queries_per_tile=16", 2 * (BJ // 16) * 4, 10, 0, (2048 + 32 * BJ) * 4, 4)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def _res_kvfold_r2[BJ: Int](ctx: DeviceContext, label: String) raises:
    comptime kern = fused_bwd_kvfold_r2_kernel[ATTN_STASH_HD, BJ, False]
    _res_begin(label, "keys=" + String(BJ) + " queries_per_tile=16", (BJ // 16) * 4, 5, 0, (1024 + 16 * BJ) * 4, 2)
    var f = ctx.compile_function[kern]()
    print("RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print("RESOURCES label=", label, " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")


def run_resources(ctx: DeviceContext) raises:
    """The RESOURCES section: eleven backward kernels at head_dim 64, each in
    its own try (the two DEVIATION 2598 zdot schedule copies beside the zdot
    stash copy they reschedule)."""
    print("RESOURCES_DEVICE column=" + column_name(TARGET_COLUMN) + " source_counts=per_thread_hd64")
    try:
        _res_zdot_stash_pf(ctx)
    except e:
        print("RESOURCES_ERROR label=zdot_stash_pf error=", e, sep="")
    try:
        _res_zdot_sched_pf[False](ctx, String("zdot_zdefer_pf"))
    except e:
        print("RESOURCES_ERROR label=zdot_zdefer_pf error=", e, sep="")
    try:
        _res_zdot_sched_pf[True](ctx, String("zdot_zlag_pf"))
    except e:
        print("RESOURCES_ERROR label=zdot_zlag_pf error=", e, sep="")
    try:
        _res_dq_tiled_pf(ctx)
    except e:
        print("RESOURCES_ERROR label=dq_tiled_pf error=", e, sep="")
    try:
        _res_dkdv_recompute(ctx)
    except e:
        print("RESOURCES_ERROR label=dkdv_recompute error=", e, sep="")
    try:
        _res_dkdv_tiled(ctx)
    except e:
        print("RESOURCES_ERROR label=dkdv_tiled error=", e, sep="")
    try:
        _res_dkdv_tiled_pf(ctx)
    except e:
        print("RESOURCES_ERROR label=dkdv_tiled_pf error=", e, sep="")
    try:
        _res_dkdv_r2[64](ctx, String("kvgrid_r64"))
    except e:
        print("RESOURCES_ERROR label=kvgrid_r64 error=", e, sep="")
    try:
        _res_dkdv_r2[32](ctx, String("kvgrid_r32"))
    except e:
        print("RESOURCES_ERROR label=kvgrid_r32 error=", e, sep="")
    try:
        _res_kvfold_r2[64](ctx, String("kvsplit_r64_fold"))
    except e:
        print("RESOURCES_ERROR label=kvsplit_r64_fold error=", e, sep="")
    try:
        _res_kvfold_r2[32](ctx, String("kvsplit_r32_fold"))
    except e:
        print("RESOURCES_ERROR label=kvsplit_r32_fold error=", e, sep="")
    print("RESOURCES_DONE")


def time_ms(ctx: DeviceContext, mut c: Case, arm: Int, with_backward: Bool, label: String) raises -> Float64:
    ctx.synchronize()
    var t0 = perf_counter_ns()
    var sf = c.forward(ctx, arm)
    var sb = FUSED_RAN
    if with_backward:
        sb = c.backward(ctx, arm)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    if sf != FUSED_RAN or sb != FUSED_RAN:
        raise Error(label + ": a timed launch did not report FUSED_RAN")
    return Float64(t1 - t0) / 1000000.0


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("attention_step_price requires IDENTICAL mode (-D MOJOLEARN_NUMERIC_IDENTICAL=1)")
    var b = _env_int("MOJOLEARN_ATTN_B", 1)
    var l = _env_int("MOJOLEARN_ATTN_L", 2048)
    var nh = _env_int("MOJOLEARN_ATTN_NH", 12)
    var nkv = _env_int("MOJOLEARN_ATTN_NKV", 12)
    var hd = _env_int("MOJOLEARN_ATTN_HD", 64)
    var window = _env_int("MOJOLEARN_ATTN_WINDOW", 0)
    var rounds = _env_int("MOJOLEARN_ATTN_ROUNDS", 7)
    var warmups = _env_int("MOJOLEARN_ATTN_WARMUPS", 2)
    var want_oracle = _env_int("MOJOLEARN_ATTN_ORACLE", 1) != 0
    var want_reach = _env_int("MOJOLEARN_ATTN_REACH", 1) != 0
    var want_timing = _env_int("MOJOLEARN_ATTN_TIMING", 1) != 0
    var cand_raw = _env_str("MOJOLEARN_ATTN_ARM", "bwd_stash")
    var base_raw = _env_str("MOJOLEARN_ATTN_BASELINE", "baseline")
    # DEVIATION 2534: `default` becomes the column's explicit arm name here,
    # before any line is printed.
    var cand_name = _resolve_arm_name(cand_raw)
    var base_name = _resolve_arm_name(base_raw)
    var kinds = _split_list(_env_str("MOJOLEARN_ATTN_KINDS", "hashed"))
    var cand = fused_attention_arm_parse(cand_name)
    var base = fused_attention_arm_parse(base_name)
    if fused_attention_arm_name(cand) != cand_name or fused_attention_arm_name(base) != base_name:
        raise Error(
            "attention_step_price: arm names do not round-trip ('" + cand_name
            + "' -> '" + fused_attention_arm_name(cand) + "', '" + base_name
            + "' -> '" + fused_attention_arm_name(base) + "'); the parser and"
            + " the name function must be inverses (brief section 12.1)"
        )
    var env_shape = (
        "B" + String(b) + "_L" + String(l) + "_nh" + String(nh) + "_nkv"
        + String(nkv) + "_hd" + String(hd) + "_win" + String(window)
    )
    print(
        "=== attention step price, mode " + numeric_mode_name() + ", column "
        + column_name(TARGET_COLUMN) + ", generator shape " + env_shape
        + " (a file: kind carries its own shape)"
    )
    print("trial_hook=" + String(ATTN_ARM_TRIAL) + " phase_timers_built=" + String(ATTN_PHASE_TIMERS))
    # DEVIATION 2534: the column's shipped default, beside what this run
    # asked for, so baseline, stash_tiled and the default are never confused.
    print(
        "DEFAULT column=" + column_name(TARGET_COLUMN) + " arm="
        + fused_attention_arm_name(ATTN_ARM_DEFAULT)
        + " source=kernel_matrix.attn_default_arm_for baseline_is_default="
        + String(base == ATTN_ARM_DEFAULT) + " candidate_is_default="
        + String(cand == ATTN_ARM_DEFAULT) + " baseline_requested=" + base_raw
        + " candidate_requested=" + cand_raw
    )
    comptime if ATTN_PHASE_TIMERS:
        print("NOTE: built with MOJOLEARN_ATTN_PHASE_TIMERS; under MOJOLEARN_TRANSFORMER_TIMING=1 every launch is serialized and PRICE lines are a breakdown, not a price")
    print("baseline=" + base_name + " candidate=" + cand_name + " rounds=" + String(rounds) + " warmups=" + String(warmups))
    print(_path_line("baseline", base))
    print(_path_line("candidate", cand))
    if not ATTN_ARM_TRIAL and cand != ATTN_ARM_BASELINE:
        print("NOTE: no -D MOJOLEARN_ATTN_ARM_TRIAL=1: the candidate arm runs the shipped kernels; reach will FAIL")

    var ctx = DeviceContext()
    # DEVIATIONS 2596 and 2597 (brief section 16.3): the compiled backward
    # kernels' attributes, once per process, before anything launches.
    if _env_int("MOJOLEARN_ATTN_RESOURCES", 1) != 0:
        run_resources(ctx)
    var failures = List[String]()
    for ki in range(len(kinds)):
        var kind_spec = String(kinds[ki])
        var kind = _kind_label(kind_spec)
        print("--- kind " + kind)
        var c = Case(ctx, kind_spec, b, l, nh, nkv, hd, window)
        var shape = (
            "B" + String(c.b) + "_L" + String(c.l) + "_nh" + String(c.nh) + "_nkv"
            + String(c.nkv) + "_hd" + String(c.hd) + "_win" + String(c.window)
        )
        var hd_c = c.hd
        var cells = c.visible_cells() * c.b * c.nh
        print("shape=" + shape + " visible_cells=" + String(cells) + " S=" + String(c.s) + " key_lo=" + String(c.key_lo) + " pos0=" + String(c.pos0))

        # 1. the baseline, and the eager oracle against it
        c.clear_outputs(ctx)
        c.run_both(ctx, base, kind + " " + base_name)
        var refout = c.download(ctx)
        digest_line(kind, base_name, refout)
        if want_oracle:
            var orc = eager_oracle(ctx, c)
            digest_line(kind, "eager", orc)
            var moved = compare_outputs(kind, base_name + "_vs_eager", orc, refout)
            if moved > 0:
                failures.append(kind + ": " + base_name + " differs from the eager oracle in " + String(moved) + " cells")

        # 2. the candidate, bit for bit
        c.clear_outputs(ctx)
        c.run_both(ctx, cand, kind + " " + cand_name)
        var got = c.download(ctx)
        digest_line(kind, cand_name, got)
        var moved_c = compare_outputs(kind, cand_name + "_vs_" + base_name, refout, got)
        if moved_c > 0:
            failures.append(kind + ": " + cand_name + " differs from " + base_name + " in " + String(moved_c) + " cells")

        # 3. reach by sabotage, then the clean arm restores the bits
        if want_reach and cand != ATTN_ARM_BASELINE:
            # A second-round arm proves reach with ATTN_ARM_SABOTAGE_NEW,
            # which flips only the new kernel (brief section 12.1).
            var reach_bit = fused_attention_arm_reach_bit(cand)
            var sab_arm = cand | reach_bit
            var sab_name = fused_attention_arm_name(sab_arm)
            c.clear_outputs(ctx)
            c.run_both(ctx, sab_arm, kind + " " + sab_name)
            var sab = c.download(ctx)
            var flipped = compare_outputs(kind, sab_name + "_vs_" + base_name, refout, sab)
            var amax_den_moved = moved_cells(refout.amax, sab.amax) + moved_cells(refout.denom, sab.denom)
            var fwd_moved = moved_cells(refout.ctxv, sab.ctxv) + amax_den_moved
            var bwd_moved = flipped - fwd_moved
            var ctx_hi_moved = moved_cells_from_column(refout.ctxv, sab.ctxv, c.hd, 16)
            var zdot_moved = moved_cells(refout.zdot, sab.zdot)
            var dv_moved = moved_cells(refout.dv, sab.dv)
            var zdot_even = moved_cells_at_parity(refout.zdot, sab.zdot, 0)
            var zdot_odd = moved_cells_at_parity(refout.zdot, sab.zdot, 1)
            var zs = fused_attention_arm_zsched(cand) != 0
            # DEVIATION 2598 (brief section 17.4): the zdot schedule copy's
            # flip with the forward clean, so the backward reads clean amax
            # and denom even on an arm whose forward flip moves them.
            if zs:
                var zb_arm = cand | ATTN_ARM_SABOTAGE_NEW
                var zb_name = fused_attention_arm_name(zb_arm)
                c.clear_outputs(ctx)
                c.run_pair(ctx, cand, zb_arm, kind + " " + cand_name + " forward and " + zb_name + " backward")
                var zo = c.download(ctx)
                var z_flipped = compare_outputs(kind, zb_name + "_backward_only_vs_" + base_name, refout, zo)
                var z_fwd = moved_cells(refout.ctxv, zo.ctxv) + moved_cells(refout.amax, zo.amax) + moved_cells(refout.denom, zo.denom)
                var z_even = moved_cells_at_parity(refout.zdot, zo.zdot, 0)
                var z_odd = moved_cells_at_parity(refout.zdot, zo.zdot, 1)
                var z_dv = moved_cells(refout.dv, zo.dv)
                print(
                    "REACH_Z " + kind + " " + cand_name + " sabotage_flipped_cells=" + String(z_flipped)
                    + " reach_bit=" + zb_name + " backward_only=True forward_moved=" + String(z_fwd)
                    + " zdot_moved_even_rows=" + String(z_even) + " zdot_moved_odd_rows=" + String(z_odd)
                    + " dv_moved=" + String(z_dv)
                )
                if z_odd == 0:
                    failures.append(kind + ": ZDOT SCHEDULE REACH NOT PROVEN for " + cand_name + " (" + zb_name + " on the backward moved zdot at no odd row)")
                if z_even > 0 or z_dv > 0 or z_fwd > 0:
                    failures.append(kind + ": " + zb_name + " on the backward moved zdot at " + String(z_even) + " even rows, dv in " + String(z_dv) + " and the forward in " + String(z_fwd) + " cells; the DEVIATION 2598 flip reaches zdot at odd rows (and dq, dk through it) only")
            # DEVIATIONS 2596 and 2597 (brief section 16.4): the dk/dv
            # launch's own flip, before the clean restore below.
            if fused_attention_arm_kv(cand):
                var kv_arm = cand | ATTN_ARM_SABOTAGE_KV
                var kv_name = fused_attention_arm_name(kv_arm)
                c.clear_outputs(ctx)
                c.run_both(ctx, kv_arm, kind + " " + kv_name)
                var kvo = c.download(ctx)
                var kv_flipped = compare_outputs(kind, kv_name + "_vs_" + base_name, refout, kvo)
                var kv_fwd = moved_cells(refout.ctxv, kvo.ctxv) + moved_cells(refout.amax, kvo.amax) + moved_cells(refout.denom, kvo.denom)
                var kv_z = moved_cells(refout.zdot, kvo.zdot)
                var kv_dq = moved_cells(refout.dq, kvo.dq)
                var kv_dk = moved_cells(refout.dk, kvo.dk)
                var kv_dv = moved_cells(refout.dv, kvo.dv)
                print(
                    "REACH_KV " + kind + " " + cand_name + " sabotage_flipped_cells=" + String(kv_flipped)
                    + " reach_bit=" + kv_name + " dk_moved=" + String(kv_dk) + " dv_moved=" + String(kv_dv)
                    + " zdot_moved=" + String(kv_z) + " dq_moved=" + String(kv_dq)
                    + " forward_moved=" + String(kv_fwd)
                )
                if kv_dk == 0 or kv_dv == 0:
                    failures.append(kind + ": DK/DV REACH NOT PROVEN for " + cand_name + " (" + kv_name + " moved dk in " + String(kv_dk) + " and dv in " + String(kv_dv) + " cells; both must move)")
                if kv_fwd + kv_z + kv_dq > 0:
                    failures.append(kind + ": " + kv_name + " moved " + String(kv_fwd + kv_z + kv_dq) + " forward, zdot or dq cells; the DEVIATION 2596 / 2597 flips reach dk and dv only")
            c.clear_outputs(ctx)
            c.run_both(ctx, cand, kind + " " + cand_name + " (restore)")
            var again = c.download(ctx)
            var restored = compare_outputs(kind, cand_name + "_restored_vs_" + base_name, refout, again) == 0
            print(
                "REACH " + kind + " " + cand_name + " sabotage_flipped_cells=" + String(flipped)
                + " clean_restored=" + String(restored) + " reach_bit=" + sab_name
                + " forward_moved=" + String(fwd_moved) + " backward_moved="
                + String(bwd_moved) + " amax_denom_moved=" + String(amax_den_moved)
                + " ctx_moved_columns_16_up=" + String(ctx_hi_moved)
                + " zdot_moved=" + String(zdot_moved) + " dv_moved=" + String(dv_moved)
                + " zdot_moved_even_rows=" + String(zdot_even)
                + " zdot_moved_odd_rows=" + String(zdot_odd)
            )
            if flipped == 0:
                failures.append(kind + ": REACH NOT PROVEN for " + cand_name + " (sabotage moved nothing: build lacks -D MOJOLEARN_ATTN_ARM_TRIAL=1, or the arm is not wired at this head dim)")
            if reach_bit == ATTN_ARM_SABOTAGE_NEW:
                # Per branch (brief sections 12.4 and 14.5). The backward
                # reads the forward's amax and denom, so a backward-must-hold
                # or backward attribution is asserted only when the
                # sabotaged forward left both alone.
                var nf = fused_attention_arm_new_forward(cand)
                var nb = fused_attention_arm_new_backward(cand)
                var qres = (cand & ATTN_ARM_FWD_QRES) != 0
                var pf = (cand & ATTN_ARM_PREFLUSH) != 0
                var zt = (cand & ATTN_ARM_BWD_ZTILED) != 0
                if nf and fwd_moved == 0:
                    failures.append(kind + ": FORWARD REACH NOT PROVEN for " + cand_name + " (" + sab_name + " moved no forward cell)")
                if nb and bwd_moved == 0:
                    failures.append(kind + ": BACKWARD REACH NOT PROVEN for " + cand_name + " (" + sab_name + " moved no backward cell)")
                if not nf and fwd_moved > 0:
                    failures.append(kind + ": " + sab_name + " moved " + String(fwd_moved) + " forward cells; " + cand_name + " has no second-round forward kernel, so its sabotage must reach no forward buffer")
                if not nb and amax_den_moved == 0 and bwd_moved > 0:
                    failures.append(kind + ": " + sab_name + " moved " + String(bwd_moved) + " backward cells with amax and denom unmoved; " + cand_name + " has no second-round backward kernel, so its sabotage must reach no backward buffer")
                if nf and qres and amax_den_moved == 0:
                    failures.append(kind + ": " + sab_name + ": the DEVIATION 2530 flip (every staged Q value) moved neither amax nor denom; the Q residency instantiation did not run")
                if nf and not qres and amax_den_moved > 0:
                    failures.append(kind + ": " + sab_name + " moved " + String(amax_den_moved) + " amax/denom cells; the 2531 and 2533 forward flips reach ctx only")
                if nf and not qres and pf and ctx_hi_moved > 0:
                    failures.append(kind + ": " + sab_name + " moved " + String(ctx_hi_moved) + " ctx cells at columns 16 and up; the 2533 forward flip reaches columns 0 to 15 only")
                if nb and not zt and amax_den_moved == 0 and (zdot_moved == 0 or dv_moved > 0):
                    failures.append(kind + ": " + sab_name + ": the DEVIATION 2533 backward flip (the stored zdot) must move zdot and hold dv; zdot moved " + String(zdot_moved) + ", dv moved " + String(dv_moved))
                if nb and not zt and zs and amax_den_moved == 0 and (zdot_odd == 0 or zdot_even > 0):
                    failures.append(kind + ": " + sab_name + ": the DEVIATION 2598 flip moves zdot at odd rows only; zdot moved at " + String(zdot_odd) + " odd and " + String(zdot_even) + " even rows")
            if not restored:
                failures.append(kind + ": " + cand_name + " did not restore the baseline bits after sabotage")

        # 4. the price, arms alternated inside each round
        if want_timing:
            for _ in range(warmups):
                _ = time_ms(ctx, c, base, True, "warmup " + base_name)
                _ = time_ms(ctx, c, cand, True, "warmup " + cand_name)
            var base_fwd = List[Float64]()
            var base_both = List[Float64]()
            var cand_fwd = List[Float64]()
            var cand_both = List[Float64]()
            for r in range(rounds):
                var first_is_base = (r % 2) == 0
                for half in range(2):
                    var take_base = first_is_base == (half == 0)
                    var arm = base
                    var name = String(base_name)
                    if not take_base:
                        arm = cand
                        name = String(cand_name)
                    var f = time_ms(ctx, c, arm, False, name)
                    var fb = time_ms(ctx, c, arm, True, name)
                    print("PRICE attention " + name + " fwd " + kind + " " + String(f) + " ms")
                    print("PRICE attention " + name + " fwdbwd " + kind + " " + String(fb) + " ms")
                    if take_base:
                        base_fwd.append(f)
                        base_both.append(fb)
                    else:
                        cand_fwd.append(f)
                        cand_both.append(fb)
            var useful_fwd = 4.0 * Float64(cells) * Float64(hd_c)
            var useful_bwd = 10.0 * Float64(cells) * Float64(hd_c)
            var bf = _median(base_fwd.copy())
            var bb = _median(base_both.copy())
            var cf = _median(cand_fwd.copy())
            var cb = _median(cand_both.copy())
            print("TABLE kind=" + kind + " shape=" + shape + " column=" + column_name(TARGET_COLUMN) + " default_arm=" + fused_attention_arm_name(ATTN_ARM_DEFAULT) + " visible_cells=" + String(cells) + " (useful flops: fwd 4*cells*hd, bwd 10*cells*hd; executed multiplier of the shipped kernels: fwd dots 3 of 2 contractions, bwd dots 6 of 5)")
            print("TABLE arm | fwd median ms | fwd useful TFLOP/s | fwd+bwd median ms | bwd (derived) ms | bwd useful TFLOP/s | fwd ratio | fwd+bwd ratio")
            print(
                "TABLE " + base_name + " | " + String(bf) + " | " + String(useful_fwd / (bf * 1e9))
                + " | " + String(bb) + " | " + String(bb - bf) + " | " + String(useful_bwd / ((bb - bf) * 1e9))
                + " | 1.0 | 1.0"
            )
            print(
                "TABLE " + cand_name + " | " + String(cf) + " | " + String(useful_fwd / (cf * 1e9))
                + " | " + String(cb) + " | " + String(cb - cf) + " | " + String(useful_bwd / ((cb - cf) * 1e9))
                + " | " + String(bf / cf) + " | " + String(bb / cb)
            )
        _ = c^

    if len(failures) > 0:
        for i in range(len(failures)):
            print("FAIL " + failures[i])
        raise Error("attention_step_price: " + String(len(failures)) + " failure(s); see the FAIL lines")
    print("attention_step_price: PASS (" + fused_attention_arm_name(cand) + " vs " + fused_attention_arm_name(base) + ")")
    _ = ctx^
