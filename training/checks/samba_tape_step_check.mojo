# SPDX-License-Identifier: Apache-2.0
"""The Samba train step's tape backward (autograd/rules_samba.mojo) against stored words.

For each case (d_model 64, vocab 256, B 2, L 32; a mixed Mamba-3 / attention
stack tied and untied, and a Mamba-3-only stack), runs one resident train
step (forward, loss, backward, AdamW update) on a fresh session and requires
the loss bits, the FNV-1a digest of every registry gradient tensor and the
digest of the updated parameters to equal the stored ones
(training/checks/samba_tape_step_digests.tsv: the words the tape and the
deleted hand walk both produced on Metal, NVIDIA and AMD). Prints every
digest so columns can be compared across vendors. Run from the repo root:

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/samba_tape_step_check.mojo
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext

from checks.identity_trace_read import read_trace_lines

from mamba.checks.mamba3_fixture import M3_D_STATE, Mamba3Dims
from training.checks.loss_contract import IGNORE_INDEX_DEFAULT
from training.checks.optimizer_contract import OPT_ADAMW
from training.checks.train_loop import download_f32
from training.samba_resident import (
    SAMBA_LAYER_ATTENTION,
    SAMBA_LAYER_MAMBA3,
    SambaResidentSession,
    samba_resident_train_step,
)


@fieldwise_init
struct Case(Copyable, Movable):
    var tag: String
    var kinds: List[Int]
    var vocab: Int
    var dm: Int
    var nh: Int
    var nkv: Int
    var hd: Int
    var it: Int
    var tie: Bool
    var b: Int
    var l: Int
    var max_norm: Float32


def digest(values: List[Float32], lo: Int, hi: Int) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(lo, hi):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def registry(c: Case) raises -> Tuple[List[Int], List[Bool], List[String]]:
    """The registry offsets (SambaConfig.registry order, as the session
    builds them), whether each tensor is a norm weight, and its name."""
    var offsets = List[Int]()
    var norm = List[Bool]()
    var names = List[String]()
    var running = 0
    offsets.append(running)
    running += c.vocab * c.dm
    offsets.append(running)
    norm.append(False)
    names.append("embed.weight")
    for i in range(len(c.kinds)):
        var sizes = List[Int]()
        var is_norm = List[Bool]()
        if c.kinds[i] == SAMBA_LAYER_MAMBA3:
            var d3 = Mamba3Dims.of(c.dm)
            sizes = [
                c.dm, d3.d_in_proj() * c.dm, d3.nheads, M3_D_STATE, M3_D_STATE,
                d3.nheads * M3_D_STATE, d3.nheads * M3_D_STATE, d3.nheads, c.dm * d3.d_inner,
            ]
            is_norm = [True, False, False, True, True, False, False, False, False]
        else:
            var qw = c.nh * c.hd
            var kw = c.nkv * c.hd
            sizes = [c.dm, c.dm, qw * c.dm, kw * c.dm, kw * c.dm, c.dm * qw, c.it * c.dm, c.it * c.dm, c.dm * c.it]
            is_norm = [True, True, False, False, False, False, False, False, False]
        for k in range(9):
            running += sizes[k]
            offsets.append(running)
            norm.append(is_norm[k])
            names.append(String("layers.") + String(i) + ".w" + String(k))
    running += c.dm
    offsets.append(running)
    norm.append(True)
    names.append("norm_f.weight")
    if not c.tie:
        running += c.vocab * c.dm
        offsets.append(running)
        norm.append(False)
        names.append("lm_head.weight")
    return (offsets^, norm^, names^)


def params(offsets: List[Int], norm: List[Bool]) -> List[Float32]:
    # norm weights near 1, everything else small and deterministic
    var out = List[Float32]()
    for j in range(len(norm)):
        for i in range(offsets[j], offsets[j + 1]):
            var x = Float32((i * 17 + j * 13) % 101 - 50) / Float32(5000.0)
            out.append(x + Float32(1.0) if norm[j] else x)
    return out^


#: case, tensor, words, value (the loss's float32 bits or a tensor's FNV-1a digest)
comptime STORED_PATH = "training/checks/samba_tape_step_digests.tsv"


@fieldwise_init
struct Stored(Copyable, Movable):
    var keys: List[String]
    var values: List[String]

    def lookup(self, key: String) raises -> String:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i].copy()
        raise Error("samba_tape_step_check: no stored word for " + key + " in " + STORED_PATH)


def load_stored() raises -> Stored:
    var keys = List[String]()
    var values = List[String]()
    var lines = read_trace_lines(STORED_PATH)
    for k in range(len(lines)):
        var fields = lines[k].split("\t")
        if len(fields) != 4:
            raise Error("samba_tape_step_check: malformed stored line: " + lines[k])
        keys.append(String(fields[0]) + "\t" + String(fields[1]) + "\t" + String(fields[2]))
        values.append(String(fields[3]))
    return Stored(keys^, values^)


def step(ctx: DeviceContext, c: Case) raises -> Tuple[Float32, List[Float32], List[Float32]]:
    """One resident train step on a fresh session: (loss, gradient, updated parameters)."""
    var reg = registry(c)
    var offsets = reg[0].copy()
    var n_tensors = len(offsets) - 1
    var n_total = offsets[n_tensors]
    var p = params(offsets, reg[1])
    var s = SambaResidentSession(
        ctx, c.kinds, c.vocab, c.dm, c.nh, c.nkv, c.hd, c.it, c.tie, Float32(1e-5), n_total,
        p.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var m = c.b * c.l
    var ids = List[Int32]()
    var targets = List[Int32]()
    for i in range(m):
        ids.append(Int32((i * 127 + 11) % c.vocab))
        targets.append(Int32((i * 31 + 7) % c.vocab))
    targets[5] = Int32(IGNORE_INDEX_DEFAULT)
    var offs = List[Int32]()
    for j in range(n_tensors + 1):
        offs.append(Int32(offsets[j]))
    var init = List[Int32](length=n_tensors, fill=Int32(0))
    var info = List[Float32](length=3, fill=Float32(0.0))
    var loss = List[Float32](length=1, fill=Float32(0.0))
    var m_state = ctx.enqueue_create_buffer[DType.float32](n_total)
    var v_state = ctx.enqueue_create_buffer[DType.float32](n_total)
    m_state.enqueue_fill(Float32(0.0))
    v_state.enqueue_fill(Float32(0.0))
    var p_stage = ctx.enqueue_create_host_buffer[DType.float32](n_total)
    var g_stage = ctx.enqueue_create_host_buffer[DType.float32](n_total)
    ctx.synchronize()
    _ = samba_resident_train_step(
        s,
        p.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        ids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        targets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        loss.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        m_state, v_state, p_stage, g_stage,
        offs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        init.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        c.b, c.l, n_tensors, OPT_ADAMW, 1, 0,
        Float32(1e-3), Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.01),
        Float32(0.0), Float32(0.0), c.max_norm,
    )
    var grad = download_f32(ctx, s.grad, n_total)
    var l0 = loss[0]
    _ = ids^
    _ = targets^
    _ = offs^
    _ = init^
    _ = info^
    _ = loss^
    _ = m_state^
    _ = v_state^
    _ = p_stage^
    _ = g_stage^
    _ = s^
    return (l0, grad^, p^)


def check_word(stored: Stored, tag: String, name: String, words: Int, got: String) raises -> Int:
    var want = stored.lookup(tag + "\t" + name + "\t" + String(words))
    var same = got == want
    print("SAMBA_TAPE", tag, name, words, "tape", got, "stored", want, "SAME" if same else "DIFFER")
    return 0 if same else 1


def run_case(ctx: DeviceContext, stored: Stored, c: Case) raises -> Int:
    var tape = step(ctx, c)
    var bad = 0
    bad += check_word(stored, c.tag, "loss", 1, String(bitcast[DType.uint32](tape[0])))
    var reg = registry(c)
    var offsets = reg[0].copy()
    var names = reg[2].copy()
    if len(tape[1]) != offsets[len(names)] or len(tape[2]) != offsets[len(names)]:
        raise Error(c.tag + ": gradient or parameter length is not the registry's")
    for j in range(len(names)):
        var lo = offsets[j]
        var hi = offsets[j + 1]
        bad += check_word(stored, c.tag, String("grad.") + names[j], hi - lo, String(digest(tape[1], lo, hi)))
    bad += check_word(stored, c.tag, "params_after", len(tape[2]), String(digest(tape[2], 0, len(tape[2]))))
    return bad


def main() raises:
    var ctx = DeviceContext()
    var stored = load_stored()
    var bad = 0
    var m3 = SAMBA_LAYER_MAMBA3
    var at = SAMBA_LAYER_ATTENTION
    # tag, kinds, vocab, d_model, n_heads, n_kv, head_dim, intermediate, tie, B, L, max_norm
    bad += run_case(ctx, stored, Case("tied_m3_attn", [m3, at, m3, at], 256, 64, 8, 4, 8, 128, True, 2, 32, Float32(1.0)))
    bad += run_case(ctx, stored, Case("untied_attn_m3_m3_attn", [at, m3, m3, at], 256, 64, 8, 4, 8, 128, False, 2, 32, Float32(0.0)))
    bad += run_case(ctx, stored, Case("tied_m3_only", [m3, m3], 256, 64, 0, 0, 0, 0, True, 2, 32, Float32(0.0)))
    if bad != 0:
        raise Error("SAMBA_TAPE_STEP_FAIL " + String(bad) + " loss / gradient / parameter words differ from the stored ones")
    print("SAMBA_TAPE_STEP_PASS every loss, gradient and updated parameter digest equals the stored one")
