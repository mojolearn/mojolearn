# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PUBLIC SEQUENCE OPTIMIZERS' ONE NATIVE CALL PER METHOD (lane
neural-optimizers, 2026-10-21): `RMSprop`, `Adagrad`, `Lion`, `Adamax`,
`NAdam`, `LAMB` and `Adafactor` in python/mojolearn/_x_sequence_optim.py.

The Python methods check their arguments and pass buffer addresses, sizes
read off the buffers and the hyperparameter words; everything that used to
run in Python around the step runs here: the parameter traversal (the
packing of several tensors into one flat buffer and the unpacking after),
the size sums, the step counter, the host scalars' reset, and the state
flags. The step itself is the existing one, unchanged: `opt_step_impl`,
`lamb_step_impl` and `adafactor_step_impl` (sequence/ops_impl.mojo) on the
host route, and the resident entries of sequence/opt_resident.mojo on the
device route (sequence/opt_public_device.mojo). The same launches on the
same values in the same order: no bit moves.

THE STATE WORDS. Each optimizer owns an int64 buffer of N_WORDS words that
the native calls read and write and the Python properties read back (`t`,
`_sc_t`, ...), so a call never assigns a Python attribute: W_T the one-based
step count, W_SC_T the step the carried host scalars belong to, W_HANDLE 1
once the resident state is open (else -1), W_USED the resident slot mask,
W_FRESH 1 when the host copies of the state hold the newest values,
W_OWNED 1 when the next resident step uploads the host copies first,
W_PRISTINE 1 while the host copies are still the constructor's, W_N the
parameter count, W_FLAGS LAMB's flags. A separate int64 buffer holds the
resident handles (one, or one per tensor for Adafactor), -1 when closed.

This file is the host-safe half (generic over `Exec`, no device import):
the CPU host binding serves the optimizers from it alone. No Python here."""
from std.memory import memcpy

from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.ops_impl import adafactor_step_impl, lamb_step_impl, opt_step_impl

comptime IP64 = MutPointer[Int64, MutUntrackedOrigin]

comptime W_T = 0
comptime W_SC_T = 1
comptime W_HANDLE = 2
comptime W_USED = 3
comptime W_FRESH = 4
comptime W_OWNED = 5
comptime W_PRISTINE = 6
comptime W_N = 7
comptime W_FLAGS = 8
comptime N_WORDS = 9

#: `seq_optimizer_init`'s modes
comptime MODE_ELEMENTWISE = 0
comptime MODE_LAMB = 1
comptime MODE_ADAFACTOR = 2


def words_ptr(addr: Int) raises -> IP64:
    """The optimizer's state words at `addr`."""
    if addr == 0:
        raise Error("sequence optimizer: null state-word buffer")
    return IP64(unsafe_from_address=addr)


def word(w: IP64, k: Int) -> Int:
    return Int(w[k])


def set_word(w: IP64, k: Int, v: Int):
    w[k] = Int64(v)


def host_fp(addr: Int, what: String) raises -> FP:
    if addr == 0:
        raise Error("sequence optimizer: null float32 buffer for " + what)
    return FP(unsafe_from_address=addr)


def seq_optimizer_init(words: Int, ip: List[Int]) raises -> Int:
    """The constructor's native half. ip = [mode, J, size_0 .. size_{J-1}]
    and, for LAMB, [trust_clip, always_adapt, grad_averaging,
    bias_correction, has_max_grad_norm] (each 0 or 1). Writes the state
    words (step 0, nothing resident, the host copies fresh, owned and
    pristine, the parameter count and LAMB's flags) and returns the
    parameter count."""
    if len(ip) < 2:
        raise Error("sequence optimizer: init requires a mode and a tensor count")
    var mode = ip[0]
    var J = ip[1]
    if mode < MODE_ELEMENTWISE or mode > MODE_ADAFACTOR or J < 0:
        raise Error("sequence optimizer: unknown mode or a negative tensor count")
    var extra = 5 if mode == MODE_LAMB else 0
    if len(ip) != 2 + J + extra:
        raise Error("sequence optimizer: init requires one size per tensor")
    var n = 0
    for j in range(J):
        var s = ip[2 + j]
        if s < 0:
            raise Error("sequence optimizer: a tensor size is negative")
        n += s
    var flags = 0
    if mode == MODE_LAMB:
        var at = 2 + J
        flags = (
            (1 if ip[at] != 0 else 0)
            | (2 if ip[at + 1] != 0 else 0)
            | (4 if ip[at + 2] != 0 else 0)
            | (8 if ip[at + 3] != 0 else 0)
            | (16 if ip[at + 4] != 0 else 0)
        )
    var w = words_ptr(words)
    set_word(w, W_T, 0)
    set_word(w, W_SC_T, 0)
    set_word(w, W_HANDLE, -1)
    set_word(w, W_USED, 0)
    set_word(w, W_FRESH, 1)
    set_word(w, W_OWNED, 1)
    set_word(w, W_PRISTINE, 1)
    set_word(w, W_N, n)
    set_word(w, W_FLAGS, flags)
    return n


struct SeqStep(Movable):
    """A step call's arguments, parsed. addrs = [p_0 .. p_{J-1}, g_0 ..
    g_{J-1}] then the mode's state addresses; ip = [J, dev, gk_0 ..
    gk_{J-1}, size_0 .. size_{J-1}] then the mode's integer words. `dev` 1:
    the params are resident sequence tensors (handles) and grad j is a
    handle (gk_j = 0) or a host address (gk_j = 1)."""

    var J: Int
    var dev: Bool
    var gk: List[Int]
    var sizes: List[Int]
    var ps: List[Int]
    var gs: List[Int]
    var rest_a: List[Int]
    var rest_i: List[Int]

    def __init__(out self, addrs: List[Int], ip: List[Int]) raises:
        if len(ip) < 2:
            raise Error("sequence optimizer: a step requires the tensor count and the device flag")
        self.J = ip[0]
        self.dev = ip[1] != 0
        if self.J < 1 or len(ip) < 2 + 2 * self.J or len(addrs) < 2 * self.J:
            raise Error("sequence optimizer: a step requires J >= 1 tensors, their sizes and 2 J addresses")
        self.gk = List[Int]()
        self.sizes = List[Int]()
        self.ps = List[Int]()
        self.gs = List[Int]()
        for j in range(self.J):
            self.gk.append(ip[2 + j])
            self.sizes.append(ip[2 + self.J + j])
            self.ps.append(addrs[j])
            self.gs.append(addrs[self.J + j])
        self.rest_a = List[Int]()
        for k in range(2 * self.J, len(addrs)):
            self.rest_a.append(addrs[k])
        self.rest_i = List[Int]()
        for k in range(2 + 2 * self.J, len(ip)):
            self.rest_i.append(ip[k])

    def total(self) raises -> Int:
        var n = 0
        for j in range(self.J):
            if self.sizes[j] < 0:
                raise Error("sequence optimizer: a tensor size is negative")
            n += self.sizes[j]
        return n


def begin_step(w: IP64, sc: Int, sc_len: Int) raises -> Int:
    """The step about to run (the count plus one). The carried host scalars
    (sc_len float32 at `sc`) belong to step W_SC_T; when that is not before
    this step (a caller rewound `t`), they restart at 1.0 from step 0."""
    var t = word(w, W_T) + 1
    if word(w, W_SC_T) >= t:
        var p = host_fp(sc, "scalars")
        for k in range(sc_len):
            p[k] = Float32(1.0)
        set_word(w, W_SC_T, 0)
    return t


def _overlap(a: Int, b: Int, n: Int) -> Bool:
    """Whether the n-float ranges at a and b share memory."""
    return a < b + 4 * n and b < a + 4 * n


def gather(addrs: List[Int], sizes: List[Int], n: Int) raises -> List[Float32]:
    """The tensors at `addrs` end to end in one flat buffer (a copy)."""
    var buf = List[Float32](length=max(n, 1), fill=Float32(0.0))
    var dst = FP(unsafe_from_address=Int(buf.unsafe_ptr()))
    var off = 0
    for j in range(len(sizes)):
        if sizes[j] > 0:
            memcpy(dest=dst + off, src=host_fp(addrs[j], "tensor"), count=sizes[j])
        off += sizes[j]
    return buf^


def scatter(buf: List[Float32], addrs: List[Int], sizes: List[Int]) raises:
    """The flat buffer back into the tensors at `addrs`, in place."""
    var src = FP(unsafe_from_address=Int(buf.unsafe_ptr()))
    var off = 0
    for j in range(len(sizes)):
        if sizes[j] > 0:
            memcpy(dest=host_fp(addrs[j], "tensor"), src=src + off, count=sizes[j])
        off += sizes[j]


def _f64s(fp: List[Float64], lo: Int, hi: Int) raises -> List[Float64]:
    if len(fp) < hi:
        raise Error("sequence optimizer: too few float parameters")
    var out = List[Float64]()
    for k in range(lo, hi):
        out.append(fp[k])
    return out^


def elementwise_step_host[E: Exec](mut ex: E, w: IP64, a: SeqStep, kind: Int, flags: Int,
                                   fp: List[Float64]) raises -> Int:
    """The element-wise optimizers' per-call step (`opt_step_impl`) on host
    arrays: one tensor in place, several packed into one flat buffer and
    unpacked after. a.rest_a = [state1, state2, state3, scalars]."""
    if a.dev:
        raise Error("sequence optimizer: this binding has no device-parameter step")
    if len(a.rest_a) != 4:
        raise Error("sequence optimizer: an element-wise step requires 3 state addresses and the scalars")
    var n = a.total()
    if n != word(w, W_N):
        raise Error("sequence optimizer: the params were resized under the optimizer")
    var sc = a.rest_a[3]
    var t = begin_step(w, sc, 3)
    var pbuf = List[Float32]()
    var gbuf = List[Float32]()
    var P: Int
    var G: Int
    if a.J == 1:
        P = a.ps[0]
        G = a.gs[0]
        if _overlap(P, G, n):
            gbuf = gather(a.gs, a.sizes, n)
            G = Int(gbuf.unsafe_ptr())
    else:
        pbuf = gather(a.ps, a.sizes, n)
        gbuf = gather(a.gs, a.sizes, n)
        P = Int(pbuf.unsafe_ptr())
        G = Int(gbuf.unsafe_ptr())
    var addrs: List[Int] = [P, G, a.rest_a[0], a.rest_a[1], a.rest_a[2], sc]
    var ip: List[Int] = [n, kind, flags, t, word(w, W_SC_T)]
    _ = opt_step_impl(ex, addrs, ip, _f64s(fp, 0, 6))
    if a.J > 1:
        scatter(pbuf, a.ps, a.sizes)
    _ = pbuf^
    _ = gbuf^
    set_word(w, W_T, t)
    set_word(w, W_SC_T, t)
    return n


def lamb_offsets_of(sizes: List[Int]) -> List[Int]:
    """[0, size_0, size_0 + size_1, ...]: the tensors' offsets."""
    var offs: List[Int] = [0]
    for j in range(len(sizes)):
        offs.append(offs[j] + sizes[j])
    return offs^


def lamb_step_host[E: Exec](mut ex: E, w: IP64, a: SeqStep, fp: List[Float64]) raises -> Int:
    """LAMB's per-call step (`lamb_step_impl`) over the tensors packed into
    one flat buffer. a.rest_a = [exp_avg, exp_avg_sq, scalars]; fp = [lr,
    beta1, beta2, eps, weight_decay, max_grad_norm]."""
    if a.dev:
        raise Error("LAMB: this binding has no device-parameter step")
    if len(a.rest_a) != 3:
        raise Error("LAMB: a step requires 2 state addresses and the scalars")
    var n = a.total()
    if n != word(w, W_N):
        raise Error("LAMB: the params were resized under the optimizer")
    var sc = a.rest_a[2]
    var flags = word(w, W_FLAGS)
    var t = begin_step(w, sc, 2)
    var pbuf = gather(a.ps, a.sizes, n)
    var gbuf = gather(a.gs, a.sizes, n)
    var addrs: List[Int] = [Int(pbuf.unsafe_ptr()), Int(gbuf.unsafe_ptr()), a.rest_a[0], a.rest_a[1], sc]
    var ip: List[Int] = [a.J, t, flags]
    var offs = lamb_offsets_of(a.sizes)
    for k in range(len(offs)):
        ip.append(offs[k])
    var f = _f64s(fp, 0, 6)
    f.append(Float64(word(w, W_SC_T)))
    _ = lamb_step_impl(ex, addrs, ip, f)
    scatter(pbuf, a.ps, a.sizes)
    _ = pbuf^
    _ = gbuf^
    set_word(w, W_T, t)
    if (flags & 8) != 0:
        set_word(w, W_SC_T, t)
    return n


def adafactor_step_host[E: Exec](mut ex: E, w: IP64, a: SeqStep, fp: List[Float64]) raises -> Int:
    """Adafactor's per-call step (`adafactor_step_impl`), tensor by tensor
    in order. a.rest_a = [s1_0 .. s1_{J-1}, s2_0 .. s2_{J-1}] (row_var or
    the variance, col_var or the variance again); a.rest_i = [R_0 ..
    R_{J-1}, C_0 .. C_{J-1}] (C = 0 for a vector)."""
    if a.dev:
        raise Error("Adafactor: this binding has no device-parameter step")
    if len(a.rest_a) != 2 * a.J or len(a.rest_i) != 2 * a.J:
        raise Error("Adafactor: a step requires two state addresses and (R, C) per tensor")
    var t = word(w, W_T) + 1
    var f = _f64s(fp, 0, 6)
    for j in range(a.J):
        var addrs: List[Int] = [a.ps[j], a.gs[j], a.rest_a[j], a.rest_a[a.J + j]]
        var ip: List[Int] = [a.rest_i[j], a.rest_i[a.J + j], t]
        _ = adafactor_step_impl(ex, addrs, ip, f)
    set_word(w, W_T, t)
    return a.J


def state_read_host(w: IP64) raises -> Int:
    """A state read on a binding with nothing resident: the reader may write
    into the host copies, so the next step treats them as the newest."""
    if word(w, W_HANDLE) >= 0:
        raise Error("sequence optimizer: the resident state lives on another binding")
    set_word(w, W_OWNED, 1)
    set_word(w, W_PRISTINE, 0)
    return 0


struct StateCopy(Movable):
    """`seq_optimizer_export` / `_import` arguments. ip = [k, n, sc_len, m,
    t, sc_given, sc_t]; addrs = [slot_0 .. slot_{k-1}, x_0 .. x_{m-1},
    scalars, scalars_x] (x: the export's outputs, m = k, or the import's
    sources, m <= k)."""

    var k: Int
    var n: Int
    var sc_len: Int
    var m: Int
    var t: Int
    var sc_given: Bool
    var sc_t: Int
    var slots: List[Int]
    var xs: List[Int]
    var sc: Int
    var sc_x: Int

    def __init__(out self, addrs: List[Int], ip: List[Int]) raises:
        if len(ip) != 7:
            raise Error("sequence optimizer: a state copy requires 7 integer parameters")
        self.k = ip[0]
        self.n = ip[1]
        self.sc_len = ip[2]
        self.m = ip[3]
        self.t = ip[4]
        self.sc_given = ip[5] != 0
        self.sc_t = ip[6]
        if self.k < 1 or self.k > 3 or self.n < 0 or self.sc_len < 1 or self.m < 0 or self.m > self.k:
            raise Error("sequence optimizer: a state copy names 1 to 3 slots")
        if len(addrs) != self.k + self.m + 2:
            raise Error("sequence optimizer: a state copy requires k + m + 2 addresses")
        self.slots = List[Int]()
        for i in range(self.k):
            self.slots.append(addrs[i])
        self.xs = List[Int]()
        for i in range(self.m):
            self.xs.append(addrs[self.k + i])
        self.sc = addrs[self.k + self.m]
        self.sc_x = addrs[self.k + self.m + 1]


def _copy(dst: Int, src: Int, n: Int, what: String) raises:
    if n > 0:
        memcpy(dest=host_fp(dst, what), src=host_fp(src, what), count=n)


def export_copy(w: IP64, c: StateCopy) raises -> Int:
    """The host copies of the state (already current) and the scalars into
    the export's outputs."""
    for i in range(c.m):
        _copy(c.xs[i], c.slots[i], c.n, "state")
    _copy(c.sc_x, c.sc, c.sc_len, "scalars")
    return c.m


def import_copy(w: IP64, c: StateCopy) raises -> Int:
    """`load_state_dict`'s copies (the host copies already current): the m
    sources into the first m slots, the step count, and the carried
    scalars when they belong to a step not after it (else they restart at
    1.0 from step 0). The host copies are then the newest and the next
    resident step uploads them."""
    for i in range(c.m):
        _copy(c.slots[i], c.xs[i], c.n, "state")
    set_word(w, W_T, c.t)
    var p = host_fp(c.sc, "scalars")
    for i in range(c.sc_len):
        p[i] = Float32(1.0)
    set_word(w, W_SC_T, 0)
    if c.sc_given and c.sc_t >= 0 and c.sc_t <= c.t:
        _copy(c.sc, c.sc_x, c.sc_len, "scalars")
        set_word(w, W_SC_T, c.sc_t)
    set_word(w, W_FRESH, 1)
    set_word(w, W_OWNED, 1)
    set_word(w, W_PRISTINE, 0)
    return c.m


def export_host(w: IP64, c: StateCopy) raises -> Int:
    if word(w, W_HANDLE) >= 0:
        raise Error("sequence optimizer: the resident state lives on another binding")
    return export_copy(w, c)


def import_host(w: IP64, c: StateCopy) raises -> Int:
    if word(w, W_HANDLE) >= 0:
        raise Error("sequence optimizer: the resident state lives on another binding")
    return import_copy(w, c)


# ---- the host binding's entries (one per public method; HostExec) -----------


def seq_optimizer_step_on[E: Exec](mut ex: E, words: Int, addrs: List[Int], ip: List[Int],
                                   fp: List[Float64]) raises -> Int:
    """An element-wise step on a binding without resident state.
    ip = [J, dev, gk .., size .., kind, flags]."""
    var a = SeqStep(addrs, ip)
    if len(a.rest_i) != 2:
        raise Error("sequence optimizer: a step requires the kind and the flags")
    return elementwise_step_host(ex, words_ptr(words), a, a.rest_i[0], a.rest_i[1], fp)


def seq_lamb_step_on[E: Exec](mut ex: E, words: Int, addrs: List[Int], ip: List[Int],
                              fp: List[Float64]) raises -> Int:
    """A LAMB step on a binding without resident state."""
    var a = SeqStep(addrs, ip)
    return lamb_step_host(ex, words_ptr(words), a, fp)


def seq_adafactor_step_on[E: Exec](mut ex: E, words: Int, addrs: List[Int], ip: List[Int],
                                   fp: List[Float64]) raises -> Int:
    """An Adafactor step on a binding without resident state."""
    var a = SeqStep(addrs, ip)
    return adafactor_step_host(ex, words_ptr(words), a, fp)


def seq_optimizer_state_on(words: Int) raises -> Int:
    return state_read_host(words_ptr(words))


def seq_optimizer_export_on(words: Int, addrs: List[Int], ip: List[Int]) raises -> Int:
    return export_host(words_ptr(words), StateCopy(addrs, ip))


def seq_optimizer_import_on(words: Int, addrs: List[Int], ip: List[Int]) raises -> Int:
    return import_host(words_ptr(words), StateCopy(addrs, ip))


def seq_optimizer_close_on(words: Int) raises -> Int:
    """Nothing is resident on a host binding."""
    _ = words_ptr(words)
    return 0
