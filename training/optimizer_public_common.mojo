# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public optimizers' native half, shared by the GPU binding
(training/optimizer_public.mojo) and the CPU binding
(training/host/optimizer_public_host.mojo). No Python here.

`mojolearn.SGD` / `Adam` / `AdamW` (python/mojolearn/_training_impl.py)
make ONE native call per public method: the door
(bindings/optimizer_public_door.mojo) reads the optimizer and the caller's
tensors, refuses by name, and calls one entry here or in its siblings. What used to be Python moved here (lane neural-optimizers,
2026-10-21), the same operations in the same order:

  * the parameter TRAVERSAL: the registry's flat buffer is the caller's
    tensors end to end (`gather`: borrowed when they already are, else one
    copy per tensor), and a step's parameters and clipped gradient go back
    into the caller's tensors (`scatter`);
  * the HYPERPARAMETER resolution: the step's learning rate from a built-in
    schedule (sequence/lr_table.mojo), the one-based step counter, the
    float32 learning rate the step reports;
  * the registry's offsets and size checks.

The update itself is the existing native step (training/estimator.mojo,
training/optimizer_multi_gpu.mojo, the host oracle): nothing here does
arithmetic on parameter or gradient values."""
from std.memory import memcpy, memset_zero
from std.os import getenv

from sequence.lr_table import LrSpec, lr_spec_value

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime F64P = MutPointer[Float64, MutUntrackedOrigin]

#: `PublicStep.g_mode`: where the step's gradient is
comptime GRADS_HOST = 0        # the caller's host tensors
comptime GRADS_DEVICE = 1      # device arrays (handles), the device-parameter step only
comptime GRADS_RESIDENT = 2    # already in the handle (`grads_to_device`, the previous step)

#: `PublicStepOut` as the float64 words the Python glue reads, in this order
comptime STEP_OUT_WORDS = 10


@fieldwise_init
struct OptHyper(Copyable, ImplicitlyCopyable, Movable):
    """One step's configuration, as the Python optimizer holds it (float64
    attributes; the step rounds each to float32 exactly as the binding's
    params list always did)."""
    var kind: Int
    var nesterov: Int
    var maximize: Bool
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var momentum: Float64
    var dampening: Float64
    #: <= 0: no clip
    var max_norm: Float64


struct PublicStep(Movable):
    """Everything `step`, `step_device` and `step_accumulated` pass."""
    var j: Int
    var t: Int
    var hyper: OptHyper
    var has_sched: Bool
    var sched: LrSpec
    var n_total: Int
    var offsets: Int
    var init: Int
    var p_addr: List[Int]
    var p_bytes: List[Int]
    var g_mode: Int
    #: host tensor addresses (a_count * j of them, microbatch-major) or
    #: device array handles
    var g_addr: List[Int]
    #: bytes per host tensor, or floats per device array
    var g_bytes: List[Int]
    var a_count: Int
    var tokens: Int
    var handle: Int
    #: -1 None, 0 False, 1 True (the optimizer's `resident=` argument)
    var resident_wanted: Int
    var pristine: Bool
    var m_addr: Int
    var v_addr: Int
    var p_dev: Bool
    var g_dev: Bool
    var device_op: Bool

    def __init__(out self):
        self.j = 0
        self.t = 0
        self.hyper = OptHyper(0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.has_sched = False
        self.sched = LrSpec(-1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        self.n_total = 0
        self.offsets = 0
        self.init = 0
        self.p_addr = List[Int]()
        self.p_bytes = List[Int]()
        self.g_mode = GRADS_HOST
        self.g_addr = List[Int]()
        self.g_bytes = List[Int]()
        self.a_count = 1
        self.tokens = -1
        self.handle = -1
        self.resident_wanted = -1
        self.pristine = True
        self.m_addr = 0
        self.v_addr = 0
        self.p_dev = False
        self.g_dev = False
        self.device_op = False


@fieldwise_init
struct PublicStepOut(Copyable, ImplicitlyCopyable, Movable):
    """What the Python glue assigns after a step."""
    var t: Int
    var handle: Int
    var on_device: Bool
    var clip_ran: Bool
    var total_norm: Float32
    var coef: Float32
    #: the learning rate the step used, as a float32
    var lr: Float32
    #: -1: not reported (the device-parameter step), else 0 / 1
    var packed: Int
    var g_dev: Bool

    def write(self, dst: F64P):
        dst[0] = Float64(self.t)
        dst[1] = Float64(self.handle)
        dst[2] = 1.0 if self.on_device else 0.0
        dst[3] = 1.0 if self.clip_ran else 0.0
        dst[4] = Float64(self.total_norm)
        dst[5] = Float64(self.coef)
        dst[6] = Float64(self.lr)
        dst[7] = Float64(self.packed)
        dst[8] = 1.0 if self.g_dev else 0.0
        dst[9] = 0.0


def step_lr(req: PublicStep) raises -> Float64:
    """The learning rate of step t + 1: the built-in schedule's value when
    the optimizer has one (`lr_schedule.lr_at(t + 1)`, the float32 the
    schedule defines), else the optimizer's `lr`."""
    if not req.has_sched:
        return req.hyper.lr
    var r = lr_spec_value(req.sched, req.t + 1)
    if r.status != 0:
        raise Error("ValueError: mojolearn optimizer: the learning-rate schedule refused step "
                    + String(req.t + 1) + " (status " + String(r.status) + ")")
    return Float64(r.value)


def registry_check(req: PublicStep) raises:
    """The parameter tensors must still hold the registry's floats (a caller
    can resize its own arrays between steps)."""
    var total = 0
    for j in range(req.j):  # driver-loop: over parameter tensors
        total += req.p_bytes[j] // 4
    if total != req.n_total:
        raise Error(
            "ValueError: mojolearn optimizer step: the parameter tensors hold " + String(total)
            + " floats now and held " + String(req.n_total) + " when this optimizer was built; "
            + "a parameter buffer was resized underneath its registry"
        )


def sizes_check(nbytes: List[Int], count: Int, n_total: Int) raises:
    """`count` tensors must hold the registry's `n_total` floats."""
    var total = 0
    for j in range(count):  # driver-loop: over parameter tensors
        total += nbytes[j] // 4
    if total != n_total:
        raise Error("ValueError: mojolearn optimizer: the tensors do not match the optimizer's registry ("
                    + String(count) + " tensors, " + String(n_total) + " floats)")


struct Flat(Movable):
    """A registry-sized float32 buffer: the caller's own storage (borrowed)
    or a packed copy owned here."""
    var own: List[Float32]
    var addr: Int
    var packed: Bool

    def __init__(out self):
        self.own = List[Float32]()
        self.addr = 0
        self.packed = False

    def ptr(self) -> F32P:
        if self.packed:
            return self.own.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        return F32P(unsafe_from_address=self.addr)


def gather(addrs: List[Int], nbytes: List[Int], first: Int, count: Int, total: Int) -> Flat:
    """`count` tensors (addrs[first ..]) as one flat buffer of `total`
    floats, in registry order. One tensor, or tensors already END TO END in
    registry order (a caller's views of one flat buffer), are borrowed whole
    and nothing is copied; otherwise one memcpy per tensor into a packed
    copy."""
    var f = Flat()
    if count == 1:
        f.addr = addrs[first]
        return f^
    var end = addrs[first]
    var adjacent = True
    for j in range(count):  # driver-loop: over parameter tensors
        if addrs[first + j] != end:
            adjacent = False
            break
        end += nbytes[first + j]
    if adjacent and total > 0:
        f.addr = addrs[first]
        return f^
    f.own = List[Float32](unsafe_uninit_length=max(total, 1))
    f.packed = True
    var base = f.own.unsafe_ptr()
    var at = 0
    for j in range(count):  # driver-loop: over parameter tensors (one memcpy each)
        var n = nbytes[first + j] // 4
        if n > 0:
            memcpy(dest=base + at, src=F32P(unsafe_from_address=addrs[first + j]), count=n)
        at += n
    return f^


def scatter(f: Flat, addrs: List[Int], nbytes: List[Int], first: Int, count: Int):
    """Write a packed flat buffer back into the caller's tensors, in place
    (a borrowed buffer already is the caller's storage)."""
    if not f.packed:
        return
    var src = f.ptr()
    var at = 0
    for j in range(count):  # driver-loop: over parameter tensors (one memcpy each)
        var n = nbytes[first + j] // 4
        if n > 0:
            memcpy(dest=F32P(unsafe_from_address=addrs[first + j]), src=src + at, count=n)
        at += n


def stack_microbatches(req: PublicStep) -> List[Float32]:
    """The A microbatch gradients as one [A, N] buffer, microbatch-major (the
    clause 9.2 tree's input, `accumulate`'s `parts`)."""
    var n = req.n_total
    var out = List[Float32](unsafe_uninit_length=max(req.a_count * n, 1))
    var base = out.unsafe_ptr()
    for i in range(req.a_count):  # driver-loop: over microbatches
        var at = i * n
        for j in range(req.j):  # driver-loop: over parameter tensors (one memcpy each)
            var k = i * req.j + j
            var c = req.g_bytes[k] // 4
            if c > 0:
                memcpy(dest=base + at, src=F32P(unsafe_from_address=req.g_addr[k]), count=c)
            at += c
    return out^


def zero_tensors(addrs: List[Int], nbytes: List[Int]):
    """`zero_grad`: all-zero bytes (+0.0) into every tensor."""
    for j in range(len(addrs)):  # driver-loop: over gradient tensors (one memset each)
        var n = nbytes[j] // 4
        if n > 0:
            memset_zero(F32P(unsafe_from_address=addrs[j]), n)


def registry_offsets(nbytes: List[Int], offsets: I32P) raises -> Int:
    """`offsets[0 .. J]` of a registry of tensors of `nbytes` bytes each;
    returns N = offsets[J]."""
    var at = 0
    offsets[0] = Int32(0)
    for j in range(len(nbytes)):  # driver-loop: over parameter tensors
        var b = nbytes[j]
        if b < 4 or b % 4 != 0:
            raise Error("ValueError: mojolearn optimizer: tensor " + String(j) + " is not a non-empty float32 buffer")
        at += b // 4
        if at > 2147483647:
            raise Error("ValueError: mojolearn optimizer: the registry holds more than 2^31 - 1 floats")
        offsets[j + 1] = Int32(at)
    return at


def resident_wanted_by_env() -> Bool:
    """`MOJOLEARN_OPTIMIZER_RESIDENT=0` keeps the optimizer state off the
    device (the per-call step; the A/B's before arm), and so does a
    multi-device optimizer count (that route lives in the per-call step)."""
    if String(getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1")) == "0":
        return False
    if String(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1")) != "1":
        return False
    return True


def eager_moments() -> Bool:
    """MOJOLEARN_OPT_LAZY_MOMENTS=0: the optimizer's host state is allocated
    with the optimizer and uploaded on the first step (the lazy
    allocation's A/B, lane gap-train-utils); otherwise it is allocated on
    first use."""
    return String(getenv("MOJOLEARN_OPT_LAZY_MOMENTS")) == "0"
