"""PyTorch custom ops over `mojolearn.identical.gemm.fp32.v1`.

**THIS FILE HAS NEVER BEEN COMPILED AND NEVER BEEN EXECUTED.** Not once, not
partially, not as a syntax check. It was written from a read of

  - `max/experimental/torch/torch.py` in the pinned pixi environment,
  - the published `max/develop/custom-kernels-pytorch` tutorial,
  - the published `max.gpu.host.device_context.DeviceContext` and
    `.DeviceBuffer` API pages,
  - and `gemm/host_entry.mojo`, which is the shape this file copies.

Every place below where this lane could not verify an API's exact spelling
from the pinned environment is marked `SPELLING NOT VERIFIED` and names the
alternatives it would try next. Being wrong and flagged is the intent; being
wrong and confident is not. `torchbridge/TORCH_BRIDGE_PLAN.md` section 10
item 12 says the same thing in the owed list.

WHAT IS OWED, IN ONE LIST
-------------------------
1. Compiling this file. Nobody has.
2. `bindings/build_torchbridge.sh`, which does not exist. Without it
   `CustomOpLibrary` JIT-compiles this source with NO build defines, so
   `GLOBAL_NUMERIC_MODE` is `NUMERIC_FAST` and the ops below compute the FAST
   arithmetic while the caller believes otherwise. `identical_gemm_mode_probe`
   is the only thing standing between that and a silent wrong claim; gate T7
   in the plan is what makes it fire.
3. The three staging copies in `_run`. `identical_gemm` wants
   `DeviceBuffer`s, a custom op receives pointers, and `DeviceBuffer` has no
   public constructor that wraps a foreign device pointer. Plan section 2.2
   and owed item 11.
4. Gates. None exist. Plan section 8, T1 through T9.

WHAT IS NOT HERE, AND MUST NEVER BE
------------------------------------
**No arithmetic.** Not one multiply, not one add, not one `ftz`, no kernel of
its own, no second summation order. This file allocates, copies, dispatches
and copies back. `gemm/host_entry.mojo` states the rule for the host-pointer
surface and it is the same rule here -- *"a multiply or an add appearing below
would be a second implementation of a bit-exact contract, which is the one
thing this lane cannot afford."* The bits come from
`gemm/mojo_only/gemm_identical.mojo` and from nowhere else.

**No new profile.** `gemm/IDENTICAL_FP32_CONTRACT.md` is frozen and this file
consumes it. Nothing here touches the leaf rule, the fold topology, the
multiply-add policy, the flush policy or either profile constant.

THE OP NUMBERING, WHICH IS A TRAP
----------------------------------
Two conventions live in this repository and they are a full permutation of
each other.

    gemm/mojo_only/gemm_oracle.mojo   OP_NN = 0   OP_NT = 1   OP_TN = 2
    bench/gemm_shapes.mojo            OP_NT = 0   OP_TN = 1   OP_NN = 2

**THIS BOUNDARY SPEAKS THE `gemm_oracle.mojo` CONVENTION.** The three op
codes below are imported from `gemm_oracle` rather than retyped, so there is
no literal here to get wrong, and the three ops are registered under NAMES
(`identical_gemm_nn`, `_nt`, `_tn`) rather than taking an integer op code
across the boundary at all. A Python caller therefore cannot hand this file a
number from the other convention, because it cannot hand it a number.

That was a deliberate choice over the alternative, which was one
`identical_gemm` op parameterized with `CustomOp.__getitem__({"op": 1})`
(`max/experimental/torch/torch.py` line 263, `ParametersDict`). The
parameterized form is more compact and this lane could not verify how a Mojo
`Int` parameter is matched by name through `ops.custom(..., parameters=...)`.
Three names cost twenty lines and rest on nothing unverified. There is also a
gate in this tree whose entire job is catching a raw pass-through between the
two conventions (`check_op_encodings_are_not_interchangeable`), which is
reason enough not to put an integer op code on a new boundary.

DEVIATIONS 1200 (the two lanes; this file is the CUSTOM_OP one), 1204 (the
`target == "cpu"` refusal below), 1205 (the mode probe).
"""

import extensibility
from extensibility import InputTensor, OutputTensor

from max.gpu.host import DeviceContext

from gemm.mojo_only.gemm_identical import identical_gemm
from gemm.mojo_only.gemm_oracle import OP_NN, OP_NT, OP_TN
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
)


#: The profile major version this file implements, mirroring
#: `bindings/_mojolearn_linalg.mojo::linalg_profile_version_binding`. The
#: version is part of the claim (contract preamble); the leaf rule of section
#: 7.1 and the fold topology of 7.2 are what it is about. Changing either
#: creates v2 and this becomes 2. It does not amend v1.
comptime PROFILE_VERSION = 1


# ===========================================================================
# THE ONE FUNCTION THAT DISPATCHES
# ===========================================================================
# `gemm_identical.mojo` gives `(L, P)` exactly one producer so that launch
# geometry cannot reach the arithmetic, and `gemm_backward.mojo` gives the
# backward shape exactly two. The same discipline here, for the quantity this
# file can get wrong: THE SHAPES AND THE OP CODE HAVE EXACTLY ONE PRODUCER,
# `_run`. All three registered ops call it and none of them computes an
# extent of its own.


def _run[
    target: StaticString, op: Int
](
    c: OutputTensor[dtype = DType.float32, rank=2],
    a: InputTensor[dtype = DType.float32, rank=2],
    b: InputTensor[dtype = DType.float32, rank=2],
    ctx: DeviceContext,
) raises:
    """`C = op(A) . op(B)` under the profile, from three torch tensors.

    Row-major and fully contiguous throughout (contract section 2). The
    Python side (`python/mojolearn/torch_ops.py`) refuses a non-contiguous
    tensor by name BEFORE it gets here, because nothing at this boundary can
    see a stride -- `max_tensor_type` in `max/experimental/torch/torch.py`
    builds `TensorType(dtype, shape, device)` and there is no stride in a
    `TensorType`. A strided tensor arriving here is read as if it were
    contiguous, which is a wrong answer and not a crash.

    The element counts, from `gemm/host_entry.mojo`'s table, which is the
    same three rows for all three ops and is why the sizing below does not
    branch:

        op       A elements   B elements   C elements
        OP_NN    m * k        k * n        m * n
        OP_NT    m * k        n * k        m * n
        OP_TN    k * m        k * n        m * n
    """

    # -----------------------------------------------------------------------
    # DEVIATION 1204. A "cpu" target is REFUSED, and this is a reach guard,
    # not a limitation.
    # -----------------------------------------------------------------------
    # `max_device_ref` (torch.py line 672) maps a torch `cpu` tensor to
    # `DeviceRef.CPU(0)`, `MaxOp.graph` builds the graph at that device, and
    # the graph compiler then dispatches this op with `target = "cpu"`. The
    # computation would be CORRECT and, under this profile, BIT IDENTICAL to
    # the GPU answer -- that is the property the profile sells, and it means
    # no output check anywhere can tell you the GPU was skipped.
    #
    # So the only place that fact can be caught is here, at compile time, and
    # a silent CPU run is turned into a build error. A caller who wants a CPU
    # tensor computed goes through the HOST lane (`mojolearn.linalg.matmul`),
    # which reaches whatever device `DeviceContext()` picks and says so.
    comptime if target != "gpu":
        raise Error(
            "torchbridge: this op was dispatched with target='"
            + String(target)
            + "', not 'gpu'. That happens when the torch tensors are on the"
            " CPU (max_device_ref maps a cpu tensor to DeviceRef.CPU, and"
            " the graph then targets the CPU). The answer would be correct"
            " AND bit-identical, so nothing downstream could tell you the"
            " GPU never ran. Use mojolearn.torch_ops, which routes cpu and"
            " mps tensors to the host lane instead. See"
            " torchbridge/TORCH_BRIDGE_PLAN.md sections 1.1 and 8.0(a)."
        )

    # -----------------------------------------------------------------------
    # THE SHAPES. `m` and `n` come from the OUTPUT, `k` from `A`, and which
    # axis of `A` carries `k` is the only thing that varies across the three
    # ops (contract section 0.1).
    # -----------------------------------------------------------------------
    #
    # SPELLING NOT VERIFIED: `dim_size(i)` on a `ManagedTensorSlice`. This
    # lane could not fetch the `extensibility/managed_tensor_slice` docs page
    # (404 on every published form of that URL) and the shipped `.mojoc` is
    # compressed. If this does not compile, the alternatives to try, in
    # order, are `c.shape()[0]`, `c.dim_size[0]()`, `c._spec.shape[0]` and
    # `c.dims()[0]`. The SEMANTICS are certain -- these are the runtime
    # extents of a rank-2 tensor -- only the accessor's name is not.
    var m = c.dim_size(0)
    var n = c.dim_size(1)
    var k = a.dim_size(1)
    comptime if op == OP_TN:
        # OP_TN: A is stored `k x m`, so the contracted extent is axis 0.
        k = a.dim_size(0)

    # Refused rather than answered, for `gemm/host_entry.mojo`'s reason:
    # contract section 8 specifies every degenerate shape and
    # `identical_gemm_with_plan` implements them, but no gate in this tree
    # has run them through a torch boundary, and an answer nothing has
    # checked is not something to hand a PyTorch caller.
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "torchbridge: m, n and k must all be positive, got m="
            + String(m)
            + " n="
            + String(n)
            + " k="
            + String(k)
            + " (contract section 8 specifies the degenerate shapes; no gate"
            " in this tree has run them through a torch boundary, so they"
            " are refused rather than guessed)"
        )

    # The cross-check that a wrong orientation is a loud error here rather
    # than a plausible wrong matrix downstream. `B`'s contracted axis is the
    # one the op names.
    var kb = b.dim_size(0)
    comptime if op == OP_NT:
        kb = b.dim_size(1)
    if kb != k:
        raise Error(
            "torchbridge: contracted extents disagree, A gives k="
            + String(k)
            + " and B gives k="
            + String(kb)
            + ". The row-major shapes for each op are in contract section"
            " 0.1 and in this function's docstring. This is checked because"
            " a k mismatch that reaches the kernel is an out-of-bounds read"
            " on the device, not an exception."
        )

    # -----------------------------------------------------------------------
    # STAGING. Three device-to-device copies, and this lane could not avoid
    # them. Plan section 2.2.
    # -----------------------------------------------------------------------
    # `identical_gemm` takes `DeviceBuffer[DType.float32]`, and `DeviceBuffer`
    # has no public constructor that wraps a foreign device pointer -- its
    # documented construction routes are `enqueue_create_buffer`,
    # `create_buffer_sync`, the reference copy constructor and
    # `create_sub_buffer`, and that is the whole list. A custom op receives
    # tensor arguments, never a `DeviceBuffer`. So the operands are staged.
    #
    # This costs `m*k + n*k + m*n` floats of device-to-device bandwidth per
    # call. It is NOT a host round trip -- the overload used is the one
    # documented as "an async copy of `size` elements from a device pointer
    # to another device pointer" -- and it moves no bit of any value.
    # Removing it needs a pointer-taking entry point in `gemm/`, which is
    # owed item 11 in the plan and is not this lane's file.
    #
    # Reimplementing the launch here against raw pointers instead is refused
    # outright. That would be a second implementation of a bit-exact
    # contract.
    var a_dev = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](n * k)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](m * n)

    # SPELLING NOT VERIFIED: `a.unsafe_ptr()` on a `ManagedTensorSlice`. Same
    # missing docs page as `dim_size` above. Alternatives to try, in order,
    # are `a._ptr`, `a.data()`, `a.unsafe_ptr[DType.float32]()` and
    # `a.to_layout_tensor().ptr`. What is needed is the device address of
    # element (0, 0), and the tensor is guaranteed contiguous by the Python
    # side.
    ctx.enqueue_copy(a_dev.unsafe_ptr(), a.unsafe_ptr(), m * k)
    ctx.enqueue_copy(b_dev.unsafe_ptr(), b.unsafe_ptr(), n * k)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # THE ONE LINE THAT COMPUTES ANYTHING.
    # -----------------------------------------------------------------------
    # `identical_gemm` and NOT `identical_gemm_into`, for the reason
    # `gemm/host_entry.mojo` gives at length: the `_into` form is
    # ASYNCHRONOUS and takes a CALLER-OWNED workspace, and a custom op has no
    # caller to own one. Sizing a workspace for one plan and letting
    # `choose_gemm_plan` pick another is an out-of-bounds write that a small
    # shape will not show you -- a one-float workspace against a SPLITK
    # dispatch returned right answers at 64 x 4 and regions of `+0.0` at
    # 64 x 64. `identical_gemm` sizes its own with
    # `identical_gemm_workspace_max_floats`, keeps it alive past the wait,
    # and synchronizes before it returns.
    #
    # The cost is one synchronize per matmul. That is real and it is priced
    # in the plan rather than hidden; making this op asynchronous is a design
    # change with a buffer-lifetime hazard attached, not an edit.
    identical_gemm(ctx, c_dev, a_dev, b_dev, m, n, k, op)

    ctx.enqueue_copy(c.unsafe_ptr(), c_dev.unsafe_ptr(), m * n)
    ctx.synchronize()

    # `[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` is dead at its
    # `.unsafe_ptr()`, and every one of the three had `.unsafe_ptr()` called
    # on it above. Without these three lines the buffers are freed before the
    # copies and the kernel have run. `identical_gemm` synchronizes
    # internally as well, so the operands are already safe by the time it
    # returns; these are here because the hazard is invisible at review time
    # and free at run time, which is `gemm/host_entry.mojo`'s argument for
    # the same three lines.
    _ = a_dev
    _ = b_dev
    _ = c_dev


# ===========================================================================
# THE THREE REGISTERED OPS
# ===========================================================================
# One per orientation, named rather than numbered, each two lines long. The
# argument order is OUTPUT FIRST, which is what `max.experimental.torch`
# requires (destination-passing style: `torch.library.custom_op(...,
# mutates_args=...)` marks the leading `num_outputs` arguments as mutated,
# `max/experimental/torch/torch.py` lines 493 and 529) AND is the order
# `identical_gemm(ctx, c, a, b, ...)` already reads in. The two agreeing is
# luck this file gets to keep; swapping `c` with `a` would write the device's
# output over the caller's input tensor, which is memory corruption in the
# caller's process and not an exception.


@extensibility.register("identical_gemm_nn")
struct IdenticalGemmNN:
    """`C[m x n] = A[m x k] . B[k x n]`. Contract op `OP_NN`."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContext,
    ) raises:
        _run[target, OP_NN](c, a, b, ctx)


@extensibility.register("identical_gemm_nt")
struct IdenticalGemmNT:
    """`C[m x n] = A[m x k] . B[n x k]^T`. Contract op `OP_NT`.

    This is the linear layer. `F.linear(x, W)` is `x @ W.T` with `x` of shape
    `(tokens, in_features)` and `W` of shape `(out_features, in_features)`,
    both C-contiguous as PyTorch stores them, so it lands here with no copy
    and no materialized transpose. It is also `gemv` at `n == 1`, which is
    not a fourth operation (contract 0.1).
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContext,
    ) raises:
        _run[target, OP_NT](c, a, b, ctx)


@extensibility.register("identical_gemm_tn")
struct IdenticalGemmTN:
    """`C[m x n] = A[k x m]^T . B[k x n]`. Contract op `OP_TN`.

    `A` is STORED `k x m`. Passing a PyTorch `.t()` view of an `m x k` tensor
    is a non-contiguous tensor and is refused on the Python side, because
    nothing at this boundary can see a stride (see `_run`'s docstring).
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContext,
    ) raises:
        _run[target, OP_TN](c, a, b, ctx)


# ===========================================================================
# THE PROBES. DEVIATION 1205, AND THIS IS THE POINT OF THE FILE.
# ===========================================================================
# `mojolearn.identical.gemm.fp32.v1` is what the IDENTICAL build computes.
# The FAST build runs the SAME kernels on the SAME path with the fused
# multiply-add pin and the flush-to-zero pin compiled away. It is a correct
# GEMM and it makes NO identity claim of any kind, and contract section 11.4
# declines to promise even that it DIFFERS from the identical one. On Apple
# the two coincide at both pinned seams (contract 4.1), so a caller comparing
# numbers on one Mac learns nothing at all.
#
# THE CUSTOM-OP LANE MAKES THIS WORSE THAN IT IS FOR THE CPython BINDING.
# `bindings/build_linalg.sh` passes `-D MOJOLEARN_NUMERIC_IDENTICAL=1` and
# lands the result in a directory the Python side selects by name. This file
# is loaded through `CustomOpLibrary` -> `KernelLibrary.load_paths`, and THIS
# LANE FOUND NO WAY TO PASS A BUILD DEFINE THROUGH THAT PATH. Point
# `CustomOpLibrary` at this .mojo source and it JIT-compiles with no defines,
# `GLOBAL_NUMERIC_MODE` is `NUMERIC_FAST`, and every op above computes the
# fast arithmetic under the identical name.
#
# So the mode is READ BACK OUT OF THE COMPILED KERNEL, by the kernel, and the
# Python side gates its guarantee on what this returns rather than on an
# environment variable or a build flag it was told about. That is
# `bindings/_mojolearn_linalg.mojo::linalg_numeric_mode_binding`'s argument
# and it is the same argument here with a worse hole behind it.
#
# The fix for the hole is `bindings/build_torchbridge.sh`, running
# `mojo package -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . torchbridge -o
# python/mojolearn/identical/torchbridge.mojopkg` and pointing
# `CustomOpLibrary` at the package instead of the source. That script does
# not exist. Until it does, these probes return 0 and `require_identical()`
# on the Python side raises, which is the right failure and still a failure.


def _fill_one_int32[
    target: StaticString
](
    out_t: OutputTensor[dtype = DType.int32, rank=1],
    value: Int32,
    ctx: DeviceContext,
) raises:
    """Write one Int32 into a length-1 device tensor.

    A device fill and a device-to-device copy, because there is no host store
    into a device tensor at this boundary and `enqueue_fill` is the only
    verified way to put a scalar into device memory without one.

    One value and not two, deliberately. Packing the mode and the profile
    version into one tensor would need `create_sub_buffer` or a pointer
    offset, and this lane has verified neither spelling at this boundary. Two
    one-element probes cost nothing and rest on nothing unverified.
    """
    comptime if target != "gpu":
        raise Error(
            "torchbridge: probe dispatched with target='"
            + String(target)
            + "', not 'gpu'. A probe that runs somewhere other than where"
            " the gemm runs answers a question nobody asked. See _run."
        )
    if out_t.dim_size(0) != 1:
        raise Error(
            "torchbridge: probe output must hold exactly 1 int32, got "
            + String(out_t.dim_size(0))
        )
    var buf = ctx.enqueue_create_buffer[DType.int32](1)
    buf.enqueue_fill(value)
    ctx.enqueue_copy(out_t.unsafe_ptr(), buf.unsafe_ptr(), 1)
    ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`, as in `_run`.
    _ = buf


@extensibility.register("identical_gemm_mode_probe")
struct IdenticalGemmModeProbe:
    """Writes 1 if THIS BINARY was compiled under `NUMERIC_IDENTICAL`, else 0.

    Compile-time, from `is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]` through
    `GLOBAL_NUMERIC_MODE`, exactly as
    `bindings/_mojolearn_linalg.mojo::linalg_numeric_mode_binding` does it.

    **It is a property of the kernel library that actually loaded**, which is
    the only honest source for it, and it is the ONLY way a caller of the
    custom-op lane can learn what arithmetic they got. Nothing in the answer
    distinguishes the two builds; on Apple they coincide at both pinned
    seams. `python/mojolearn/torch_ops.py::require_identical` refuses to
    deliver the profile's guarantee unless this returns 1.

    `int32` and not `bool`, so that a future third mode is a value rather
    than a new op.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_t: OutputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContext,
    ) raises:
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            _fill_one_int32[target](out_t, Int32(1), ctx)
            return
        _fill_one_int32[target](out_t, Int32(0), ctx)


@extensibility.register("identical_gemm_profile_probe")
struct IdenticalGemmProfileProbe:
    """Writes the MAJOR VERSION of the GEMM profile this binary implements.

    1, for `mojolearn.identical.gemm.fp32.v1`. An Int and not the profile
    string, for `linalg_profile_version_binding`'s reason -- the profile name
    has no Mojo constant anywhere in this tree, so a string returned here
    would be a literal typed twice rather than a fact read once.

    The Python side cross-checks it against its own constant, so a stale
    `.mojopkg` beside a newer wrapper is a loud error rather than a
    mislabeled answer. That matters more on this lane than on the CPython
    one, because a `.mojopkg` has no import machinery to notice it is old.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_t: OutputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContext,
    ) raises:
        _fill_one_int32[target](out_t, Int32(PROFILE_VERSION), ctx)
