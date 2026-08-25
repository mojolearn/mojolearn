"""`mojolearn.torch_ops`: the bit-identical FP32 matrix product, from PyTorch.

**THIS FILE HAS NEVER BEEN IMPORTED, COMPILED OR EXECUTED.** Not once, not
partially, not as a syntax check. Neither has the Mojo half
(`torchbridge/identical_ops.mojo`). It was written from a read of
`python/mojolearn/_linalg_impl.py`, `gemm/IDENTICAL_FP32_CONTRACT.md`,
`gemm/mojo_only/gemm_backward.mojo`, `gemm/IDENTICAL_BACKWARD_PLAN.md` and
`max/experimental/torch/torch.py` in the pinned environment.

The design, the two transport lanes, the refusals and the gate plan are in
`torchbridge/TORCH_BRIDGE_PLAN.md`. Read it before trusting anything here.
DEVIATIONS 1200 through 1213.

WHAT IS OWED
------------
1. Every gate. T1 through T9 in the plan, section 8. None is written, no
   sabotage arm has been fired, and until one has, each of them is a claim
   that a passing check has never been made to fail.
2. `python/mojolearn/identical/_mojolearn_linalg.so` is not built in this
   tree, so the HOST lane raises `ImportError` in identical mode today. That
   is the ONLY lane Apple has.
3. `bindings/build_torchbridge.sh` does not exist, so the CUSTOM_OP lane
   JIT-compiles `torchbridge/identical_ops.mojo` with no build defines and
   computes the FAST arithmetic. `require_identical()` below reads the
   compiled mode back out of the kernel and refuses; that refusal is the only
   thing between this hole and a silent wrong claim.
4. `python/mojolearn/__init__.py` does not export `torch_ops`. It should be a
   LAZY submodule import so torch stays optional; that file is not this
   lane's to edit.
5. The cross-microbatch accumulator. See `accumulation_split_is_aligned`.

THE ONE THING TO READ BEFORE USING THIS MODULE
-----------------------------------------------
`_linalg_impl`'s module docstring carries it in full and it applies here
unchanged. **The identity claim belongs to the IDENTICAL build, not to a
function name.** The FAST build runs the same kernels on the same path with
the fused-multiply-add pin and the flush-to-zero pin compiled away. It is a
correct GEMM that makes no identity claim of any kind, and on Apple the two
coincide at both pinned seams, so a caller comparing numbers on one Mac
learns nothing at all.

So `identical_matmul` REQUIRES THE IDENTICAL BUILD BY DEFAULT, on whichever
lane it ran, and refuses loudly and by name when that is not what loaded.
Pass `identical=False` to say in your source that you want whichever build is
present and are making no identity claim about the result.

WHAT THIS BUYS YOU, AND WHERE IT STOPS
---------------------------------------
For ONE call, the output bits are a pure function of the input bits, of `k`,
and of the profile. Not of `m`, not of `n`, not of the launch geometry, not
of the block count, and not of the vendor.

It is NOT an identical model, NOT an identical training run, NOT dropout,
NOT data order and NOT the optimizer. It does not promise NaN payload bits
(contract 9.1), it does not survive a `min` / `max` / `argmax` you take over
the output (9.2(e), because `-0.0 == +0.0`), it will not match
`torch.matmul` bit for bit, and it carries no performance claim of any kind.
Plan section 6 is the itemized version and contract section 11 is the
authority.
"""

import os

from . import _linalg_impl as linalg

#: The profile this module speaks, taken from `_linalg_impl` rather than
#: retyped. One literal in the Python tree, not two.
PROFILE = linalg.PROFILE
PROFILE_FAMILY = linalg.PROFILE_FAMILY
PROFILE_VERSION = linalg.PROFILE_VERSION

#: The three operations of contract section 0.1, IMPORTED and not retyped.
#:
#: THE OP NUMBERING IS A TRAP. `gemm/mojo_only/gemm_oracle.mojo` says
#: OP_NN=0, OP_NT=1, OP_TN=2; `bench/gemm_shapes.mojo` says OP_NT=0, OP_TN=1,
#: OP_NN=2. Those are a full permutation of each other and there is a gate in
#: this tree whose whole job is catching a raw pass-through between them
#: (`check_op_encodings_are_not_interchangeable`).
#:
#: **THIS MODULE SPEAKS THE `gemm_oracle` CONVENTION**, and it never puts an
#: integer op code on a boundary: the host lane passes named `transpose_a` /
#: `transpose_b` flags to `linalg.matmul`, and the custom-op lane calls three
#: separately NAMED Mojo ops (`identical_gemm_nn`, `_nt`, `_tn`). The
#: integers below are used only inside this file, as dictionary keys.
OP_NN = linalg.OP_NN
OP_NT = linalg.OP_NT
OP_TN = linalg.OP_TN

#: `transpose_a`, `transpose_b` for each op, the inverse of `_linalg_impl._OPS`
#: and kept beside it so the two cannot drift apart silently.
_FLAGS = {
    OP_NN: (False, False),
    OP_NT: (False, True),
    OP_TN: (True, False),
}
_OPS = {v: k for k, v in _FLAGS.items()}

#: The Mojo op name each code registers under in
#: `torchbridge/identical_ops.mojo`.
_OP_NAMES = {
    OP_NN: "identical_gemm_nn",
    OP_NT: "identical_gemm_nt",
    OP_TN: "identical_gemm_tn",
}

#: The two profile constants (`gemm/mojo_only/gemm_oracle.mojo`).
#: **They are PROFILE constants, not tuning knobs** -- changing either changes
#: the answer's bits and is a contract revision. Mirrored here only so
#: `accumulation_split_is_aligned` can answer without a device.
CONTRACT_K_LEAF_MIN = 128
CONTRACT_MAX_LEAVES = 1024

#: Lane names. Plan section 1.
LANE_CUSTOM_OP = "CUSTOM_OP"
LANE_HOST = "HOST"

_ops_cache = None
_mode_cache = {}


# ===========================================================================
# TORCH, IMPORTED LAZILY AND NEVER AT MODULE SCOPE
# ===========================================================================


def _torch():
    """The `torch` module, imported on first use.

    NEVER at module scope. `mojolearn` is a scikit-learn-shaped package whose
    users mostly do not have PyTorch, and importing torch at package load
    would add a hard dependency (and about a second of import time) to every
    one of them for a module they will not call. It would also make
    `import mojolearn` fail on a box with a broken CUDA install, which is a
    failure with nothing to do with anything this package computes.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "mojolearn.torch_ops needs PyTorch, which is not installed. It "
            "is an OPTIONAL dependency of this package: nothing else in "
            "mojolearn imports it. Install it and try again."
        ) from e
    return torch


# ===========================================================================
# LANE SELECTION. DEVIATION 1200.
# ===========================================================================


def lane_for_device(device):
    """`LANE_CUSTOM_OP` or `LANE_HOST` for a `torch.device`, or raise.

    THE WHOLE REASON THERE ARE TWO LANES is in
    `max/experimental/torch/torch.py` lines 672-688, which is the entirety of
    the torch-device translation in the pinned environment:

        def max_device_ref(device):
            if type == "cpu":  return DeviceRef.CPU(index)
            elif type == "cuda": return DeviceRef.GPU(index)
            else: raise TypeError(...)

        def max_device(device):
            DeviceType = {"cuda": Accelerator, "cpu": CPU}[device.type]

    **There is no `"mps"` arm in either.** A PyTorch tensor on an Apple GPU
    lives on `torch.device("mps")`, so it cannot cross the custom-op boundary
    at all, and moving it to `cpu` first does not help -- the graph would then
    be built at `DeviceRef.CPU`, the op would be dispatched with
    `target = "cpu"`, and the Metal GPU would never be touched. The answer
    would be correct AND, under this profile, BIT IDENTICAL, so no number
    anywhere could tell you the GPU was skipped.

    That is why `mps` and `cpu` are routed to the HOST lane EXPLICITLY
    (DEVIATION 1204) rather than being allowed to fall through, and why
    `torchbridge/identical_ops.mojo` refuses a `target != "gpu"` dispatch at
    compile time as a second line of defense.

    PyTorch built for ROCm reports `device.type == "cuda"`, so AMD lands on
    the custom-op lane with NVIDIA. **This lane did not verify that against a
    ROCm install**; gate T8 does.
    """
    t = device.type
    if t == "cuda":
        return LANE_CUSTOM_OP
    if t in ("cpu", "mps"):
        return LANE_HOST
    raise ValueError(
        f"mojolearn.torch_ops: device type {t!r} is not supported. This "
        "profile runs on CUDA and ROCm through the custom-op lane, and on "
        "CPU and MPS through the host lane. See "
        "torchbridge/TORCH_BRIDGE_PLAN.md section 5."
    )


def _custom_ops():
    """The loaded `CustomOpLibrary` for `torchbridge/`.

    `CustomOpLibrary` accepts a `.mojo` path or a `.mojoc` / `.mojopkg`, and
    JIT-compiles the former. **A JIT compile carries no build defines**, so
    a library loaded from source computes the FAST arithmetic and
    `identical_gemm_mode_probe` reports 0. The packaged form is what an
    identical run needs, and the packaging script does not exist yet (owed
    item 3 at the top of this file). The preference order below reflects
    that: package first, source only as a fallback, and the mode probe is
    what decides whether the result may carry the claim.
    """
    global _ops_cache
    if _ops_cache is not None:
        return _ops_cache
    from max.experimental.torch import CustomOpLibrary

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))
    candidates = [
        os.path.join(pkg_dir, "identical", "torchbridge.mojopkg"),
        os.path.join(pkg_dir, "torchbridge.mojopkg"),
        os.path.join(repo_root, "torchbridge"),
    ]
    for path in candidates:
        if os.path.exists(path):
            _ops_cache = CustomOpLibrary(path)
            return _ops_cache
    raise ImportError(
        "mojolearn.torch_ops: no torchbridge kernel library found. Looked "
        "at:\n    " + "\n    ".join(candidates) + "\n"
        "The packaged form is what an identical run needs; build it with\n"
        "    mojo package -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . torchbridge "
        "-o python/mojolearn/identical/torchbridge.mojopkg\n"
        "That build script does not exist in this tree yet "
        "(torchbridge/TORCH_BRIDGE_PLAN.md section 10 item 3)."
    )


# ===========================================================================
# WHAT ACTUALLY LOADED. DEVIATIONS 1205 AND 1213.
# ===========================================================================


def numeric_mode(lane=LANE_HOST):
    """`'identical'` or `'fast'`: what THIS LANE actually compiled with.

    **Read back from the binary, per lane, never from the environment.**

    - `LANE_HOST` asks `_linalg_impl.numeric_mode()`, which reads
      `linalg_numeric_mode()` out of the loaded `.so` (a compile-time
      `is_defined` answer) and cross-checks it against the directory the
      loader chose.
    - `LANE_CUSTOM_OP` runs `identical_gemm_mode_probe`, a one-element
      device op that returns `GLOBAL_NUMERIC_MODE` from the kernel library
      that actually loaded.

    THE TWO LANES CAN DISAGREE, and that is not a hypothetical: the host
    lane's mode is a directory choice made by `bindings/build_linalg.sh`
    while the custom-op lane's is a property of a `.mojopkg` (or of a JIT
    compile with no defines at all). A process can hold an identical `.so`
    and a fast kernel library at the same time. So the mode is asked PER
    LANE, cached per lane, and reported per lane in `profile()`.

    **Nothing in the ANSWER distinguishes the two builds.** On Apple they
    coincide at both pinned seams (contract 4.1), and contract 11.4 declines
    to promise even that they differ anywhere. A caller who checks by
    comparing numbers learns nothing; this function is the only honest source.
    """
    if lane in _mode_cache:
        return _mode_cache[lane]
    if lane == LANE_HOST:
        mode = linalg.numeric_mode()
    elif lane == LANE_CUSTOM_OP:
        torch = _torch()
        ops = _custom_ops()
        # Probed on `cuda:0`. A multi-GPU box whose devices somehow held
        # different kernel libraries would be mislabeled by this, which
        # cannot happen today (one `CustomOpLibrary` per process, one
        # `InferenceSession` over every accelerator) and is worth knowing if
        # that ever changes.
        probe = torch.empty(1, dtype=torch.int32, device="cuda")
        ops.identical_gemm_mode_probe(probe)
        mode = "identical" if int(probe.item()) == 1 else "fast"
        ver = torch.empty(1, dtype=torch.int32, device="cuda")
        ops.identical_gemm_profile_probe(ver)
        got = int(ver.item())
        if got != PROFILE_VERSION:
            raise RuntimeError(
                f"mojolearn.torch_ops: the loaded kernel library implements "
                f"profile {PROFILE_FAMILY}.v{got} but this wrapper describes "
                f"{PROFILE}. The version names the ARITHMETIC (contract 7.1 "
                "and 7.2), so this is two different answers and not a "
                "packaging detail. Repackage torchbridge."
            )
    else:
        raise ValueError(f"mojolearn.torch_ops: unknown lane {lane!r}")
    _mode_cache[lane] = mode
    return mode


def require_identical(lane=LANE_HOST):
    """Raise unless THIS LANE loaded the IDENTICAL build.

    Call it once at start-up if your program's correctness depends on the
    profile, so the failure lands at start-up rather than at the first
    matmul. `identical_matmul` calls it for you unless you passed
    `identical=False`.
    """
    if numeric_mode(lane) != "identical":
        if lane == LANE_HOST:
            fix = (
                "  Set MOJOLEARN_NUMERIC_MODE=identical in the environment "
                "BEFORE importing mojolearn, having built the identical "
                "binary with\n      MOJOLEARN_NUMERIC_MODE=identical bash "
                "bindings/build_linalg.sh"
            )
        else:
            fix = (
                "  The custom-op lane loads a Mojo kernel library, and a JIT "
                "compile from source carries NO build defines, so it is the "
                "FAST arithmetic by construction. Package it with the define "
                "instead:\n      mojo package -D "
                "MOJOLEARN_NUMERIC_IDENTICAL=1 -I . torchbridge -o "
                "python/mojolearn/identical/torchbridge.mojopkg\n"
                "  That build script does not exist in this tree yet."
            )
        raise RuntimeError(
            f"mojolearn.torch_ops: the {lane} lane loaded the FAST build, "
            f"which makes NO identity claim; {PROFILE} is what the IDENTICAL "
            "build computes. Nothing in the numbers would have told you "
            "(contract 4.1, 11.4).\n"
            + fix
            + "\n  Or pass identical=False to say in the source that you "
            "want the fast product and are making no identity claim about it."
        )


def profile(lane=LANE_HOST):
    """What this process is holding on `lane`, as a dict you can print or
    assert on. The profile name is part of the claim, so a result quoted
    anywhere should carry it.

    `identity_claimed` is the field that matters. Every other field describes
    the SHAPE of the computation and none of them is a guarantee about bits.
    """
    mode = numeric_mode(lane)
    d = dict(linalg.profile()) if lane == LANE_HOST else {}
    d.update(
        {
            "profile": PROFILE,
            "profile_version": PROFILE_VERSION,
            "lane": lane,
            "numeric_mode": mode,
            "identity_claimed": mode == "identical",
            "dtype": "float32",
            "ops": ("OP_NN", "OP_NT", "OP_TN"),
            "contract": "gemm/IDENTICAL_FP32_CONTRACT.md",
            "kernel": "gemm/mojo_only/gemm_identical.mojo",
            "backward": "gemm/mojo_only/gemm_backward.mojo",
            "bridge_plan": "torchbridge/TORCH_BRIDGE_PLAN.md",
            "bridge_gated": False,
            "bridge_gates_owed": "T1..T9, TORCH_BRIDGE_PLAN.md section 8",
        }
    )
    return d


# ===========================================================================
# OPERAND CHECKING. DEVIATIONS 1201, 1202, 1203, 1211, 1212.
# ===========================================================================


def _check_operand(x, name, allow_copy):
    """A float32, 2-D, contiguous torch tensor, or an exception naming why not.

    Returns the tensor to use. It is the SAME OBJECT as `x` unless
    `allow_copy` was passed and a copy was needed, in which case the caller
    asked for it in their own source.
    """
    torch = _torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"mojolearn.torch_ops: {name} must be a torch.Tensor, got "
            f"{type(x).__name__}"
        )

    # DTYPE. DEVIATION 1202. REFUSED, NEVER CAST.
    if x.dtype != torch.float32:
        raise TypeError(
            f"mojolearn.torch_ops: {name} has dtype {x.dtype}, and only "
            f"float32 is in this profile ({PROFILE}; contract section 1 makes "
            "FP32 a hard requirement and 0.5 excludes FP16, BF16, TF32 and "
            "float64). Refused rather than cast, because the product this "
            "module sells is your control over exactly which bits go in, and "
            "a cast performed in here takes that away and returns an answer "
            "to a question you did not ask.\n"
            "  If this arrived as bfloat16 or float16 without you typing a "
            "dtype, you are probably inside torch.autocast. Wrap the call in "
            "`with torch.autocast(device_type=..., enabled=False):` and cast "
            f"yourself with {name}.float(), where you can see it -- silently "
            "upcasting in here would run an arithmetic your surrounding model "
            "does not use, at twice the memory traffic, invisibly."
        )

    # RANK. DEVIATION 1203. Batched GEMM is DEFERRED, contract section 0.3.
    if x.dim() != 2:
        raise ValueError(
            f"mojolearn.torch_ops: {name} must be 2-D, got {x.dim()}-D shape "
            f"{tuple(x.shape)}. Batched GEMM is DEFERRED by contract section "
            "0.3 and this function will NOT loop over a batch dimension for "
            "you.\n"
            "  The loop would be numerically indistinguishable from a real "
            "batched GEMM -- the per-cell arithmetic depends on k and the "
            "profile alone -- and arbitrarily slower, one kernel launch and "
            "one synchronize per batch item, with no way for you to find "
            "out. So it is refused instead of hidden.\n"
            "  If your batch dimension is contractible, it is not a batch: "
            "a (B, T, K) @ (K, N) linear layer is ONE OP_NN at m = B*T. "
            f"Reshape with {name}.reshape(-1, {name}.shape[-1]) and reshape "
            "the result back. That is a view, not a copy, on a contiguous "
            "tensor.\n"
            "  A vector product is OP_NT at n == 1 and is not a fourth "
            "operation (contract 0.1); pass it as 2-D."
        )

    if x.numel() == 0:
        raise ValueError(
            f"mojolearn.torch_ops: {name} has shape {tuple(x.shape)} and no "
            "elements. Contract section 8 does specify the degenerate shapes, "
            "but no gate in this tree has run them through a torch boundary, "
            "so they are refused here rather than answered unchecked."
        )

    # CONTIGUITY. DEVIATION 1201. REFUSED, NEVER SILENTLY REPAIRED.
    if not x.is_contiguous():
        if not allow_copy:
            raise ValueError(
                f"mojolearn.torch_ops: {name} is not contiguous "
                f"(shape {tuple(x.shape)}, strides {tuple(x.stride())}), and "
                "contract section 2 requires row-major fully contiguous "
                "operands -- no leading dimension, no stride, no offset, no "
                "sub-matrix view.\n"
                "  This function will NOT call .contiguous() for you. That is "
                "one line, it produces the RIGHT ANSWER, every numerical "
                "check in this repository passes with it in place, and it is "
                "a full copy of your operand on your device that you did not "
                "write and cannot see. On a 4096x4096 fp32 activation it is "
                "64 MB per call, every call, forever.\n"
                f"  Write {name}.contiguous() yourself, where you can price "
                "it, or pass allow_copy=True to say in your source that the "
                "copy is intended.\n"
                "  Note that the common case needs neither: F.linear(x, W) is "
                "x @ W.T with both operands stored C-contiguous, which is "
                "OP_NT with no copy and no materialized transpose "
                "(identical_matmul(x, W, transpose_b=True))."
            )
        x = x.contiguous()

    return x


def _check_out(out, m, n, device, a, b):
    """A caller-supplied `out=`, checked and never repaired. DEVIATION 1212."""
    torch = _torch()
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "mojolearn.torch_ops: out must be a torch.Tensor, got "
            f"{type(out).__name__}"
        )
    if out.dtype != torch.float32:
        raise TypeError(
            f"mojolearn.torch_ops: out has dtype {out.dtype}, and the "
            "profile's output is float32 (contract section 1). Refused "
            "rather than cast on the way out."
        )
    if tuple(out.shape) != (m, n):
        raise ValueError(
            f"mojolearn.torch_ops: out has shape {tuple(out.shape)}, want "
            f"({m}, {n})"
        )
    if out.device != device:
        raise ValueError(
            f"mojolearn.torch_ops: out is on {out.device} and the operands "
            f"are on {device}. Moving it for you would be a copy you did not "
            "write."
        )
    if not out.is_contiguous():
        raise ValueError(
            "mojolearn.torch_ops: out must be contiguous; the device writes "
            "it directly (contract section 2). A non-contiguous out cannot "
            "be written in place, and copying into it afterwards would make "
            "`out` a lie about where the result was produced."
        )
    # ALIASING. The device writes `out` while reading `a` and `b`; an `out`
    # that shares storage with either is a read-write hazard that produces a
    # plausible wrong matrix and never an exception.
    for name, t in (("a", a), ("b", b)):
        if out.data_ptr() == t.data_ptr() or (
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        ):
            raise ValueError(
                f"mojolearn.torch_ops: out shares storage with {name}. The "
                "device writes out while reading the operands; overlapping "
                "them is a wrong answer, not an exception."
            )
    return out


# ===========================================================================
# THE DISPATCH. ONE DOOR, BOTH LANES.
# ===========================================================================


def _dispatch(a, b, op, out, identical):
    """`out = op(a) . op(b)`, on whichever lane `a.device` selects.

    **EXACTLY ONE PRODUCER OF THE LANE CHOICE AND THE OP CODE.**
    `gemm_identical.mojo` gives `(L, P)` one producer so launch geometry
    cannot reach the arithmetic; `gemm_backward.mojo` gives the backward
    shape two. The same discipline here for the two things this file can get
    wrong.
    """
    torch = _torch()
    lane = lane_for_device(a.device)
    if identical:
        require_identical(lane)
    else:
        numeric_mode(lane)

    if lane == LANE_CUSTOM_OP:
        # Destination-passing style. `torch.library.custom_op(...,
        # mutates_args=...)` marks the leading arguments as mutated
        # (max/experimental/torch/torch.py lines 493 and 529), and the Mojo
        # signature is `(c, a, b, ctx)` with the OUTPUT FIRST, mirroring
        # `identical_gemm(ctx, c, a, b, ...)`. Swapping `out` with `a` writes
        # the device's output over the caller's input tensor, which is memory
        # corruption in this process and not an exception.
        getattr(_custom_ops(), _OP_NAMES[op])(out, a, b)
        return out

    # HOST LANE. Four copies on mps, two on cpu, and they are counted out
    # loud in TORCH_BRIDGE_PLAN.md section 2.3 rather than hidden.
    #
    # `linalg.matmul` is the entry, NOT a second front door of our own: its
    # guards, its refusals, its op mapping and its `require_identical` are
    # reused as they stand. **This bridge adds no numerical surface.**
    ta, tb = _FLAGS[op]
    an = a.detach().cpu().numpy()
    bn = b.detach().cpu().numpy()
    cn = linalg.matmul(
        an, bn, transpose_a=ta, transpose_b=tb, identical=identical
    )
    # `torch.from_numpy` shares memory with `cn`, which is alive in this
    # frame; `copy_` moves it to `out`'s device and `out` is the caller's.
    out.copy_(torch.from_numpy(cn))
    _ = an, bn
    return out


def _shapes(a, b, op):
    """`(m, n, k)` for the forward call, with `k` CHECKED against both
    operands.

    A `k` mismatch that is not caught here is an out-of-bounds read on the
    device. The row-major shapes are contract section 0.1:

        OP_NN   C = A . B      A is m x k,  B is k x n
        OP_NT   C = A . B^T    A is m x k,  B is n x k
        OP_TN   C = A^T . B    A is k x m,  B is k x n
    """
    if op == OP_TN:
        k, m = a.shape
    else:
        m, k = a.shape
    if op == OP_NT:
        n, kb = b.shape
    else:
        kb, n = b.shape
    if k != kb:
        raise ValueError(
            f"mojolearn.torch_ops: contracted extents disagree, a gives k={k} "
            f"and b gives k={kb} (a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}, op={_OP_NAMES[op]}). The row-major "
            "shapes for each op are in contract section 0.1."
        )
    return int(m), int(n), int(k)


# ===========================================================================
# THE BACKWARD ROUTING TABLE. DEVIATION 1206.
# ===========================================================================
# A LINE-FOR-LINE MIRROR of `gemm/mojo_only/gemm_backward.mojo`'s
# `gemm_backward_a_call` and `gemm_backward_b_call`, ported so a torch
# autograd node can route without a second Mojo entry point. The names match
# on purpose so gate T5 can compare the two functions case by case rather
# than comparing this file against its own docstring.
#
# **THERE IS NO ARITHMETIC IN EITHER FUNCTION AND THERE MUST NEVER BE.** They
# return which of the contract's three operations computes the gradient, at
# what shape, with which operand on which side. Every actual number comes
# back through `_dispatch` onto the same certified forward kernel. That is
# the claim `gemm_backward.mojo` makes about itself and it is falsifiable
# rather than decorative -- if a multiply or an add appears here, it is false.
#
#     forward   A       B       dA                      dB
#     OP_NN     m x k   k x n   OP_NT(dC, B) @ (m,k,n)  OP_TN(A, dC) @ (k,n,m)
#     OP_NT     m x k   n x k   OP_NN(dC, B) @ (m,k,n)  OP_TN(dC, A) @ (n,k,m)
#     OP_TN     k x m   k x n   OP_NT(B, dC) @ (k,m,n)  OP_NN(A, dC) @ (k,n,m)
#
# Two of the six put `dC` on the RIGHT, so operand order is part of the table
# and not a convention that can be assumed.

#: `dC` is the LEFT operand of the backward call.
BWD_DC_LEFT = 0
#: `dC` is the RIGHT operand of the backward call.
BWD_DC_RIGHT = 1


def gemm_backward_a_call(op, m, n, k):
    """`(op', m', n', k', dc_side)` for `dA`, given the FORWARD `(op,m,n,k)`.

    `dA` always has `A`'s shape. `k'` is `n` in all three rows -- the `dA`
    product contracts over the OUTPUT width, so for a linear layer that is
    `out_features` and nothing new happens to the leaf rule.
    """
    bop, bm, bn, side = OP_NT, m, k, BWD_DC_LEFT
    if op == OP_NT:
        bop = OP_NN
    elif op == OP_TN:
        bm, bn, side = k, m, BWD_DC_RIGHT
    return (bop, bm, bn, n, side)


def gemm_backward_b_call(op, m, n, k):
    """`(op', m', n', k', dc_side)` for `dB`, given the FORWARD `(op,m,n,k)`.

    `dB` always has `B`'s shape.

    **`k'` IS `m` IN ALL THREE ROWS, AND THAT IS THE FINDING OF THE BACKWARD
    LANE.** `m` is the forward's batch dimension, so the weight gradient
    contracts over the TOKENS. Contract 6.1 forbids the leaf size from
    depending on `m` precisely because the forward must be batch invariant;
    here the batch dimension arrives as `k`, where section 6 REQUIRES the
    leaf size to depend on it. Both are the contract working correctly, and
    together they say `dB` is bit identical across vendors at a FIXED token
    count and is a different number at a different one. See
    `accumulation_split_is_aligned` for what that costs a microbatched loop
    and for the split that costs nothing.
    """
    bop, bm, bn, side = OP_TN, k, n, BWD_DC_RIGHT
    if op == OP_NT:
        bm, bn, side = n, k, BWD_DC_LEFT
    elif op == OP_TN:
        bop = OP_NN
    return (bop, bm, bn, m, side)


def _run_backward_call(call, dc, w, identical):
    """Run one row of the table. `w` is the OTHER forward operand."""
    bop, bm, bn, _bk, side = call
    left, right = (dc, w) if side == BWD_DC_LEFT else (w, dc)
    out = left.new_empty((bm, bn))
    # REACH CHECK, not decoration. The table says what shape the result has;
    # if the routing and the operands ever disagree this is where it shows,
    # rather than as a plausible wrong matrix in someone's optimizer.
    gm, gn, gk = _shapes(left, right, bop)
    if (gm, gn) != (bm, bn) or gk != _bk:
        raise RuntimeError(
            "mojolearn.torch_ops: the backward routing table says this call "
            f"is {_OP_NAMES[bop]} at ({bm}, {bn}, {_bk}) but the operands "
            f"give ({gm}, {gn}, {gk}). One of the two is wrong and neither "
            "can be trusted. gemm/mojo_only/gemm_backward.mojo is the "
            "authority on the table."
        )
    _dispatch(left, right, bop, out, identical)
    return out


_autograd_cache = None


def _autograd_fn():
    """The `torch.autograd.Function`, built on first use.

    It is defined inside a function because it subclasses
    `torch.autograd.Function`, and torch is imported lazily. That is the only
    reason.

    The classic combined `forward(ctx, ...)` form is used rather than the
    newer `forward()` + `setup_context()` split. Both are supported; the
    combined one is the one this lane could write without guessing at which
    torch versions accept which. **Untested either way.**
    """
    global _autograd_cache
    if _autograd_cache is not None:
        return _autograd_cache
    torch = _torch()

    class IdenticalMatmul(torch.autograd.Function):
        """`C = op(A) . op(B)` under the profile, differentiable.

        WHAT THE GRADIENTS ARE, AND WHAT THEY ARE NOT
        ----------------------------------------------
        `dA` and `dB` are computed by routing onto the SAME certified forward
        kernel through the table above. There is no second arithmetic and no
        second profile. At a fixed shape each of them is bit identical across
        vendors, launches, plans and block counts.

        THE ALIGNMENT REQUIREMENT THIS PLACES ON YOU
        ---------------------------------------------
        `dB` contracts over the token count, so its bits are a function of
        how many tokens were in the call. If you accumulate `dB` across
        microbatches, the schedule is part of your numerical specification
        unless the split is aligned; `accumulation_split_is_aligned` is the
        predicate and its docstring carries the measurement. This class does
        NOT enforce it. It cannot -- it sees one call and has no idea what
        you do with the gradient afterwards.
        """

        @staticmethod
        def forward(ctx, a, b, op, identical):
            m, n, k = _shapes(a, b, op)
            out = a.new_empty((m, n))
            _dispatch(a, b, op, out, identical)
            ctx.save_for_backward(a, b)
            ctx.gemm_op = op
            ctx.gemm_shape = (m, n, k)
            ctx.gemm_identical = identical
            return out

        @staticmethod
        def backward(ctx, dc):
            a, b = ctx.saved_tensors
            op = ctx.gemm_op
            m, n, k = ctx.gemm_shape
            identical = ctx.gemm_identical

            # `dC` arrives from downstream and torch does not promise it is
            # contiguous. Refusing here would be refusing something the
            # CALLER did not do, so this one IS made contiguous -- and it is
            # the only copy in this file that is not either refused or
            # asked for. Said out loud rather than hidden, because it is the
            # exception to DEVIATION 1201.
            if not dc.is_contiguous():
                dc = dc.contiguous()

            da = db = None
            if ctx.needs_input_grad[0]:
                da = _run_backward_call(
                    gemm_backward_a_call(op, m, n, k), dc, b, identical
                )
            if ctx.needs_input_grad[1]:
                db = _run_backward_call(
                    gemm_backward_b_call(op, m, n, k), dc, a, identical
                )
            # `op` and `identical` are not differentiable.
            return da, db, None, None

    _autograd_cache = IdenticalMatmul
    return _autograd_cache


# ===========================================================================
# THE PUBLIC SURFACE
# ===========================================================================


def identical_matmul(
    a,
    b,
    *,
    transpose_a=False,
    transpose_b=False,
    out=None,
    identical=True,
    allow_copy=False,
    return_meta=False,
):
    """`C = op(a) . op(b)` on the GPU, in float32, under `PROFILE`.

    Parameters
    ----------
    a, b : torch.Tensor
        2-D, dtype float32, contiguous, on the same device. Any other dtype
        is refused by name and never cast; a non-contiguous tensor is refused
        unless `allow_copy=True`; a 3-D or higher tensor is refused rather
        than looped over.
    transpose_a, transpose_b : bool
        Which of the contract's three operations to run. The flags describe
        the TENSORS you pass, so `transpose_a=True` means `a` is stored
        `k x m` and the left operand is its transpose.

            transpose_a  transpose_b   op      C          a.shape   b.shape
            False        False         OP_NN   a @ b      (m, k)    (k, n)
            False        True          OP_NT   a @ b.T    (m, k)    (n, k)
            True         False         OP_TN   a.T @ b    (k, m)    (k, n)
            True         True          REFUSED

        **Both true is refused by name** (DEVIATION 1211, mirroring 913).
        The contract has three operations and `a.T @ b.T` is not one of them.
    out : torch.Tensor, optional
        Where to write. float32, contiguous, shape `(m, n)`, same device, not
        aliasing either operand. A fresh tensor is allocated when None.
        **Ignored when either operand requires grad**, because autograd owns
        the output tensor in that case; that is stated rather than silently
        tolerated.
    identical : bool, default True
        Whether you are asking for the profile's guarantee. True requires
        that the LANE THIS CALL RUNS ON loaded the identical build and raises
        if it did not. False says in your source that you want whichever
        build is present and are making no identity claim about the result.
    allow_copy : bool, default False
        Permit a `.contiguous()` on a strided operand. Passing it is a
        statement; omitting it is not.
    return_meta : bool, default False
        Also return a dict naming the lane, the device, the numeric mode and
        the op that actually ran. **Read back from what happened, never
        inferred from the inputs**, which matters more here than it looks:
        because this profile is device independent, a wrongly routed device
        returns THE SAME BITS, so no numerical check anywhere can see it.
        `torchbridge/TORCH_BRIDGE_PLAN.md` section 8.0(a).

    Returns
    -------
    torch.Tensor, or `(torch.Tensor, dict)` when `return_meta=True`.

    What you are getting, stated once
    ---------------------------------
    Under `identical=True` the answer is `gemm_oracle`'s, bit for bit: leaves
    of `contract_leaf_size(k)` accumulated serially ascending with a fused
    multiply-add and a flush-to-zero at every seam, folded by a fixed
    balanced tree with adjacent pairing and a carried odd tail. A pure
    function of the input bits, `k` and the profile.

    It is NOT the most accurate way to sum `k` products and is not trying to
    be (contract section 1 is explicit that sameness rather than accuracy is
    what is bought). It is not `torch.matmul`'s answer and will not match it
    bit for bit.

    Where the measurement stops, honestly
    -------------------------------------
    The kernel's certified sweep is 62 shapes and the three-vendor card is 60
    stages (`gemm/README.md`). **This function accepts shapes outside that
    sweep**, where identity rests on the construction (contract 6 and 0.3)
    rather than on a measurement at your shape.

    **And the BRIDGE itself has never been gated at all.** Every gate in
    `torchbridge/TORCH_BRIDGE_PLAN.md` section 8 is unwritten. A wrapper can
    lose identity in the copy, the stride handling, the dtype path or the
    device routing without the kernel changing one instruction, and nothing
    in this tree has yet checked that this one does not.
    """
    key = (bool(transpose_a), bool(transpose_b))
    if key not in _OPS:
        raise ValueError(
            "mojolearn.torch_ops.identical_matmul: transpose_a=True with "
            f"transpose_b=True is refused. {PROFILE} has exactly three "
            "operations (contract section 0.1: OP_NN, OP_NT, OP_TN) and "
            "a.T @ b.T is not one of them. This function will not "
            "materialize a transpose to fake a fourth. If you want it, write "
            "the identity yourself:\n"
            "    identical_matmul(b, a).T.contiguous()\n"
            "which is (b @ a).T == a.T @ b.T, runs as OP_NN, and leaves the "
            "extra step visible in your source where you can price it."
        )
    op = _OPS[key]

    a = _check_operand(a, "a", allow_copy)
    b = _check_operand(b, "b", allow_copy)
    if a.device != b.device:
        raise ValueError(
            f"mojolearn.torch_ops: a is on {a.device} and b is on {b.device}. "
            "Moving one for you would be a copy you did not write, and on the "
            "host lane it would be four."
        )
    m, n, k = _shapes(a, b, op)
    lane = lane_for_device(a.device)

    if a.requires_grad or b.requires_grad:
        if out is not None:
            raise ValueError(
                "mojolearn.torch_ops: out= cannot be combined with an operand "
                "that requires grad; autograd owns the output tensor. Drop "
                "out=, or detach the operands if you did not want a gradient."
            )
        c = _autograd_fn().apply(a, b, op, identical)
    else:
        if out is None:
            c = a.new_empty((m, n))
        else:
            c = _check_out(out, m, n, a.device, a, b)
        _dispatch(a, b, op, c, identical)

    if not return_meta:
        return c
    meta = {
        "lane": lane,
        "device": str(a.device),
        "numeric_mode": numeric_mode(lane),
        "identity_claimed": identical and numeric_mode(lane) == "identical",
        "op": _OP_NAMES[op],
        "shape": (m, n, k),
        "profile": PROFILE,
        "bridge_gated": False,
    }
    return c, meta


def identical_linear(x, weight, bias=None, *, identical=True, **kw):
    """`F.linear(x, weight)` under the profile, plus an UNPINNED bias add.

    `x` is `(tokens, in_features)` and `weight` is `(out_features,
    in_features)`, both as PyTorch stores them, which is `OP_NT` with no copy
    and no materialized transpose.

    **THE BIAS ADD IS NOT IN THE PROFILE.** Contract section 10 excludes
    alpha, beta, bias and epilogues. `+ bias` below is an ordinary PyTorch
    elementwise add, performed by PyTorch, and it is a single rounding per
    cell so it is very likely identical across vendors -- but "very likely"
    is not what this module is for, and nothing has measured it. If your
    claim depends on it, drop the bias here and add it yourself where it is
    visible.
    """
    y = identical_matmul(x, weight, transpose_b=True, identical=identical, **kw)
    if bias is None:
        return y
    return y + bias


def identical_bias_grad(dc, *, identical=True):
    """`dbias[j] = sum over i of dC[i, j]`, as an `OP_NN` GEMM at `(1, n, m)`.

    **A REDUCTION ROUTED THROUGH THE PRODUCT, DELIBERATELY** (DEVIATION 851,
    which is the backward lane's number for the same decision).
    `db = ones[1 x m] . dC[m x n]` is `OP_NN` at `(1, n, m)`, and the leaf
    loop then runs `fma(ftz(1.0), ftz(dC[p, j]), acc)`. `fma(1, x, acc)` is
    ONE rounding of `x + acc` because the product `1 * x` is exact, so the
    ones vector turns the reduction into the contract's own ascending flushed
    chain inside a leaf and the contract's own balanced tree across leaves,
    with no second fold shape anywhere and nothing new to certify.

    The cost is stated rather than hidden -- `m * n` multiplications by 1.0
    that a hand-written reduction would not perform, and `m` floats of ones.
    It buys one arithmetic instead of two.

    The ones vector is built as EXACTLY 1.0 and is not cached across calls
    here. **A wrong value in it is a wrong gradient with no symptom**,
    because any vector produces a plausible weighted column sum; the gate for
    it is the finite-difference check of IDENTICAL_BACKWARD_PLAN.md gate B1,
    which is not written for this lane.

    This reduces `dC`'s ROW axis, which is the linear-layer bias case (one
    bias per output column).
    """
    dc = _check_operand(dc, "dc", allow_copy=False)
    m, n = int(dc.shape[0]), int(dc.shape[1])
    ones = dc.new_ones((1, m))
    out = dc.new_empty((1, n))
    _dispatch(ones, dc, OP_NN, out, identical)
    return out.reshape(n)


# ===========================================================================
# THE MICROBATCH ALIGNMENT PREDICATE. DEVIATION 1207.
# ===========================================================================


def contract_leaf_size(k):
    """`L`, the logical k leaf size. A pure function of `k` and the two
    profile constants (contract section 6), mirroring
    `gemm/mojo_only/gemm_oracle.mojo::contract_leaf_size`.

        k <= 0                                  -> 1
        k <= K_LEAF_MIN                         -> k
        ceil(k / K_LEAF_MIN) <= MAX_LEAVES      -> K_LEAF_MIN
        otherwise                               -> ceil(k / MAX_LEAVES)
    """
    if k <= 0:
        return 1
    if k <= CONTRACT_K_LEAF_MIN:
        return k
    p = -(-k // CONTRACT_K_LEAF_MIN)
    if p <= CONTRACT_MAX_LEAVES:
        return CONTRACT_K_LEAF_MIN
    return -(-k // CONTRACT_MAX_LEAVES)


def _fold_children(p):
    """The leaf span of the ROOT's two children, for `p` leaves, or None.

    Contract 7.2: a fixed balanced binary tree over ADJACENT leaves, with a
    carried odd tail. Level 0 is the `p` leaves; each level pairs adjacent
    nodes and carries the last one unchanged when the count is odd.

    Returns `(left_leaves, right_leaves)` or None when `p < 2` (no fold
    addition happens at all at `P == 1`, contract 7.3).
    """
    if p < 2:
        return None
    spans = [(i, i + 1) for i in range(p)]
    prev = spans
    while len(spans) > 1:
        prev = spans
        nxt = []
        i = 0
        while i + 1 < len(spans):
            nxt.append((spans[i][0], spans[i + 1][1]))
            i += 2
        if i < len(spans):
            nxt.append(spans[i])
        spans = nxt
    if len(prev) != 2:
        return None
    return (prev[0][1] - prev[0][0], prev[1][1] - prev[1][0])


def accumulation_split_is_aligned(total_tokens, parts):
    """Is accumulating `dB` over `parts` bit-equal to one call at
    `total_tokens`?

    Returns `(aligned: bool, reason: str)`.

    WHY THIS EXISTS
    ---------------
    `gemm_backward_b_call` routes `dB` at `k' = m`, the forward's batch
    dimension, so the weight gradient's bits are a function of the token
    count. The obvious conclusion is that microbatching destroys identity,
    and the backward lane's gate G5 MEASURED THAT TO BE FALSE for an aligned
    split, on 2026-08-25, host and device agreeing:

        T = 512 split 256/256   moved 0 of 35 gradient cells
        T = 384 split 256/128   moved 0 of 35
        T = 300 split 150/150   moved 31
        T = 512 split 200/312   moved 30
        T = 384 split 192/192   moved 31
        dA moved 0 cells under EVERY split, aligned or not.

    THE RULE THIS IMPLEMENTS, AND ITS ASSUMPTIONS
    ----------------------------------------------
    Two conditions, and BOTH are required.

    1. Every part is a positive multiple of `L = contract_leaf_size(total)`,
       and `L` is the same for the total and for each part. A split inside a
       leaf shares no boundary with the unsplit partition at all.
    2. The parts decompose the balanced tree LEFT-ASSOCIATIVELY. Accumulating
       `p1, p2, ..., pN` in order is `((p1 + p2) + p3) + ...`, so the tree
       must lean that way: the LAST part must be exactly the root's right
       child, and the parts before it must recursively satisfy the same
       condition on the left child.

       **Condition 2 is stricter than "every boundary is a subtree
       boundary", and that difference is not obvious.** At `P = 8`, splitting
       into four parts of two leaves each puts every boundary on a subtree
       boundary, and it is still NOT free -- the unsplit root is
       `(n0 + n1) + (n2 + n3)` while a left-associative accumulation is
       `((n0 + n1) + n2) + n3`. Those are different sums.

    WHAT IS MEASURED AND WHAT IS CONSTRUCTION
    ------------------------------------------
    The five cases above are MEASURED. Everything else this function returns
    True for is CONSTRUCTION from contract 7.2, and it has not been checked
    at your split. Gate T6 in the bridge plan is what would check it.

    **It also assumes your accumulator is the contract's flushed add.** A
    plain `+=` into a PyTorch `.grad` buffer is not that. On a flush-to-zero
    backend they coincide; on one that is not, they do not, and that gap is
    owed (item 5 at the top of this file). If you need the guarantee today,
    accumulate through `identical_matmul` yourself or keep the parts and add
    them once.

    A False here does NOT mean your gradient is vendor-dependent. Each part
    is still bit identical everywhere. It means the accumulated total is a
    different number from the unsplit one, so your reproducibility claim has
    to name the schedule.
    """
    total = int(total_tokens)
    parts = [int(p) for p in parts]
    if total <= 0:
        return False, "total_tokens must be positive"
    if not parts:
        return False, "no parts given"
    if any(p <= 0 for p in parts):
        return False, "every part must be a positive token count"
    if sum(parts) != total:
        return False, (
            f"the parts sum to {sum(parts)}, not {total}; this predicate "
            "answers about a partition of the same tokens"
        )
    if len(parts) == 1:
        return True, "one part is the unsplit call"

    leaf = contract_leaf_size(total)
    for i, p in enumerate(parts):
        if p % leaf:
            return False, (
                f"part {i} is {p} tokens, which is not a multiple of the leaf "
                f"size L={leaf} at k={total}. A split inside a leaf shares no "
                "boundary with the unsplit partition (contract sections 6 "
                "and 7.1)."
            )
        if contract_leaf_size(p) != leaf:
            return False, (
                f"part {i} is {p} tokens, whose leaf size is "
                f"{contract_leaf_size(p)} rather than the total's {leaf}. "
                "Different leaf sizes mean different leaves, so no boundary "
                "can line up."
            )

    counts = [p // leaf for p in parts]

    def _ok(p_leaves, cs):
        if len(cs) == 1:
            return cs[0] == p_leaves
        children = _fold_children(p_leaves)
        if children is None:
            return False
        left, right = children
        if cs[-1] != right:
            return False
        return _ok(left, cs[:-1])

    if not _ok(sum(counts), counts):
        return False, (
            f"the leaf counts {counts} do not decompose the balanced fold "
            "left-associatively: the last part must be exactly the root's "
            "right subtree, and the parts before it must do the same on the "
            "left subtree (contract 7.2). Note that this is STRICTER than "
            "'every boundary is a subtree boundary' -- see this function's "
            "docstring for the P=8 counterexample."
        )
    return True, (
        f"leaf size L={leaf}, leaf counts {counts}, which decompose the "
        "balanced fold left-associatively; CONSTRUCTION from contract 7.2 "
        "unless this is one of gate G5's five measured cases, and it assumes "
        "your cross-part accumulator is the contract's flushed add"
    )
