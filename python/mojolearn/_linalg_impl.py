# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.linalg`: the bit-identical FP32 matrix product.

Profile `mojolearn.identical.gemm.fp32.v1`. Contract
`gemm/IDENTICAL_FP32_CONTRACT.md`; kernel `gemm/checks/gemm_identical.mojo`;
oracle `gemm/host/gemm_oracle.mojo::gemm_oracle`.

Everything else this package exposes is an estimator. This is a numerical
primitive, and its audience is anyone who needs a matrix product that returns
the same bits on Apple, NVIDIA and AMD, including people who will never fit a
model here.

THE ONE THING TO READ BEFORE USING THIS MODULE
-----------------------------------------------
**The identity claim belongs to the IDENTICAL build, not to this function
name.** Two builds of the extension can sit in the package
(`python/mojolearn/_backend.py`):

    python/mojolearn/_mojolearn_linalg.so            NUMERIC_FAST, the default
    python/mojolearn/identical/_mojolearn_linalg.so  NUMERIC_IDENTICAL

and `MOJOLEARN_NUMERIC_MODE=identical` in the environment AT IMPORT TIME is
what selects the second. The FAST build runs the SAME kernels on the SAME
path with the fused-multiply-add pin and the flush-to-zero pin compiled away
(`gemm/checks/gemm_identical.mojo`, "WHAT `NUMERIC_FAST` DOES HERE"). It is
a correct GEMM. **It makes no identity claim of any kind**, and contract
section 11.4 declines to promise even that it DIFFERS from the identical one.

Nothing in the returned array distinguishes the two. On Apple they coincide at
both pinned seams -- contract section 4.1 measured Metal fused in both modes,
and `ftz` is a no-op on a flush-to-zero backend -- so a caller who checks by
comparing numbers on one Mac learns nothing at all.

So `matmul` REQUIRES THE IDENTICAL BUILD BY DEFAULT and refuses, loudly and by
name, when it is not the one loaded. DEVIATION 911. The argument, because a
default that raises deserves one:

- The failure being designed against is a user who installs this package for a
  reproducible matrix product, calls `matmul`, gets the fast answer, and
  believes otherwise. That is a wrong answer with no symptom, and the
  contract's own preamble is about exactly this class of thing: a claim that
  cannot be checked is a belief rather than a property.
- The alternative defaults are worse. Defaulting to the fast product means the
  obvious call quietly under-delivers the one guarantee the module exists for.
  Two function names (`matmul` and `matmul_identical`) means the shorter name
  is the trap, and the shorter name is what people type.
- The cost of this default is an exception, on the first call, naming both
  ways forward. That is the loudest possible failure and the cheapest to fix.
  A caller who genuinely wants the fast product asks for it with
  `identical=False`, which is a statement rather than an omission.

`numeric_mode()` and `profile()` report what actually loaded, read back from
the binary's own compile-time answer, not from the environment variable. A
binary in the wrong directory is caught there and nowhere else.

WHAT THE PROFILE COVERS, AND WHERE THE MEASUREMENT STOPS
---------------------------------------------------------
The certified sweep is 62 shapes across eight execution plans, with launch
invariance, batch invariance and batch-composition invariance gated and six
sabotages shown to fail; the three-vendor card (Apple M4, NVIDIA H100, AMD
MI325X) is 60 stages, judged by `tools/e3_round_judge.sh`. `gemm/README.md`
carries that status and is the authority on it.

**This function accepts shapes outside that sweep, and 62 shapes is not all
shapes.** What holds outside it is CONSTRUCTION, not measurement: contract
section 6 makes the leaf partition a pure function of `k` and two profile
constants, and section 0.3 derives from that the statement that a cell's
arithmetic depends on `k` and the profile alone, not on `m`, not on `n` and
not on how many cells shared the launch. That is a strong argument and it is
not a measurement at your shape. Say so if you quote this module in a paper.

Two things the contract explicitly does NOT promise, both of which reach a
caller of this function:

- **NaN payload bits are not promised** (section 9.1). If your output can
  contain NaN, compare those cells as "is NaN", not by bits.
- **A downstream `min`, `max` or `argmin` over this output reintroduces order
  dependence** (section 9.2(e)), because `-0.0 == +0.0` compares equal. The
  profile guarantees the sign it hands you does not depend on the launch; it
  cannot stop you from creating an order dependence afterwards.
"""

import array
import importlib.machinery
import importlib.util
import os
import sys

from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, as_i8_c, as_i32_c, as_u16_c, empty
from ._bufcheck import dtype_name, is_native_f32, nelems, probe

#: The profile family and the version, kept apart because the VERSION is the
#: part the contract makes load-bearing: "a bit-identity claim with no version
#: on it is a claim about whichever revision the reader happens to be
#: holding." The leaf rule (contract 7.1) and the fold topology (7.2) are what
#: the number is about; changing either creates v2 and does not amend v1.
PROFILE_FAMILY = "mojolearn.identical.gemm.fp32"
PROFILE_VERSION = 1
PROFILE = f"{PROFILE_FAMILY}.v{PROFILE_VERSION}"
# API-shell metadata only. Unknown or accidentally installed numerical versions
# still refuse; an experimental binary also needs its exact expected profile.
# This opt-in does not turn a source experiment into accepted identity evidence.
_EXPERIMENT_PROFILES = {
    2: "mojolearn.identical.gemm.fp32.i04-leaf64",
    3: "mojolearn.identical.gemm.fp32.ni08-leaf256",
}
_EXPECTED_PROFILE_ENV = "MOJOLEARN_EXPERIMENT_GEMM_PROFILE"

#: The two low-bit profiles (gemm/IDENTICAL_LOWBIT_CONTRACT.md,
#: lane/identical-lowbit-inference, 2026-09-17). Same version discipline.
LOWBIT_PROFILE_VERSION = 1
PROFILE_BF16 = f"mojolearn.identical.gemm.bf16f32.v{LOWBIT_PROFILE_VERSION}"
PROFILE_INT8 = f"mojolearn.identical.gemm.int8i32.v{LOWBIT_PROFILE_VERSION}"

#: The fifteen-bit profile (gemm/IDENTICAL_LOWBIT_CONTRACT.md section 6,
#: lane/lowbit-int15, 2026-09-29). Its own version, read back from the
#: binary on its own: a change to it makes v2 of it and of nothing else.
INT15_PROFILE_VERSION = 1
PROFILE_INT15 = f"mojolearn.identical.gemm.int15i64.v{INT15_PROFILE_VERSION}"
#: The largest contracted extent the profile accepts (contract clause W-4).
INT15_MAX_K = 65536

#: The three operations of contract section 0.1, and their `op` codes as
#: `gemm/host/gemm_oracle.mojo` defines them. `gemv` is `OP_NT` at
#: `n == 1` and is NOT a fourth operation.
OP_NN = 0
OP_NT = 1
OP_TN = 2

_MODULE_NAME = "_mojolearn_linalg"
_BUILD_SCRIPT = "bindings/build_linalg.sh"
_binding_cache = None
_mode_cache = None
_profile_cache = None


def _load():
    """The linalg extension for the tier this process SELECTED, through
    `_backend.binding`: the one choke point that refuses an identical-only
    lane by name under a lower tier, loads the set the tier and vendor axes
    name, and cross-checks the binary's compiled tier. `numeric_mode()`
    below adds the profile-version check on top.

    DEVIATION 912 gave this module a private by-path loader because
    `_backend._MODULES` did not list `_mojolearn_linalg` then. It does now,
    and since DEVIATION 2490 (2026-09-10) this binding exists in the
    identical tier alone, so under `fast` or `deterministic` this raises the
    identical-only refusal before `require_identical` ever runs. A missing
    binary still raises HERE, on first use, rather than at import.
    """
    global _binding_cache
    if _binding_cache is None:
        _binding_cache = _backend.binding(_MODULE_NAME)
    return _binding_cache


def numeric_mode():
    """'fast', 'deterministic' or 'identical': what this process LOADED.

    Read back from the binary through `linalg_numeric_mode()`, which is a
    compile-time answer (`is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]`), and
    cross-checked against the directory the loader chose. A `.so` in the wrong
    directory is caught here; nothing else in the process can see it.

    Also checks the exact binary profile. The default accepts v1; an experimental
    profile needs its exact name in MOJOLEARN_EXPERIMENT_GEMM_PROFILE. Unknown
    versions and stale extensions refuse rather than return mislabeled answers.
    """
    global _mode_cache, _profile_cache
    if _mode_cache is not None:
        return _mode_cache
    binding = _load()
    raw = int(binding.linalg_numeric_mode())
    # A NAME lookup: the binary reports the NUMERIC_* code, and the middle
    # tier is 2. `raw == 1 else "fast"` called a deterministic binary
    # "fast", which then AGREED with a fast selector -- a cross-check that
    # passes on the wrong arm is worse than no cross-check.
    compiled = _backend._CODE_MODE.get(raw, "unknown")
    selected = _backend.numeric_mode()
    if compiled != selected:
        raise RuntimeError(
            "mojolearn.linalg: the loaded binary was compiled "
            f"{compiled} but this process selected the {selected} set -- a "
            f".so is in the wrong directory ({binding.__file__}); rebuild "
            f"both sets with {_BUILD_SCRIPT}"
        )
    version = int(binding.linalg_profile_version())
    getter = getattr(binding, "linalg_numerical_profile", None)
    actual = str(getter()) if getter is not None else (PROFILE if version == PROFILE_VERSION else None)
    known = PROFILE if version == PROFILE_VERSION else _EXPERIMENT_PROFILES.get(version)
    expected = os.environ.get(_EXPECTED_PROFILE_ENV, PROFILE)
    if known is None or actual != known or actual != expected:
        raise RuntimeError(
            f"mojolearn.linalg: binary numerical profile {actual!r} "
            f"(version {version}) does not match the supported expected profile "
            f"{expected!r}. The version names the arithmetic; unknown or stale "
            f"profiles cannot be substituted. Experimental arms require their "
            f"exact {_EXPECTED_PROFILE_ENV} value and remain unqualified."
        )
    if version != PROFILE_VERSION and compiled != "identical":
        raise RuntimeError("Experimental neural GEMM profiles require IDENTICAL mode")
    _profile_cache = (actual, version)
    _mode_cache = compiled
    return compiled


def profile():
    """What this process is actually holding, as a dict a caller can print or
    assert on. The profile name is part of the claim, so a result quoted
    anywhere should carry it.

    `identity_claimed` is the field that matters. It is True only when the
    IDENTICAL binary loaded; in FAST mode every other field still describes
    the shape of the computation and NONE of them is a guarantee about bits.
    """
    mode = numeric_mode()
    actual, version = _profile_cache or (PROFILE, PROFILE_VERSION)
    result = {
        "profile": actual,
        "profile_version": version,
        "numeric_mode": mode,
        "identity_claimed": mode == "identical",
        "dtype": "float32",
        "ops": ("OP_NN", "OP_NT", "OP_TN"),
        "contract": "gemm/IDENTICAL_FP32_CONTRACT.md",
        "kernel": "gemm/checks/gemm_identical.mojo",
        "oracle": "gemm/host/gemm_oracle.mojo::gemm_oracle",
        # The measured extent of the claim, not the extent of the API. Both
        # numbers are the lane's own, from gemm/README.md; read it rather than
        # quoting these.
        "certified_shapes": 62,
        "certified_vendors": ("Apple M4", "NVIDIA H100", "AMD MI325X"),
        "certification_source": "gemm/README.md",
        "binary": _load().__file__,
    }
    if version != PROFILE_VERSION:
        # v1's evidence does not establish this different graph. Identity is
        # the intended per-version contract; qualification remains outstanding.
        result.update(
            contract="gemm/contract.mojo",
            identity_claimed=False,
            identity_required=True,
            certified_shapes=0,
            certified_vendors=(),
            certification_source=None,
            qualification="UNVERIFIED_EXPERIMENT",
            experiment_source="experiments/neural_identical_20261006/gemm_cnn.json",
        )
    return result


def require_identical():
    """Raise unless this process loaded the IDENTICAL build.

    Call it once at start-up if your program's correctness depends on the
    profile, so the failure lands at start-up rather than at the first
    matmul. `matmul` calls it for you unless you passed `identical=False`.
    """
    loaded = numeric_mode()
    if loaded != "identical":
        # Names the tier that actually loaded. This said "the FAST build"
        # for every non-identical tier and so mislabeled the deterministic
        # build on the 2026-08-29 Apple stability run.
        raise RuntimeError(
            f"mojolearn.linalg: this process loaded the {loaded.upper()} "
            f"build, which makes NO cross-vendor identity claim; {PROFILE} "
            "is what the IDENTICAL build computes.\n"
            "  Select it with mojolearn.set_numeric_mode('identical') "
            "before the call, or set MOJOLEARN_NUMERIC_MODE=identical "
            "before importing mojolearn; "
            "the identical binary builds with\n      "
            f"MOJOLEARN_NUMERIC_MODE=identical bash {_BUILD_SCRIPT}\n"
            "  Or pass identical=False to say in the source that you want "
            f"the {loaded} product and are making no identity claim about it."
        )


# The op mapping, contract section 0.1, written out once. `transpose_a` and
# `transpose_b` describe the operand ARRAYS the caller passes, so
# `transpose_a=True` means "the array `a` is `k x m` and the left operand is
# its transpose", which is exactly `OP_TN`.
#
#   transpose_a  transpose_b   op       C = ...      a.shape   b.shape
#   False        False         OP_NN    a @ b        (m, k)    (k, n)
#   False        True          OP_NT    a @ b.T      (m, k)    (n, k)
#   True         False         OP_TN    a.T @ b      (k, m)    (k, n)
#   True         True          REFUSED  -- see below
_OPS = {
    (False, False): OP_NN,
    (False, True): OP_NT,
    (True, False): OP_TN,
}


def _operand(x, name):
    """A float32, C-contiguous, 2-D `Array` over `x`, to keep alive across
    the call. DEVIATION 2400: the operand is read through the BUFFER
    PROTOCOL (`_buffer.view`), so a NumPy array, an `array.array('f')`, a
    `mojolearn.Array` or anything else exporting a float32 buffer is an
    operand; there is no NumPy on this path.

    **A non-float32 buffer is REFUSED BY NAME, never cast.** `_buffer.as_f32_c`
    converts float64 silently-but-reported, which is the right trade for an
    estimator whose answer is approximate anyway. It is the wrong trade here:
    this module's entire product is the caller's control over which bits go
    in, and a float64 input downcast on the way through has already lost 29
    mantissa bits before the profile sees it. The caller does the cast, and
    then it is in their source where they can see it. So the FORMAT is
    checked here first and `as_f32_c` is reached only by a float32 buffer,
    where its one remaining job is the layout.

    Non-contiguous IS accepted with a copy, because reordering float32 values
    changes no bit. The contract requires contiguity (section 2) and a
    layout copy supplies it without touching a value; a float32 buffer
    already in C order is borrowed with no copy at all.
    """
    try:
        pb = probe(x)
    except TypeError:
        raise TypeError(
            f"mojolearn.linalg: {name} is a {type(x).__name__}, which does "
            "not support the buffer protocol, and only a float32 buffer is "
            f"in this profile ({PROFILE}; contract section 1 makes FP32 a "
            "hard requirement). Pass a float32 array -- a NumPy array, an "
            "array.array('f') or a mojolearn.Array; convert with "
            f"np.asarray({name}, dtype=np.float32) if that is what you want."
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn.linalg: {name} has dtype {dtype_name(x, pb)}, and "
            f"only float32 is in this profile ({PROFILE}; contract section 1 "
            "makes FP32 a hard requirement, and section 0.5 excludes FP16, "
            "BF16, TF32 and float64). Refused rather than cast, because a "
            "cast from float64 drops mantissa bits you may care about. "
            f"Convert it yourself with np.asarray({name}, dtype=np.float32) "
            "if that is what you want."
        )
    if pb.ndim != 2:
        raise ValueError(
            f"mojolearn.linalg: {name} must be 2-D, got {pb.ndim}-D shape "
            f"{pb.shape}. A vector product is OP_NT at n == 1 (contract 0.1); "
            f"pass it as a 2-D array of shape (n, k) or (k, 1)."
        )
    if nelems(pb.shape) == 0:
        raise ValueError(
            f"mojolearn.linalg: {name} has shape {pb.shape} and no elements. "
            "Contract section 8 does specify the degenerate shapes (k == 0 "
            "writes +0.0 into every cell; m == 0 or n == 0 writes nothing), "
            "but no gate in this tree has run them through the Python "
            "surface, so they are refused here rather than answered "
            "unchecked."
        )
    # A layout copy at most, never a cast (the format was checked above):
    # contiguity is a layout and reordering float32 values moves no bit.
    # Contract section 2 requires it.
    a, _copied = as_f32_c(x, ndim=2, name=name)
    return a


def matmul(a, b, *, transpose_a=False, transpose_b=False, out=None,
           identical=True):
    """`C = op(a) @ op(b)` on the GPU, in float32, under profile
    `mojolearn.identical.gemm.fp32.v1`.

    Parameters
    ----------
    a, b : any float32 buffer
        2-D, dtype float32, read through the buffer protocol: a NumPy
        array, an `array.array('f')`, a `mojolearn.Array`, or any other
        object exporting a float32 buffer (DEVIATION 2400; NumPy is not
        required). Any other dtype is refused by name rather than cast
        (see `_operand`). Non-contiguous input is copied, which moves no
        bit; a C-contiguous float32 buffer is borrowed with no copy.
    transpose_a, transpose_b : bool
        Which of the contract's three operations to run. The flags describe
        the ARRAYS you pass, so `transpose_a=True` means `a` is stored `k x m`
        and the left operand is its transpose.

            transpose_a  transpose_b   op      C          a.shape   b.shape
            False        False         OP_NN   a @ b      (m, k)    (k, n)
            False        True          OP_NT   a @ b.T    (m, k)    (n, k)
            True         False         OP_TN   a.T @ b    (k, m)    (k, n)
            True         True          REFUSED

        **Both true is refused by name.** The contract has three operations
        and `a.T @ b.T` is not one of them (section 0.1); `gemv` is `OP_NT` at
        `n == 1` and is likewise not a fourth. The refusal message names the
        identity that gets you there through an operation the contract DOES
        cover, as your own expression rather than as a transpose this function
        materialized behind your back. DEVIATION 913.
    out : any writable float32 buffer, optional
        Where to write. Float32, C-contiguous, writable, shape `(m, n)`:
        a NumPy array, a `mojolearn.Array`, or any other object exporting
        such a buffer (DEVIATION 2401 widened this from `numpy.ndarray`
        to the buffer protocol; the checks are the same four, read off
        `_buffer.view`). A fresh `mojolearn.Array` is allocated when this
        is None.
    identical : bool, default True
        Whether you are asking for the profile's guarantee. True requires that
        this process loaded the IDENTICAL build and raises if it did not; see
        `require_identical` and the module docstring for why that is the
        default. False says in your source that you want whichever build is
        loaded and are making no identity claim about the result.

    Returns
    -------
    mojolearn.Array
        `(m, n)`, float32, C-contiguous; `numpy.asarray` on it is
        zero-copy through `__array_interface__`. `out` itself -- the very
        object you passed, whatever its type -- when `out` was given.

    What you are getting, stated once
    ---------------------------------
    Under `identical=True` the answer is `gemm_oracle`'s, bit for bit: leaves
    of `contract_leaf_size(k)` accumulated serially ascending with a fused
    multiply-add and a flush-to-zero at every seam, folded by a fixed balanced
    tree with adjacent pairing and a carried odd tail. That is a pure function
    of the input bits, `k` and the profile. It does not depend on `m`, on `n`,
    on the launch geometry, on the block count, on how many rows shared the
    call, or on the vendor.

    It is NOT the most accurate way to sum `k` products, and it is not trying
    to be; contract section 1 is explicit that sameness rather than accuracy is
    what is being bought. It is also not numpy's answer and will not match
    `a @ b` bit for bit.

    Where the measurement stops, honestly
    -------------------------------------
    The certified sweep is 62 shapes, and the three-vendor card is 60 stages
    (`gemm/README.md`). **This function accepts shapes outside that sweep.**
    Outside it, identity rests on the construction -- the per-cell arithmetic
    is a pure function of `k` and the profile, contract sections 6 and 0.3 --
    and on the gates, not on a measurement at your shape. Certification at 62
    shapes is not certification at all shapes.

    NaN cells must be compared as "is NaN" and not by bits (contract 9.1
    declines to promise NaN payloads), and a `min`, `max` or `argmin` you take
    over this output is yours to make order-independent (9.2(e)).
    """
    if identical:
        require_identical()
    else:
        # Still loads and version-checks the binary, so `identical=False` is
        # an opt-out of the GUARANTEE and not of the sanity checks.
        numeric_mode()

    key = (bool(transpose_a), bool(transpose_b))
    if key not in _OPS:
        raise ValueError(
            "mojolearn.linalg.matmul: transpose_a=True with transpose_b=True "
            f"is refused. {PROFILE} has exactly three operations (contract "
            "section 0.1: OP_NN, OP_NT, OP_TN) and a.T @ b.T is not one of "
            "them. This function will not materialize a transpose to fake a "
            "fourth. If you want it, write the identity yourself:\n"
            "    np.ascontiguousarray(np.asarray(matmul(b, a)).T)\n"
            "which is (b @ a).T == a.T @ b.T, runs as OP_NN, and leaves the "
            "extra step visible in your source where you can price it."
        )
    op = _OPS[key]

    a_arr = _operand(a, "a")
    b_arr = _operand(b, "b")

    # THE SHAPES, contract section 0.1. `k` is read off `a` and CHECKED
    # against `b`, because a k mismatch that is not caught here is an
    # out-of-bounds read on the device.
    if transpose_a:
        k, m = a_arr.shape          # OP_TN: a is k x m
    else:
        m, k = a_arr.shape          # OP_NN, OP_NT: a is m x k
    if transpose_b:
        n, kb = b_arr.shape         # OP_NT: b is n x k
    else:
        kb, n = b_arr.shape         # OP_NN, OP_TN: b is k x n
    if k != kb:
        raise ValueError(
            f"mojolearn.linalg.matmul: contracted extents disagree, a gives "
            f"k={k} and b gives k={kb} (a.shape={a_arr.shape}, "
            f"b.shape={b_arr.shape}, transpose_a={bool(transpose_a)}, "
            f"transpose_b={bool(transpose_b)}). The row-major shapes for each "
            "op are in this function's table and in contract section 0.1."
        )

    if out is None:
        out_arr = empty((m, n), "<f4")
    else:
        # DEVIATION 2401: `out` is ANY writable buffer with the right shape
        # and a float32 format, judged through `_buffer.view` -- the four
        # checks are the ones the `isinstance(out, np.ndarray)` spelling
        # made, in the same order and the same words, minus the type name.
        out_arr = out
        try:
            pb = probe(out_arr)
        except TypeError:
            raise TypeError(
                "mojolearn.linalg.matmul: out must be a numpy array, a "
                "mojolearn.Array or any other writable float32 buffer "
                f"(anything supporting the buffer protocol), got "
                f"{type(out_arr).__name__}"
            ) from None
        if not is_native_f32(pb.format):
            raise TypeError(
                "mojolearn.linalg.matmul: out has dtype "
                f"{dtype_name(out_arr, pb)}, and the profile's output is "
                "float32 (contract section 1). Refused rather than cast on "
                "the way out."
            )
        if pb.shape != (m, n):
            raise ValueError(
                f"mojolearn.linalg.matmul: out has shape {pb.shape}, "
                f"want ({m}, {n})"
            )
        if not pb.c_contiguous:
            raise ValueError(
                "mojolearn.linalg.matmul: out must be C-contiguous; the "
                "device writes it directly (contract section 2). A "
                "non-contiguous out cannot be written in place, and copying "
                "into it afterwards would make `out` a lie about where the "
                "result was produced."
            )
        if pb.readonly:
            raise ValueError(
                "mojolearn.linalg.matmul: out is read-only, refusing to "
                "write to it; the device writes `out` directly (contract "
                "section 2) and a read-only buffer handed to it is memory "
                "corruption, not an exception."
            )

    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_linalg.mojo::gemm_binding`):
    #
    #     0  m       rows of C
    #     1  n       columns of C
    #     2  k       the contracted extent
    #     3  op      0 = OP_NN, 1 = OP_NT, 2 = OP_TN
    #
    # A silent reorder here is a WRONG ANSWER and not a crash: swap m and n on
    # a square shape and the call still returns a full matrix of plausible
    # floats. If you change this list, change the comment in the binding in
    # the same edit.
    params = [int(m), int(n), int(k), int(op)]

    binding = _load()
    # `a_arr`, `b_arr` and `out_arr` are held in locals across the call. The
    # Mojo side takes raw addresses, borrows and retains nothing, which is
    # only sound while the owning objects are alive (`_buffer.py`).
    #
    # THE OUTPUT ADDRESS COMES FIRST, mirroring `identical_gemm(ctx, c, a, b,
    # ...)`. Swapping it with `a` writes the device's output over the caller's
    # input matrix, which is memory corruption and not an exception.
    # `addr` (writable) for the output, `addr_ro` for the operands.
    binding.gemm(addr(out_arr, name="out"), addr_ro(a_arr, name="a"),
                 addr_ro(b_arr, name="b"), params)
    return out_arr



# ===========================================================================
# THE LOW-BIT PROFILES (lane/identical-lowbit-inference, 2026-09-17)
# gemm/IDENTICAL_LOWBIT_CONTRACT.md. Every function below refuses a lower
# tier the way `matmul` does, reads back the profile version from the
# binary, and passes addresses plus one `params` list whose order is written
# out in the same words in `bindings/_mojolearn_linalg.mojo`.
# ===========================================================================


def _lowbit_binding():
    b = _load()
    require_identical()
    got = getattr(b, "lowbit_profile_version", None)
    if got is None:
        raise RuntimeError(
            "mojolearn.linalg: the loaded linalg extension predates the "
            "low-bit profiles; rebuild it with bindings/build_linalg.sh"
        )
    v = int(got())
    if v != LOWBIT_PROFILE_VERSION:
        raise RuntimeError(
            f"mojolearn.linalg: the loaded extension implements low-bit "
            f"profile version {v}, this wrapper expects {LOWBIT_PROFILE_VERSION}"
        )
    return b


def _operand_bits(x, name):
    """A 2-D uint16 buffer of bf16 bits, C-contiguous, kept alive."""
    try:
        pb = probe(x)
    except TypeError:
        raise TypeError(
            f"mojolearn.linalg: {name} must be a uint16 buffer of bf16 bits "
            f"(profile {PROFILE_BF16}); got {type(x).__name__}"
        ) from None
    if dtype_name(x, pb) != "uint16":
        raise TypeError(
            f"mojolearn.linalg: {name} has dtype {dtype_name(x, pb)}; the "
            f"bf16 profile takes bf16 BITS in a uint16 buffer, which "
            "to_bf16() produces. Refused rather than reinterpreted."
        )
    if pb.ndim != 2:
        raise ValueError(f"mojolearn.linalg: {name} must be 2-D, got {pb.ndim}-D")
    if nelems(pb.shape) == 0:
        raise ValueError(f"mojolearn.linalg: {name} has no elements")
    a, _ = as_u16_c(x, ndim=2, name=name)
    return a


def to_bf16(x):
    """float32 to bf16 bits (contract L-2: flush, then round to nearest
    even), on the GPU, as a uint16 `mojolearn.Array` of the same shape."""
    a, _ = as_f32_c(x, ndim=None, name="x")
    pb = probe(a)
    if not is_native_f32(pb.format):
        raise TypeError("mojolearn.linalg.to_bf16: x must be float32")
    n = nelems(pb.shape)
    if n == 0:
        raise ValueError("mojolearn.linalg.to_bf16: x has no elements")
    out = empty(pb.shape, "<u2")
    _lowbit_binding().to_bf16(addr(out, name="out"), addr_ro(a, name="x"), [int(n)])
    return out


def from_bf16(bits):
    """bf16 bits to float32 (contract L-1, exact), on the GPU, as a float32
    `mojolearn.Array` of the same shape."""
    a, _ = as_u16_c(bits, ndim=None, name="bits")
    pb = probe(a)
    n = nelems(pb.shape)
    if n == 0:
        raise ValueError("mojolearn.linalg.from_bf16: bits has no elements")
    out = empty(pb.shape, "<f4")
    _lowbit_binding().from_bf16(addr(out, name="out"), addr_ro(a, name="bits"), [int(n)])
    return out


def from_f16(bits):
    """IEEE float16 bits (a `'<u2'` or `'<f2'` buffer) to float32, exact, by
    bit construction (`gemm/contract.mojo::f16_bits_to_f32`): on the GPU on
    a GPU install, the linalg host binding on a CPU-only one. A float32
    `mojolearn.Array` of the same shape."""
    a, _ = as_u16_c(bits, ndim=None, name="bits")
    pb = probe(a)
    n = nelems(pb.shape)
    if n == 0:
        raise ValueError("mojolearn.linalg.from_f16: bits has no elements")
    out = empty(pb.shape, "<f4")
    _lowbit_binding().from_f16(addr(out, name="out"), addr_ro(a, name="bits"), [int(n)])
    return out


def quantize_int8(x):
    """Row-wise int8 codes and per-row power-of-two exponents of a 2-D
    float32 matrix (contract L-3, L-4), on the GPU. Returns `(codes,
    exponents)`: an int8 Array of x's shape and an int32 Array of `rows`."""
    a = _operand(x, "x")
    rows, cols = probe(a).shape
    codes = empty((rows, cols), "<i1")
    exps = empty((rows,), "<i4")
    _lowbit_binding().quantize_int8(addr(codes, name="codes"), addr(exps, name="exponents"),
                                    addr_ro(a, name="x"), [int(rows), int(cols)])
    return codes, exps


def dequantize_int8(codes, exponents):
    """`codes * 2**exponents[row]`, exact, on the GPU, as float32."""
    q, _ = as_i8_c(codes, ndim=2, name="codes")
    e, _ = as_i32_c(exponents, ndim=1, name="exponents")
    rows, cols = probe(q).shape
    if probe(e).shape != (rows,):
        raise ValueError(
            f"mojolearn.linalg.dequantize_int8: exponents has shape "
            f"{probe(e).shape}, want ({rows},)"
        )
    out = empty((rows, cols), "<f4")
    _lowbit_binding().dequantize_int8(addr(out, name="out"), addr_ro(q, name="codes"),
                                      addr_ro(e, name="exponents"), [int(rows), int(cols)])
    return out


def _op_and_shape(a_shape, b_shape, transpose_a, transpose_b, who):
    if transpose_a and transpose_b:
        raise ValueError(
            f"mojolearn.linalg.{who}: transpose_a and transpose_b together "
            "is not one of the contract's three operations (section 0.1); "
            "write (b @ a).T with the flags swapped, in your own source."
        )
    if transpose_a:
        op = OP_TN
        k, m = a_shape
        kb, n = b_shape
    elif transpose_b:
        op = OP_NT
        m, k = a_shape
        n, kb = b_shape
    else:
        op = OP_NN
        m, k = a_shape
        kb, n = b_shape
    if k != kb:
        raise ValueError(
            f"mojolearn.linalg.{who}: contracted extents differ, a gives "
            f"k={k} and b gives k={kb} (a.shape={a_shape}, b.shape={b_shape}, "
            f"transpose_a={transpose_a}, transpose_b={transpose_b})"
        )
    return op, m, n, k


def matmul_bf16(a, b, *, transpose_a=False, transpose_b=False, out=None):
    """`C = op(a) @ op(b)` under `mojolearn.identical.gemm.bf16f32.v1`.

    `b` is bf16 bits (a uint16 buffer, from `to_bf16`); `a` is float32, or
    bf16 bits too. The output is float32: the exactly widened operands
    through the fp32 profile's arithmetic, so a caller who widens the bits
    with `from_bf16` and calls `matmul` gets the same bits. The transpose
    flags and `out` follow `matmul`.
    """
    a_bits = False
    try:
        pa = probe(a)
        a_bits = dtype_name(a, pa) == "uint16"
    except TypeError:
        pass
    a_arr = _operand_bits(a, "a") if a_bits else _operand(a, "a")
    b_arr = _operand_bits(b, "b")
    op, m, n, k = _op_and_shape(probe(a_arr).shape, probe(b_arr).shape,
                                transpose_a, transpose_b, "matmul_bf16")
    out_arr = _out_or_new(out, m, n, "matmul_bf16")
    # `params` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_linalg.mojo::gemm_bf16_binding`):
    #     0 m, 1 n, 2 k, 3 op, 4 a_bf16
    params = [int(m), int(n), int(k), int(op), 1 if a_bits else 0]
    _lowbit_binding().gemm_bf16(addr(out_arr, name="out"), addr_ro(a_arr, name="a"),
                                addr_ro(b_arr, name="b"), params)
    return out_arr


def matmul_int8(a, b, *, out=None):
    """`C = a @ b.T` under `mojolearn.identical.gemm.int8i32.v1` (OP_NT).

    Each of `a` (m x k) and `b` (n x k) is either a float32 matrix, which is
    quantized on the GPU by the profile's own rule, or a `(codes,
    exponents)` pair from `quantize_int8`. The output is float32: the exact
    Int32 sum of the codes, dequantized by one multiply by a power of two.
    It is NOT the float32 product; it is the product of the rounded
    operands, and the profile pins that, not its distance from the
    unrounded one.
    """
    qa, ea = _int8_operand(a, "a")
    qb, eb = _int8_operand(b, "b")
    m, k = probe(qa).shape
    n, kb = probe(qb).shape
    if k != kb:
        raise ValueError(
            f"mojolearn.linalg.matmul_int8: contracted extents differ, a "
            f"gives k={k} and b gives k={kb}"
        )
    out_arr = _out_or_new(out, m, n, "matmul_int8")
    # `params` is `[m, n, k]`, mirrored in `gemm_int8_binding`.
    _lowbit_binding().gemm_int8(addr(out_arr, name="out"), addr_ro(qa, name="a codes"),
                                addr_ro(ea, name="a exponents"), addr_ro(qb, name="b codes"),
                                addr_ro(eb, name="b exponents"), [int(m), int(n), int(k)])
    return out_arr


def _int8_operand(x, name):
    if isinstance(x, tuple) and len(x) == 2:
        q, _ = as_i8_c(x[0], ndim=2, name=name + " codes")
        e, _ = as_i32_c(x[1], ndim=1, name=name + " exponents")
        if probe(e).shape != (probe(q).shape[0],):
            raise ValueError(
                f"mojolearn.linalg.matmul_int8: {name} exponents has shape "
                f"{probe(e).shape}, want ({probe(q).shape[0]},)"
            )
        return q, e
    return quantize_int8(_operand(x, name))


# ===========================================================================
# THE FIFTEEN-BIT PROFILE (lane/lowbit-int15, 2026-09-29)
# gemm/IDENTICAL_LOWBIT_CONTRACT.md section 6. An operand is THREE arrays:
# its two int8 planes `hi` and `lo` (code = hi * 128 + lo, `lo` in
# [0, 127], `hi` in [-128, 127]) and one int32 exponent per row. The planes
# are what the product reads on every vendor, and what a weight is kept as.
# ===========================================================================


def _int15_binding():
    b = _load()
    require_identical()
    got = getattr(b, "int15_profile_version", None)
    if got is None:
        raise RuntimeError(
            "mojolearn.linalg: the loaded linalg extension predates the "
            "fifteen-bit profile; rebuild it with bindings/build_linalg.sh"
        )
    v = int(got())
    if v != INT15_PROFILE_VERSION:
        raise RuntimeError(
            f"mojolearn.linalg: the loaded extension implements fifteen-bit "
            f"profile version {v}, this wrapper expects {INT15_PROFILE_VERSION}"
        )
    return b


def quantize_int15(x):
    """Row-wise fifteen-bit codes of a 2-D float32 matrix, as planes
    (contract W-1 to W-3). Returns `(hi, lo, exponents)`: two int8 Arrays
    of x's shape and an int32 Array of `rows`.

    Per row `e = floor(log2 absmax) - 13`, so the row's largest magnitude
    lands in `[8192, 16384)`; the code is `clamp(rne(x * 2**-e), -16383,
    16383)`; `hi = code >> 7` and `lo = code & 127`, so `code = hi * 128 +
    lo`. The same planes and exponents on every vendor."""
    a = _operand(x, "x")
    rows, cols = probe(a).shape
    hi = empty((rows, cols), "<i1")
    lo = empty((rows, cols), "<i1")
    exps = empty((rows,), "<i4")
    # `addrs` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_linalg.mojo::quantize_int15_binding`):
    #     0 hi, 1 lo, 2 exponents, 3 x
    addrs = [addr(hi, name="hi"), addr(lo, name="lo"), addr(exps, name="exponents"),
             addr_ro(a, name="x")]
    _int15_binding().quantize_int15(addrs, [int(rows), int(cols)])
    return hi, lo, exps


def _int15_planes(x, name, who):
    hi, _ = as_i8_c(x[0], ndim=2, name=name + " hi")
    lo, _ = as_i8_c(x[1], ndim=2, name=name + " lo")
    e, _ = as_i32_c(x[2], ndim=1, name=name + " exponents")
    shape = probe(hi).shape
    if probe(lo).shape != shape:
        raise ValueError(
            f"mojolearn.linalg.{who}: {name} lo has shape {probe(lo).shape}, "
            f"hi has {shape}"
        )
    if probe(e).shape != (shape[0],):
        raise ValueError(
            f"mojolearn.linalg.{who}: {name} exponents has shape "
            f"{probe(e).shape}, want ({shape[0]},)"
        )
    return hi, lo, e


def dequantize_int15(hi, lo, exponents):
    """`(hi * 128 + lo) * 2**exponents[row]`, as float32: the matrix a
    store of fifteen-bit planes stands for."""
    h, l, e = _int15_planes((hi, lo, exponents), "operand", "dequantize_int15")
    rows, cols = probe(h).shape
    out = empty((rows, cols), "<f4")
    # `addrs`: 0 y, 1 hi, 2 lo, 3 exponents (mirrored in the binding).
    addrs = [addr(out, name="out"), addr_ro(h, name="hi"), addr_ro(l, name="lo"),
             addr_ro(e, name="exponents")]
    _int15_binding().dequantize_int15(addrs, [int(rows), int(cols)])
    return out


def _int15_operand(x, name):
    if isinstance(x, tuple) and len(x) == 3:
        return _int15_planes(x, name, "matmul_int15")
    if isinstance(x, tuple):
        raise TypeError(
            f"mojolearn.linalg.matmul_int15: {name} is a tuple of {len(x)}; a "
            "fifteen-bit operand is (hi, lo, exponents) from quantize_int15, "
            "or a float32 matrix"
        )
    return quantize_int15(_operand(x, name))


def matmul_int15(a, b, *, out=None):
    """`C = a @ b.T` under `mojolearn.identical.gemm.int15i64.v1` (OP_NT).

    Each of `a` (m x k) and `b` (n x k) is either a float32 matrix, which is
    quantized by the profile's own rule, or a `(hi, lo, exponents)` triple
    from `quantize_int15`. The output is float32: the exact integer sum of
    the products of the codes, converted to float32 with round to nearest
    even and scaled by one power of two. It is NOT the float32 product; it
    is the product of the rounded operands, and the profile pins that.

    `k` is at most 65536 (`INT15_MAX_K`); a larger `k` is refused by name.
    The same bits on every vendor and from every execution plan: the
    integer matrix unit where the vendor has one, a kernel of integer
    multiplies elsewhere.
    """
    ah, al, ea = _int15_operand(a, "a")
    bh, bl, eb = _int15_operand(b, "b")
    m, k = probe(ah).shape
    n, kb = probe(bh).shape
    if k != kb:
        raise ValueError(
            f"mojolearn.linalg.matmul_int15: contracted extents differ, a "
            f"gives k={k} and b gives k={kb}"
        )
    if k > INT15_MAX_K:
        raise ValueError(
            f"mojolearn.linalg.matmul_int15: k={k} is above {INT15_MAX_K}, the "
            f"largest contracted extent {PROFILE_INT15} accepts (contract W-4)"
        )
    out_arr = _out_or_new(out, m, n, "matmul_int15")
    # `addrs` is, in this exact order (mirrored word for word in
    # `bindings/_mojolearn_linalg.mojo::gemm_int15_binding`):
    #     0 c, 1 a hi, 2 a lo, 3 a exponents, 4 b hi, 5 b lo, 6 b exponents
    # THE OUTPUT ADDRESS COMES FIRST, as in every product of this module.
    addrs = [addr(out_arr, name="out"),
             addr_ro(ah, name="a hi"), addr_ro(al, name="a lo"), addr_ro(ea, name="a exponents"),
             addr_ro(bh, name="b hi"), addr_ro(bl, name="b lo"), addr_ro(eb, name="b exponents")]
    _int15_binding().gemm_int15(addrs, [int(m), int(n), int(k)])
    return out_arr


def _out_or_new(out, m, n, who):
    if out is None:
        return empty((m, n), "<f4")
    pb = probe(out)
    if not is_native_f32(pb.format):
        raise TypeError(f"mojolearn.linalg.{who}: out must be float32")
    if pb.shape != (m, n):
        raise ValueError(f"mojolearn.linalg.{who}: out has shape {pb.shape}, want ({m}, {n})")
    if not pb.c_contiguous or pb.readonly:
        raise ValueError(f"mojolearn.linalg.{who}: out must be C-contiguous and writable")
    return out


# ===========================================================================
# THE THREE DECOMPOSITIONS UNDER THEIR OWN NAMES (lane/linalg-public,
# 2026-09-19)
#
# `PCA`, `TruncatedSVD`, `Nystroem`, `SpectralClustering`, `KernelRidge` and
# the ARIMA least squares have run a Householder QR, a one-sided Jacobi SVD
# and a symmetric Jacobi eigensolver for months, each gated and sabotaged as
# somebody's internal step. A caller who wanted the decomposition itself had
# no way to ask. These three doors are that ask, and nothing more: the
# arithmetic is `decomposition/host/linalg_public.mojo`, which adds none of
# its own on top of the shipping oracles.
#
# THE DEVICE ON A GPU BOX, THE HOST ON A CPU-ONLY ONE, ONE SET OF NAMES.
#
# THIS PARAGRAPH REPLACES A WRONG ONE (2026-09-19). Until this edit these
# three called `_mojolearn_linalg_host` UNCONDITIONALLY, on a GPU box
# included, and the comment here said that was deliberate "because there is
# no one-shot device door for them yet, and inventing one here would be a
# second way to run the same arithmetic". The first half described a missing
# file rather than a decision; the second half was backwards. What it cost
# is exact and was measured the same day: a Metal round recorded the vendor
# `arm64`, because the host binding is what served it. A decomposition that
# only ever runs on the host can never have a GPU column, can never be held
# against another vendor, and so cannot take part in the only claim this
# library makes. THIS IS A GPU-FIRST LIBRARY AND THESE ARE NOW GPU DOORS.
#
# `_door()` picks the binding the way `_cholesky_impl.py`'s does:
#
#     a GPU install   `_mojolearn_linalg`       the device kernels
#                                               (decomposition/
#                                               linalg_public_device.mojo)
#     a CPU-only one  `_mojolearn_linalg_host`  the host oracles
#                                               (decomposition/host/
#                                               linalg_public.mojo)
#
# SAME NAMES, SAME ADDRESS AND PARAM CONTRACTS, SAME ANSWER. The two routes
# run the same arithmetic reached two ways -- the host oracles are the
# serial replay of exactly these kernels -- and the ordering they read is
# ONE function (`eigh_ascending`, `svdvals_descending`), not a copy each.
# That is what makes "the device and the host agree bit for bit" a
# statement a run can check rather than a hope.
#
# `HostQR`-style second names do not exist and should not: a caller who
# wants the host route on a GPU box is asking for the device/host
# comparison, which is a verification job and loads `_mojolearn_linalg_host`
# itself (`_backend.load_host_module`, the verification side); this module
# never loads a host binding (cpu-gpu-cleanup c-linear, 2026-10-02).


def _door():
    """The module that serves `qr_r`, `eigh` and `svdvals` for THIS install:
    `_backend.binding`, which is the device binding where there is a GPU and,
    on a CPU-only install, the host binding `_select_cpu_only` installed
    under the canonical name (`_mojolearn_linalg_host`, host_surface's
    linalg family). Chosen by ROUTE, never probed.

    The device binding is IDENTICAL-only (`_backend._IDENTICAL_ONLY`), so
    `numeric_mode='fast'` on a GPU box is refused here by name rather than
    quietly answered by the host. Before 2026-09-19 it was quietly answered
    by the host, which is the behaviour this door exists to end.
    """
    return _load()


def _two_d(x, name):
    """`_operand`, plus the 2-D shape it promises, as `(arr, rows, cols)`."""
    arr = _operand(x, name)
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(
            f"mojolearn.linalg: {name} must be 2-D, got shape {shape!r}"
        )
    return arr, int(shape[0]), int(shape[1])


def _refuse_wide(rows, cols, who):
    if rows < cols:
        raise ValueError(
            f"mojolearn.linalg.{who}: needs at least as many rows as columns, "
            f"got {rows} x {cols}. The route for a wide matrix is an LQ "
            "factorization of the transpose, which this tree does not carry "
            "(DEVIATION 593). It is REFUSED BY NAME rather than transposed "
            "for you, because the singular values of the transpose are the "
            "same and the VECTORS are not."
        )


def _xd_kit():
    """The decomp lane's cells (x_decomp) in the process's numeric mode:
    IDENTICAL is GPU == CPU bit for bit (the x-decomp lanes), so Q and U are
    the same bytes on every column there. lane/apple-fast-gap-linalg2
    (2026-10-03): this pinned "identical", so a FAST process (the M3, FAST
    bindings only) failed qr/eigh/svd with the identical binding's
    ImportError; the kit now follows `_backend.default_mode()`."""
    from ._expansion_decomp import _Kit
    return _Kit(_backend.default_mode())


# TOMBSTONE: MOJOLEARN_QR_FAST_DEV / MOJOLEARN_SVD_FAST_CHOLQR deleted 2026-10-09 on lane/owed-deletions-D1 took their
# only caller-facing helper, `_fast_apple_kit`; code recoverable at b639a2bd2.


def _xd_matrix(a_arr, rows, cols):
    import array as _array
    from ._expansion_decomp import _M
    st = _array.array("f")
    st.frombytes(a_arr.tobytes())
    return _M(st, rows, cols)


def _triu(M, r):
    """The first r rows of M with everything below the diagonal zeroed (pure
    data movement)."""
    import array as _array
    from ._expansion_decomp import _M
    out = _array.array("f", M.s[:r * M.c])
    # a slice assignment per row (one C-level copy each) instead of a
    # Python store per zeroed cell (lane/neural-net-experiment)
    c = M.c
    for i in range(1, r):  # glue: one C-level zero fill per row (r-sized: matrix rows)
        w = min(i, c)
        out[i * c:i * c + w] = _array.array("f", bytes(4 * w))
    return _M(out, r, c)


class QRResult(tuple):
    """numpy's `QRResult(Q, R)` named tuple: unpacks as (Q, R)."""
    __slots__ = ()

    def __new__(cls, Q, R):
        return tuple.__new__(cls, (Q, R))

    Q = property(lambda self: self[0])
    R = property(lambda self: self[1])


class SVDResult(tuple):
    """numpy's `SVDResult(U, S, Vh)` named tuple: unpacks as (U, S, Vh)."""
    __slots__ = ()

    def __new__(cls, U, S, Vh):
        return tuple.__new__(cls, (U, S, Vh))

    U = property(lambda self: self[0])
    S = property(lambda self: self[1])
    Vh = property(lambda self: self[2])


def _qr_q_factor(a_arr, rows, cols):
    return _xd_kit().geqrf(_xd_matrix(a_arr, rows, cols))


# ===========================================================================
# THE BLOCKED TSQR (lane neural-pass140, 2026-10-02): x_decomp/tsqr_core.mojo
# (the order), x_decomp/tsqr_device.mojo (the kernels), x_decomp/
# tsqr_host.mojo (their host replay, the whole route on a CPU-only install).
#
# A tall matrix (rows >= cols, cols <= _TS_MAX_N) is cut into blocks of 4096
# rows, each block factored by panel-blocked Householder reflectors in its
# own threadgroup, the block R factors combined in a fixed binary tree; Q is
# never formed by itself: `x_decomp_tsqr_q` applies the kept reflectors to a
# small n x k C (the tree top down, then every block), so `qr` asks for
# Q diag(+-1) and `svd` for Q U_R in ONE pass over the rows. No Gram matrix
# anywhere. It replaced (as the default; MOJOLEARN_LINALG_TSQR=0 keeps the
# old device routes for the A/B) geqrf + orgqr (`qr` 'reduced': one dependent
# chain of m multiply-adds per column per step) and `_svd_tall` on the whole
# matrix (a 64-slice QR, then two more for U's orthonormalization): at
# 1,000,000 x 220 on the L40S, 8.4 s and 15.1 s.
#
# SEMANTICS. `qr(mode='reduced')`: R's diagonal is NON-NEGATIVE (each row of R
# whose diagonal came out negative is negated, and Q's column with it: an
# exact sign flip). With a full-rank A that is THE QR factorization, a
# function of A alone; LAPACK's signs (numpy's) come out of its own
# reflector sequence and differ row by row; neither promises more than
# A = Q R with Q orthonormal and R upper triangular. A zero diagonal (a
# dependent column, DEVIATION 588) stays zero. `svd(full_matrices=False)`:
# S and V from the SVD of R (`_svd_tall` on the n x n R: the QR + one-sided
# Jacobi and U_R's orthonormalization), U = Q U_R; the signs are the
# Jacobi's as before.
# ===========================================================================

#: x_decomp/tsqr_core.mojo TS_MAX_N: the widest matrix the TSQR takes
_TS_MAX_N = 512


def _tsqr_on(rows, cols, extra=0):
    """Whether the blocked TSQR serves a rows x (cols + extra) factorization:
    tall and at most _TS_MAX_N wide."""
    n = cols + extra
    return cols >= 1 and n <= _TS_MAX_N and rows >= n


def _tsqr_r(b, a_arr, rows, cols, keep):
    """R (cols x cols, an Array) of the TSQR of a_arr; `keep` holds the
    factorization in the binding for one `_tsqr_q`."""
    R = empty((cols, cols), "<f4")
    # ORDER MATCHES x_decomp/api.mojo tsqr_r_py: (a, b, r_out), params
    # (m, d, nrhs, keep); b is not read when nrhs is 0
    b.x_decomp_tsqr_r(addr_ro(a_arr, name="a"), addr_ro(a_arr, name="a"), addr(R, name="r_out"),
                      [int(rows), int(cols), 0, 1 if keep else 0])
    return R


def _tsqr_q(b, C, rows, cols, k):
    """Q C (rows x k, an Array) for the kept factorization; C is cols x k."""
    out = empty((rows, k), "<f4")
    b.x_decomp_tsqr_q(addr_ro(C, name="c"), addr(out, name="q_out"), [int(rows), int(cols), int(k)])
    return out


def _tsqr_release(b):
    try:
        one = empty((1,), "<f4")
        b.x_decomp_tsqr_q(addr_ro(one, name="c"), addr(one, name="q_out"), [1, 1, 0])
    except Exception:
        pass


def _qr_tsqr(a_arr, rows, cols):
    """`qr(a, 'reduced')` through the blocked TSQR: (Q, R), R's diagonal
    non-negative."""
    b = _xd_kit().b
    R = _tsqr_r(b, a_arr, rows, cols, True)
    try:
        # R's rows signed so its diagonal is non-negative, and C the
        # diagonal of those signs, in Mojo (x_decomp/api.mojo r_signs_py)
        C = empty((cols, cols), "<f4")
        b.x_decomp_r_signs(addr(R, name="r"), addr(C, name="c"), [int(cols)])
    except BaseException:
        _tsqr_release(b)
        raise
    Q = _tsqr_q(b, C, rows, cols, cols)
    return QRResult(Q, R)


def _svd_tsqr(a_arr, rows, cols):
    """`svd(a, full_matrices=False)` of a tall a through the blocked TSQR:
    the SVD of its R, then U = Q U_R."""
    k = _xd_kit()
    b = k.b
    R = _tsqr_r(b, a_arr, rows, cols, True)
    try:
        if k.qfix_flags() & 1:
            # SVD_QFIX (see _SVD_QFIX_NULL_RTOL): every direction above
            # 2^-40 s_0 kept, orthonormalized by Householder QR of the small
            # n x n A V / s (the orth route's two A R^-1 passes lose
            # orthogonality on its ill-conditioned columns)
            Ur, S, Vt = _svd_tall(k, _xd_matrix(R, cols, cols), False,
                                  null_rtol=_SVD_QFIX_NULL_RTOL, householder=True)
        else:
            Ur, S, Vt = _svd_tall(k, _xd_matrix(R, cols, cols), False)
        C = Ur.out()
    except BaseException:
        _tsqr_release(b)
        raise
    U = _tsqr_q(b, C, rows, cols, cols)
    return SVDResult(U, S.out((cols,)), Vt.out())


def _tsvd_cholqr_r(k, x, rows, cols):
    """lane/apple-fast-s-linalg TSVD_FAST_CHOLQR3 (default off, FAST + Apple;
    x_decomp/tsvd_fast.mojo): R of a tall x by shifted CholeskyQR3 on the
    matrix unit, or None (the caller runs the TSQR) when the binding lacks
    `x_decomp_tsvd_cholqr_r` (define off) or its device guard tripped."""
    try:
        fn = getattr(k._raw(), "x_decomp_tsvd_cholqr_r")
    except Exception:
        return None
    R = empty((cols, cols), "<f4")
    ok = fn(addr_ro(x, name="x"), addr(R, name="r_out"), [int(rows), int(cols)])
    return R if int(ok) else None


def _tsvd_tsqr_components(x, nc, mode):
    """lane/apple-fast-q-linalg TSVD_QFIX (x_decomp/qfix.mojo bit 2; FAST
    default, -D MOJOLEARN_TSVD_QOLD restores the Gram route; KEPT for quality
    2026-10-04, rab8-tsvd: istella reconstruction 2.554e-3 -> 1.219e-4 vs
    sklearn 1.22e-4 at 362.9 -> 864.1 ms, speed follow-up in lane
    apple-fast-s-linalg): TruncatedSVD's
    (components (nc, d), singular values (nc,)) of a tall x as the top right
    singular vectors of its TSQR R (the one-sided Jacobi of R, `Kit.svd`),
    each row signed so its largest-|.| entry (first on a tie) is positive
    (DEVIATION 525's rule, the Gram route's). None when the binding does not
    carry the repair or x is not a TSQR shape: the caller runs the Gram.
    #: audit 2026-10-04 tsvd istella relative_reconstruction_error 2.55e-03
    #: (sklearn 1.22e-04): the float32 Gram's rounding, about eps lambda_0 an
    #: entry, swamps every direction under ~1e-5 lambda_0; R carries X's
    #: conditioning, not its square."""
    from ._expansion_decomp import _Kit
    try:
        k = _Kit(mode)
        on = k.qfix_flags() & 2
    except Exception:       # no decomp binding for this tier: the Gram route
        return None
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if not on or not _tsqr_on(rows, cols) or not 1 <= nc <= cols:
        return None
    R = _tsvd_cholqr_r(k, x, rows, cols)
    if R is None:
        R = _tsqr_r(k.b, x, rows, cols, False)
    S, Vt = k.svd(_xd_matrix(R, cols, cols))
    V = Vt.rows(0, nc)
    V = k.ew("mul", V, k.absmax_signs(V, False))
    return V.out(), S.take_cols(list(range(nc))).out((nc,))


def _qr_q(a, mode):
    """numpy.linalg.qr's 'reduced', 'complete' and 'raw' modes: LAPACK geqrf
    (the reflectors kept, dlarfg's signs) and orgqr, through the decomp
    lane's sliced order (x_decomp/qr_sliced.mojo, DEVIATION 5320;
    lane/algos-decomp, 2026-09-27; sliced by lane hr-qr, 2026-10-02). Any shape, wide included."""
    a_arr, rows, cols = _two_d(a, "a")
    # TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
    # grid-fold geqrf / orgqr ahead of the TSQR); code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch; record in docs/TOMBSTONES.md.
    if mode == "reduced" and _tsqr_on(rows, cols):
        return _qr_tsqr(a_arr, rows, cols)
    k = _xd_kit()
    h, tau = k.geqrf(_xd_matrix(a_arr, rows, cols))
    kk = min(rows, cols)
    if mode == "raw":
        # numpy returns geqrf's Fortran-ordered array seen in C order: the
        # factored matrix transposed, (N, M); tau (K,)
        return h.T.out(), tau.out((kk,))
    if mode == "complete":
        return QRResult(k.orgqr(h, tau, rows).out(), _triu(h, rows).out())
    return QRResult(k.orgqr(h, tau, kk).out(), _triu(h, kk).out())


def qr(a, mode="r"):
    """`numpy.linalg.qr(a, mode)`.

    Parameters
    ----------
    a : float32 buffer, shape (M, N)
        C-contiguous or copied into C order; a non-float32 buffer is refused
        by name and never cast (see `_operand`).
    mode : {'reduced', 'complete', 'r', 'raw'}
        The default 'r' preserves MojoLearn's existing R-only return value.
        Request 'reduced' explicitly for NumPy's default (Q, R) form.
        'reduced' returns (Q (M, K), R
        (K, N)), 'complete' (Q (M, M), R (M, N)), 'raw' (h (N, M), tau (K,)),
        K = min(M, N): LAPACK's geqrf with its reflectors kept, dlarfg's sign
        convention, and orgqr for Q, in the decomp lane's cells (IDENTICAL on
        every column; DEVIATION 5320). EXCEPT 'reduced' of a tall matrix with
        N <= 512: the blocked TSQR (`_qr_tsqr`, lane neural-pass140), R's
        diagonal non-negative (see THE BLOCKED TSQR above `_qr_tsqr`);
        MOJOLEARN_LINALG_TSQR=0 keeps geqrf + orgqr there too. 'r' is the TSQR route below (M >= N
        only), which never forms Q; its R agrees with 'reduced''s to float32
        rounding but is not the same bits (a different reduction tree).

    Returns
    -------
    For mode='r': Array, shape (N, N)
        R, row major, upper triangular; ``R.T @ R`` equals ``a.T @ a`` to
        float32. For M >= N it is the TSQR route's R, whose rows may differ
        in SIGN from LAPACK's (the tree's combine steps choose their own
        reflector signs; measured 2026-09-27 on a 48 x 5 input), so it can
        differ from ``qr(a, mode='reduced')[1]`` by a sign per row.
        Its bits predate the Q modes and are kept (the linalg-qr lane). A
        wide input's R is geqrf's, the same as ``qr(a, mode='reduced')[1]``.

    Notes
    -----
    Runs `core/householder_qr.mojo::qr_factor` ON THE DEVICE where there is
    one (`decomposition/linalg_public_device.mojo`), and the host replay of
    that same factorization on a CPU-only install. The two are held to each
    other bit for bit, which is the point of having both.
    """
    if mode in ("reduced", "complete", "raw"):
        return _qr_q(a, mode)
    if mode != "r":
        raise ValueError(
            f"mojolearn.linalg.qr: unrecognized mode {mode!r}; numpy's modes "
            "are 'reduced', 'complete', 'r' and 'raw'"
        )
    a_arr, rows, cols = _two_d(a, "a")
    if rows < cols:
        # a wide R (M x N, upper trapezoidal) is geqrf's; the TSQR route below
        # is tall only
        return _triu(_qr_q_factor(a_arr, rows, cols)[0], rows).out()
    out = empty((cols, cols), "<f4")
    # ORDER MATCHES qr_r_binding IN BOTH BINDINGS, bindings/
    # _mojolearn_linalg.mojo (device) and _mojolearn_linalg_host.mojo:
    # addrs are (a, r_out) and params are (n_rows, n_cols). Swapping the two
    # addresses writes R over the caller's matrix, which is corruption and
    # not an exception.
    _door().qr_r([addr_ro(a_arr, name="a"), addr(out, name="r_out")],
                 [int(rows), int(cols)])
    return out


def eigh(a, UPLO="L"):
    """`numpy.linalg.eigh(a, UPLO)`: eigenvalues ASCENDING and their vectors.

    Parameters
    ----------
    a : float32 buffer, shape (N, N)
        Read as symmetric from ONE triangle, as `numpy.linalg.eigh` reads it:
        UPLO='L' (numpy's default) mirrors the lower triangle, 'U' the upper;
        the other triangle is never read (lane/algos-decomp, 2026-09-27; it
        used to feed the whole matrix to the Jacobi, which agreed with numpy
        only on a symmetric input).

    Returns
    -------
    (w, v) : Array (N,), Array (N, N)
        ``w`` ascending, ``v[:, i]`` the unit eigenvector for ``w[i]`` --
        numpy's layout and numpy's order. `PCA` sorts the same spectrum
        DESCENDING for its own reasons; this door carries numpy's name so it
        carries numpy's order, and `decomposition/host/linalg_public.mojo`
        derives one from the other rather than sorting twice.

        The sign of each vector is pinned by the same `host_sign_flip` `PCA`
        applies, so a vector is a function of the matrix and not of the sweep
        order; without it two boxes agreeing bit for bit could still return
        ``v`` and ``-v``.

    Raises
    ------
    RuntimeError
        If the Jacobi sweep did not converge. An unconverged decomposition is
        not returned as though it were one (DEVIATION 590). On the device
        route a kernel that NEVER RAN is reported as its own failure and not
        as a convergence failure, so a broken build is never read as bad
        data.

    Notes
    -----
    Runs x_decomp's round-robin Jacobi (`eigh_par_*` kernels) and
    `sign_flip_kernel` ON THE DEVICE where there is one, and their host
    replay (x_decomp's host binding, the same rounds) on a CPU-only install.
    """
    a_arr, rows, cols = _two_d(a, "a")
    if rows != cols:
        raise ValueError(
            f"mojolearn.linalg.eigh: a must be square, got {rows} x {cols}"
        )
    if UPLO not in ("L", "U"):
        raise ValueError("mojolearn.linalg.eigh: UPLO argument must be 'L' or 'U'")
    # lane fix-eigh-main (2026-10-02): x_decomp's eigh, the round-robin
    # Jacobi (x_decomp/jacobi_par.mojo, the pinned order of x_decomp/rr.mojo:
    # n/2 independent rotations per round, one thread per updated cell; the
    # host binding replays the same rounds), then `device_eigh`'s own tail
    # (sign_flip_kernel, eigh_ascending). It replaces the linalg binding's
    # `device_eigh`, ONE threadblock for the whole cyclic sweep
    # (grid_dim=(1, 1, 1)), which never finished n = 4096 on the L40S inside
    # the board's 2400 s (0.8.34 board: 1,984 s). `svd(hermitian=True)`
    # already took this route.
    # cgr-decomp (2026-10-03): the triangle is mirrored by the binding, on
    # the device where there is one (`sym_from_triangle_kernel`), not by a
    # host loop over the rows
    w, v = _xd_kit().eigh(_xd_matrix(a_arr, rows, rows), uplo=1 if UPLO == "L" else 2)
    return w.out((rows,)), v.out()


def svdvals(a):
    """`numpy.linalg.svdvals(a)`: the singular values, DESCENDING.

    The route is `PCA(svd_solver='full')`'s without the centering -- the
    Householder QR of ``a``, then the one-sided Jacobi SVD of R. The singular
    values of R ARE those of ``a`` because Q is orthogonal, and that identity
    is why this can exist without forming Q at all. Both kernels run ON THE
    DEVICE where there is one, and as their host replay on a CPU-only
    install.

    Parameters
    ----------
    a : float32 buffer, shape (M, N); a wide one (M < N) is read through its
        transpose, whose singular values are the same

    Returns
    -------
    Array, shape (min(M, N),)
        Descending, numpy's order.

    Notes
    -----
    `svd` returns ``(U, S, Vh)`` (lane/algos-decomp, 2026-09-27); its
    ``compute_uv=False`` is this function.
    """
    a_arr, rows, cols = _two_d(a, "a")
    if rows < cols:
        # the singular values of a wide A are those of A^T, exactly: the
        # transpose is data movement (lane/algos-decomp, 2026-09-27)
        from ._buffer import frombytes
        t = _xd_matrix(a_arr, rows, cols).T
        a_arr, rows, cols = frombytes(t.s.tobytes(), "<f4", (cols, rows)), cols, rows
    out = empty((cols,), "<f4")
    _door().svdvals([addr_ro(a_arr, name="a"), addr(out, name="s_out")],
                    [int(rows), int(cols)])
    return out


#: A singular value at most this fraction of the largest (and exactly zero
#: ones) has its U column taken from the orthogonal complement instead of
#: A v / s, which would divide rounding noise (numpy's gesdd returns SOME
#: orthonormal basis of that null space too).
_SVD_NULL_RTOL = 2.0 ** -20
_SVD_NULL_RTOL_MODULE = _SVD_NULL_RTOL


def _svd_stage_timer():
    """MOJOLEARN_LINALG_TIMING=1: each stage of `_svd_tall` and its wall on
    stderr (the device waited for, since every stage's result is read)."""
    if os.environ.get("MOJOLEARN_LINALG_TIMING") != "1":
        return lambda name: None
    import sys
    import time
    last = [time.perf_counter()]

    def tick(name):
        now = time.perf_counter()
        print("svd stage %-28s %9.1f ms" % (name, (now - last[0]) * 1e3), file=sys.stderr)
        last[0] = now
    return tick


#: lane/apple-fast-q-linalg SVD_QFIX (x_decomp/qfix.mojo bit 1; REVERTED
#: 2026-10-04 to opt-in -D MOJOLEARN_SVD_QFIX: rab5-svd showed no quality
#: gain and taxi +13.2%, so _SVD_NULL_RTOL and the orth route are the
#: default again; this cut applies only when the bit is set): the
#: null cut for U_R on the TSQR route. Directions with 2^-40 s_0 < s_j were
#: replaced by arbitrary complement columns under the 2^-20 cut, up to 2 s_j
#: of error each in U S V^T. #: audit 2026-10-04 svd istella
#: relative_reconstruction_error_100k_rows 3.84e-05 (numpy 4.10e-08), taxi
#: 1.83e-06 (numpy 4.31e-08); float32 model of the route
#: (~/mojolearn-evidence/q-linalg/sim_svd2.py): 2^-20 2.1e-06, 2^-30 and
#: below 3.2e-07.
_SVD_QFIX_NULL_RTOL = 2.0 ** -40


def _svd_tall(k, A, full, null_rtol=None, householder=False):
    """(U, S, Vt) of a tall A (m >= n) as _M: S and V from the decomp lane's
    QR + one-sided Jacobi (`Kit.svd`, descending, ties to the lower index);
    U from the geqrf + orgqr of the columns A v_j / s_j with s_j > 2^-20 s_0
    (the cells' gemm and division), each Q column signed by its R[j, j],
    the null directions and the m - n more of full_matrices its trailing
    columns. `null_rtol` overrides _SVD_NULL_RTOL and `householder` skips
    the orth route (SVD_QFIX, `_svd_tsqr` only: A is then the n x n R)."""
    from ._expansion_decomp import _M
    m, n = A.r, A.c
    tick = _svd_stage_timer()
    S, Vt = k.svd(A)
    tick("svd (sliced QR + Jacobi of R)")
    s0 = S.s[0] if n else 0.0
    # a local of the module constant's name: SVD_QFIX's override, if any
    _SVD_NULL_RTOL = _SVD_NULL_RTOL_MODULE if null_rtol is None else null_rtol
    # the numerical rank, counted in Mojo (x_decomp/api.mojo rank_above_py)
    r = int(k.b.x_decomp_rank_above(S.s.buffer_info()[0], [n], float(_SVD_NULL_RTOL))) if n else 0
    AV = k.mm(A, Vt, tb=True)                                   # m x n
    tick("A V (gemm)")
    Ug = k.ew("div", AV.take_cols(list(range(r))) if r < n else AV, S.take_cols(list(range(r))) if r < n else S)
    tick("A V / s")
    width = m if full else n
    if r == n and width == n and not householder:
        # lane neural-pass17: with every direction kept and no trailing
        # columns wanted, U is the orthonormalized A V / s: the kit's orth
        # (two sliced-QR passes and a row-parallel A R^-1, DEVIATION 5309),
        # whose folds run over the row slices in parallel, where geqrf +
        # orgqr ran 4 n serial one-thread chains over the m rows (at
        # 1,000,000 x 220, 880 chains of a million dependent steps: most
        # of the bench board's svd race on every GPU). Column j is signed
        # by the product of the passes' R diagonals, which is Q_j . Ug_j:
        # the same orientation geqrf's R[j, j] gave (Ug_j's). A guarded
        # (zero) product means a dependent column the orth cannot build;
        # the Householder route then stands.
        Qo, d = k.orth_diag(Ug)
        tick("U = orth(A V / s)")
        # every product nonzero (a NaN counts as nonzero, as `!=` did), then
        # each column signed by its product's sign, on the kit
        if not k.total(k.ew("le", k.ew("abs", d), _M.zeros(1, 1))).s[0]:   # one count word down
            out = k.ew("mul", Qo, k.neg_signs(d))
            tick("U column signs")
            return out, S, Vt
    import array as _array
    if not r:
        # the m x width identity: a zero store and one strided C-level fill of its diagonal
        eye = _array.array("f", bytes(4 * m * width))
        nd = min(m, width)
        eye[0:nd * (width + 1):width + 1] = _array.array("f", [1.0]) * nd
        return _M(eye, m, width), S, Vt
    # A v / s loses orthogonality as s_0 / s_j grows (its error is V's
    # rounding times that ratio); the Householder QR of those columns
    # restores it: Q's column j signed by R[j, j] (an exact negation) is
    # A v_j / s_j to that same error, orthonormal to float32, and Q's
    # trailing columns are the complement's basis.
    h, tau = k.geqrf(Ug)
    tick("geqrf(A V / s)")
    Qc = k.orgqr(h, tau, width)
    tick("orgqr")
    # R[j, j]'s sign per kept column (+1 for the trailing complement), on the kit
    sg = k.neg_signs(k.strided(h, r, r + 1, 0))
    if width > r:
        sg = k.hstack([sg, k.const(1.0, 1, width - r)])
    # lane/neural-net-experiment (2026-09-30): the sign flip of Q's columns
    # was a Python loop over every value of each flipped column (m Python
    # negations per column: at 1,000,000 rows and 220 columns, most of the
    # bench board's 95.8 s svd on an MI325X). The flip is the kit's
    # elementwise multiply by -1.0 / +1.0 per column: an exact sign flip of
    # values a device kernel already stored (so already flushed), the same
    # bits the loop produced (the loop and its env switch are gone).
    return k.ew("mul", Qc, sg), S, Vt


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """`numpy.linalg.svd(a, full_matrices, compute_uv, hermitian)`
    (lane/algos-decomp, 2026-09-27): ``SVDResult(U, S, Vh)``, S descending.

    The decomp lane's cells, IDENTICAL on every column: the Householder QR
    and the one-sided Jacobi SVD of R give S and V; U's column j is
    ``A v_j / s_j``; a numerically null direction (s_j <= 2^-20 s_0) and the
    extra columns of ``full_matrices=True`` come from the complete Q of the
    Householder QR of the columns already found (geqrf + orgqr, DEVIATION
    5320). A wide A (M < N) is the transposed problem: ``A.T = U' S V'^T``,
    so U = V' and Vh = U'^T. ``hermitian=True`` is numpy's route through
    eigh (the lower triangle): S = |w| descending, U = v, Vh = (sign(w) v)^T.
    ``compute_uv=False`` returns `svdvals` (of ``a.T`` when wide), as numpy's
    svdvals is svd(compute_uv=False). The signs of U's and V's columns are
    the Jacobi's, not LAPACK's: numpy promises none either.
    ``full_matrices=False`` of a tall A with N <= 512 is the blocked TSQR
    (`_svd_tsqr`, lane neural-pass140): the same SVD of A's R, U = Q U_R in
    one pass over the rows (MOJOLEARN_LINALG_TSQR=0 keeps the route above)."""
    a_arr, rows, cols = _two_d(a, "a")
    if not compute_uv:
        return svdvals(a_arr)
    # TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1 (the FAST kit's
    # CholeskyQR2 whole-matrix route ahead of the TSQR); code recoverable at b639a2bd2.
    # Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch; record in docs/TOMBSTONES.md.
    if not hermitian and not full_matrices and _tsqr_on(rows, cols):
        return _svd_tsqr(a_arr, rows, cols)
    k = _xd_kit()
    A = _xd_matrix(a_arr, rows, cols)
    if hermitian:
        if rows != cols:
            raise ValueError("mojolearn.linalg.svd: hermitian=True needs a square matrix")
        w, v = k.eigh(A, uplo=1)
        from ._expansion_decomp import _M
        # numpy: argsort(|w|) (ascending) reversed, so equal magnitudes come
        # higher index first
        # (lane py-runtime round 2: the order and the signs on the kit) the
        # stable ascending order of -|w| read back to front is |w| descending
        # with equal magnitudes higher index first
        rev = _M.of(array.array("f", range(rows - 1, -1, -1)), 1, rows)   # the index table n-1 .. 0
        wr = k.take_cols_m(w, rev, rows)
        o = k.order_small(k.ew("scale", k.ew("abs", wr), s=-1.0))
        order = k.ew("adds", k.ew("scale", o.reshape(1, rows), s=-1.0), s=float(rows - 1))
        S = k.take_cols_m(k.ew("abs", w), order, rows)
        U = k.take_cols_m(v, order, rows)
        Vt = k.ew("mul", U.T, k.neg_signs(k.take_cols_m(w, order, rows)).reshape(rows, 1))
        return SVDResult(U.out(), S.out((rows,)), Vt.out())
    if rows >= cols:
        U, S, Vt = _svd_tall(k, A, bool(full_matrices))
        return SVDResult(U.out(), S.out((cols,)), Vt.out())
    U2, S, Vt2 = _svd_tall(k, A.T, bool(full_matrices))
    return SVDResult(Vt2.T.out(), S.out((rows,)), U2.T.out())
