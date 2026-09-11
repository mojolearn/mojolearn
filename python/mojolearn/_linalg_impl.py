# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.linalg`: the bit-identical FP32 matrix product.

Profile `mojolearn.identical.gemm.fp32.v1`. Contract
`gemm/IDENTICAL_FP32_CONTRACT.md`; kernel `gemm/checks/gemm_identical.mojo`;
oracle `gemm/checks/gemm_oracle.mojo::gemm_oracle`.

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

import importlib.machinery
import importlib.util
import os
import sys

from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._bufcheck import dtype_name, is_native_f32, nelems, probe

#: The profile family and the version, kept apart because the VERSION is the
#: part the contract makes load-bearing: "a bit-identity claim with no version
#: on it is a claim about whichever revision the reader happens to be
#: holding." The leaf rule (contract 7.1) and the fold topology (7.2) are what
#: the number is about; changing either creates v2 and does not amend v1.
PROFILE_FAMILY = "mojolearn.identical.gemm.fp32"
PROFILE_VERSION = 1
PROFILE = f"{PROFILE_FAMILY}.v{PROFILE_VERSION}"

#: The three operations of contract section 0.1, and their `op` codes as
#: `gemm/checks/gemm_oracle.mojo` defines them. `gemv` is `OP_NT` at
#: `n == 1` and is NOT a fourth operation.
OP_NN = 0
OP_NT = 1
OP_TN = 2

_MODULE_NAME = "_mojolearn_linalg"
_BUILD_SCRIPT = "bindings/build_linalg.sh"
_binding_cache = None
_mode_cache = None


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

    Also checks the binary's profile version against `PROFILE_VERSION`, so a
    stale extension beside a newer wrapper is an error rather than a
    mislabeled answer.
    """
    global _mode_cache
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
    if version != PROFILE_VERSION:
        raise RuntimeError(
            f"mojolearn.linalg: the loaded binary implements profile "
            f"{PROFILE_FAMILY}.v{version} but this wrapper describes "
            f"{PROFILE}. The version names the ARITHMETIC (contract 7.1 and "
            f"7.2), so this is two different answers, not a packaging "
            f"detail. Rebuild with {_BUILD_SCRIPT}."
        )
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
    return {
        "profile": PROFILE,
        "profile_version": PROFILE_VERSION,
        "numeric_mode": mode,
        "identity_claimed": mode == "identical",
        "dtype": "float32",
        "ops": ("OP_NN", "OP_NT", "OP_TN"),
        "contract": "gemm/IDENTICAL_FP32_CONTRACT.md",
        "kernel": "gemm/checks/gemm_identical.mojo",
        "oracle": "gemm/checks/gemm_oracle.mojo::gemm_oracle",
        # The measured extent of the claim, not the extent of the API. Both
        # numbers are the lane's own, from gemm/README.md; read it rather than
        # quoting these.
        "certified_shapes": 62,
        "certified_vendors": ("Apple M4", "NVIDIA H100", "AMD MI325X"),
        "certification_source": "gemm/README.md",
        "binary": _load().__file__,
    }


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
