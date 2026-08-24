"""CPython boundary for `mojolearn.identical.gemm.fp32.v1`, the bit-identical
FP32 matrix product.

DEVIATION 910 (the host-pointer entry this calls) and DEVIATION 912 (the
mode read-back below). Written 2026-08-24.

WHAT THIS EXTENSION IS FOR, AND WHY IT IS ITS OWN EXTENSION
------------------------------------------------------------
Everything else in this tree that Python can reach is an ESTIMATOR. This is
a numerical primitive, and its audience is anyone who needs a reproducible
matrix product, including people who will never fit a model. It gets its own
extension for the reason `bindings/build.sh` states for all of them: an
independently changing binding must not become a merge point. It also gets
its own for a second reason particular to this one -- it is the only module
in the package whose VALUE is a bit-level claim, so it must be possible to
build, gate and ship it without touching anything that is not the claim.

Arrays cross as BORROWED NumPy addresses. Every device buffer and every
`DeviceContext` lives for exactly one call and no pointer is retained past
the return; `python/mojolearn/_linalg_impl.py` keeps each array alive across
the call, which is the whole point of `_arrays.py` returning the array
beside its address.

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working somewhere around nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent reorder
here is a WRONG ANSWER, not a crash -- swap `m` and `n` on a square shape
and every call still returns a full matrix of plausible floats -- so the two
comments are the only thing standing between a caller and a quiet lie. If
you change one, change the other in the same edit.

THE GIL IS RELEASED AROUND THE DEVICE WORK, and nothing inside a
`GILReleased` block touches a `PythonObject`.

THE MODE READ-BACK IS THE POINT OF `linalg_numeric_mode` (DEVIATION 912)
------------------------------------------------------------------------
`mojolearn.identical.gemm.fp32.v1` is what the IDENTICAL build computes.
The FAST build runs the SAME kernels on the SAME path with the two pins
compiled away (`gemm/mojo_only/gemm_identical.mojo`'s header, "WHAT
`NUMERIC_FAST` DOES HERE"), which is a correct GEMM that makes no identity
claim at all. Nothing in the ANSWER distinguishes the two on Apple at these
seams -- contract section 4.1 measured Metal fused in both modes -- so a
caller cannot tell by looking at the numbers which one it got.

`linalg_numeric_mode` is therefore the only honest way for the Python side
to know what it is holding, and it is a COMPILE-TIME answer from the binary
that actually loaded rather than a restatement of an environment variable.
`python/mojolearn/_linalg_impl.py` refuses to deliver the profile's
guarantee unless this returns 1. This mirrors `gbdt_numeric_mode` in
`bindings/_mojolearn_gbdt.mojo`, which exists for the same reason and whose
wrapper reads it once.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from gemm.host_entry import identical_gemm_host
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def linalg_numeric_mode_binding() raises -> PythonObject:
    """1 when this binding was built under NUMERIC_IDENTICAL, else 0.

    Compile-time, from `is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]` through
    `GLOBAL_NUMERIC_MODE`. The wrapper reads it once per process and gates
    the identity guarantee on it. A `.so` that lands in the wrong directory
    is caught here and nowhere else."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return PythonObject(1)
    return PythonObject(0)


def linalg_profile_version_binding() raises -> PythonObject:
    """The MAJOR VERSION of the GEMM profile this binary implements: 1, for
    `mojolearn.identical.gemm.fp32.v1`.

    An Int and not the profile string, because the profile name has no Mojo
    constant anywhere in this tree today -- it lives in
    `gemm/IDENTICAL_FP32_CONTRACT.md` and in docstrings -- so a string
    returned here would be a literal typed twice rather than a fact read
    once, and Int is a conversion the other bindings already prove.

    **The version is part of the claim**, contract preamble: *"a bit-identity
    claim with no version on it is a claim about whichever revision the
    reader happens to be holding."* The leaf rule of section 7.1 and the fold
    topology of section 7.2 are what this number is about. Changing either
    creates v2 and this returns 2; it does not amend v1. The wrapper
    cross-checks it against its own constant, so a stale `.so` beside a newer
    wrapper is a loud error rather than a mislabeled answer."""
    return PythonObject(1)


def gemm_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`C = op(A) . op(B)` under `mojolearn.identical.gemm.fp32.v1`.
    Returns the number of output cells written, `m * n`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_linalg_impl.py`):

        0  m       rows of C
        1  n       columns of C
        2  k       the contracted extent
        3  op      0 = OP_NN, 1 = OP_NT, 2 = OP_TN

    and the row-major shapes those name are contract section 0.1:

        OP_NN   C = A . B      A is m x k,  B is k x n
        OP_NT   C = A . B^T    A is m x k,  B is n x k
        OP_TN   C = A^T . B    A is k x m,  B is k x n

    **THE OUTPUT ADDRESS COMES FIRST**, which is the opposite of
    `_mojolearn_estimators.mojo`'s inputs-then-outputs habit and deliberate:
    it mirrors `identical_gemm(ctx, c, a, b, ...)`, the certified function
    this ultimately calls, so the two argument lists read the same way down
    the page. Swapping `c_addr` with `a_addr` writes the device's output over
    the caller's input matrix, which is memory corruption in the caller's
    process and not an exception here.

    `c_addr` is written, `a_addr` and `b_addr` are read. All three are
    float32, row-major and fully contiguous (contract section 2); no leading
    dimension, no stride, no sub-view. `gemm/host_entry.mojo` refuses a
    non-positive extent and an unknown `op` by name."""
    if len(params) != 4:
        raise Error(
            "gemm: params must contain 4 values (m, n, k, op), got "
            + String(len(params))
        )
    var cp = _f32_ptr(Int(py=c_addr))
    var ap = _f32_ptr(Int(py=a_addr))
    var bp = _f32_ptr(Int(py=b_addr))
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        identical_gemm_host(ctx, cp, ap, bp, m, n, k, op)
    return PythonObject(m * n)


@export
def PyInit__mojolearn_linalg() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_linalg")
        m.def_function[gemm_binding]("gemm")
        m.def_function[linalg_numeric_mode_binding]("linalg_numeric_mode")
        m.def_function[linalg_profile_version_binding](
            "linalg_profile_version"
        )
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_linalg: ", e))
