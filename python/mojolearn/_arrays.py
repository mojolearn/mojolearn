"""Buffer handling at the boundary, and the contract it has to honor.

The Mojo side takes RAW ADDRESSES. It borrows, never owns, and it holds
nothing after a call returns. That is only sound if the Python object owning
the memory is alive for the whole call, so every function here returns the
array alongside its address and every caller keeps that array in a local for
the duration. `_addr` exists so there is one place where an address is taken
and one place to look when that contract is broken.

FLOAT32, NOT FLOAT64, AND THAT IS NOT NEGOTIABLE HERE. Metal has no float64
and neither does the Mojo side; every kernel in this library is float32. A
float64 array is converted, which COPIES. That copy is reported by
`as_f32_c` returning `copied=True` rather than being hidden, because on a
4,000,000 x 32 matrix it is 512 MB the caller did not ask for.
"""

import numpy as np


def _addr(a):
    """The address of an array's first byte.

    `__array_interface__` rather than `ctypes.data` because it also carries
    the read-only flag, and handing a read-only buffer to a function that
    writes it is a segfault rather than an exception.
    """
    iface = a.__array_interface__
    ptr, read_only = iface["data"]
    if read_only:
        raise ValueError(
            "mojolearn: output buffer is read-only, refusing to write to it"
        )
    return ptr


def _addr_ro(a):
    """The address of an array that will only be read."""
    return a.__array_interface__["data"][0]


def as_f32_c(x, name):
    """A C-contiguous float32 view of `x`, and whether that cost a copy.

    Returns `(array, copied)`. The caller MUST keep `array` alive across the
    Mojo call; that is the whole reason this returns the array rather than
    just an address.
    """
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(
            f"mojolearn: {name} must be 2-D, got {a.ndim}-D shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn: {name} is empty, shape {a.shape}")
    copied = False
    if a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
        # Named rather than silent: on a large matrix this is the dominant
        # cost of the call and the caller can avoid it by passing float32
        # C-order in the first place.
        a = np.ascontiguousarray(a, dtype=np.float32)
        copied = True
    return a, copied


def as_f32_colmajor(x, name):
    """A COLUMN-MAJOR float32 flat buffer over `x`, at most one copy.

    Returns `(array, flat, copied)`: `array` is 2-D F-contiguous float32,
    `flat` its flat column-major view (`flat[f * n_rows + r] ==
    array[r, f]`), and the two share ONE buffer -- `flat.base` keeps it
    alive, so holding either in a local satisfies the borrow contract at
    the top of this module.

    DEVIATION 1887 -- ONE HOST MATERIALIZATION OF X, NOT TWO. The fit
    path used to run `as_f32_c` (a copy to C order for anything not
    already float32 C-contiguous) and then `ascontiguousarray(Xa.T)` (a
    second, always-taken copy back to column-major). `asfortranarray`
    converts dtype and layout in one pass straight from the caller's
    buffer, so a C-order or float64 input costs ONE copy -- the floor,
    since the quantizer walks columns -- and a float32 F-contiguous
    input is a ZERO-COPY borrow, which is what `ensemble.py`'s module
    docstring had promised all along ("pass `X` already in Fortran order
    to avoid it") while the old pair still copied it twice. Same element
    values, same flat order: the float64 -> float32 cast is the same
    elementwise cast either way, so no output bit can move.
    """
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(
            f"mojolearn: {name} must be 2-D, got {a.ndim}-D shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn: {name} is empty, shape {a.shape}")
    copied = False
    if a.dtype != np.float32 or not a.flags["F_CONTIGUOUS"]:
        a = np.asfortranarray(a, dtype=np.float32)
        copied = True
    # `a.T` of an F-contiguous array is C-contiguous, so this reshape is
    # a VIEW; nothing after this line copies.
    return a, a.T.reshape(-1), copied
