# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`maximize=True` on SGD, Adam and AdamW: the gradient's SIGN FLIP.

REFERENCE. torch/optim/sgd.py `_single_tensor_sgd`: `grad = grads[i] if not
maximize else -grads[i]`, the first statement of the per-tensor body, BEFORE
the coupled weight decay and the momentum buffer. torch/optim/adam.py
`_single_tensor_adam`: `grad = grads[i] if not maximize else -grads[i]`, again
first, before the coupled decay (Adam) and the moments; AdamW's decoupled
decay multiplies the PARAMETER and never reads the sign. So a maximizing step
IS the minimizing step on `-g`, and that is how the two bindings run it: the
step reads the negated gradient and nothing else changes.

THE SEAM (DEVIATION 6200, IDENTITY_PATHS row 200). Negation is exact on every
IEEE machine, and there are two legal spellings of it that are NOT the same
bits: `-g` (the sign bit flipped; torch.neg, what the reference writes) and
`0.0 - g` (a subtraction). They differ at exactly one input, `g = +0.0`:
`-(+0.0)` is `-0.0` and `0.0 - 0.0` is `+0.0`. The sign of that zero reaches
the answer: SGD's first step COPIES the gradient into the momentum buffer, so
the buffer holds it; and `identical_mul_add(-lr, g, p)` at `p = -0.0` gives
`+0.0` for `g = -0.0` and `-0.0` for `g = +0.0`. The PIN: the sign-bit XOR
below, on every column. It also flips a NaN's sign bit exactly as torch.neg
does, although a NaN gradient is refused before any step reads it.

THE CLIP. With `max_norm` on, the binding clips the NEGATED gradient and
writes it back negated again. Norms read squares, so `||-g|| == ||g||` bit for
bit, and round-to-nearest is sign-symmetric, so `(-g) * c == -(g * c)`: the
caller sees the same clipped gradient as with `maximize=False`.

The oracle is the definition itself: a `maximize=True` step on `g` equals, bit
for bit, a `maximize=False` step on `-g` negated by the sign bit
(python/mojolearn/tests/test_optim_maximize_seam.py, which also refuses as
VACUOUS a fixture on which the two spellings agree).
"""

from std.memory import bitcast

comptime _SIGN_BIT = UInt32(0x80000000)


@always_inline
def maximize_negate(x: Float32) -> Float32:
    """torch's `-grad`: the sign bit flipped (DEVIATION 6200). Exact."""
    return bitcast[DType.float32](bitcast[DType.uint32](x) ^ _SIGN_BIT)


def maximize_negated_copy(
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int
) -> List[Float32]:
    """`n` floats of `src`, each negated by `maximize_negate`."""
    var out = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        out[i] = maximize_negate(src[i])
    return out^
