# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane cnn-dx-wgrad (2026-10-11, ideas-all-gaps G1/G2): the two conv
backward switches, shared by the device path (x_cnn/device.mojo) and the
host twin (x_cnn/host/ops_host.mojo) so both columns read ONE gate and ONE
leaf rule. Both default OFF; both change bits on NVIDIA, AMD and the host
together (same contract, same fold order).

`MOJOLEARN_IDN_CNN_DX_IMPLICIT` (G1): dx as an implicit transposed
convolution. Each dx cell (n, c, h, w) is one contract over
k = OC * KH * KW (k index `oc * KH*KW + kh * KW + kw`), the A operand
gathered from the rows-layout output gradient g at (oh, ow) =
((h + PH - kh DH) / SH, (w + PW - kw DW) / SW) (zero where not integral or
out of range), the B operand W[oc, c, kh, kw]; leaves
`contract_partition(k)`, the contract's balanced fold. Structural reason:
today dcols = g . W (rows x C*KH*KW, 462 MB at conv2d) is written and
re-read uncoalesced by col2im; the implicit form never materializes it and
is one forward-shaped launch. Expected: conv2d NV 6.3 -> ~4.3 ms, AMD
6.2 -> ~4.0 ms; conv1d 10.1 -> ~7. B arm: the define off (dcols + col2im,
including `MOJOLEARN_NI14_COL2IM`).

`MOJOLEARN_IDN_CNN_WGRAD_LEAF1024` (G2): the dW contract over the
N*OH*OW rows takes leaves of `CNN_WGRAD_LEAF` = 1024 rows (raised only to
keep P <= CONTRACT_MAX_LEAVES = 1024) instead of `contract_partition(rows)`
(leaf ~196 at conv2d, P capped at 1024). Structural reason: the leaf
partial workspace is P * OC * C*KH*KW floats (151 MB at conv2d, written
then re-read by the fold); 1024-row leaves cut P ~5x (~29 MB) and reuse
each staged tile 5x longer. The 1024 is a cost constant (partial traffic
vs leaf-kernel parallelism; at 1024 rows a leaf is 64 staged pages, and
every conv with >= 64k rows keeps >= 64 leaves of parallelism), never a
shape rule. The fold is the contract's balanced tree over the P partials
(the GEMM's fold kernels, unchanged). Expected: dW share ~1.5 -> ~0.6 ms per
conv on NV; conv2d 6.3 -> ~5.3 ms.
"""

from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_cnn.ops import NN14_BOUNDED_IM2COL

#: the implicit-conv route of x_cnn/device.mojo (NI12_IMPLICIT_CONV), spelled
#: here without its GPU imports so the host twin reads the same gate
comptime _CNN_IMPLICIT_ROUTE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not is_defined["MOJOLEARN_NI12_IMPLICIT_CONV_OFF"]()
    and not NN14_BOUNDED_IM2COL
)
comptime IDN_CNN_DX_IMPLICIT = _CNN_IMPLICIT_ROUTE and is_defined["MOJOLEARN_IDN_CNN_DX_IMPLICIT"]()
comptime IDN_CNN_WGRAD_LEAF1024 = _CNN_IMPLICIT_ROUTE and is_defined["MOJOLEARN_IDN_CNN_WGRAD_LEAF1024"]()

#: G2's leaf (rows per dW leaf partial), see the module doc
comptime CNN_WGRAD_LEAF = 1024
#: the GEMM fold kernels' leaf cap (contract 13.4, CONTRACT_MAX_LEAVES)
comptime CNN_WGRAD_MAX_LEAVES = 1024


def cnn_wgrad_partition(rows: Int) -> Tuple[Int, Int]:
    """(leaf, P) of G2's dW contract over `rows`: leaf = max(1024,
    ceil(rows / 1024)), P = ceil(rows / leaf). The device leaf kernel and
    the host twin both call this; nothing else produces G2's partition."""
    if rows <= 0:
        return (CNN_WGRAD_LEAF, 0)
    var leaf = CNN_WGRAD_LEAF
    var need = (rows + CNN_WGRAD_MAX_LEAVES - 1) // CNN_WGRAD_MAX_LEAVES
    if need > leaf:
        leaf = need
    return (leaf, (rows + leaf - 1) // leaf)
