# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CTR estimation order, one definition for the device and the host
column (lane cpu4-gbdt, 2026-10-04).

WHAT MOVED. `train` built one CTR estimation order per permutation on the
host inside the fit: `ctrs_estimation_permutation(n, p).fill_order()`, a
serial `TRandom` Fisher-Yates over every row, then uploaded it once per
categorical feature per permutation. A serial shuffle has no parallel form
with the same output, so the order is now a keyed bijection computed one
row per GPU thread (`gbdt/ctrs/kernel/ctr_order.mojo`), written once per
fit and kept resident.

THE ORDER. Position `i` holds row `ctr_order_row(i, n, key)`:

  * permutation 0 is the identity, as their `TDataPermutation` makes it
    (`permutation.cpp:14-17`);
  * every other permutation is `border_sample_row(i, n, key)`
    (`gbdt/grid_creator/gls_borders.mojo`): a 4-round Feistel bijection on
    `[0, 2^(2h))` cycle-walked into `[0, n)`, so distinct positions give
    distinct rows and the order is a permutation of the rows;
  * `key` is the permutation's own seed, `TDataPermutation.get_seed()`
    (`1664525 * id + 1013904223 + blockSize`), so permutations differ.

Integer work only: every vendor and the host column compute the same words.
BITS MOVE against the old Fisher-Yates order (a different, equally uniform
permutation per id); old bits do not matter within this version, and the
host column (`gbdt/host/gbdt_oracle_ctr.mojo`) takes
`ctr_estimation_order_host` from this file, so all four columns moved
together. `TDataPermutation.fill_order` and `shuffle` are unchanged (their
other callers and `checks/ctr_permutation_check.mojo` still use them).

GPU-free: this file imports no device API, so the host column can import it.
"""

from gbdt.data.permutation import (
    IDENTITY_PERMUTATION_ID,
    ctrs_estimation_permutation,
)
from gbdt.grid_creator.gls_borders_cells import border_sample_row


def ctr_order_key(permutation_id: Int) -> UInt64:
    """The Feistel key of permutation `permutation_id`: its
    `TDataPermutation` seed (`get_seed`, which does not read the row
    count)."""
    return ctrs_estimation_permutation(1, permutation_id).get_seed()


@always_inline
def ctr_order_row(i: Int, n_rows: Int, key: UInt64, identity: Bool) -> Int:
    """The row at position `i` of the order (`identity` for permutation 0)."""
    if identity:
        return i
    return border_sample_row(i, n_rows, key)


def ctr_estimation_order_host(
    n_rows: Int, permutation_id: Int
) -> List[UInt32]:
    """The host column's copy of the order the device writes: the same
    `ctr_order_row` per position."""
    var key = ctr_order_key(permutation_id)
    var identity = permutation_id == IDENTITY_PERMUTATION_ID
    var out = List[UInt32](capacity=n_rows)
    for i in range(n_rows):
        out.append(UInt32(ctr_order_row(i, n_rows, key, identity)))
    return out^
