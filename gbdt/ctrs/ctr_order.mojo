# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learn orders of a fit, one definition for the device and the host
column: the CTR estimation order of every permutation and the learn
permutations of Ordered boosting (lane cpu4-gbdt, 2026-10-04; seeded from
`random_state` by lane cat-ctr, 2026-11-21,
docs/plans/FUTURE_WORK_2026-11-21.md section 1, step 2).

WHAT THE REFERENCE DOES. It shuffles the learn pool once at load
(`ShuffleLearnDataIfNeeded`, `private/libs/algo/preprocess.cpp:161-199`),
seeded from `random_seed`, and then draws its permutations over the
SHUFFLED pool: permutation 0 is the identity of that pool
(`permutation.cpp:14-17`), every other one a shuffle of whole blocks
(`cuda/data/data_utils.h:21-47`).

WHAT THIS FILE DOES. The same composition, as a function of the position:

    order_p[i] = load_row(perm_p_row(i))

  * `load_row(j) = border_sample_row(j, n, load_key)`: a 4-round Feistel
    bijection on `[0, 2^(2h))` cycle-walked into `[0, n)`
    (`gbdt/grid_creator/gls_borders_cells.mojo`), keyed by `random_state`.
    It is the load shuffle: the row at position `j` of the shuffled pool;
  * `perm_0_row(i) = i`; for `p > 0`, `perm_p_row` is the same bijection
    over the BLOCKS of the pool, keyed by `(random_state, p, block size)`,
    the blocks laid out in their shuffled order with a short last block
    wherever it lands (`block_order_row`). With a block size of 1 it is the
    bijection over the rows.

A COUNTER-BASED GENERATOR, NOT A SEQUENTIAL SHUFFLE: every position is a
function of `(random_state, permutation, position)` alone, so the device
writes one position per thread, the host column writes its positions across
all workers, and both write the same words. Integer work only: every vendor
and the host column agree.

THE CALLER NO LONGER SHUFFLES. Until this lane the CTR order of permutation
0 was the identity over the rows as passed, so rows sorted by target leaked
into the statistic and the caller had to shuffle and record a seed. Every
order is now drawn from `random_state`, which is a constructor parameter:
a receipt (`estimator.receipt`) records it with the other parameters.

BITS MOVE on every column together for every fit that builds a CTR and for
every Ordered fit (the orders are different, equally uniform permutations).
`TDataPermutation.fill_order` and `shuffle` (`gbdt/data/permutation.mojo`)
are unchanged; the fit no longer calls them for these orders.

GPU-free: this file imports no device API, so the host column imports it.
"""

from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from gbdt.grid_creator.gls_borders_cells import _mix64, border_sample_row

#: the load shuffle's key salt ("LOADSHUF"): the pool order of a fit
comptime LEARN_LOAD_SALT = UInt64(0x4C4F414453485546)
#: the per-permutation key salt ("PERMUTE!")
comptime LEARN_PERMUTATION_SALT = UInt64(0x5045524D55544521)
comptime _GOLDEN = UInt64(0x9E3779B97F4A7C15)


def learn_load_key(random_seed: UInt64) -> UInt64:
    """The key of the load shuffle of a fit seeded `random_seed`."""
    return _mix64(random_seed ^ LEARN_LOAD_SALT)


def learn_permutation_key(random_seed: UInt64, permutation_id: Int, block_size: Int) -> UInt64:
    """The key of permutation `permutation_id` (> 0) over blocks of
    `block_size` rows: distinct permutations and block sizes draw distinct
    orders from one `random_seed`."""
    return _mix64(
        _mix64(random_seed ^ LEARN_PERMUTATION_SALT)
        + UInt64(permutation_id) * _GOLDEN
        + UInt64(block_size)
    )


def border_sample_position(row: Int, n: Int, key: UInt64) -> Int:
    """The inverse of `border_sample_row`: the position `i` with
    `border_sample_row(i, n, key) == row`. The Feistel rounds run backwards
    and the cycle walk is the same walk in the other direction."""
    var h = 1
    while (1 << (2 * h)) < n:
        h += 1
    var mask = (UInt64(1) << UInt64(h)) - 1
    var x = UInt64(row)
    while True:
        var l = x >> UInt64(h)
        var r = x & mask
        for back in range(4):
            var rnd = 3 - back
            # forward round: (l, r) -> (r, l ^ f(r)); so r_old = l, l_old = r ^ f(l)
            var f = _mix64(l ^ (key + UInt64(rnd) * _GOLDEN)) & mask
            var t = r ^ f
            r = l
            l = t
        x = (l << UInt64(h)) | r
        if x < UInt64(n):
            return Int(x)


def short_block_slot(n_rows: Int, block_size: Int, key: UInt64) -> Int:
    """Where the short last block of the pool lands among the shuffled
    blocks, or -1 when every block is whole (or the block size is 1). Once
    per permutation, on the host; `block_order_row` takes it as a word."""
    if block_size <= 1 or n_rows % block_size == 0:
        return -1
    var n_blocks = (n_rows + block_size - 1) // block_size
    return border_sample_position(n_blocks - 1, n_blocks, key)


@always_inline
def block_order_row(i: Int, n_rows: Int, block_size: Int, key: UInt64, short_slot: Int) -> Int:
    """Position `i` of a keyed shuffle of whole blocks: slot `q` holds block
    `border_sample_row(q, n_blocks, key)`, the blocks are laid out in slot
    order, and the short last block (at `short_slot`, -1 for none) takes
    only its own rows. A bijection on `[0, n_rows)`."""
    if block_size <= 1:
        return border_sample_row(i, n_rows, key)
    var n_blocks = (n_rows + block_size - 1) // block_size
    var slot: Int
    var offset: Int
    if short_slot < 0 or i < short_slot * block_size:
        slot = i // block_size
        offset = i - slot * block_size
    else:
        var short_rows = n_rows - (n_blocks - 1) * block_size
        if i < short_slot * block_size + short_rows:
            # inside the short block: it is block `n_blocks - 1` by construction
            return (n_blocks - 1) * block_size + (i - short_slot * block_size)
        var past = i - short_rows
        slot = past // block_size + 1
        offset = past - (slot - 1) * block_size
    return border_sample_row(slot, n_blocks, key) * block_size + offset


@always_inline
def learn_order_row(
    i: Int, n_rows: Int, block_size: Int, permuted: Bool, key: UInt64, short_slot: Int,
    load_key: UInt64,
) -> Int:
    """The original row at position `i` of a learn order (module
    docstring): `permuted` is False for permutation 0."""
    var j = i
    if permuted:
        j = block_order_row(i, n_rows, block_size, key, short_slot)
    return border_sample_row(j, n_rows, load_key)


def learn_order_host(
    n_rows: Int, permutation_id: Int, block_size: Int, random_seed: UInt64
) -> List[UInt32]:
    """Permutation `permutation_id`'s learn order as a host list
    (`order[position]` = the original row): the same `learn_order_row` per
    position as the device kernel, positions split across the workers."""
    if n_rows <= 0:
        return List[UInt32]()
    var load_key = learn_load_key(random_seed)
    var key = learn_permutation_key(random_seed, permutation_id, block_size)
    var permuted = permutation_id != 0
    var short_slot = -1
    if permuted:
        short_slot = short_block_slot(n_rows, block_size, key)
    var order = List[UInt32](unsafe_uninit_length=n_rows)
    var op = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(order.unsafe_ptr()))
    # a position costs two cycle walks of four mixing rounds each
    var units = max(1, host_predict_task_count_work(n_rows, n_rows * 32))
    var chunk = host_predict_chunk(n_rows, units)

    def fill(unit: Int) {imm chunk, imm n_rows, imm block_size, imm permuted, imm key, imm short_slot, imm load_key, imm op}:
        for i in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            op[i] = UInt32(learn_order_row(i, n_rows, block_size, permuted, key, short_slot, load_key))

    host_parallelize(fill, units)
    return order^


def ctr_estimation_order_host(
    n_rows: Int, permutation_id: Int, random_seed: UInt64
) -> List[UInt32]:
    """The host column's copy of the CTR estimation order the device writes
    (`gbdt/ctrs/kernel/ctr_order.mojo`): block size 1."""
    return learn_order_host(n_rows, permutation_id, 1, random_seed)
