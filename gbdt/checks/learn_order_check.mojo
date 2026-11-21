# SPDX-License-Identifier: Apache-2.0
"""Native check of the seeded learn orders (`gbdt/ctrs/ctr_order.mojo`):
every order is a permutation of the rows, the inverse walk inverts the
forward one, whole blocks stay whole, the seed and the permutation id move
the order, and the Ordered plan returns the same words. Host only."""
from gbdt.ctrs.ctr_order import (
    learn_load_key,
    learn_permutation_key,
    border_sample_position,
    short_block_slot,
    block_order_row,
    learn_order_host,
    ctr_estimation_order_host,
)
from gbdt.grid_creator.gls_borders_cells import border_sample_row
from gbdt.data.ordered_plan import ordered_permutations


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("learn order check: " + what)


def _is_permutation(order: List[UInt32], n: Int, what: String) raises:
    _need(len(order) == n, what + ": length")
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var row = Int(order[i])
        _need(row >= 0 and row < n, what + ": row in range")
        _need(not seen[row], what + ": a row appears twice")
        seen[row] = True


def inverse() raises:
    var sizes: List[Int] = [1, 2, 3, 17, 64, 1000, 4097]
    for s in range(len(sizes)):
        var n = sizes[s]
        var key = learn_permutation_key(UInt64(7), s + 1, 1)
        for i in range(n):
            var row = border_sample_row(i, n, key)
            _need(border_sample_position(row, n, key) == i, "the inverse walk inverts the forward walk")


def blocks() raises:
    # 1003 rows in blocks of 64: 15 whole blocks and a short one of 43
    var n = 1003
    var bs = 64
    var key = learn_permutation_key(UInt64(11), 2, bs)
    var slot = short_block_slot(n, bs, key)
    _need(slot >= 0 and slot < 16, "the short block has a slot")
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var row = block_order_row(i, n, bs, key, slot)
        _need(row >= 0 and row < n and not seen[row], "the block order is a bijection")
        seen[row] = True
        if i > 0:
            var before = block_order_row(i - 1, n, bs, key, slot)
            # inside a block consecutive positions hold consecutive rows
            if row % bs != 0:
                _need(row == before + 1, "a block stays whole and in order")
    _need(short_block_slot(1024, bs, key) == -1, "no short block when the rows divide")
    _need(short_block_slot(n, 1, key) == -1, "no short block at block size 1")


def orders() raises:
    var n = 5000
    var zero = ctr_estimation_order_host(n, 0, UInt64(0))
    var three = ctr_estimation_order_host(n, 3, UInt64(0))
    var other_seed = ctr_estimation_order_host(n, 0, UInt64(1))
    _is_permutation(zero, n, "permutation 0")
    _is_permutation(three, n, "permutation 3")
    _is_permutation(other_seed, n, "seed 1")
    var fixed = 0
    var same_three = 0
    var same_seed = 0
    var load_key = learn_load_key(UInt64(0))
    for i in range(n):
        if Int(zero[i]) == i:
            fixed += 1
        if zero[i] == three[i]:
            same_three += 1
        if zero[i] == other_seed[i]:
            same_seed += 1
        _need(Int(zero[i]) == border_sample_row(i, n, load_key), "permutation 0 is the load shuffle")
    # a uniform permutation of 5000 rows has about one fixed point
    _need(fixed < 50, "permutation 0 is not the caller's row order")
    _need(same_three < 50, "permutation 3 is not permutation 0")
    _need(same_seed < 50, "the seed moves the order")
    var again = ctr_estimation_order_host(n, 3, UInt64(0))
    for i in range(n):
        _need(again[i] == three[i], "the order is a function of (seed, permutation, position)")
    # the Ordered plan: the same function, with blocks at 50,000 rows and up
    var plan = ordered_permutations(n, 3, 1, UInt64(5))
    _need(len(plan) == 3, "one order per permutation")
    for p in range(3):
        var direct = learn_order_host(n, p, 1, UInt64(5))
        _is_permutation(plan[p], n, "ordered permutation")
        for i in range(n):
            _need(plan[p][i] == direct[i], "the Ordered plan is learn_order_host")
    var big = 60037
    var blocked = ordered_permutations(big, 2, 64, UInt64(9))
    _is_permutation(blocked[0], big, "blocked permutation 0")
    _is_permutation(blocked[1], big, "blocked permutation 1")


def run_checks() raises:
    inverse()
    blocks()
    orders()
