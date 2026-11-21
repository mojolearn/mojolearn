# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HOST half of Ordered boosting's plan, shared by the device fit
(`gbdt/methods/ordered_boosting.mojo`) and the CPU host oracle
(`gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit`): the
permutation block size, the learn permutations and the score-noise
multiplier. Host control-plane code, like the border search; nothing here
imports a device module, so both sides call the same function rather than
two spellings of it (lane/catboost-parity, 2026-09-19). The account of what
each restates is in `ordered_boosting.mojo`'s docstring.
"""

from checks.numerics import portable_exp64, portable_log64
from gbdt.ctrs.ctr_order import learn_order_host

# TOMBSTONE-NOTE (lane cat-ctr, 2026-11-21): ORDERED_LOAD_SHUFFLE_SALT moved to
# gbdt/ctrs/ctr_order.mojo as LEARN_LOAD_SALT (same value) with the load
# shuffle itself; nothing else read it.
#: the fit's own host stream ("ORDERED!"): one learn-permutation draw (when
#: there is a choice) and one tree noise seed per iteration
comptime ORDERED_STREAM_SALT = UInt64(0x4F52444552454421)
#: the bootstrap's per-thread seeds ("ORDBOOTS"), apart from the plain fit's
comptime ORDERED_BOOTSTRAP_SALT = UInt64(0x4F5244424F4F5453)
#: `MinFoldSize("min_fold_size", 100)` (`boosting_options.cpp:24`)
comptime ORDERED_MIN_FOLD_SIZE = 100


def _int_log2_ceil(v: Int) -> Int:
    """`NCB::IntLog2` (`libs/helpers/math_utils.h:14-16`): the ceiling."""
    var bit = 0
    while (1 << bit) < v:
        bit += 1
    return bit


def ordered_permutation_block_size(n_rows: Int, suggested: Int) -> Int:
    """`GetPermutationBlockSize` (`dynamic_boosting.h:115-128`) after
    `UpdateGpuSpecificDefaults` (`cuda/train_lib/train.cpp:115-118`) set an
    unset or zero `fold_permutation_block` to 64."""
    if n_rows < 50000:
        return 1
    var block = suggested if suggested > 0 else 64
    if block > 1:
        block = 1 << _int_log2_ceil(block)
        while block * 128 > n_rows:
            block >>= 1
    return block


def ordered_permutations(
    n_rows: Int, permutation_count: Int, block_size: Int, random_seed: UInt64
) raises -> List[List[UInt32]]:
    """The `permutation_count` learn orders of their Ordered fit, each as
    `order[position] = original row`: the load shuffle
    (`ShuffleLearnDataIfNeeded`) composed with the permutation's shuffle of
    whole blocks (permutation 0 the identity of the SHUFFLED pool).

    Lane cat-ctr, 2026-11-21: each order is `learn_order_host`
    (`gbdt/ctrs/ctr_order.mojo`), a counter-based function of
    `(random_seed, permutation, position)` written across all workers, in
    place of a serial Fisher-Yates load shuffle composed with a serial
    `TDataPermutation.fill_order` per permutation. The device fit and the
    host column both call this function, so they read the same words; bits
    move for every Ordered fit, on every column together."""
    var orders = List[List[UInt32]]()
    for p in range(permutation_count):  # small-loop(permutation_count: permutations): each order fills across all workers
        orders.append(learn_order_host(n_rows, p, block_size, random_seed))
    return orders^


def ordered_model_length_mult(n_rows: Int, model_size: Float64) -> Float64:
    """`CalcScoreModelLengthMult` (`random_score_helper.h:18-22`), through
    the portable binary64 log and exp so every host forms the same bits:
    `L = exp(log(n) - model_size); L / (1 + L)`."""
    var left = portable_exp64(portable_log64(Float64(n_rows)) - model_size)
    return left / (Float64(1.0) + left)
