# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels behind the user-object reader (bindings/user_objects_native.mojo).

The shell reads a user's Python objects once into typed values (here: one
byte key per row); these functions do the work on them. Host Mojo, no Python.

cpu-route: user-object input step (group ids of a ranking pool), before any fit.
"""


def _fnv1a(keys: List[UInt8], start: Int, length: Int) -> UInt64:
    var h: UInt64 = 14695981039346656037
    for k in range(length):  # driver-loop: over one key's bytes
        h = (h ^ UInt64(keys[start + k])) * 1099511628211
    return h


def _same_key(keys: List[UInt8], a: Int, la: Int, b: Int, lb: Int) -> Bool:
    if la != lb:
        return False
    for k in range(la):  # driver-loop: over one key's bytes
        if keys[a + k] != keys[b + k]:
            return False
    return True


def group_run_sizes(keys: List[UInt8], starts: List[Int], lengths: List[Int], mut sizes: List[Int]) -> Int:
    """The run lengths of equal consecutive keys, in row order, into `sizes`.

    Row i's key is `keys[starts[i] : starts[i] + lengths[i]]`. A key that
    starts a new run after an earlier run of the same key is a group split
    into two runs: the function returns that row (sizes then holds the runs
    before it); -1 when every group is one run."""
    var n = len(starts)
    var cap = 16
    while cap < 2 * n + 2:  # driver-loop: over table doublings
        cap *= 2
    var mask = cap - 1
    # slot -> the first row of a run (its key), -1 empty
    var slots = List[Int](length=cap, fill=-1)
    var last = -1
    for i in range(n):  # cpu-route: group ids read once by user_objects_c (group_runs), one key per row
        var s = starts[i]
        var ln = lengths[i]
        if last >= 0 and _same_key(keys, s, ln, starts[last], lengths[last]):
            sizes[len(sizes) - 1] += 1
            continue
        var h = Int(_fnv1a(keys, s, ln) & UInt64(mask))
        while slots[h] >= 0:  # driver-loop: over one probe sequence
            var r = slots[h]
            if _same_key(keys, s, ln, starts[r], lengths[r]):
                return i
            h = (h + 1) & mask
        slots[h] = i
        sizes.append(1)
        last = i
    return -1
