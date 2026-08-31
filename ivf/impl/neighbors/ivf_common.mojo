# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pieces every IVF index shares: chunk offsets and the two postprocesses.

PORT OF `cuvs/src/neighbors/ivf_common.cuh` (`kOutOfBoundsRecord` :31,
`calc_chunk_indices` :49-77 with its kernel in `ivf_common.cu:22-51`,
`find_chunk_ix` :94-109, `postprocess_neighbors` :113-165,
`postprocess_distances` :175-...) at cuVS `6ba2ce2`. Partial, and the
partiality is the interesting part.

WHAT THIS FILE'S UPSTREAM IS FOR, AND WHICH HALF SURVIVES
----------------------------------------------------------
Their candidate numbering is PROBE-CONCATENATED: for one query, sample `s`
means "the `s`-th vector of the probed lists laid end to end, in probe
order". Two things follow, and both are in their file.

  - `calc_chunk_indices` is the segmented inclusive scan of the probed
    list sizes, so `chunk_indices[p]` is where probe `p` ends and
    `n_samples[q]` is how many candidates query `q` has at all.
  - `find_chunk_ix` binary-searches those offsets to turn a sample index
    back into `(which probe, which position in that list)`, and
    `postprocess_neighbors_kernel` then reads
    `db_indices[clusters_to_probe[chunk_ix]][data_ix]` to get the database
    id. That whole path exists because their scan's output is a position
    and the caller wants a row id.

THE FIRST HALF IS PORTED AND THE SECOND HALF IS NOT, **DEVIATION 1790**.
This port's candidate row is not probe-concatenated: it is the ascending-
original-index merge of the probed lists (`ivf/checks/list_layout.mojo`,
DEVIATION 1786), carried alongside a `cand_orig` array of original row ids.
So the "turn a position into a row id" step is one array read, and their
binary search would be a second, slower spelling of a mapping this port
already holds explicitly. Recorded in `ivf/NOT_IMPLEMENTED.tsv` with this
sentence.

`calc_chunk_indices` survives with its job intact -- `n_samples[q]`, the
candidate count, is a number the search needs and a stage the card
records -- and it runs on the host here for DEVIATION 1800's reason. It is
INTEGER ARITHMETIC END TO END: a sum of `uint32` list sizes has one value
on every machine, so their `cub::BlockScan` and this loop are the same
function and there is nothing for `IDENTICAL` to pin.
"""

from cluster.impl.cluster.kmeans_params import (
    METRIC_L2_EXPANDED,
    METRIC_L2_SQRT_EXPANDED,
)


comptime IVF_OUT_OF_BOUNDS_RECORD: UInt32 = 0xFFFFFFFF
"""`kOutOfBoundsRecord<IdxT>`, `ivf_common.cuh:31`
(`std::numeric_limits<IdxT>::max()`).

RECORDED AND NOT WRITTEN. `postprocess_neighbors_kernel` stores it wherever
`find_chunk_ix` returns `n_probes`, which happens when the probed lists
between them hold fewer than `k` vectors (`:106-108`). This port REFUSES
that case by name instead (DEVIATION 1794): a caller who gets `k` slots
back, some of them a sentinel, has to know to test for the sentinel, and
the ported selector cannot take `k > len` at all -- the same hole
`neighbors/impl/neighbors/detail/knn_brute_force.mojo` refuses `k >
n_index` for, for the same reason, in the same words."""


def calc_chunk_indices(
    list_sizes: List[Int32], probe_list_ids: List[UInt32], n_probes: Int
) raises -> List[Int32]:
    """`calc_chunk_indices_kernel`, `ivf_common.cu:22-51`, for one query.

    Returns `n_probes` INCLUSIVE prefix sums of the probed lists' sizes, in
    probe order, exactly as theirs does. `out[n_probes - 1]` is their
    `n_samples[blockIdx.x]`, the total candidate count for this query.

    Their kernel is a `cub::BlockScan` per query block with a running
    `total` carried between tiles when `n_probes` exceeds the block width;
    ours is the same scan spelled as one ascending loop, because it is a
    sum of unsigned integers and integer addition is associative on every
    machine. There is no float here and therefore no pathway for
    `IDENTITY_PATHS` to have a row about.
    """
    if len(probe_list_ids) < n_probes:
        raise Error(
            "calc_chunk_indices: probe_list_ids has "
            + String(len(probe_list_ids))
            + " entries, expected at least "
            + String(n_probes)
        )
    var out = List[Int32]()
    var total = Int32(0)
    for p in range(n_probes):
        var l = Int(probe_list_ids[p])
        if l < 0 or l + 1 >= len(list_sizes) + 1:
            raise Error(
                "calc_chunk_indices: probe " + String(p) + " names list "
                + String(l) + ", which has no size entry"
            )
        total = total + list_sizes[l]
        out.append(total)
    return out^


def n_samples_from_chunks(chunk_indices: List[Int32], n_probes: Int) -> Int:
    """Their `n_samples[q]`. Zero when every probed list is empty, which is
    a reachable state (`check_empty_list`) and not an error here; the
    refusal for "fewer candidates than k" is the search's, so that the
    message can name `k`."""
    if n_probes <= 0:
        return 0
    return Int(chunk_indices[n_probes - 1])


def postprocess_neighbors(
    selected_positions: List[UInt32], cand_orig: List[UInt32], k: Int
) raises -> List[UInt32]:
    """`postprocess_neighbors`, `ivf_common.cuh:148-165`, through the carry.

    Theirs: `find_chunk_ix` the sample index into `(chunk, position)`, then
    `db_indices[clusters_to_probe[chunk]][position]`. Ours: `cand_orig` was
    built by `gather_candidate_indices` at the same time as the candidate
    vectors and holds the row id for every candidate position, so the
    lookup is one read. **DEVIATION 1790.**

    THE BOUNDS CHECK IS NOT DEFENSIVE TIDINESS. A selected position outside
    the candidate row means the selector returned a slot it never filled,
    which is what happens when `k` exceeds the candidate count -- the
    silent-wrong-answer failure `knn_brute_force.mojo` documents at its own
    `k > n_index` refusal. The search refuses that case before it reaches
    here; this raise is what would catch a NEW way of producing it.
    """
    if len(selected_positions) < k:
        raise Error(
            "postprocess_neighbors: the selector returned "
            + String(len(selected_positions))
            + " positions, expected k = "
            + String(k)
        )
    var out = List[UInt32]()
    for i in range(k):
        var p = Int(selected_positions[i])
        if p < 0 or p >= len(cand_orig):
            raise Error(
                "postprocess_neighbors: slot "
                + String(i)
                + " holds candidate position "
                + String(p)
                + ", outside [0, "
                + String(len(cand_orig))
                + "). The selector wrote a slot it never filled."
            )
        out.append(cand_orig[p])
    return out^


def postprocess_distances_is_identity(metric: Int) raises -> Bool:
    """Whether `postprocess_distances` has anything to do on this metric.

    `ivf_common.cuh:175-...` switches on the metric and, for `L2Expanded`
    at `scaling_factor == 1.0` with `in == out` and no type change, does
    NOTHING -- `needs_cast` is false, `needs_copy` is false, and the
    `scaling_factor != 1.0` arm is not taken. Their `search_impl` only
    calls it at all on the `!manage_local_topk` path (`:295-298`), which is
    the `k > warpsort::kMaxCapacity` path this port does not have.

    So on the two metrics this port carries it is the identity, and this
    function says so with the citation rather than a comment nobody
    reads. **DEVIATION 1791.** It raises rather than returning `False` for
    anything else, because a `False` here would be read as "some other arm
    ran" and no other arm exists in this tree.

    The square root that `L2SqrtExpanded` wants is NOT theirs to do here:
    it is applied at the distance seam, inside
    `neighbors/checks/pinned_distance_tile.mojo` under `IDENTICAL` and
    `core/expand_distances.mojo` under `FAST`, through `identical_sqrt`
    (DEVIATION 550). That is where the k-NN lane put it and this lane calls
    those kernels rather than re-spelling either.
    """
    if metric == METRIC_L2_EXPANDED or metric == METRIC_L2_SQRT_EXPANDED:
        return True
    raise Error(
        "postprocess_distances: metric "
        + String(metric)
        + " is not one this port carries; there is no arm to run."
    )
