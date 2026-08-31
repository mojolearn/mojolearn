# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT driver: one hashed build, one search, the card, the mode.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.ivf.card \\
        tools/with_build_lock.sh pixi run mojo run -I . ivf/ivf_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.ivf.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . ivf/ivf_main.mojo

    python3 tools/identity_trace_diff.py \\
        /tmp/mac.ivf.identical.card /tmp/<other>.ivf.identical.card

Environment knobs (all optional): `MOJOLEARN_IVF_N_LISTS` (default 8),
`MOJOLEARN_IVF_N_PROBES` (default 3), `MOJOLEARN_IVF_K` (default 8),
`MOJOLEARN_IVF_SEED` (default 0).

THE CARD: sixteen stages plus the coarse fit's, FNV-1a64 over raw bytes.
A cross-vendor run that diverges has an address:

    ivf.quantizer.*   `cluster/`'s k-means fit under this prefix. Its
                      identity status is UNSUPERVISED_IDENTITY.md's, NOT
                      this lane's, and a divergence here is that lane's
                      to read.
    ivf.centers       the coarse centroids. Hazard 2: everything below is
                      conditional on these bytes.
    ivf.center_norms  `core/row_norms.mojo`, one block per row.
    ivf.assign        hazard 1, the list assignment and its tie rule.
    ivf.list_offsets  the CSR row pointer. Integers; a divergence here is
                      a divergence in `assign` one stage up.
    ivf.list_indices  hazard 3, the CARRY. Diverging while `list_data`
                      agrees means the layout permuted without the carry
                      following, which is the classic IVF bug.
    ivf.list_data     the permutation itself.
    ivf.query_norm    `core/row_norms.mojo` again, on the queries.
    ivf.coarse_dist   IDENTITY_PATHS rows 9/10/24 (the per-cell
                      contraction, the flush, and the vendor matmul the
                      IDENTICAL arm does not use).
    ivf.probe_dist    the coarse selection's values.
    ivf.probe_lists   hazard 1 on the query side, and hazard 4's `n_probes`
                      is what makes this stage the length it is.
    ivf.cand_counts   integers; a pure function of the assignment and
                      `n_probes`.
    ivf.cand_idx      hazard 3 again, downstream: the candidate row's
                      ORIGINAL ids in merge order.
    ivf.cand_dist     the candidate distances.
    ivf.out_dist      the answer's values.
    ivf.out_idx       the answer's identities. DIVERGING WHILE `out_dist`
                      AGREES IS THE TIE CLASS AND NOTHING ELSE
                      (IDENTITY_PATHS row 11's signature, and the reason
                      the pair is two tags rather than one).

The fixture performs no host floating-point operation (see
`ivf/checks/ivf_fixture.mojo`). Prints the first eight neighbours as
decimal AND hex, because `String(Float32)` does not round-trip.

NO NUMBER PRINTED HERE IS A PERFORMANCE NUMBER. This driver has never been
timed, `ivf/README.md` says so in those words, and a traced run is not a
measurement in any case (`core/identity_trace.mojo` rule 4).
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from cluster.impl.cluster.kmeans_params import METRIC_L2_EXPANDED
from ivf.estimator import ivf_flat_build_and_search_host
from ivf.checks.ivf_fixture import ivf_index_fixture, ivf_query_fixture
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime IVF_MAIN_N_ROWS = 512
comptime IVF_MAIN_N_QUERIES = 64
comptime IVF_MAIN_DIM = 8


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _env_int(name: String, fallback: Int) -> Int:
    var raw = String(getenv(name))
    if raw == "":
        return fallback
    var value = 0
    for i in range(raw.byte_length()):
        var c = ord(String(raw[byte=i]))
        if c < 48 or c > 57:
            return fallback
        value = value * 10 + (c - 48)
    return value


def main() raises:
    var n_lists = _env_int(String("MOJOLEARN_IVF_N_LISTS"), 8)
    var n_probes = _env_int(String("MOJOLEARN_IVF_N_PROBES"), 3)
    var k = _env_int(String("MOJOLEARN_IVF_K"), 8)
    var seed = UInt64(_env_int(String("MOJOLEARN_IVF_SEED"), 0))

    print(
        "== ivf/ivf_main.mojo ["
        + _mode_name()
        + "] n_rows="
        + String(IVF_MAIN_N_ROWS)
        + " n_queries="
        + String(IVF_MAIN_N_QUERIES)
        + " d="
        + String(IVF_MAIN_DIM)
        + " n_lists="
        + String(n_lists)
        + " n_probes="
        + String(n_probes)
        + " k="
        + String(k)
        + " seed="
        + String(seed)
        + " =="
    )

    var index = ivf_index_fixture(IVF_MAIN_N_ROWS, IVF_MAIN_DIM, 1)
    var queries = ivf_query_fixture(
        index, IVF_MAIN_N_ROWS, IVF_MAIN_N_QUERIES, IVF_MAIN_DIM, 1
    )

    var ctx = DeviceContext()
    var result = ivf_flat_build_and_search_host(
        ctx,
        index,
        IVF_MAIN_N_ROWS,
        IVF_MAIN_DIM,
        n_lists,
        queries,
        IVF_MAIN_N_QUERIES,
        k,
        n_probes,
        20,
        METRIC_L2_EXPANDED,
        seed,
    )

    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print(
            "no MOJOLEARN_IDENTITY_TRACE set: index built and searched, no"
            " card written"
        )
    else:
        print(
            "card written to "
            + trace_path
            + " (16 ivf.* stages plus the coarse fit's ivf.quantizer.*)"
        )

    print("  query 0's " + String(k) + " neighbours:")
    var shown = k
    if shown > 8:
        shown = 8
    for i in range(shown):
        print(
            "    idx="
            + String(result.indices[i])
            + "  dist="
            + String(result.distances[i])
            + "  "
            + _hex32(result.distances[i])
        )

    var total = 0
    for q in range(IVF_MAIN_N_QUERIES):
        total += Int(result.n_candidates[q])
    print(
        "  candidates scored: "
        + String(total)
        + " of "
        + String(IVF_MAIN_N_QUERIES * IVF_MAIN_N_ROWS)
        + " a brute force would have scored ("
        + String(n_probes)
        + " of "
        + String(n_lists)
        + " lists probed)"
    )
    print(
        "  THAT RATIO IS NOT A SPEEDUP. It is a count of distance cells,"
        " taken under a trace, on a path nothing has timed."
    )
