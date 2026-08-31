# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Print the kernel matrix as every vendor resolves it. NO GPU REQUIRED.

`probe_main.mojo` also shows the table, but it launches kernels on the way, so
it can only ever tell you about the machine you are sitting at. This one
touches no device: it resolves the columns and prints the constants each
section's kernels would compile against.

**That is the point.** The table's whole job is to answer questions about
hardware nobody here owns. A readout that needs the hardware cannot do that.

    mojo build -I . matrix_main.mojo -o /tmp/matrix && /tmp/matrix

To see another vendor's build, change ONE line -- `TARGET_COLUMN` in
`checks/kernel_matrix.mojo` -- and rebuild. Every constant below moves with
it, in six sections, with no other edit anywhere in the tree.
"""

from core.column_stats import STATS_TPB
from core.row_norms import NORM_TPB
from cluster.checks.plus_plus import PLUS_PLUS_TPB
from cluster.impl.distance.unfused_distance_nn import (
    REDUCE_MIN_LANES,
    REDUCE_MIN_TPB,
)
from dbscan.impl.dbscan.vertexdeg.algo import VD_TPB
from decomposition.checks.jacobi_eigh_device import JACOBI_TPB
from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_COUNT,
    COLUMN_NVIDIA,
    IDENTITY_FLOOR_BLOCK,
    IDENTITY_FLOOR_LANES,
    IDENTITY_FLOOR_SHARED_BYTES,
    IDENTITY_PROFILE,
    K_LIB_SELECT_WARPSORT,
    PINNED_LIB_REDUCE_LANES,
    TARGET_COLUMN,
    column_has_dedicated_shared_memory,
    column_has_float_atomics,
    column_has_threadgroup_int_atomics,
    column_is_buildable,
    column_spec_guarantees_onchip_shared,
    column_lane_width,
    column_lane_width_is_fixed,
    column_max_block_size,
    column_meets_identity_floor,
    column_name,
    column_shared_limit,
    identity_refusal_reason,
    lib_block_size_for,
    lib_lane_width_for,
    lib_smem_pages_for,
)


def _yn(b: Bool) -> String:
    return String("yes") if b else String("NO")

#: RAFT's Policy4x4 shared page, in bytes. Two of these is their double
#: buffer: 36,992, which is over Metal's 32 KB and under NVIDIA's 48 KB.
comptime POLICY4X4_PAGE = 18496


def main() raises:
    print("kernel matrix, resolved per column. No device was touched.")
    print()
    print("  column          shared   lanes   fold   smem pages   warpsort blk")
    var cols = List[Int]()
    cols.append(COLUMN_BIT_IDENTICAL)
    cols.append(COLUMN_APPLE)
    cols.append(COLUMN_NVIDIA)
    cols.append(COLUMN_AMD)
    var pages = List[Int]()
    pages.append(lib_smem_pages_for[COLUMN_BIT_IDENTICAL, POLICY4X4_PAGE]())
    pages.append(lib_smem_pages_for[COLUMN_APPLE, POLICY4X4_PAGE]())
    pages.append(lib_smem_pages_for[COLUMN_NVIDIA, POLICY4X4_PAGE]())
    pages.append(lib_smem_pages_for[COLUMN_AMD, POLICY4X4_PAGE]())
    var wsb = List[Int]()
    wsb.append(lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_BIT_IDENTICAL]())
    wsb.append(lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_APPLE]())
    wsb.append(lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_NVIDIA]())
    wsb.append(lib_block_size_for[K_LIB_SELECT_WARPSORT, COLUMN_AMD]())

    for i in range(len(cols)):
        var c = cols[i]
        print(
            "  ",
            column_name(c),
            "\t",
            column_shared_limit(c) // 1024,
            "KB\t",
            column_lane_width(c),
            "\t",
            PINNED_LIB_REDUCE_LANES,
            "\t",
            pages[i],
            "\t\t",
            wsb[i],
        )

    print()
    print("  THIS BUILD compiles every kernel against:", column_name(TARGET_COLUMN))
    print()
    print("  constant            section                  value")
    print("    NORM_TPB          core/row_norms            ", NORM_TPB)
    print("    STATS_TPB         core/column_stats         ", STATS_TPB)
    print("    PLUS_PLUS_TPB     cluster/plus_plus         ", PLUS_PLUS_TPB)
    print("    REDUCE_MIN_TPB    cluster/unfused_distance  ", REDUCE_MIN_TPB)
    print("    REDUCE_MIN_LANES  cluster/unfused_distance  ", REDUCE_MIN_LANES)
    print("    VD_TPB            dbscan/vertexdeg          ", VD_TPB)
    print("    JACOBI_TPB        decomposition/jacobi      ", JACOBI_TPB)
    print(
        "    smem pages        contraction (Policy4x4)   ",
        lib_smem_pages_for[TARGET_COLUMN, POLICY4X4_PAGE](),
    )
    print(
        "    lane width        indexing only             ",
        lib_lane_width_for[TARGET_COLUMN](),
    )
    print()
    print(
        "  Not one of those is a literal in its own file. Each is"
        " lib_*_for[..., TARGET_COLUMN]()."
    )

    # ---------------------------------------------------------------------
    # THE VENDOR MINIMUMS, INCLUDING THE COLUMNS NOTHING CAN BUILD FOR YET.
    #
    # This is the table to read before adding a target, and the reason it
    # prints the unbuildable columns is that the question it answers is due
    # BEFORE the target exists: does admitting this vendor move anybody's
    # bits? The floor below is frozen, so the answer is no by construction --
    # a vendor either meets it and joins, or is refused by name.
    # ---------------------------------------------------------------------
    print()
    print(
        "  vendor minimums. 'build' = Mojo emits code for it today;"
        " everything else is the vendor's documented floor."
    )
    print()
    print(
        "  column         build  smem   lanes    maxblk  f32atom"
        "  i32atom  smem-hw  IDENTICAL"
    )
    for c in range(COLUMN_COUNT):
        var lanes = String(column_lane_width(c))
        if not column_lane_width_is_fixed(c):
            #: The width is the COMPILER's decision on this vendor, not the
            #: device's. Printed with the marker because a reader who takes
            #: it for a constant has misread the most dangerous cell here.
            lanes = lanes + "+var"
        print(
            "  ",
            column_name(c),
            "\t",
            _yn(column_is_buildable(c)),
            "\t",
            column_shared_limit(c) // 1024,
            "KB\t",
            lanes,
            "\t",
            column_max_block_size(c),
            "\t",
            _yn(column_has_float_atomics(c)),
            "\t",
            _yn(column_has_threadgroup_int_atomics(c)),
            "\t",
            #: Both halves: what the vendor built (Mali: cached system
            #: RAM) and what the spec promises (nothing about where
            #: Workgroup storage lives). Printing `yes` for a column that
            #: merely has not been contradicted would be the misleading
            #: half of the truth.
            _yn(
                column_has_dedicated_shared_memory(c)
                and column_spec_guarantees_onchip_shared(c)
            ),
            "\t",
            _yn(column_meets_identity_floor(c)),
        )
        var why = identity_refusal_reason(c)
        if why.byte_length() > 0:
            print("      refused:", why)

    print()
    print(
        "  identity floor, profile",
        IDENTITY_PROFILE,
        ": ",
        IDENTITY_FLOOR_SHARED_BYTES // 1024,
        "KB threadgroup,",
        IDENTITY_FLOOR_LANES,
        "logical lanes, block",
        IDENTITY_FLOOR_BLOCK,
        ", threadgroup int atomics.",
    )
    print(
        "  FROZEN. A vendor that misses it is refused for IDENTICAL and runs"
        " FAST; the floor never drops to fit one, because dropping it"
        " changes every model already produced under this profile."
    )
