"""Print the kernel matrix as every vendor resolves it. NO GPU REQUIRED.

`probe_main.mojo` also shows the table, but it launches kernels on the way, so
it can only ever tell you about the machine you are sitting at. This one
touches no device: it resolves the columns and prints the constants each
section's kernels would compile against.

**That is the point.** The table's whole job is to answer questions about
hardware nobody here owns. A readout that needs the hardware cannot do that.

    mojo build -I . matrix_main.mojo -o /tmp/matrix && /tmp/matrix

To see another vendor's build, change ONE line -- `TARGET_COLUMN` in
`mojo_only/kernel_matrix.mojo` -- and rebuild. Every constant below moves with
it, in six sections, with no other edit anywhere in the tree.
"""

from core.column_stats import STATS_TPB
from core.row_norms import NORM_TPB
from cluster.mojo_only.plus_plus import PLUS_PLUS_TPB
from cluster.ported.distance.unfused_distance_nn import (
    REDUCE_MIN_LANES,
    REDUCE_MIN_TPB,
)
from dbscan.ported.dbscan.vertexdeg.algo import VD_TPB
from decomposition.mojo_only.jacobi_eigh_device import JACOBI_TPB
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    K_LIB_SELECT_WARPSORT,
    PINNED_LIB_REDUCE_LANES,
    TARGET_COLUMN,
    column_lane_width,
    column_name,
    column_shared_limit,
    lib_block_size_for,
    lib_lane_width_for,
    lib_smem_pages_for,
)

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
