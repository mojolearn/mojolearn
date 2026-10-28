# SPDX-License-Identifier: Apache-2.0
# Private solver linkage transport over the shared NativeDescriptorCall ABI.
# Borrowed address/integer spans remain owned by the public operation. The
# numerical owners retain their existing validation, algorithms and outputs.
comptime HIERARCHY_SOLVER_ABI = "mojolearn.hierarchy.solver.v1"
comptime HIERARCHY_SOLVER_OPERATIONS = 3
comptime HIERARCHY_SOLVER_LINKAGE = 0
comptime HIERARCHY_SOLVER_MODE = 1
comptime HIERARCHY_SOLVER_CAPABILITIES = 2
comptime HIERARCHY_SOLVER_CAP_LINKAGE = 1
comptime HIERARCHY_SOLVER_CAP_DEVICE = 2


def hierarchy_solver_validate_call(operation: Int, na: Int, np: Int, nf: Int,
                                   result: Int) raises:
    if operation < 0 or operation >= HIERARCHY_SOLVER_OPERATIONS:
        raise Error("hierarchy solver provider: unknown operation")
    if result == 0 or nf != 0:
        raise Error("hierarchy solver provider: null result or unexpected scalar span")
    if operation == HIERARCHY_SOLVER_LINKAGE:
        # A: row-major f32 X, i32 children, i32 labels, i32 info[2].
        # I: n_rows, n_cols, n_clusters, cuML metric, use_knn. F: empty.
        if na != 4 or np != 5:
            raise Error("hierarchy solver provider: linkage requires four buffers and five integers")
    elif na != 0 or np != 0:
        raise Error("hierarchy solver provider: metadata takes no operands")


def hierarchy_solver_validate_buffers(addresses: List[Int], params: List[Int]) raises:
    for i in range(4):  # small-loop(four admitted linkage input/output addresses)
        if addresses[i] == 0:
            raise Error("hierarchy solver provider: null linkage buffer")
    # These are byte-span overflow guards, not workload routing thresholds.
    # Ordinary dimension/metric refusals remain in the existing numerical owner.
    var n = params[0]
    var d = params[1]
    if n > 0 and d > 0 and n > (9223372036854775807 // 4) // d:
        raise Error("hierarchy solver provider: X byte span overflows native size")
    if n > 1 and n - 1 > 9223372036854775807 // 8:
        raise Error("hierarchy solver provider: children byte span overflows native size")
