# SPDX-License-Identifier: Apache-2.0
"""RUN of the kernel SVC's inference and of `coef_`: typed plans over the
SVM binding's entry tables (core/svc_native_api.mojo SvcDrivers), no Python.
The caller (bindings/svc_kernel_native_door.mojo) releases the GIL around
each run. Every launch is the one the door always made, with the same
operands in the same order: one query per fitted machine (the cross-kernel's
support columns gathered first for kernel='precomputed'), then one per-row
epilogue. Buffers are addresses whose owners the caller holds."""
from core.svc_native_api import SvcDrivers, SvcQueryRequest
from core.hostptr import i32_ptr
from svm.operation_metadata import SvcOperationSettings

# No epilogue: the machine's decisions are the result (binary decision_function).
comptime SVC_EPILOGUE_NONE = -1
# kernel='precomputed' (svm/operation_metadata.mojo kernel codes)
comptime SVC_KERNEL_PRECOMPUTED = 4
# `svc_pair_epilogue`'s GATHER mode (svm/impl/svc_rows.mojo GLUE_GATHER; a plain word here so this
# file imports no kernel module).
comptime SVC_GLUE_GATHER = 6


@fieldwise_init
struct SvcMachine(Copyable, Movable):
    """One fitted binary machine: its duals (1 x supports float32), its
    support rows (supports x features float32) and its intercept. With
    kernel='precomputed' the query is the cross-kernel: `columns` are the
    machine's int32 support indices and `gathered` a float32
    (rows x supports) buffer for those columns of X."""
    var dual: Int
    var vectors: Int
    var supports: Int
    var bias: Float64
    var columns: Int
    var gathered: Int


struct SvcQueryPlan(Movable):
    """One inference operation. The machines write their decisions
    machine-major at `decisions` (float32, machines x rows); `epilogue`
    (a `svc_pair_epilogue` mode, or SVC_EPILOGUE_NONE) then reads them with
    the pair codes (`pair_codes`: int32 (i, j) per pair) and, for the
    probabilities, the sigmoid parameters (`ab`: float64 (A, B) per pair),
    and writes the result at `dst`. `resident`: the device entry reads each
    machine from slot (its index) of `lease` and writes the query's finite
    and range words at `status` (two int32); the query is then the host
    words at `x` or the fold gathered on the device."""
    var machines: List[SvcMachine]
    var settings: SvcOperationSettings
    var x: Int
    var rows: Int
    var cols: Int
    var label0: Float64
    var label1: Float64
    var predict_class: Bool
    var resident: Bool
    var lease: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var status: Int
    var decisions: Int
    var epilogue: Int
    var pairs: Int
    var pair_codes: Int
    var ab: Int
    var classes: Int
    var epilogue_label1: Float64
    var dst: Int

    def __init__(out self, settings: SvcOperationSettings):
        self.machines = List[SvcMachine]()
        self.settings = settings.copy()
        self.x = 0
        self.rows = 0
        self.cols = 0
        self.label0 = 0.0
        self.label1 = 1.0
        self.predict_class = False
        self.resident = False
        self.lease = 0
        self.base_id = -1
        self.base_rows = 0
        self.index_id = -1
        self.status = 0
        self.decisions = 0
        self.epilogue = SVC_EPILOGUE_NONE
        self.pairs = 0
        self.pair_codes = 0
        self.ab = 0
        self.classes = 0
        self.epilogue_label1 = 0.0
        self.dst = 0


def svc_machines_run(drivers: SvcDrivers, plan: SvcQueryPlan) raises -> Bool:
    """Every machine's decisions (or labels). Returns False when the device
    entry flagged the query (the words are at `plan.status`): the caller
    refuses it; no later machine runs, as before."""
    for m in range(len(plan.machines)):  # driver-loop: over the fitted machines (one launch each)
        var supports = plan.machines[m].supports
        var x = plan.x
        var features = plan.settings.features
        var vectors = plan.machines[m].vectors
        if plan.settings.kernel == SVC_KERNEL_PRECOMPUTED and supports > 0:
            # X is the cross-kernel against the TRAINING rows; the decision
            # reads its columns at the support vectors and no support rows.
            if plan.rows > 0:
                _ = drivers.epilogue(SVC_GLUE_GATHER, plan.x, 0, plan.machines[m].columns, plan.machines[m].gathered,
                                     plan.rows, plan.cols, plan.rows, supports)
            x = plan.machines[m].gathered
            features = supports
            vectors = plan.machines[m].dual
        _ = drivers.query(SvcQueryRequest(x, plan.machines[m].dual if supports > 0 else 0,
            vectors if supports > 0 else 0, plan.decisions + 4 * plan.rows * m, plan.status,
            plan.rows, features, supports, plan.settings.kernel, plan.settings.degree, plan.predict_class,
            plan.lease, m, plan.base_id, plan.base_rows, plan.index_id, plan.resident,
            plan.machines[m].bias, plan.label0, plan.label1, plan.settings.gamma, plan.settings.cache_mib,
            plan.settings.coef0))
        if plan.resident:
            var words = i32_ptr(plan.status)
            if words[0] != 0 or words[1] != 0:
                return False
    return True


def svc_query_run(drivers: SvcDrivers, plan: SvcQueryPlan) raises -> Bool:
    """The machines, then the operation's per-row epilogue (none for a
    binary decision_function, and none for zero rows). False: the query was
    flagged and nothing after the flagged machine ran."""
    if not svc_machines_run(drivers, plan):
        return False
    if plan.epilogue != SVC_EPILOGUE_NONE and plan.rows > 0:
        _ = drivers.epilogue(plan.epilogue, plan.decisions, plan.pair_codes if plan.pairs > 0 else 0, plan.ab,
                             plan.dst, plan.rows, plan.pairs, plan.classes, 0, 0.0, plan.epilogue_label1)
    return True


@fieldwise_init
struct SvcCoefRow(Copyable, Movable):
    """One machine's `coef_` row: duals (1 x supports) times support rows."""
    var dual: Int
    var vectors: Int
    var supports: Int


def svc_coef_run(drivers: SvcDrivers, rows: List[SvcCoefRow], features: Int, sign: Int, dst: Int) raises:
    """`coef_` of a linear kernel: one GEMV per fitted machine into row
    `r` of the float32 (machines x features) matrix at `dst`."""
    for r in range(len(rows)):  # driver-loop: over the fitted machines (one GEMV each)
        if features > 0 and rows[r].supports > 0:
            _ = drivers.dual_gemv(rows[r].dual, rows[r].vectors, dst + 4 * r * features, rows[r].supports,
                                  features, sign)


def svc_gather_columns_run(drivers: SvcDrivers, x: Int, rows: Int, cols: Int, columns: Int, count: Int,
                           dst: Int) raises:
    """kernel='precomputed': columns `columns` (int32, `count`) of the
    float32 (rows x cols) cross-kernel at `x`, into `dst` (rows x count)."""
    if rows > 0 and count > 0:
        _ = drivers.epilogue(SVC_GLUE_GATHER, x, 0, columns, dst, rows, cols, rows, count)
