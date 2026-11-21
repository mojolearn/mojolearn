from core.shared_slot import SharedSlot
from bindings.operation_names_density import operation_names_density
from bindings.operation_names_par import operation_names_par
from bindings.operation_names_kmeans import operation_names_kmeans
from bindings.operation_names_km import operation_names_km
from bindings.operation_names_svm import operation_names_svm
from bindings.operation_names_xc import operation_names_xc
from bindings.operation_names_metric import operation_names_metric
from bindings.operation_names_xn import operation_names_xn
from bindings.operation_names_chol import operation_names_chol
from bindings.operation_names_msel import operation_names_msel
from bindings.operation_names_svc import operation_names_svc
from bindings.operation_names_gp import operation_names_gp
from bindings.operation_names_mcls import operation_names_mcls
from bindings.operation_names_mexp import operation_names_mexp
from bindings.operation_names_hdbscan import operation_names_hdbscan
from bindings.operation_names_cfa import operation_names_cfa
from bindings.operation_names_cfb import operation_names_cfb
from bindings.operation_names_mix import operation_names_mix
from bindings.operation_names_mani import operation_names_mani
from bindings.operation_names_low import operation_names_low
from bindings.operation_names_linops import operation_names_linops
from bindings.operation_names_classical import operation_names_classical
from bindings.operation_names_trees import operation_names_trees
# SPDX-License-Identifier: Apache-2.0
# The one table of integer operation codes for the classical owner's door
# (lane ne1-h-labels, plan NATIVE_ENTRY_POINTS_AFTER_0866 section H, lesson
# 12: dispatch on integer codes through a table, not string chains).
#
# The Python shells (`_expansion_{neighbors,trees,metrics}.py`,
# `_metrics_impl.py`, `_forest_host.py`, `_gbdt_host.py`) passed each
# operation as a str; `classical_native_call` copied it into a Mojo String and
# matched it against a chain of exact and prefix tests, then
# bindings/family_native.mojo ran about 32 more prefix tests before the
# family owner. Now a shell reads this table once at import
# (python/mojolearn/_native_ops.py, door operation "operation_codes") and
# passes the integer code; the door reads the route, the family route and
# the name from the table (filled once per process) and calls the owner
# directly.
#
# THE LEAF COLUMN (lane bw2-core, 2026-11-20: the prefix route of the boundary
# conversion). A dispatcher that tested name prefixes and forwarded the name
# String to another owner now resolves the name ONCE to its code here, and a
# converted owner switches on the name's LEAF code (`leaves`), an integer the
# owner's own file defines (bindings/operation_names_<owner>.mojo:
# `<FAMILY>_OPN_<NAME upper>`, numbered by sub-owner so a group is one range
# test). A str caller (a shell that passes the name, the parallel driver's
# owner, model selection) is resolved by ONE dictionary lookup with its own
# str object (`operation_code`, bindings/native_operation_names_unpack.mojo);
# no String is built and no prefix chain runs for a name of the table. An
# owner that is not converted yet has `OP_LEAF_NONE` and still receives the
# operation name: the table's own String, borrowed, never built per call. A
# name the table does not hold keeps the String path (`family_route` on the
# name), which is also where an unknown name gets its error text.
#
# Codes are the index in `names` (registration order: the trees names, then
# each family owner's file); they hold within one build only, and Python
# never hard-codes them. Every `family_*` name a caller can pass belongs in
# its owner's file; `trees_*` names are in bindings/operation_names_trees.mojo
# and the `host_tree_*` names passed by code in `_operation_names` below.

#: The leaf of a name whose owner still matches the name.
comptime OP_LEAF_NONE = -1

comptime OP_ROUTE_FAMILY = 1
comptime OP_ROUTE_TREES = 2
comptime OP_ROUTE_HOST_TREE = 3
# The classical door's and its tree door's own string operations (lane
# bw2-classical): registered LAST (bindings/operation_names_classical.mojo),
# so they follow every code Python maps; never passed by code (the coded
# doors refuse this route) and never in `operation_code_names`.
comptime OP_ROUTE_CLASSICAL = 4

# Family owner routes (`family_route`), in the priority order of the former
# prefix chain of bindings/family_native.mojo.
comptime FR_UNKNOWN = 0
comptime FR_DENSITY = 1
comptime FR_PAR_PUBLIC = 2
comptime FR_PAR_RUN = 3
comptime FR_KMEANS = 4
comptime FR_KM = 5
comptime FR_SVM = 6
comptime FR_XC = 7
comptime FR_METRIC = 8
comptime FR_XN = 9
comptime FR_CHOL = 10
comptime FR_MSEL = 11
comptime FR_SVC = 12
comptime FR_GP = 13
comptime FR_MCLS = 14
comptime FR_MEXP = 15
comptime FR_HDBSCAN_PUBLIC = 16
comptime FR_CFA = 17
comptime FR_CFB = 18
comptime FR_MIX = 19
comptime FR_MANI = 20
comptime FR_LOW = 21
comptime FR_LINALG_OPS = 22
# The estimator-argument binding step that runs before the owner.
comptime FR_PRE_NONE = 0
# TOMBSTONE: FR_PRE_ESTIMATORS (1: the per-call `estimator_arguments(model, "_mojolearn_estimators", ...)` step of
# every `family_svc_linear_*` / `family_svm_linear_*` call, a binding-route read and a new args tuple per call)
# deleted 2026-11-21 by lane ma-pub-linear: LinearSVC / LinearSVR hold their driver on their device route
# (bindings/linear_svc_boundary_unpack.mojo `linear_svc_route`). Word 1 stays unused.
comptime FR_PRE_SVM = 2
comptime FR_PRE_BASE = 3
comptime FR_PRE_X_NEIGHBORS = 4
# A packed family route word: owner + 64 * pre + 1024 * cluster flag.
comptime FR_PRE_SCALE = 64
comptime FR_CLUSTER_FLAG = 1024


def family_route_owner(route: Int) -> Int:
    return route % FR_PRE_SCALE


def family_route_pre(route: Int) -> Int:
    return (route // FR_PRE_SCALE) % (FR_CLUSTER_FLAG // FR_PRE_SCALE)


def family_route_cluster(route: Int) -> Bool:
    return route >= FR_CLUSTER_FLAG


def family_route(operation: String) -> Int:
    """The packed route word of one `family_*` operation name: the same
    tests, in the same order, the family dispatcher ran on every call (now
    run once per name for coded calls, at table fill)."""
    var pre = FR_PRE_NONE
    if operation.startswith("family_svc_svc_") or operation.startswith("family_svc_svr_"):
        pre = FR_PRE_SVM
    elif operation.startswith("family_kmeans_"):
        pre = FR_PRE_BASE
    if (operation.startswith("family_xn_loc_") or operation.startswith("family_xn_semi_")
            or operation.startswith("family_xn_graph_") or operation.startswith("family_svc_ocsvm_")):
        pre = FR_PRE_X_NEIGHBORS
    var cluster = ((operation.startswith("family_cfa_") and not operation.startswith("family_cfa_hdbscan_"))
                   or (operation.startswith("family_xc_") and operation != "family_xc_dbscan_predict"
                       and operation != "family_xc_agglo_predict") or operation.startswith("family_mix_bgmm_"))
    var owner = FR_UNKNOWN
    if operation.startswith("family_density_"):
        owner = FR_DENSITY
    elif operation == "family_par_public":
        owner = FR_PAR_PUBLIC
    elif operation == "family_par_run":
        owner = FR_PAR_RUN
    elif operation.startswith("family_kmeans_"):
        owner = FR_KMEANS
    elif operation.startswith("family_km_"):
        owner = FR_KM
    elif operation.startswith("family_svm_"):
        owner = FR_SVM
    elif operation.startswith("family_xc_"):
        owner = FR_XC
    elif operation.startswith("family_metric_"):
        owner = FR_METRIC
    elif operation.startswith("family_xn_"):
        owner = FR_XN
    elif operation.startswith("family_chol_"):
        owner = FR_CHOL
    elif operation.startswith("family_msel_"):
        owner = FR_MSEL
    elif operation.startswith("family_svc_"):
        owner = FR_SVC
    elif operation.startswith("family_gp_"):
        owner = FR_GP
    elif operation.startswith("family_mcls_"):
        owner = FR_MCLS
    elif operation.startswith("family_mexp_"):
        owner = FR_MEXP
    elif operation.startswith("family_hdbscan_public_"):
        owner = FR_HDBSCAN_PUBLIC
    elif operation.startswith("family_cfa_"):
        owner = FR_CFA
    elif operation.startswith("family_cfb_"):
        owner = FR_CFB
    elif operation.startswith("family_mix_"):
        owner = FR_MIX
    elif operation.startswith("family_mani_"):
        owner = FR_MANI
    elif operation.startswith("family_low_"):
        owner = FR_LOW
    elif (operation.startswith("family_lin_") or operation.startswith("family_rs_")
            or operation.startswith("family_pg_")):
        owner = FR_LINALG_OPS
    return owner + FR_PRE_SCALE * pre + (FR_CLUSTER_FLAG if cluster else 0)


def _operation_names(mut names: List[String], mut leaves: List[Int]):
    """The `host_tree_*` operations a shell passes by code, then every
    `trees_*` name with its leaf (bindings/operation_names_trees.mojo, lane
    bw3-trees). Straight-line appends (no branch per name)."""
    names.append("host_tree_count")
    names.append("host_tree_ctr_dim")
    names.append("host_tree_ctr_predict")
    names.append("host_tree_engine")
    names.append("host_tree_getstate")
    names.append("host_tree_hash")
    names.append("host_tree_init_forest")
    names.append("host_tree_init_gbdt")
    names.append("host_tree_parse")
    names.append("host_tree_predict")
    names.append("host_tree_proba")
    names.append("host_tree_setstate")
    names.append("host_tree_vote")
    _leaves_pad(names, leaves)
    operation_names_trees(names, leaves)


def _family_names(mut names: List[String], mut leaves: List[Int]):
    """Every family owner's names, in the owners' route order. An owner
    that is not converted appends names only; its leaves are filled with
    `OP_LEAF_NONE` here."""
    operation_names_density(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_par(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_kmeans(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_km(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_svm(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_xc(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_metric(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_xn(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_chol(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_msel(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_svc(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_gp(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_mcls(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_mexp(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_hdbscan(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_cfa(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_cfb(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_mix(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_mani(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_low(names, leaves)
    _leaves_pad(names, leaves)
    operation_names_linops(names, leaves)
    _leaves_pad(names, leaves)


def _leaves_pad(names: List[String], mut leaves: List[Int]):
    """`OP_LEAF_NONE` for every name registered without a leaf."""
    while len(leaves) < len(names):  # driver-loop: over the fixed operation names, once per process
        leaves.append(OP_LEAF_NONE)


struct OperationTable(Defaultable, Movable):
    """The names, routes, family route words and leaf codes, filled once per
    process. No Python here: the name -> code dictionary of the str callers
    is bindings/native_operation_names_unpack.mojo's."""
    var names: List[String]
    var routes: List[Int]
    var family_routes: List[Int]
    var leaves: List[Int]

    def __init__(out self):
        var names = List[String]()
        var leaves = List[Int]()
        _operation_names(names, leaves)
        _leaves_pad(names, leaves)
        _family_names(names, leaves)
        var first_classical = len(names)
        operation_names_classical(names, leaves)
        _leaves_pad(names, leaves)
        self.names = names^
        self.leaves = leaves^
        self.routes = List[Int]()
        self.family_routes = List[Int]()
        for i in range(len(self.names)):  # driver-loop: over the fixed operation names, once per process
            if i >= first_classical:
                self.routes.append(OP_ROUTE_CLASSICAL)
                self.family_routes.append(FR_UNKNOWN)
            elif self.names[i].startswith("trees_"):
                self.routes.append(OP_ROUTE_TREES)
                self.family_routes.append(FR_UNKNOWN)
            elif self.names[i].startswith("host_tree_"):
                self.routes.append(OP_ROUTE_HOST_TREE)
                self.family_routes.append(FR_UNKNOWN)
            else:
                self.routes.append(OP_ROUTE_FAMILY)
                self.family_routes.append(family_route(self.names[i]))


# Shared by every Mojo extension in the process; bump the version whenever a
# name is added, removed or reordered (the codes are indices) or a field
# changes. V4: the leaf column and every family name. (bw2-classical adds the
# OP_ROUTE_CLASSICAL names at the end; the integrator bumps once per wave.)
comptime OPERATION_TABLE = SharedSlot[OperationTable, "MojolearnOperationCodesV4"]


def operation_table() raises -> UnsafePointer[OperationTable, MutAnyOrigin]:
    """The process-wide table, filled at first use."""
    return OPERATION_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
