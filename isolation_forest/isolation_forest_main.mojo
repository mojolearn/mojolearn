"""Isolation Forest driver: one hashed fit, the identity card, the mode it
ran in.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.if.card \\
        tools/with_build_lock.sh pixi run mojo run -I . isolation_forest/isolation_forest_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.if.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . isolation_forest/isolation_forest_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.if.identical.card /tmp/<other>.if.identical.card

THE CARD: `if.rng.probe` (the first 16 XORWOW draws of tree 0, i32 bits),
then per tree `if.treeNNN.rows` (the subsample's source rows),
`if.treeNNN.features` (only when max_features < n_cols), `if.treeNNN.meta`
(n_nodes, max_depth), `if.treeNNN.structure.{feat,thr,left,right}` (the
USED nodes), then `if.pathlen` and `if.scores` for the query batch. A
cross-vendor run that diverges has an address: the RNG probe is pure
integer (a difference there is a port defect, not a vendor), `.rows` /
`.features` are the Floyd sampler over that stream, `.structure.thr` is
the per-node strict-compare min/max + `identical_mul_add` threshold
(IDENTITY_PATHS rows 9/39) and `compute_c_n`'s `identical_log` (row 12),
`if.pathlen` is a serial per-row fold, `if.scores` is DEVIATION 681's
`identical_pow`.

Environment knobs (optional): `MOJOLEARN_IF_N_ESTIMATORS` (default 16),
`MOJOLEARN_IF_MAX_SAMPLES` (default 256), `MOJOLEARN_IF_SEED` (default 42).
The fixture performs no host floating-point operation
(`isolation_forest/mojo_only/if_fixture.mojo`). Prints the first eight
scores as decimal AND hex, because `String(Float32)` does not round-trip.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.mojo_only.if_fixture import (
    blob_fixture,
    plant_constant_column,
    plant_duplicates,
    plant_signed_zero_column,
    to_column_major,
)
from isolation_forest.ported.isolation_forest.isolation_forest import (
    IF_params,
    IsolationForestModel,
    fit,
    score_samples,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IF_MAIN_N = 1024
comptime IF_MAIN_D = 8
comptime IF_MAIN_N_OUTLIERS = 16
comptime IF_MAIN_N_QUERY = 256


def _mode_name() -> String:
    """The mode this binary COMPILED in."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _env_int(name: String, default: Int) -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    try:
        return Int(s)
    except:
        return default


def main() raises:
    print("isolation_forest_main [" + _mode_name() + "]")
    var n_estimators = _env_int("MOJOLEARN_IF_N_ESTIMATORS", 16)
    var max_samples = _env_int("MOJOLEARN_IF_MAX_SAMPLES", 256)
    var seed = _env_int("MOJOLEARN_IF_SEED", 42)

    var x = blob_fixture(IF_MAIN_N, IF_MAIN_D, 11, IF_MAIN_N_OUTLIERS)
    plant_constant_column(x, IF_MAIN_N, IF_MAIN_D, 5, 11)
    plant_signed_zero_column(x, IF_MAIN_N, IF_MAIN_D, 6)
    plant_duplicates(x, IF_MAIN_N, IF_MAIN_D, 40, 6, 97)  # LAST, so the copies stay copies
    var x_col = to_column_major(x, IF_MAIN_N, IF_MAIN_D)
    var query = List[Float32]()
    for i in range(IF_MAIN_N_QUERY * IF_MAIN_D):
        query.append(x[i])

    var params = IF_params.default()
    params.n_estimators = n_estimators
    params.max_samples = max_samples
    params.seed = UInt64(seed)

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "isolation_forest_main mode="
        + _mode_name()
        + " n="
        + String(IF_MAIN_N)
        + " d="
        + String(IF_MAIN_D)
        + " n_estimators="
        + String(n_estimators)
        + " max_samples="
        + String(max_samples)
        + " seed="
        + String(seed)
    )
    var model = IsolationForestModel(ctx)
    fit(ctx, model, x_col, IF_MAIN_N, IF_MAIN_D, params, trace)
    var scores = score_samples(ctx, model, query, IF_MAIN_N_QUERY, IF_MAIN_D, trace)
    var total_nodes = 0
    for t in range(n_estimators):
        total_nodes += Int(model.tree_n_nodes_host[t])
    print(
        "fit: "
        + String(n_estimators)
        + " trees, "
        + String(total_nodes)
        + " nodes, c(n)="
        + String(model.c_normalization)
        + " max_nodes_per_tree="
        + String(model.max_nodes_per_tree)
    )
    for i in range(8):
        print("score[" + String(i) + "] = " + String(scores[i]) + " " + _hex32(scores[i]))
    print("card: " + (trace.path if trace.enabled else String("(MOJOLEARN_IDENTITY_TRACE unset)")))
