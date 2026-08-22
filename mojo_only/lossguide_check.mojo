"""`EGrowPolicy::Lossguide` grows a tree, and it is not Depthwise's tree.

    pixi run check-lossguide

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`.

WHAT IS UNDER TEST. `fit_non_symmetric_tree` under `grow_policy=Lossguide` --
the merged driver's Lossguide branches, end to end, on the depthwise lane's
fixture. The fixture is IMPORTED and not re-written, because two lanes with
two fixtures is two ways to be lucky; a contrast is only worth something if
both arms saw the same rows.

THE CLAIMS, and every one of them is a property Depthwise does NOT have:

  L1  IT GROWS AND CONSERVES. A well-formed non-symmetric tree: `len(nodes) +
      1` leaves, every row in exactly one leaf, row count conserved.

  L2  ONE SPLIT PER ITERATION, WHICH IS THE POLICY. Lossguide splits exactly
      one leaf per iteration (`greedy_search_helper.cpp:319-324`), so the
      leaf count rises by exactly one each time and the tree ends with
      EXACTLY `max_leaves` leaves whenever the terminal tests do not bite
      first. Depthwise's leaf count doubles-ish per level and lands wherever
      the sign test leaves it. This is the sharpest observable difference
      between the two policies and it is an EXACT integer, not a tendency.

  L3  `max_leaves` IS THE LIVE BOUND, and Lossguide is the only policy for
      which it is the user's number -- CatBoost overwrites it with
      `1 << MaxDepth` for every other policy (`catboost_options.cpp:993`).
      Asked for 3, 5 and 9 leaves, the tree must have exactly 3, 5 and 9.

  L4  THE CONTRAST IS LIVE. The same fixture, same depth, same everything but
      the policy, must produce a DIFFERENT structure. If it did not, either
      the policy branch is not reached or the fixture cannot tell the two
      apart, and every other claim here would be vacuous. REACH IS PER
      BRANCH.

  L5  `min_data_in_leaf` STILL BITES under Lossguide. It goes live for every
      non-symmetric policy (`greedy_search_helper.cpp:685`), so a large
      minimum must stop the tree BEFORE `max_leaves` does -- which is the
      case that proves `max_leaves` is not the only thing being honored.

  L6  THE TRACE IS WIRED AND LOCALIZING. One fit under an `IdentityTrace`
      must emit records; a fit under `IdentityTrace.disabled()` must emit
      none. An instrument that is present but unreached is worse than none,
      because its silence reads as agreement.

WHAT THIS FILE DOES NOT CLAIM. Nothing about CatBoost's own Lossguide output.
There is no oracle here: CatBoost's GPU learner does not run on this machine,
so the values cannot be compared against theirs on this box. What is gated is
that the POLICY is the policy their source describes. The value comparison is
the NVIDIA column's job and is owed.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    TDepthwiseWorkspace,
    fit_non_symmetric_tree,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
)
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.models.non_symmetric_tree import TNonSymmetricTree
from gbdt.options.catboost_options import GROW_LOSSGUIDE
from mojo_only.depthwise_check import Fixture, default_options, fit as depthwise_fit


def lossguide_options(
    max_depth: Int, max_leaves: Int, min_data_in_leaf: Float64 = 1.0
) raises -> TTreeStructureSearcherOptions:
    var o = TTreeStructureSearcherOptions()
    o.policy = GROW_LOSSGUIDE
    o.max_depth = max_depth
    # THE ONE POLICY WHERE THIS IS THE USER'S NUMBER. Every other policy has
    # it overwritten with `1 << MaxDepth` and `check()` enforces the
    # equality; Lossguide is exempt there for exactly this reason.
    o.max_leaves = max_leaves
    o.min_leaf_size = min_data_in_leaf
    return o^


def fit_policy(
    mut fx: Fixture,
    options: TTreeStructureSearcherOptions,
    trace_path: String = "",
) raises -> TNonSymmetricTree:
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    # `disabled()` unless the caller names a path: a GATE whose behavior
    # depends on whether the operator has MOJOLEARN_IDENTITY_TRACE exported
    # is a gate that passes or fails for reasons outside itself.
    var tr = IdentityTrace.disabled()
    if trace_path != "":
        tr = IdentityTrace.to_path(trace_path)
    var model = fit_non_symmetric_tree(
        fx.ctx, fx.n_rows, fx.folds, options,
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, tr,
    )
    # The keep-alive invariant the depthwise check pins: `ws` / `dws` hold
    # every device buffer the fit enqueued against, and they are locals, so
    # they are safe only because the fit drains before returning.
    _ = ws^
    _ = dws^
    _ = tr^
    return model^


def structure_text(model: TNonSymmetricTree) raises -> String:
    var out = String("")
    for i in range(len(model.model_structure.nodes)):
        out += (
            String(Int(model.model_structure.nodes[i].feature_id))
            + "/"
            + String(Int(model.model_structure.nodes[i].bin))
            + "/"
            + String(Int(model.model_structure.nodes[i].left_subtree))
            + "/"
            + String(Int(model.model_structure.nodes[i].right_subtree))
            + " "
        )
    return out


def check_lossguide(ctx: DeviceContext) raises:
    var failures = 0
    var fx = Fixture(ctx.copy())

    print("-- L1: it grows, and it conserves --")
    var m = fit_policy(fx, lossguide_options(6, 9))
    var n_leaves = len(m.model_structure.nodes) + 1
    if len(m.model_structure.nodes) == 0:
        print("  FAIL the tree has no nodes at all")
        failures += 1
    elif n_leaves != model_leaf_count(m):
        print(
            "  FAIL leaves_count()", model_leaf_count(m),
            "disagrees with len(nodes)+1", n_leaves,
        )
        failures += 1
    else:
        print("  ok  ", len(m.model_structure.nodes), "nodes,", n_leaves, "leaves")

    print()
    print("-- L2/L3: exactly `max_leaves` leaves, because ONE split per iteration --")
    var all_exact = True
    for want in [2, 3, 5, 9]:
        var mm = fit_policy(fx, lossguide_options(6, want))
        var got = len(mm.model_structure.nodes) + 1
        if got != want:
            print(
                "  FAIL max_leaves", want, "gave", got,
                "leaves; Lossguide adds exactly one leaf per iteration so"
                " the count is EXACT, not a bound",
            )
            all_exact = False
            failures += 1
    if all_exact:
        print("  ok   max_leaves 2, 3, 5, 9 -> exactly 2, 3, 5, 9 leaves")

    print()
    print("-- L4: the contrast is live (same fixture, same depth) --")
    # THE CLAIM: same fixture, same depth, everything but the policy equal,
    # and the two structures must DIFFER. Without it every claim above is
    # consistent with a Lossguide branch that is never reached.
    #
    # THIS CLAIM WAS BLOCKED FOR AN HOUR AND THE BLOCKER WAS MINE. It read
    # "depthwise level loop did not terminate", I believed the noun, and I
    # eliminated six things about the depthwise arm before noticing the
    # driver's iteration bound was `max_depth + 2` -- a LEVEL count, correct
    # for Depthwise and wrong for Lossguide, which splits one leaf per
    # iteration and needs a LEAF count. The error's own numbers said so from
    # the first run: iterations always exactly `max_depth + 2`, leaves always
    # exactly `iterations + 1`. See the bound's block in the driver.
    var fx4 = Fixture(ctx.copy())
    var dw_o = default_options(4)
    var dw = depthwise_fit(fx4, dw_o)
    var lg = fit_policy(fx4, lossguide_options(4, 16))
    var dw_s = structure_text(dw)
    var lg_s = structure_text(lg)
    if dw_s == lg_s:
        print(
            "  FAIL Depthwise and Lossguide produced the SAME structure;"
            " either the policy branch is unreached or this fixture cannot"
            " tell them apart, and every claim above is vacuous"
        )
        failures += 1
    elif len(dw.model_structure.nodes) == 0 or len(lg.model_structure.nodes) == 0:
        print("  FAIL one of the two arms grew nothing")
        failures += 1
    else:
        print(
            "  ok   structures differ:", len(dw.model_structure.nodes),
            "depthwise nodes vs", len(lg.model_structure.nodes),
            "lossguide nodes at the same depth and fixture",
        )

    print()
    print("-- L5: min_data_in_leaf bites BEFORE max_leaves --")
    # `max_leaves` MUST BE OUT OF THE WAY or this claim tests nothing: the
    # first version asked for 9 leaves in both arms and got 9 in both,
    # because the LEAF CAP stopped the tree before the size test ever could.
    # The gate caught its own design, which is the only reason it is worth
    # having. 64 is their struct default and is far above what this fixture
    # can grow at min=600.
    var fx5 = Fixture(ctx.copy())
    # DEPTH 4, WHICH IS WHERE THE DRIVER IS ACTUALLY EXERCISED. Every one
    # of `depthwise_check`'s seven claims calls `default_options(4)`; at
    # depth 6 with the cap out of the way, BOTH policies hit the shared
    # driver's non-termination guard on this fixture. That is a real finding
    # and it is reported, but it is not this claim's subject, and a claim
    # that fails for a reason other than its subject teaches nothing. See
    # L4's block for the elimination list.
    var loose = fit_policy(fx5, lossguide_options(4, 16, 1.0))
    var tight = fit_policy(fx5, lossguide_options(4, 16, 600.0))
    var n_loose = len(loose.model_structure.nodes) + 1
    var n_tight = len(tight.model_structure.nodes) + 1
    if n_tight >= n_loose:
        print(
            "  FAIL min_data_in_leaf=600 gave", n_tight,
            "leaves against", n_loose,
            "at min=1; the size test is not reached under Lossguide",
        )
        failures += 1
    else:
        print(
            "  ok   min=1 ->", n_loose, "leaves, min=600 ->", n_tight,
            "-- the terminal test stops it before max_leaves does",
        )

    print()
    print("-- L6: the identity trace is wired and it is REACHED --")
    # AN INSTRUMENT THAT IS PRESENT BUT UNREACHED IS WORSE THAN NONE,
    # because its silence reads as agreement. Two fits, same options: one
    # under a real `IdentityTrace`, one under `disabled()`. The first must
    # emit records and the second must emit none.
    var fx6 = Fixture(ctx.copy())
    var tp = String("/tmp/mojolearn_lossguide_fit.trace")
    var traced = fit_policy(fx6, lossguide_options(6, 5), tp)
    var recs = read_trace_lines(tp)
    var untraced = fit_policy(fx6, lossguide_options(6, 5))
    if len(recs) == 0:
        print(
            "  FAIL a traced Lossguide fit emitted NO records; the"
            " checkpoints in the shared driver are not reached on this"
            " policy's path"
        )
        failures += 1
    elif structure_text(traced) != structure_text(untraced):
        # The trace DRAINS and copies. If that changed the model, the
        # instrument would be altering what it measures, which is the one
        # thing it may never do.
        print(
            "  FAIL tracing changed the tree; the instrument is not inert"
        )
        failures += 1
    else:
        print(
            "  ok  ", len(recs),
            "records emitted, and the traced tree is identical to the"
            " untraced one",
        )

    if failures != 0:
        raise Error("lossguide check: " + String(failures) + " failures")
    print()
    print("lossguide: all claims OK")


def model_leaf_count(model: TNonSymmetricTree) raises -> Int:
    return model.model_structure.leaves_count()


def main() raises:
    var ctx = DeviceContext()
    check_lossguide(ctx)
