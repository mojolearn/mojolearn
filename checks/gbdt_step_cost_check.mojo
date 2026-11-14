# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The non-symmetric grower's step cost: host drains per tree and pool size.

    pixi run check-gbdt-step-cost

WHAT IS UNDER TEST. `fit_non_symmetric_tree` grown 50 times per policy on the
depthwise lane's fixture (4,096 rows x 16 features, the fixture
`checks/lossguide_check.mojo` also uses), with ONE pair of pools kept across
the 50 trees, as the boosting loop keeps them. After every tree the grower
records `TDepthwiseWorkspace.last_tree_drains` (the manager's counted waits
plus its own `ctx.synchronize` calls) and `last_tree_rounds`.

THE CLAIMS:

  S1  DEPTHWISE: at most two host waits per level (the score read and the
      split sizes, `RebuildLeavesSizes`) plus the leaf-value wait, so at most
      `2 * (max_depth + 1) + 1` drains a tree, and at most `max_depth + 2`
      rounds.
  S2  LOSSGUIDE: at most two host waits per round plus the leaf-value wait and
      the exact replay's one tail drain, so at most `2 * rounds + 2` drains a
      tree, and at most `max_leaves + 1` rounds. A drain per node or per leaf
      beyond that fails here.
  S3  LOSSGUIDE POOL: the tree pool holds the fit's leaf capacity (at most
      `2 * max_leaves` slots), not `1 << max_depth`; at max_depth 16 the
      depth-sized pool was 65,536 slots zeroed every tree. Skipped under
      `-D MOJOLEARN_IDN_GBDT_STEP_OLD` (the A/B arm that restores it).
  S4  POOL REUSE: the second through fiftieth trees reuse the first tree's
      pools (no per-tree rebuild).

WHY NOT "AT MOST TWO DRAINS PER TREE" FOR LOSSGUIDE. The grower is CatBoost's
host-driven round loop (`greedy_search_helper.cpp`), two waits per round, and
it was so in the measured winner too; the device-resident Lossguide frontier
(I17/T16) was measured and lost on NVIDIA and AMD on 2026-10-07 and is
deleted. S2 bounds the drains by the rounds, which is the structure that runs.
"""

from std.sys.compile import is_defined

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
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
from gbdt.options.catboost_options import GROW_DEPTHWISE, GROW_LOSSGUIDE
from checks.depthwise_check import Fixture

comptime TREES = 50
comptime STEP_OLD = is_defined["MOJOLEARN_IDN_GBDT_STEP_OLD"]()


def policy_options(
    policy: Int, max_depth: Int, max_leaves: Int
) raises -> TTreeStructureSearcherOptions:
    var o = TTreeStructureSearcherOptions()
    o.policy = policy
    o.max_depth = max_depth
    o.max_leaves = max_leaves
    o.min_leaf_size = 1.0
    return o^


def run_policy(
    mut fx: Fixture,
    name: String,
    options: TTreeStructureSearcherOptions,
    lossguide: Bool,
) raises -> Int:
    """Grow `TREES` trees with one pair of pools; returns the failure count."""
    var failures = 0
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    var tr = IdentityTrace.disabled()
    var first_slots = -1
    var max_drains = 0
    var max_rounds = 0
    for t in range(TREES):  # driver-loop: over the check's trees
        fx.reset()
        var model = fit_non_symmetric_tree(
            fx.ctx, fx.n_rows, fx.folds, options,
            fx.cindex, fx.stats, fx.row_index,
            fx.total_weight, fx.total_gradient,
            ws, dws, tr,
        )
        _ = model^
        var drains = dws[0].last_tree_drains
        var rounds = dws[0].last_tree_rounds
        if drains > max_drains:
            max_drains = drains
        if rounds > max_rounds:
            max_rounds = rounds
        var drain_cap: Int
        var round_cap: Int
        if lossguide:
            drain_cap = 2 * rounds + 2
            round_cap = options.max_leaves + 1
        else:
            drain_cap = 2 * (options.max_depth + 1) + 1
            round_cap = options.max_depth + 2
        if drains > drain_cap or drains < 1:
            print("FAIL", name, "tree", t, "drains", drains, "cap", drain_cap)
            failures += 1
        if rounds > round_cap or rounds < 1:
            print("FAIL", name, "tree", t, "rounds", rounds, "cap", round_cap)
            failures += 1
        var slots = ws[0].max_leaves_key
        if t == 0:
            first_slots = slots
        elif slots != first_slots:
            print("FAIL", name, "S4 pool rebuilt at tree", t)
            failures += 1
        comptime if not STEP_OLD:
            if lossguide and slots > 2 * options.max_leaves:
                print(
                    "FAIL", name, "S3 pool slots", slots,
                    "past the capacity", 2 * options.max_leaves,
                )
                failures += 1
    print(
        name, "trees", TREES, "max drains/tree", max_drains,
        "max rounds/tree", max_rounds, "pool slots", first_slots,
    )
    _ = ws^
    _ = dws^
    _ = tr^
    return failures


def main() raises:
    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())
    var failures = 0
    failures += run_policy(
        fx, "depthwise", policy_options(GROW_DEPTHWISE, 6, 1 << 6), False
    )
    failures += run_policy(
        fx, "lossguide", policy_options(GROW_LOSSGUIDE, 16, 64), True
    )
    if failures > 0:
        raise Error(String("gbdt_step_cost_check: ") + String(failures) + " failures")
    print("gbdt_step_cost_check: PASS")
