"""Print the depthwise stage ladder, and localize a bitwise divergence.

    pixi run check-depthwise-digest

NO CATBOOST COUNTERPART: a probe, so `mojo_only/`.

WHAT IT IS FOR. `depthwise_check` claim 6 asks whether two configurations
build the same model and answers yes or no. When the answer is NO -- and the
answer WILL be no the first time this port runs on a backend that honours
denormals, or fuses differently, or has float threadgroup atomics -- the next
question is *starting where*, and a model diff cannot answer it. Every stage
of the fit emits a digest line here; the first line that differs is the
stage, and everything after it is a consequence.

TWO WAYS TO USE IT.

**Across two machines**, which is the case this exists for:

    # Apple
    pixi run check-depthwise-digest > /tmp/dw-metal.txt
    # CUDA column, via tools/remote_gpu.sh
    pixi run check-depthwise-digest > /tmp/dw-cuda.txt

    diff /tmp/dw-metal.txt /tmp/dw-cuda.txt | head -4

**In one process**, which is what this file also does unprompted: it grows
the same tree at this device's real core count and at 108 (an A100's) and
reports the first differing tag. The core count is the only
machine-dependent input the algorithm has, so this is the cross-GPU question
asked locally, and `depthwise_check` claim 6 is the same question as a GATE
rather than as a report.

READING THE RESULT. The tag names the stage:

    d2.hist.scanned   the histogram diverged at level 2. Downstream stages
                      differ as a CONSEQUENCE and say nothing.
    d2.best.gain      the device agreed, the HOST reduce did not -- look at
                      `best_split_properties_less` and at the single sign
                      conversion at its call site, not at a kernel.
    d2.rowindex       arithmetic agreed, row ORDER did not -- the stable
                      partition.
    model.nodes       every device stage AND every host reduce agreed and
                      the MODEL differs -- the path fold or the pre-order
                      flatten, both pure host code.

FIRST HOUR ON A NEW BACKEND, in order: `check-ieee-arith` (denormals and
contraction, IDENTITY_PATHS rows 9 and 10), then `check-depthwise`, then
this. Running this first tells you a stage disagrees without telling you
whether the arithmetic was ever going to agree.

WHAT THIS IS NOT. It is not a benchmark and must never be one: it drains and
copies at every stage. It is not a correctness gate either -- `check-depthwise`
is. It is the thing you run when a gate has already gone red.
"""

from max.gpu.host import DeviceContext

from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    TDepthwiseWorkspace,
    fit_depthwise_tree,
)
from mojo_only.depthwise_check import Fixture, default_options
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mojo_only.stage_digest import TStageDigest, first_difference


def run_ladder(
    mut fx: Fixture, scope: String, sm_count_override: Int, quiet: Bool
) raises -> TStageDigest:
    """One fit with the ladder on. Returns the digest so it can be compared.

    The fixture is RESET first. Growth permutes `stats` and `row_index` in
    place -- their boosting loop re-supplies both per tree -- and a probe
    that compares two fits without resetting is comparing the second fit
    against a permuted input. That mistake cost this lane an hour on
    2026-08-22 and produced three wrong hypotheses before it was found; see
    `Fixture.pristine_stats`.
    """
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    var dg = List[TStageDigest]()
    dg.append(TStageDigest(fx.ctx.copy(), True, scope, quiet=quiet))
    var model = fit_depthwise_tree(
        fx.ctx, fx.n_rows, fx.folds, default_options(4),
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, dg,
        sm_count_override=sm_count_override,
    )
    _ = model^
    _ = ws^
    _ = dws^
    return dg.pop()


def main() raises:
    var mode = String(
        "IDENTICAL"
    ) if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else String("FAST")
    print("# depthwise stage ladder, numeric mode", mode)
    print(
        "# 4096 rows, 8 binary + 4 half-byte + 4 one-byte, hashed bins,"
        " depth 4"
    )
    print(
        "# tag format: #<scope>/<stage> <fnv1a64 of the raw bits>"
        " n=<elements hashed>"
    )

    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())

    # The ladder for THIS machine, printed. Redirect and diff against
    # another column's.
    var here = run_ladder(fx, String("dw"), -1, False)

    # THE LADDER MUST HAVE BEEN WALKED. A run that emits nothing and a run
    # whose stages all agree produce the same empty diff, and only this
    # number tells them apart. It is the `reached-but-inert` rule applied to
    # a diagnostic: a localizer that localizes nothing is worse than none,
    # because it reads as a clean bill of health.
    if here.count < 10:
        raise Error(
            String("the ladder emitted only ")
            + String(here.count)
            + " tags; a depth-4 fit walks dozens. The digest is disabled or"
            " the stages are not calling it, and an empty ladder reads as"
            " agreement."
        )

    # ================= THE CONTROL, AND IT RUNS FIRST =================
    # THE SAME CONFIGURATION TWICE. A ladder can only be used to compare two
    # configurations if it agrees with ITSELF, and on a nondeterministic
    # backend it does not: under FAST, a family whose flush is a float
    # `atomicAdd` gives a different histogram every run on the same machine.
    #
    # Without this control, run-to-run noise is indistinguishable from
    # machine dependence, and the probe would confidently name a stage that
    # simply is not reproducible. This lane has already been burned once by
    # a differential in which the thing under test was not the only thing
    # that differed (see `Fixture.pristine_stats`); this is that lesson
    # applied to the diagnostic itself.
    # ==================================================================
    var again = run_ladder(fx, String("dw"), -1, True)
    var self_diff = first_difference(here, again)

    print("")
    print("# ---- control: same configuration, twice ----")
    if self_diff.byte_length() == 0:
        print(
            "# REPRODUCIBLE:", here.count,
            "stages agree with themselves [" + mode + "]"
        )
    else:
        print("# NOT REPRODUCIBLE run to run. First unstable stage:")
        print("#   " + self_diff)
        print(
            "# every comparison below is unusable until this is clean."
            " Under FAST that is expected on a backend with float"
            " threadgroup atomics; build IDENTICAL and re-run."
        )

    # The same tree as a 108-SM machine would schedule it, quietly, then the
    # first tag on which the two disagree.
    var a100 = run_ladder(fx, String("dw"), 108, True)

    print("")
    print("# ---- in-process localization: this device vs 108 SMs ----")
    if self_diff.byte_length() != 0:
        print(
            "# SKIPPED: the control above says this build is not"
            " reproducible run to run, so a difference here would not mean"
            " machine dependence."
        )
        return
    var diff = first_difference(here, a100)
    if diff.byte_length() == 0:
        print(
            "# no divergence:", here.count,
            "stages agree bit for bit at both core counts [" + mode + "]"
        )
    else:
        print("# FIRST DIVERGING STAGE:")
        print("#   " + diff)
        print(
            "# everything after that tag is a consequence. Read the tag"
            " against the table in this file's docstring."
        )
