# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
from core.identity_trace import IdentityTrace, first_divergence
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def run_ladder(
    mut fx: Fixture, path: String, sm_count_override: Int
) raises -> Int:
    """One fit with `core/identity_trace.mojo` writing to `path`.

    Returns the record count so the caller can assert the ladder was
    actually walked.

    THE FIXTURE IS RESET FIRST. Growth permutes `stats` and `row_index` in
    place -- their boosting loop re-supplies both per tree -- so a probe
    that compares two fits without resetting is comparing the second fit
    against a permuted input. That mistake cost this lane an hour on
    2026-08-22 and produced three wrong hypotheses before it was found; see
    `Fixture.pristine_stats`.

    `to_path` and not `IdentityTrace()`: this probe must not depend on
    whether the operator happens to have `MOJOLEARN_IDENTITY_TRACE`
    exported, and `to_path` truncates, so re-running does not read back the
    previous run concatenated with this one.
    """
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    var tr = IdentityTrace.to_path(path)
    tr.header(
        String("depthwise, 4096 rows, 8 binary + 4 half-byte + 4 one-byte,")
        + " hashed bins, depth 4, sm_count_override="
        + String(sm_count_override)
    )
    var model = fit_depthwise_tree(
        fx.ctx, fx.n_rows, fx.folds, default_options(4),
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, tr,
        sm_count_override=sm_count_override,
    )
    _ = model^
    _ = ws^
    _ = dws^
    return tr.seq


def main() raises:
    var mode = String(
        "IDENTICAL"
    ) if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else String("FAST")

    # Plain `/tmp` paths, the same convention `identity_trace_check` uses,
    # so this runs on a bare machine with no directory to create first.
    # `to_path` truncates, so a re-run does not read back its own last run.
    var p_here = String("/tmp/mojolearn_dw_here.trace")
    var p_again = String("/tmp/mojolearn_dw_again.trace")
    var p_a100 = String("/tmp/mojolearn_dw_a100.trace")

    print("depthwise identity trace, numeric mode", mode)

    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())

    var n_here = run_ladder(fx, p_here, -1)

    # THE LADDER MUST HAVE BEEN WALKED. A run that emits nothing and a run
    # whose stages all agree produce the same empty diff, and only this
    # number tells them apart. `reached-but-inert` applied to a diagnostic:
    # a localizer that localizes nothing reads as a clean bill of health.
    if n_here < 10:
        raise Error(
            String("the trace emitted only ")
            + String(n_here)
            + " records; a depth-4 fit walks dozens. The trace is disabled"
            " or the stages are not calling it, and an empty trace reads as"
            " agreement."
        )

    # ================= THE CONTROL, AND IT RUNS FIRST =================
    # THE SAME CONFIGURATION TWICE. A trace can only compare two
    # configurations if it agrees with ITSELF, and on a nondeterministic
    # backend it does not: under FAST, a family whose flush is a float
    # `atomicAdd` gives a different histogram every run on the same machine.
    # Without this, run-to-run noise is indistinguishable from machine
    # dependence and the probe would confidently name a stage that simply is
    # not reproducible.
    _ = run_ladder(fx, p_again, -1)
    var self_diff = first_divergence(p_here, p_again)

    print("")
    print("-- control: same configuration, twice --")
    if self_diff.byte_length() == 0:
        print(
            "   REPRODUCIBLE:", n_here,
            "records agree with themselves [" + mode + "]",
        )
    else:
        print("   NOT REPRODUCIBLE run to run. First unstable stage:")
        print("   " + self_diff)
        print(
            "   every comparison below is unusable until this is clean."
            " Under FAST that is expected on a backend with float"
            " threadgroup atomics; build IDENTICAL and re-run."
        )

    # The same tree as a 108-SM machine would schedule it. The core count is
    # the only machine-dependent input this algorithm has, so this is the
    # cross-GPU question asked locally.
    _ = run_ladder(fx, p_a100, 108)

    print("")
    print("-- localization: this device vs 108 SMs --")
    if self_diff.byte_length() != 0:
        print(
            "   SKIPPED: the control says this build is not reproducible"
            " run to run, so a difference here would not mean machine"
            " dependence."
        )
    else:
        var diff = first_divergence(p_here, p_a100)
        if diff.byte_length() == 0:
            print(
                "   no divergence:", n_here,
                "records agree bit for bit at both core counts ["
                + mode + "]",
            )
        else:
            print("   FIRST DIVERGING STAGE:")
            print("   " + diff)
            print(
                "   everything after that tag is a consequence. Read it"
                " against the table in this file's docstring."
            )

    print("")
    print("-- artifacts for tools/identity_trace_diff.py --")
    print("   " + p_here)
    print("   " + p_a100)
    print(
        "   tools/identity_trace_diff.py " + p_here + " " + p_a100
        + "   # aligns on tag sequences and classifies each differing cell"
    )
