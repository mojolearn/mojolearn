# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `gbdt/methods/histograms_helper.mojo`.

`TComputeHistogramsHelper`'s state machine, which decides ONE bit that
matters more than anything else in the histogram family:

    BuildFromScratch is the FULL-PASS flag.

True computes every part of the level. False computes only the SMALLER child
of each sibling pair and files it under the right one, leaving a subtraction
to recover the other. Get it wrong and nothing crashes: the histogram is
right at one depth and quietly wrong after it.

The machine has no GPU in it. Everything it decides is a pure function of the
sequence of depths it is called with, which is why this check needs no
device and can enumerate transitions a fit would take days to reach.

GATES -- each is a distinct TRANSITION, not a distinct assertion:

  F1  a fresh tree, depths 0..5: depth 0 full, every deeper level partial.
      This is the only sequence a healthy fit produces.
  F2  a SECOND tree on the same helper. Depth 0 must be full again. Their
      helper is constructed per tree upstream, so this is the transition a
      port that reuses helpers depends on and theirs never exercises.
  F3  a SKIPPED level, 0 then 2. Must be full: the counter advanced to 1 and
      the subsets say 2, so they disagree.
  F4  a RE-ENTRY at the same depth, 0 then 0. Must be full, same reason.
  F5  a policy with NO FEATURES does not clear the flag. Their assignment
      `BuildFromScratch = false` sits inside
      `if (DataSet->GetGridSize(Policy))`, so a grid with no features leaves
      the flag set and the next level that does have features rebuilds. A
      port that cleared it unconditionally would compute a partial pass onto
      histograms nothing ever filled.
  F6  the two SIZES are different expressions. The allocation uses
      `MaxDepth` and the view uses `CurrentBit`; at depth 0 of a depth-6
      tree they differ by 64x. Sizing the allocation with `CurrentBit` works
      at every level except the ones that grow.

A NOTE ON WHAT IS *NOT* GATED, because saying so is part of the result. The
`|| CurrentBit == 0` clause in their condition is REDUNDANT with the
`BuildFromScratch = true` field initialiser, and this check demonstrates it
rather than hiding it: `CurrentBit` reaches 0 only by incrementing from its
initial -1, and at that moment the flag is already true. Deleting the clause
leaves every gate below green -- measured, not assumed. It is transcribed
because it is theirs and because it is the clause that still holds if a
future caller ever resets the counter without resetting the flag.
"""

from gbdt.methods.histograms_helper import (
    POLICY_BINARY,
    POLICY_ONE_BYTE,
    ComputeHistogramsHelper,
)


def main() raises:
    var failures = 0

    # ---------------------------------------------------------------- F1
    var h = ComputeHistogramsHelper(POLICY_ONE_BYTE, 1, 6)
    var want_full: List[Bool] = [True, False, False, False, False, False]
    for d in range(6):
        var p = h.plan(d)
        h.clear_from_scratch()
        if p.build_from_scratch != want_full[d]:
            print(
                "FAIL F1: depth", d, "full-pass =", p.build_from_scratch,
                "expected", want_full[d],
            )
            failures += 1
        if p.current_bit != d:
            print("FAIL F1: depth", d, "current_bit =", p.current_bit)
            failures += 1
        if p.part_count != (1 << d):
            print("FAIL F1: depth", d, "part_count =", p.part_count)
            failures += 1
    if failures == 0:
        print("  ok   F1 -- depth 0 full, depths 1-5 partial")

    # ---------------------------------------------------------------- F2
    # the same helper, a second tree
    var p2 = h.plan(0)
    if not p2.build_from_scratch:
        print(
            "FAIL F2: a second tree's depth 0 was a PARTIAL pass. The"
            " helper would subtract against histograms belonging to the"
            " previous tree.",
        )
        failures += 1
    elif p2.current_bit != 0:
        print("FAIL F2: second tree depth 0 left current_bit =", p2.current_bit)
        failures += 1
    else:
        print("  ok   F2 -- a second tree's depth 0 rebuilds")

    # ---------------------------------------------------------------- F3
    var h3 = ComputeHistogramsHelper(POLICY_ONE_BYTE, 1, 6)
    _ = h3.plan(0)
    h3.clear_from_scratch()
    var p3 = h3.plan(2)
    if not p3.build_from_scratch:
        print("FAIL F3: a skipped level (0 -> 2) was a partial pass")
        failures += 1
    elif p3.current_bit != 2:
        print("FAIL F3: current_bit =", p3.current_bit, "expected 2")
        failures += 1
    else:
        print("  ok   F3 -- a skipped level rebuilds, and resyncs to 2")

    # ---------------------------------------------------------------- F4
    var h4 = ComputeHistogramsHelper(POLICY_ONE_BYTE, 1, 6)
    _ = h4.plan(0)
    h4.clear_from_scratch()
    var p4 = h4.plan(0)
    if not p4.build_from_scratch:
        print("FAIL F4: a re-entry at the same depth was a partial pass")
        failures += 1
    else:
        print("  ok   F4 -- a re-entry at the same depth rebuilds")

    # ---------------------------------------------------------------- F5
    # an empty grid: `plan` runs, the launch does not, so the flag is NOT
    # cleared and the next level must still rebuild
    var h5 = ComputeHistogramsHelper(POLICY_BINARY, 1, 6)
    var p5a = h5.plan(0)
    # no clear_from_scratch() -- this policy had no features
    var p5b = h5.plan(1)
    if not p5a.build_from_scratch:
        print("FAIL F5: depth 0 was not a full pass")
        failures += 1
    elif not p5b.build_from_scratch:
        print(
            "FAIL F5: depth 1 was a PARTIAL pass after a level that never"
            " launched. It would subtract against histograms nothing"
            " filled. Their `BuildFromScratch = false` is inside"
            " `if (DataSet->GetGridSize(Policy))`.",
        )
        failures += 1
    else:
        print(
            "  ok   F5 -- an empty grid does not clear the flag; the next"
            " level still rebuilds",
        )

    # ---------------------------------------------------------------- F6
    # DRIVE IT TO A DEPTH FIRST. A freshly constructed helper has
    # `current_bit == -1`, and the first version of this gate compared the
    # sizes there -- so a sabotage that sized the allocation with
    # `current_bit` evaluated `1 << -1` and TRAPPED before the assertion
    # ran. A crash is a failure signal but a useless one: it says nothing
    # about which expression was wrong. Advancing to a real depth makes the
    # sabotage produce a WRONG NUMBER instead of a trap, which the gate can
    # name.
    var h6 = ComputeHistogramsHelper(POLICY_ONE_BYTE, 1, 6)
    _ = h6.plan(0)
    h6.clear_from_scratch()
    _ = h6.plan(1)
    h6.clear_from_scratch()
    var bin_features = 300
    var view0 = h6.histogram_view_size(0, bin_features)
    var alloc = h6.histogram_alloc_size(bin_features)
    var view5 = h6.histogram_view_size(5, bin_features)
    if view0 != 1 * 1 * bin_features * 2:
        print("FAIL F6: view at bit 0 =", view0)
        failures += 1
    if alloc != 64 * 1 * bin_features * 2:
        print("FAIL F6: alloc =", alloc, "expected 64 *", bin_features, "* 2")
        failures += 1
    if view5 != 32 * 1 * bin_features * 2:
        print("FAIL F6: view at bit 5 =", view5)
        failures += 1
    if alloc <= view5:
        print(
            "FAIL F6: the allocation is not larger than the deepest view --",
            "it is being sized with CurrentBit rather than MaxDepth, which",
            "works at every level except the ones that grow.",
        )
        failures += 1
    # fold count must scale both
    var h6f = ComputeHistogramsHelper(POLICY_ONE_BYTE, 4, 6)
    if h6f.histogram_alloc_size(bin_features) != 4 * alloc:
        print("FAIL F6: fold_count does not scale the allocation")
        failures += 1
    if failures == 0:
        print(
            "  ok   F6 -- view", view0, "at bit 0 against an allocation of",
            alloc, "; fold count scales both",
        )

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("histograms helper state machine: F1-F6 pass")
