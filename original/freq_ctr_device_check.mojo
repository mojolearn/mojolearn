# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate the DEVICE FeatureFreq calcer against the host reference, per cell.

    pixi run check-freq-ctr-device

`TWeightedBinFreqCalcerGpu` + `compute_simple_ctrs_device`
(`gbdt/ctrs/ctr_calcers.mojo`) against the host driver
`compute_simple_ctrs`, which `original/ctr_check.mojo` already gates
against PLANTED counts and an independent O(n^2) tally. This check closes
`PORTING.md` deviation 52: the last host-side piece of the CTR block now
has a device arm, and this file is what makes that a measurement.

## Why BIT-equality and not a tolerance

The trivial weights their GPU learner actually builds are 1.0 per row
(`ctr_helper.h:19-21`), and integer-valued float sums below 2^24 are
exact in ANY reduction order -- so the device's segmented-reduce order
cannot differ from the host's sequential order by even an ulp, and the
final divide runs on identical inputs. Equality is therefore the
contract, and a tolerance would only hide a defect.

## Sections

1. CARDINALITY SWEEP at n = 4096, hashed scattered codes, cards
   {2, 3, 15, 16, 17, 254, 255, 1000} -- across every packing-policy
   boundary the one-hot sweep established matters. Device vs host per
   cell, plus an independent per-row tally (count/(n+1)) that neither
   driver computes the same way.
2. TWO PRIORS in one visit ({0.0, 1} -- their GetDefaultPriors value --
   and {0.5, 2}), exercising the grouped loop's repeat-divide.
3. THE `partCount == size` ARM: n = 16 rows over 15 categories puts
   segments + fake == rows and takes the dispatcher's FillBuffer +
   skip-suffix path (`partitions.cu:165-172`), which no other fixture
   reaches. (The mid-array backfill walk, for the record, is UNREACHABLE
   from this calcer by construction: scanned segment ids are consecutive,
   so only the leading wrap and the suffix arms ever run. Stated so
   nobody plants an impossible fixture to chase the branch.)
4. SCALE: n = 100,000 at card 137 -- multi-block scan, multi-block
   reduce, and a bin count big enough that the fake-bin suffix write
   matters.
5. SABOTAGE, one per the check-must-fail rule: the device arm is run on
   an input with ONE code changed, and the comparison must go red on the
   affected categories' rows and ONLY there. A comparator that stays
   green, or one that flags rows of untouched categories, fails the
   check about the check.
"""

from max.gpu.host import DeviceContext

from gbdt.ctrs.ctr import CTR_FEATURE_FREQ, TCtrConfig, TPrior
from gbdt.ctrs.ctr_calcers import (
    compute_simple_ctrs,
    compute_simple_ctrs_device,
)


def _hashed_codes(n: Int, card: Int, salt: UInt64) -> List[UInt32]:
    var out = List[UInt32]()
    for i in range(n):
        var h = (UInt64(i) + salt) * 0x9E3779B97F4A7C15
        h ^= h >> 29
        h *= 0xBF58476D1CE4E5B9
        h ^= h >> 32
        out.append(UInt32(Int(h % UInt64(card))))
    return out^


def _configs() -> List[TCtrConfig]:
    var out = List[TCtrConfig]()
    # GetDefaultPriors(FeatureFreq) is the single {0.0, 1}
    # (cat_feature_options.cpp:127-129)
    out.append(TCtrConfig(CTR_FEATURE_FREQ, TPrior(0.0, 1.0), 0, 0))
    out.append(TCtrConfig(CTR_FEATURE_FREQ, TPrior(0.5, 2.0), 0, 0))
    return out^


def _compare(
    name: String,
    ctx: DeviceContext,
    codes: List[UInt32],
    card: Int,
) raises -> Int:
    """Runs both drivers and the independent tally; returns mismatches."""
    var n = len(codes)
    var configs = _configs()
    var host_cols = compute_simple_ctrs(
        codes, card, configs, List[UInt8](), False
    )
    var dev_cols = compute_simple_ctrs_device(ctx, codes, card, configs)

    var counts = List[Int]()
    for _ in range(card):
        counts.append(0)
    for i in range(n):
        counts[Int(codes[i])] += 1

    var wrong = 0
    for c in range(len(configs)):
        var prior = configs[c].numerator_shift()
        var prior_obs = configs[c].denumerator_shift()
        for i in range(n):
            var expected = (
                Float32(counts[Int(codes[i])]) + prior
            ) / (Float32(n) + prior_obs)
            if dev_cols[c][i] != host_cols[c][i]:
                wrong += 1
            elif dev_cols[c][i] != expected:
                # host and device agree but both differ from the tally:
                # that is a shared defect, and it counts
                wrong += 1
    print(
        "  ", name, ": n", n, "card", card, "--", wrong,
        "wrong of", n * len(configs), "cells x 2 configs",
    )
    return wrong


def main() raises:
    var ctx = DeviceContext()
    var total = 0

    print("freq ctr device check: device driver vs host driver vs tally")
    var cards = [2, 3, 15, 16, 17, 254, 255, 1000]
    for ci in range(len(cards)):
        total += _compare(
            "card sweep", ctx,
            _hashed_codes(4096, cards[ci], UInt64(11 + ci)), cards[ci],
        )

    # partCount == size: 16 rows, 15 categories (one duplicate), so
    # segments + fake == rows and the FillBuffer skip-suffix arm runs
    var pc = List[UInt32]()
    for i in range(15):
        pc.append(UInt32(i))
    pc.append(UInt32(7))
    total += _compare("partCount==size", ctx, pc, 15)

    total += _compare(
        "scale", ctx, _hashed_codes(100_000, 137, UInt64(99)), 137
    )

    if total != 0:
        raise Error("freq ctr device check FAILED")

    # SABOTAGE: one code changed on the device arm only; the red must land
    # exactly on the two affected categories' rows.
    var clean = _hashed_codes(4096, 17, UInt64(5))
    var broken = clean.copy()
    var old_cat = broken[0]
    var new_cat = UInt32((Int(old_cat) + 1) % 17)
    broken[0] = new_cat
    var configs = _configs()
    var host_cols = compute_simple_ctrs(
        clean, 17, configs, List[UInt8](), False
    )
    var dev_cols = compute_simple_ctrs_device(ctx, broken, 17, configs)
    var moved = 0
    var moved_elsewhere = 0
    for c in range(len(configs)):
        for i in range(4096):
            if dev_cols[c][i] != host_cols[c][i]:
                var cat = broken[i]
                if cat == old_cat or cat == new_cat:
                    moved += 1
                else:
                    moved_elsewhere += 1
    print(
        "  sabotage: ", moved, "cells moved in the affected categories,",
        moved_elsewhere, "elsewhere",
    )
    if moved == 0:
        raise Error(
            "sabotage did not move the comparison; the check cannot fail"
            " and is not evidence"
        )
    if moved_elsewhere != 0:
        raise Error("sabotage moved rows of untouched categories")

    print("freq ctr device check OK")
