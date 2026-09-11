# Pointwise searcher speed after DEVIATION 2624 (lane/pointwise-speed, 2026-09-11 night)

One NVIDIA H100 80GB HBM3 on RunPod (pod `eqxtzdcpctpnkh`, driver 580.126.09),
IDENTICAL tier, both trees built and timed ON THAT POD. Big logs live outside
the repo in `~/mojolearn-evidence/pointwise-speed-2026-09-11/`.

The question the lane was given: win back the time the opt-in pointwise
searcher (`use_pointwise_searcher=True`, symmetric trees) lost to DEVIATION
2624, **keeping 2624's bits**.

## 1. DEVIATION 2669 is impossible, and this is the proof

2669 was: split the document axis again, give every document block a private
scratch slot, fold the slots in a fixed block order, and land exactly on the
multiplier-1 cells.

At `M = 1` an accumulator slot is one left-to-right float sum over the
documents in stride order. At `M` blocks, block `b` sums the documents whose
stride index is `b` mod `M`, starting from zero, and the fold adds those
partials. In float32 with `x = (1e8, 1, -1e8, 1)`:

    sequential   ((1e8 + 1) + -1e8) + 1 = 1     (1e8 + 1 rounds back to 1e8)
    M = 2 fold   (1e8 + -1e8) + (1 + 1) = 2

so the two differ on ordinary data, and no order of folding the partials
repairs it, because each partial has already rounded without the other
blocks' documents. The only exact schedule carries each block's running state
into the next, which serializes the blocks and wins nothing. After tree 0 the
Logloss gradients are real-valued (that is exactly what 2624's drift was), so
there is no data-independent exact case to specialize on.

**A same-bits win therefore cannot come from splitting the document axis.**

## 2. DEVIATION 2670 (opt-in, NOT flipped): pinned multiplier, private slots

`-D MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS=1`. The ordered tiers split the
document axis at `EstimateBlockPerFeatureMultiplier` evaluated at a PINNED SM
count (`PW_2670_PINNED_SM = 128`, kernel matrix row
`pointwise_private_doc_slots_sm_for`) instead of the device's own; each
document block stores its partial into its own slot of a zeroed scratch
(`pw_private_doc_slot`), and `pw_fold_doc_slots_kernel` adds slots 0..M-1 onto
`binSums` in block order. No atomics anywhere, and the multiplier no longer
follows `MULTIPROCESSOR_COUNT`.

New bits against 2624, one schedule on every vendor and at every launch
geometry.

### Gates on the H100 (2670 build, sha256 `dadf2458b9b0…`; main `7489c4bc03ad…`)

| gate | result |
|---|---|
| `check-pointwise-identical-multiplier-2670` G1 | ok, 896 grids, multiplier independent of sm at every one, 832 of them split the document axis |
| `check-pointwise-identical-multiplier-2670` G2 | ok, 55,290 cells bit-equal across sm 1/16/132/4096 and three repeats, three policies, full and partial pass |
| `check-pointwise-dispatch-2670` F1..F7 | pass (integer stats against the host tally; F2 has the M = 4 run identical to the M = 1 run cell for cell, which is what sees a misfiled slot; F3 covers the partial pass) |
| `check-pointwise-identical-multiplier` on main, same pod | PASS (the 2624 control) |

### Synthetic fixture model hashes, 1M rows, 100 trees, 3 repeats each

| fixture | main (2624) | 2670 | repeats |
|---|---|---|---|
| binary | 5dc6a9a1c1657d39 | **9551119e7f1120ed** | 3 of 3 stable, both trees |
| halfbyte | 752e10a37cba36ca | 752e10a37cba36ca | 3 of 3 stable |
| onebyte | 16bf618caba014bb | 16bf618caba014bb | 3 of 3 stable |
| mixed | b753f4d9116a4935 | b753f4d9116a4935 | 3 of 3 stable |
| wide220 | 8c5ac18e8e5547e9 | 8c5ac18e8e5547e9 | 3 of 3 stable |
| greedy arm (all five) | unchanged | unchanged | 3 of 3 stable |

**Equal model hashes are NOT equal histograms, and the identity traces say
so.** Twenty-tree traced fits of the same fixtures, main against 2670
(`tools/identity_trace_diff.py`, logs `trace*_mojolearn*.log`):

| fixture | first divergence | verdict |
|---|---|---|
| onebyte | none, 182 of 182 stages | IDENTICAL |
| halfbyte | `tree001.depth00.hist.HalfByteFeatures` | DIVERGENT |
| binary | `tree001.depth00.hist.BinaryFeatures` | DIVERGENT |
| mixed | `tree001.depth00.hist.BinaryFeatures` | DIVERGENT |
| wide220 | `tree001.depth00.hist.BinaryFeatures` | DIVERGENT |

So 2670 moves the histogram from the first tree wherever a FLOAT accumulator
builds it (the binary and half-byte policies), and moves nothing at all where
the accumulator is Int32 fixed point (the one-byte family, DEVIATION 93: its
cells are integer multiples of `1 / fixed_scale`, so they fold exactly in any
order). Where the 100-tree model hash nevertheless matches 2624
(halfbyte, mixed, wide220), the reason is that no near-tied split flipped, not
that the bits agreed. 2670 is a bit-moving change and is gated as one.

## 3. Speed

Both trees built on the same pod and timed INTERLEAVED at the process level:
three outer rounds, the tree order swapped every round, three timed rounds per
process, `MOJOLEARN_SPEED_ROUNDS=3`, 1,000,000 rows. Each process also times
the GREEDY symmetric arm (`ours`), whose code neither deviation touches, as
the within-pod drift control. Medians over the nine rounds.

| dataset | arm | main (2624) ms | 2670 ms | after/before | logloss main | logloss 2670 | hash main | hash 2670 |
|---|---|---|---|---|---|---|---|---|
| taxi 1M x 16 | pointwise (`ours-ab`) | 1803.3 | **795.4** | **0.441** | 0.525925 | 0.525668 | c56ab803f4d1e84a | 97222a45020a166c |
| taxi 1M x 16 | greedy control (`ours`) | 309.3 | 308.9 | 0.999 | 0.525735 | 0.525735 | 90c3558501933f47 | 90c3558501933f47 |
| Istella-S 1M x 220 | pointwise (`ours-ab`) | ISTELLA_POINTWISE | | | | | | |
| Istella-S 1M x 220 | greedy control (`ours`) | ISTELLA_GREEDY | | | | | | |

On taxi the opt-in pointwise arm takes 0.441 of its 2624 time, which is the
pre-2624 speed (the 2624 merge recorded about 810 ms before and 1805 ms
after) with the fold added, and taxi's logloss is 0.525668 against 2624's
0.525925 (the pre-2624 value, since 2670's taxi model hash is the pre-2624
hash 97222a45020a166c). The greedy control moved 0.1%, so the pod was not
drifting under the pointwise rows.

### The pinned SM constant is not what the win rests on

`PW_2670_PINNED_SM` is a NUMERIC constant I chose, so it was measured rather
than argued. A third tree built from the 2670 tree with the constant set to 32
instead of 128, taxi 1M, the two interleaved in one job on the same pod (two
outer rounds, three timed rounds each):

| pinned SM | taxi pointwise ms, per round | model hash |
|---|---|---|
| 128 | 787.4, 785.7, 787.9, 793.5, 789.5, 794.2 | 97222a45020a166c |
| 32 | 798.5, 801.2, 803.4, 837.1, 794.7, 805.0 | 97222a45020a166c |

32 is about 1 to 2 percent slower on this shape and gives the SAME model
hash, so on taxi the constant moves the schedule inside 2670's bits rather
than the bits themselves (a shape whose multiplier ladder lands differently
at the two constants would move them; that is not this shape). The win is
not an artifact of picking 128.

## 3b. RUN OWED before 2670 could ever flip

2670 moves pointwise IDENTICAL bits (section 2), so the orchestrator's two
gates are owed BEFORE any flip. Both run against this branch
(`lane/pointwise-speed`), built with the define.

On the Apple M4, from a checkout of this branch:

    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
      MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS=1" \
      bash bindings/build_gbdt.sh
    pixi run check-pointwise-identical-multiplier-2670
    pixi run check-pointwise-dispatch-2670
    # then the five synthetic fixtures, which no Apple box has ever run for
    # the pointwise arm (the 2624 merge left that owed too):
    python3 ~/mojolearn-evidence/pointwise-hash-drift-2026-09-11/ptwdrift_repro.py \
      --fixtures binary,halfbyte,onebyte,mixed,wide220 --repeats 3

On an AMD box (Hot Aisle MI300X first, per the box order), the same three
commands. The pass condition is the gates green AND the five fixture hashes
equal to the H100's 2670 column in section 2, which is what makes the new
bits cross-vendor bits rather than NVIDIA's.

## 4. What is NOT done

- 2670 is opt-in and default OFF. Flipping it needs the Apple M4 and an AMD
  box, because it moves pointwise IDENTICAL model hashes.
- Same-bits ideas at multiplier 1 that were NOT tried: the half-byte
  accumulator takes 16 block-wide `barrier()` calls per point while its shared
  writes stay inside one warp slice, so a warp-scoped turn-taking sync is a
  scheduling-only candidate. It is the code path of the AMD identity bug
  (DEVIATION 2600) and Apple's `syncwarp()` carries no memory fence, so it
  needs its own cross-vendor leg.
