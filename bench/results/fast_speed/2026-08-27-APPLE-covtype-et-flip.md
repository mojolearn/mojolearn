# covtype, 2026-08-27: the et accuracy defect is fixed and the lane flips ahead of sklearn

MacBook Pro M4, FAST both sides, one alternating window per lane, `nice 19`,
three timed rounds after one warm-up, `MOJOLEARN_SPEED_ROUNDS=3`. Ours on
the M4 GPU, sklearn CPU (the only arm it has here). Logs in the session
scratchpad; the et rows were measured at `integrate/et-accuracy` f732d11
immediately before it merged to main as 7ff90bf.

## et, covtype 522,911 x 54, 7-class -- DEVIATIONS 463-465 land

| arm | median ms | accuracy |
|---|---|---|
| **ours (463-465)** | **9,843.1** | **0.679592** |
| sklearn-et-cpu | 10,454.7 | 0.676804 |

Yesterday (2026-08-26-APPLE-trees.md) this lane was 1.04x slower at
accuracy 0.6701 against sklearn's 0.6768, and the deficit's shape was the
audit's exact prediction: exact ties awarded to the highest column id, on
covtype the 44 one-hot columns over the 10 informative ones, plus a
selection hash with no avalanche. With the keyed tie-break (463), the
fmix32 finalizer (464) and the salted threshold keys (465): **accuracy
0.6796, AHEAD of sklearn, and 1.06x faster intra-window.** Tie-rate
measured on a covtype fit (`-D MOJOLEARN_ET_TIE_STATS=1`): 21,000 exact
ties in 484,476 node decisions, 4.33% -- the mass the old rule was
systematically misassigning.

The 2026-08-26 parking event is RETRACTED as an algorithmic break: the
build-gate smoke failed because a fixed-seed, 3-tree, 0.9-bar training
accuracy check fails 4 of 20 seeds on a CORRECT ExtraTrees (measured;
converges to 1.0000 by 100 trees), and DEVIATION 465 legitimately
re-rolled which forest seed 7 draws. The smoke now runs 10 trees.

## rf, covtype, fresh window at HEAD (post rf-port 0a8ba2f)

| arm | median ms | accuracy |
|---|---|---|
| ours | 23,505.2 | 0.707750 |
| sklearn-rf-cpu | 17,400.6 | 0.709506 |

1.35x slower intra-window (was 2.16x yesterday) -- but NOT claimed as the
rf-port's win: today's ambient load hit the CPU arm far harder than the GPU
arm (sklearn 9.1s -> 17.4s day over day, ours 19.7 -> 23.5), so the two
windows do not compare. What stands: accuracy parity, and covtype remains
rf's weak dataset on this box while 1M HIGGS is a 3.13x win.

## Why covtype rf is slow, with a number: the launch census

One instrumented run (`RF_LAUNCH_LOG`, warm-up + 1 round = 2 fits) logged
**214,486 enqueued operations, ~107k per covtype fit**:

| op | per fit (~) |
|---|---|
| xfer_args_upload | 20,700 |
| xfer_workload_info / xfer_work_items / xfer_splits_download | 11,100 each |
| sample_features / init_split / histogram_binned / find_best_splits | 9,500 each |

~38k compute launches plus ~57k transfers, one host round-trip
(`splits_download`) per level-cycle. cuML enqueues the same logical
schedule on CUDA streams at single-digit microseconds per op; the M4 pays
far more per enqueue, which is why the deep-narrow covtype (many cycles,
small kernels) hurts more than fat higgs rungs (few cycles, big kernels),
and why the same code is 3.42x behind cuML on covtype but only ~2.1x on
higgs (2026-08-25-nvidia-trees-PARTIAL.md). The improvement campaign this
prices: batch the K=4 in-flight trees' splits downloads into one readback
per cycle, and coalesce the ~20.7k args uploads. Neither is a today-edit;
both now have a denominator.
