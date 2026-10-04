# MCD next: batched GEMM launches and wide d (w2-mcd2)

Branch lane/apple-fast-w2-mcd2, base main b2b1c22bc. Both defines are opt-in,
FAST + Apple only, on top of the MCD_BATCH_MMA default (no effect with
MOJOLEARN_MCD_BATCH_MMA_OFF). Binding x_decomp. Code only; nothing compiled,
run or timed by the lane.

## What dominates the 3.6 s (source analysis, not measured)

Main enqueues three matrix-unit GEMMs per candidate per C-step (covariance,
weighted Gram, Mahalanobis product) in host loops over every candidate of
the phase, active or not. Taxi (n100000, d11): phase A 3,330 candidates x
3 steps, phase B 3,330 candidates x up to 31 steps, phase C 10 x up to 31.
Phase B alone is up to ~310,000 GEMM launches of 11 x 11 work. At the
measured Metal enqueue cost (~10-20 us per launch) that is seconds; the rest
of a step is ~10 phase-wide launches and one host read of the stop word.
Hypothesis: launch count, not arithmetic, is most of the 3.6 s.

Wide d: fast_mcd_fast refused d > 64, so istella (d220) ran fast_mcd_dev:
per candidate and per C-step, kit launches plus host syncs, and a 220-wide
round-robin eigh with a sync per sweep (219 rounds x 2 launches each sweep)
for every pinvh. 3,330 candidates in phase A and phase B each take at least
one such pinvh (the istella covariances have constant columns, so the first
determinant is -inf), which explains > 20 min.

## MOJOLEARN_MCD_BMMA

x_decomp/mcd_bmma.mojo: main's tile kernel (afn_gemm_mma_kernel at
AFN_TILE_SQUARE, statement for statement) with the candidate from
block_idx.z and per-candidate operand offsets. The launcher computes tiles,
splits and K chunk exactly as _launch_gemm_mma does for one candidate, so
each output cell has main's matrix-unit sums over the same K windows and
split ranges; split partials still meet in f32 atomics (order free as on
main). Gate words: covariance = active, weighted Gram = mm_ran, distance =
active (all candidates for the final distances), the same masks main's
publication kernels apply; gated-off candidates launch nothing, and
mc_center_kernel[SKIP] no longer zeroes their operands. Expected bits: main's
up to split-K atomic order. Launches per step drop from 3 x nc to 3 (+1
zero launch when the covariance splits).

## MOJOLEARN_MCD_WIDE

Allows 64 < d <= 256 in the batched search. Adds mf_det_wide_kernel (one
block per candidate: block-reduced first-largest pivot, parallel swap /
factors / trailing update with _logdet's COMPAT cells, one-thread ascending
log sum, identical C-step control) and mc_pinvh_kernel[MMA, 256] (eigen
tables of 256). Everything else is the narrow MMA path. Memory at istella
board size (phase B nc3330, r1500, d220): mm_x and mm_y 4.4 GB each, eight
nc d^2 buffers 0.65 GB each, phase A buffers still live: about 20 GB peak.
Arm A for istella has no time (does not finish), so timing is B only, gated
on a capped quality PASS against main.

## Quality gate (fixed before results)

tools/mcd_compat_quality.py compare, unchanged: location, covariance,
precision, raw location/covariance, query and training distances relative
difference <= .01; flags, support and raw support Jaccard >= .99; equal raw
rank; EE adds offset_ and decision_function at 1%. These are the gates that
accepted MCD_BATCH_MMA; BMMA keeps main's per-cell arithmetic, so a pass is
expected near 1e-7. WIDE changes istella's arithmetic (kit eigh/LU -> batched
cells) and is held to the same gates on istella cap3000.
