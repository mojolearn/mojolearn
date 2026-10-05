# sym-iter: per-iteration fixed cost of the pointwise SymmetricTree arm (Apple FAST)

Branch `lane/apple-fast-sym-iter`. Board lanes `gbdt-symmetric-1000` (deciding, taxi) and `gbdt-symmetric` (istella).
Every define is FAST + Apple only, default OFF, read with `is_defined` in `gbdt/methods/sym_iter_fast.mojo`; IDENTICAL
compiles main's loop. The profile the defines answer is `docs/apple-fast/notes/sym-iter.md`: main makes 9 host drains
and ~20 Metal allocations per tree outside the histogram and score math, at 1000 trees on taxi.

Request order (`sym-iter.txt`): the ALL arm and the four singles on `gbdt-symmetric-1000` taxi, then `gbdt-symmetric`
istella for ALL and the two singles expected to carry most of it (LEAF_FROM_STATS, DERIV_FUSED). If a different single
wins on taxi, swap it into the istella lines before queueing them. Quality: FSPEED-ACC must stay within FAST's run-to-run
spread on every arm; the two arms that change arithmetic (REUSE_PARTITION's fold order, LEAF_FROM_STATS's snapped
gradient) are the ones to read it on.

## `-D MOJOLEARN_SYM_BUF_ARENA` (alias `SYM_BUF_ARENA`)

Mechanism: a pool of one (`SymIterPool`, keyed on rows and depth, built on the first tree) for every buffer the loop
allocated per tree: the two split planes (`split_stat_planes` made two n_rows buffers), the bins staging (five host and
five device buffers), the leaf partitioner (`partition_from_bins` built a `DeviceLeafPartitioner` per call: seven
n_rows-sized buffers and a drain in its constructor), the oracle's five host staging buffers (through the estimation
workspace's `arena_host`), and the three buffers the structure searcher allocates for its fold arm and never reads on the
plain arm (one of them n_rows wide). The magnitudes are read into the fit's existing `h_mags` instead of a host buffer per
tree. The two drains whose only job was to keep a per-call buffer alive (`compute_bins_for_model`'s and
`partition_from_bins`' outer one) go with the buffers. Expected: allocations per tree 20 -> 0, drains 9 -> 7, and a flat
live-buffer count (Apple launch cost grows ~0.25 us per launch per live buffer). Risk: low; same kernels, same integers,
same partition; the staging is rewritten only after the tree's drains, as the estimation workspace already relies on.

## `-D MOJOLEARN_SYM_REUSE_PARTITION` (alias `SYM_REUSE_PARTITION`)

Mechanism: after the last level the searcher's subsets are already the partition the estimator rebuilds: `indices` holds
the rows grouped by leaf and `partitions` the (offset, size) per leaf. The searcher's one tail drain now also carries the
256 partition records to the host (`sym_parts_out`), and `compute_bins_for_model` (one launch, ten buffers, a drain) plus
`partition_from_bins` (an 8-bit radix sort of a million rows, ~28 launches, three drains) are skipped; the estimator gets
`subsets.indices` as its row order. A tree that stopped early (a repeated split: the structure is shorter than `max_depth`
and the subsets carry a redundant bit per repeat) takes main's path, decided on the host after the drain. Expected:
drains 9 -> 6, launches outside the search -40%. Bits: the row order inside a leaf is the searcher's (eight stable one-bit
sorts) rather than the full-key radix sort's ascending order, so the estimator's leaf sums fold in a different order: FAST
only, same rows per leaf, same leaves. Risk: low on the plain arm; the fallback covers the early-stop case.

## `-D MOJOLEARN_SYM_DERIV_FUSED` (alias `SYM_DERIV_FUSED`)

Mechanism: the next tree's gradient pass (`launch_approximate`, the fv and magnitude folds, the fv and magnitude
copies) and the score-noise std-dev reduce (`random_strength` 1.0 on the board makes it a drain per tree) are enqueued
behind this tree's `add_model_value_kernel`, inside the estimator's tail, and the one closing drain settles the cursor
update, the estimator's buffers and all of it. The loop head then skips its launches and its two drains (the magnitudes
drain and the std-dev drain) and reads `h_mags`, `h_fv` and the std-dev word that are already on the host. The loss word
is captured after the structure drain, before the tail overwrites `h_fv`. Gate: one permutation (the plain arm's one
cursor), no bootstrap, a `launch_approximate` loss (not multiclass, querywise, pairwise or YetiRank). The launches stay
separate kernels; what is fused is the command buffer and the wait, which on Apple is the cost (~0.2 ms per wait). Expected:
drains 9 -> 7. Bits: none; the same launches in the same order on the same buffers, one tree earlier in the stream. Risk:
the std-dev and magnitude values are read one drain later than main reads them, which is the drain that produced them;
`h_fv` is read from the captured word. The estimator's tail drain moves to the call site with nothing of the oracle's in
flight (its evaluation copies were settled by the walker's own drain).

## `-D MOJOLEARN_SYM_LEAF_FROM_STATS` (alias `SYM_LEAF_FROM_STATS`, implies `SYM_REUSE_PARTITION`)

Mechanism: DEVIATION 64 for the pointwise arm. Under Newton with one iteration the leaf is the single step from zero,
`sum(weight * der) / (sum(weight * der2) + l2)` over the leaf's rows at the current cursor. The searcher's partition
statistics already hold `sum(weight * der)` per leaf (the gradient plane is the search target), so one launch with one
block per leaf reduces the Hessian plane over the leaf's rows (`weight` for RMSE, `weight * p * (1 - p)` with
`cross_entropy_kernel`'s `p` for Logloss) and writes the leaf; a second launch applies `learning_rate * leaf` through the
searcher's partition; the 256 leaf values ride the tail drain back for the model. No gathers, no oracle construction, no
Newton walker, no walker drain: steps 9-12 of the profile become two launches and one readback. Gate: RMSE or Logloss,
Newton, one iteration, one dimension, one permutation, a full-depth tree (REUSE_PARTITION's gate). Expected: drains
9 -> 5 alone, launches outside the search ~40 -> ~8. Bits: FAST only. The gradient sum is over the SNAPPED plane
(`enqueue_snap_plane` rounds the gradient to the fixed-point grid the histogram quantizes at, a quantum of about
sum|der| / 2^28 per row, so the leaf sum moves by parts in 1e-5 at most) and both sums are Float32 device folds where the
oracle folded Float64 on the host. Quality: same formula over the same rows; the deviation is far inside FAST's
run-to-run spread, and the FSPEED-ACC line on taxi (Logloss) is the check. Risk: the Logloss Hessian kernel reproduces
`cross_entropy_kernel`'s `p` (exp, finite guard, clamp); a mismatch would show as a leaf scale error in the first trees.

## `-D MOJOLEARN_SYM_ITER_ALL`

All four. Per tree: drains 9 -> 2 (the structure drain and the closing drain), allocations 20 -> 0, launches outside the
search ~40 -> ~8. The composition is by construction: ARENA's pooled partitioner and staging are only reached on an
early-stopped tree once REUSE is on; FUSED's tail drain is the drain that delivers LEAF's values.

Not in this lane: the fixed-point scale as a device pointer into `compute_hist2` (af-sym-hist's kernel signature; it
would remove the magnitude readback entirely, where FUSED only hides its wait), test-metric batching (the board lanes fit
with no eval set), and a single fused add-model-value + gradient kernel (the separate launches share one wait already).

## Compile status (2026-10-03, gbdt binding, laptop, compile only)

- FAST `-D MOJOLEARN_SYM_ITER_ALL` (turns on every define): rc=0 at 05605a559 (after one fix:
  `sym_scale_from_mags` now `raises`, since `choose_scale` raises).
- FAST `-D MOJOLEARN_SYM_BUF_ARENA`, `-D MOJOLEARN_SYM_REUSE_PARTITION`, `-D MOJOLEARN_SYM_DERIV_FUSED`,
  `-D MOJOLEARN_SYM_LEAF_FROM_STATS` (each alone): compile owed: peer.
- FAST with no define: compile owed: peer.
- IDENTICAL: compile owed: peer.
Stopped on orders when the machine's compile slots jammed; the M3 peer compiles the rest.
