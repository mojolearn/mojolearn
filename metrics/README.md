# metrics: cuML `cpp/src/metrics/` and the RAFT `raft/stats/` headers it calls

Seventh section. **COPY, DO NOT IMPROVE.** cuML's metrics are thin wrappers
(`metrics/ported/metrics/`, one file per `.cu`) over RAFT's `raft::stats`
(`metrics/ported/stats/detail/`, one file per `.cuh`). Pinned: cuML 265b9da,
RAFT ebf9268 (the v26.08.00 checkouts; `PORTED_MAP.tsv`). sklearn is read
for SEMANTICS only and its formula stands beside each of ours in a comment.

## Status

CONSTRUCTION plus one Apple device's gates; no second vendor has run this.

All four groups are ported, gated in both modes on one Apple M4, and
sabotaged: A (the label metrics), B (r2, KL divergence), C (silhouette,
the batched path cuML dispatches) and D (trustworthiness). `metrics_main.
mojo` runs every ported metric on one hashed fixture and emits the
identity card (one stage per metric).

## Commands

No pixi task (pixi.toml is not this lane's). Every device run goes
through the build lock:

    tools/with_build_lock.sh     pixi run mojo run -I . metrics/mojo_only/label_metrics_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/label_metrics_check.mojo
    tools/with_build_lock.sh     pixi run mojo run -I . metrics/mojo_only/regression_metrics_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/regression_metrics_check.mojo
    tools/with_build_lock.sh     pixi run mojo run -I . metrics/mojo_only/silhouette_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/silhouette_check.mojo
    tools/with_build_lock.sh     pixi run mojo run -I . metrics/mojo_only/trustworthiness_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/trustworthiness_check.mojo

The card (one stage per metric; diff two machines' cards with
`tools/identity_trace_diff.py`):

    MOJOLEARN_IDENTITY_TRACE=/tmp/metrics.mac.card tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo
    python3 tools/identity_trace_diff.py /tmp/metrics.mac.card /tmp/metrics.other.card

Every printed line carries the mode the binary COMPILED in.

## The one idea, stated once: integer atomics ARE identity-safe, float atomics are not

RAFT's label metrics all start from an INTEGER contingency matrix or
histogram built with `atomicAdd` on `int`. Integer addition is associative
and commutative, so whatever interleaving the hardware produces, the
counter holds the same bits at the end -- on Metal, PTX and AMDGPU, run
after run, in BOTH numeric modes. The device half of every Group A metric
is therefore exact and needs no IDENTICAL arm, and the checks compare it
EXACTLY, per cell.

The float half of those metrics is small -- a few dozen `-p log p` or
`c (log(nc) - log(ab))` terms -- and RAFT folds it on the device in double
with `atomicAdd`, an arrival order. Under IDENTICAL we do not replace the
accumulator with a fixed tree here; we move it. DEVIATION 650: the host
reads the integer matrix back and performs the float ops serially,
ascending. DEVIATION 651: under IDENTICAL those ops are Float32 through
`identical_log` / `identical_mul_add` / `ftz` (one arithmetic on every
vendor and every host; no portable Float64 log exists in
`mojo_only/numerics.mojo`), and under FAST they are RAFT's Float64 through
the host's `log`. IDENTICAL's value therefore differs from FAST's in the
seventh digit by design, and is the same bits everywhere.

The float metrics (Groups B, C) are different: a reduction over `n` of
Float32 terms, where the fold shape IS the number. There the accumulator is
replaced (DEVIATION 653, `metrics/mojo_only/pinned_sum.mojo`): a
`PINNED_SUM_W = 256` slab tree whose width is a constant of the source and
not the launch's block size, per-chunk partials folded ascending on the
host, `ftz` at every stored seam, gated bit for bit against a host model
of the same additions and launch-invariant across two block sizes and two
grid shapes.

## Per-metric table

| metric | cuML file | RAFT file | ported | refused | gated (mode) | sabotaged |
|---|---|---|---|---|---|---|
| accuracy_score | accuracy_score.cu | scores.cuh | yes | -- | count EXACT, value bitwise (both) | via (a) below, the kernel family |
| rand_index | rand_index.cu | rand_index.cuh | yes (DEV 652) | -- | (a, b) EXACT vs O(n^2) host loop AND sklearn pair-confusion; value bitwise (both) | (a) |
| adjusted_rand_index | adjusted_rand_index.cu | adjusted_rand_index.cuh | yes | int64 overload | bitwise vs host formula (both); 4 early returns | (a) |
| entropy | entropy.cu | entropy.cuh | yes (DEV 650, 651) | -- | IDENTICAL bitwise vs oracle; both vs Float64 ref 1e-6; constant -> 0, size 0 -> 1 | (c) null, recorded |
| mutual_info_score | mutual_info_score.cu | mutual_info_score.cuh | yes (DEV 650, 651) | -- | IDENTICAL bitwise; both vs ref; MI == 0 exactly on a product-structured labeling | (b) |
| homogeneity / completeness / v_measure | homogeneity_score.cu, completeness_score.cu, v_measure.cu | homogeneity_score.cuh, v_measure.cuh | yes | -- | IDENTICAL bitwise; both vs ref; constant-truth (1,0,0), constant-pred (0,1,0), both-constant (1,1,1), independent (0,0,0), singleton | (b) |
| r2_score | r2_score.cu | scores.cuh | yes (DEV 653), float | double overload | y_bar/sse/ssto/r2 bitwise vs host tree model (IDENTICAL); vs Float64 sklearn r2 1e-5 (both); FTZ seam visible on an all-subnormal sse fixture; LAUNCH INVARIANT across block 64/256, grid 1-D/2-D, pad 0/37 NaN | (d), (e), (f) |
| kl_divergence | kl_divergence.cu | kl_divergence.cuh | yes (DEV 653), float | double overload | bitwise vs host tree model (IDENTICAL); vs Float64 scipy spelling 1e-5; p == 0 branch (42 planted); q == 0 -> +inf; LAUNCH INVARIANT as r2 | (d) by construction (same fold) |
| trustworthiness | trustworthiness.cu | trustworthiness_score.cuh | yes (DEV 655): ranks counted not sorted, embedded k-NN = neighbors' knn_search | n_neighbors + 1 > 256 by name; batchSize validated | rank sum EXACT vs host count given the same neighbors; neighbor sets == Float64 host k-NN on 523 rows; t bitwise = sklearn formula (and = sklearn 1.9's value 0.7837041712302066); perfect -> 1.0, scrambled 0.496; planted duplicate rows (exact ties) EXACT; 4 refusals; launch invariant | (j) fails |
| silhouette_score / silhouette_samples | silhouette_score_batched_float.cu | detail/batched/silhouette_score.cuh (+ SilOp, countLabels of silhouette_score.cuh) | yes (DEV 654), float, batched path, metric L2SqrtUnexpanded | double; unbatched path (never dispatched); other metrics by name | 1031 per-sample scores + mean bitwise vs host model (IDENTICAL); vs Float64 sklearn 1e-4; singleton +0.0; empty label slot; exact tie a == b -> +0.0; two-way min tie; 4 refusals; chunk is scheduling (3 values one pattern); LAUNCH INVARIANT 8 launches x 1031 cells | (g) fails; (h) (i) null on Apple |

## Group A: what the gates printed (Apple M4, 2026-08-23)

FAST and IDENTICAL, `label_metrics_check.mojo`:

    check_contingency_matrix_exact    k=7 SMEM arm, minLabel=3: 49 cells EXACT, sum 4099
                                      k=31 SMEM arm ... 961 cells EXACT
                                      k=40 GLOBAL arm ... 1600 cells EXACT
    check_accuracy_exact              count 3315 of 4099 EXACT; value 0.8087338 0x3f4f092e bitwise
    check_rand_index_exact            a=1942300 b=4572497 EXACT (two host tallies agree)
                                      rand_index 0x3fe8d258ef472429 bitwise
    check_adjusted_rand_index         0x3fe01ca6e2aee9c2 bitwise; singleton 0x3fe0192ed2b0394e bitwise
    check_entropy_epilogue            IDENTICAL: 0x3ff7ecef00000000 bitwise vs oracle, rel 4.1e-09 vs Float64
                                      FAST: 0x3ff7eceefe5c3322, rel 3.7e-13 vs Float64 (bitwise is a REPORT)
    check_mutual_info_epilogue        IDENTICAL: 0x3fdadbbb00000000 bitwise, rel 1.5e-07; FAST rel 1.1e-10
    check_homogeneity_completeness_v  IDENTICAL: h/c/v bitwise; perfect -> h 1.0 exactly;
                                      FAST: perfect -> h 0.9999999999919665 (MI and H are two
                                      spellings of one quantity; RAFT does not promise equality)
    check_label_epilogue_order_is_visible
                                      MI ascending 0x3ed6ddd8, descending 0x3ed6ddd9: the
                                      fixture separates a reversed fold by one ulp, so the
                                      bitwise gates have teeth

## Group B: what the gates printed (Apple M4, 2026-08-23)

`regression_metrics_check.mojo`, IDENTICAL (FAST prints the same lines as
REPORTs and asserts only the Float64 references and the branches):

    check_virtual_sum_equals_pinned_block_sum
                                   virtual 0xc68a95a8 pinned_block_sum 0xc68a95a8 host model 0xc68a95a8
                                   (FAST: block.sum 0xc68a95a7 twice, model 0xc68a95a8: the
                                   library fold is a different tree, as designed)
    check_sum_order_is_visible     tree 0x45c552fb shift1 0x45c552ff shift7 0x45c55301
                                   shift64 0x45c552ff serial 0x45c55331 (3 of 3 shifts and
                                   the serial fold differ)
    check_r2_matches_oracle        y_bar 0xbd9285e0, sse 0x498b256f, ssto 0x4a7b0dda,
                                   r2 0x3f390e67 bitwise; vs Float64 sklearn 0.72287604861
                                   rel 5.0e-08 (FAST: y_bar 0xbd9285fd REPORT, rel 2.3e-08)
    check_r2_ftz_seam_is_visible   sse device 0x00000000 oracle 0x00000000 (the unflushed
                                   host sum of the 64 subnormal squares is 0x000098bb)
    check_r2_launch_invariant      8 launches (b256/b64 x g1d/g2d x pad0/pad37) one byte
                                   pattern 0x3f390e67
    check_kl_matches_oracle        0x3f7b5f74 bitwise; 42 planted p == 0; q[5] = 0 -> inf
                                   (0x7f800000); vs Float64 rel 8.9e-08 (FAST 0x3f7b5f73, REPORT)
    check_kl_launch_invariant      8 launches one byte pattern 0x3f7b5f74

**A rotation inside the chunk is invisible to the halving tree** (the
first sabotage model tried it: rot 1 / 7 / 64 all equal the tree). The
tree's subtrees at level k are the residue classes of the slot index mod
2^k, and a rotation maps residue classes onto residue classes, so every
pairing adds the same multisets and addition commutes. What a launch-
dependent partition really does is move the CHUNK BOUNDARIES, and that is
what the teeth check and sabotage (d) do. `pinned_sum.mojo::
sabotage_shifted_host_tree_sum` carries the derivation.

**`r2 = 1 - sse/ssto` absorbs a last-bit move in sse whenever `sse <<
ssto`** (measured: the first fixture had r2 = 0.9957 and the shifted
partition moved sse and not r2). The checks therefore gate the three SUMS
(`r2_score_parts`) and the fixture was rebuilt so r2 sits at 0.72.

**A planted subnormal inside a 1e6 sum is reached and invisible** (1e-42
into 1e6 is 1e6); the FTZ seam is made VISIBLE by a second fixture whose
sse is a sum of 64 subnormal squares only: +0.0 under the row-10 policy,
0x000098bb (5.5e-41) on a host that keeps denormals. Under FAST the
report line shows exactly that split (device 0x00000000, oracle
0x000098bb), which is the Apple-vs-host behavior FAST is entitled to.

## Group C: what the gates printed (Apple M4, 2026-08-23)

`silhouette_check.mojo`, IDENTICAL:

    check_silhouette_matches_oracle    cluster counts 483 251 155 67 41 34; 1031 per-sample
                                       scores bitwise; mean 0x3f10f331 bitwise; vs Float64
                                       sklearn mean 0.56621081619 rel 1.7e-08, worst per-sample
                                       abs 1.4e-07
                                       (FAST: 309 of 1031 cells and the mean REPORT one ulp off
                                       the tree model; block.sum is a different fold)
    check_silhouette_singleton_and_empty
                                       1031 bitwise; singleton row 41 = 0x00000000; label 6 empty
    check_silhouette_planted_ties      row0 (a == b == 1) 0x00000000, row1 0x3e95f619 = host
                                       formula, row2 (singleton) 0x00000000; min tie b_1 == b_2
                                       = sqrt2: 0xbe95f61a, 4 cells bitwise
    check_silhouette_refusals          n_labels 1 and 64 (= n_rows), metric 1, chunk 0 RAISE by
                                       name; chunk 1 / 40000 / 7 one byte pattern 0x3f28a853
    check_silhouette_launch_invariant  8 launches (b256/b64 x g1d/g2d x pad0/pad37), 1031 cells +
                                       mean, one byte pattern 0x3f10f331
                                       (FAST: the b64 launches move 279 cells and the mean by
                                       one ulp against b256 -- block.sum is a function of the
                                       block, which is exactly what IDENTICAL must not be)

## Group D: what the gates printed (Apple M4, 2026-08-23)

`trustworthiness_check.mojo`, both modes (the metric is integer plus one
Float64 closed form; IDENTICAL and FAST print the same numbers):

    check_trust_rank_sum_exact        rank sum device 291291 host 291291; embedded neighbor
                                      sets vs Float64 host k-NN: 0 rows differ; t
                                      0.7837041712302066 (0x3fe9141ac525854c) = sklearn
                                      formula; sklearn 1.9.0 on the same fixture (Python,
                                      float64): 0.7837041712302066
    check_trust_perfect_and_scrambled perfect: rank sum 0, t 1.0; scrambled: t 0.4956
    check_trust_planted_duplicates    device 189301 = host(j < e) 189301; host(j <= e) 191658
    check_trust_refusals              n_neighbors 0, n_neighbors + 1 > n, > TRUST_MAX_K,
                                      batchSize 0 RAISE by name
    check_trust_launch_invariant      b256/b64 x g1d/g2d: 291291 x 4

## The card (`metrics_main.mojo`, Apple M4, 2026-08-23)

34 stages, one `seq`: the inputs (`metrics.input.*`), the integer
products (`metrics.contingency`, `metrics.rand.a/b`, neighbors'
`knn.*` six stages, `trust.emb_ind`, `trust.rank_sum`), the per-sample
silhouettes, and every returned value by its bits. Two IDENTICAL runs:

    python3 tools/identity_trace_diff.py /tmp/metrics.ident.card /tmp/metrics.ident2.card
    RESULT: IDENTICAL. Same stage sequence, same counts, same hashes.

IDENTICAL against FAST diverges first at stage 8, `metrics.entropy` --
DEVIATION 651's Float32-vs-Float64 epilogue, by design; the integer
stages 0-4 agree, as they must. The values, IDENTICAL / FAST:

    accuracy_score      0x3f426680 / same          rand_index        0x3fe8c307da380cc0 / same
    adjusted_rand_index 0x3fde6f24da913871 / same  entropy(y_true)   0x3ff6a22320000000 / 0x3ff6a22304271931
    mutual_info_score   0x3fe1b503a0000000 / 0x3fe1b503bebf2960
    homogeneity         0x3fd908f8572bc9a9 / 0x3fd908f8a171bee4  (completeness, v_measure likewise)
    r2_score            0x3f320e9b / 0x3f320e9c     kl_divergence     0x3f809113 / same
    silhouette_score    0x3ef548da / same           trustworthiness   0x3fe99369e91a2645 / same

**Trustworthiness's k-NN stages come from neighbors/**: the metric's
traced entry hands the card to `knn_search_traced`, so `knn.*` and
`trust.*` share one `seq` (the DEVIATION 518 lesson: a second
`IdentityTrace()` appends a second `seq 0` the differ refuses; the first
draft of this driver did exactly that and the card showed it).

**What trustworthiness rests on that is not integer:** the embedded k-NN
(`neighbors/estimator.mojo::knn_search`, the EXPANDED L2 arm, where RAFT
asks `brute_force_knn` for `L2SqrtUnexpanded`) and the rank comparison
`d(i,j) < d(i,e)` on Float32 unexpanded distances. Near-tied embedded
neighbors could order differently under the two distance spellings and
move the integer sum; on the hashed fixture the neighbor sets agree with
a Float64 host k-NN on every row. A k-NN entry for the unexpanded metric
is neighbors/'s to add (hand-off below).

**The distance is the UNEXPANDED formula, not the neighbors tile.**
`neighbors/mojo_only/pinned_distance_tile.mojo` is the EXPANDED L2 from
precomputed norms (cuVS `L2Expanded`); cuML's silhouette dispatches
`'euclidean'` to `L2SqrtUnexpanded`, a different arithmetic, so
`metrics/mojo_only/pinned_distance.mojo` mirrors that tile's discipline
(one thread per cell, ascending features, `identical_mul_add`, `ftz`,
`identical_sqrt`) on the unexpanded formula and does not call it. cuVS's
Contractions kernel for the unexpanded distance is not ported (UNPORTED).

**The max(a, b) hazard** is stated in `silhouette_score.mojo`: RAFT's
`SilOp` never calls `max`; the tie and 0/0 cases are explicit `+0.0`
branches, `-0.0` cannot arise as a or b (sums of nonnegative terms seeded
+0.0 or FLT_MAX), and the gate plants an exact tie and reads `0x00000000`.

**Completeness is the transposed fold.** `completeness_score.cu` calls
`homogeneity_score(y_hat, y)`, so RAFT computes a SECOND mutual information
over the transposed contingency matrix. A transposed fold is a different
Float32 sum: on the singleton fixture `MI(truth, pred) = 0x3fe0589f00000000`
and `MI(pred, truth) = 0x3fe0589ee0000000`. sklearn computes one MI and
uses it for both h and c. Ours mirrors RAFT (two folds), and so does the
oracle; the first draft of the oracle used one MI and the IDENTICAL gate
caught it on the singleton fixture (`v_measure(singleton) got
0x3fd8213c1ffcaab2 oracle 0x3fd8213c36e1d2cb`), which is the gate doing
its job.

**sklearn semantics that differ from RAFT and are mirrored as RAFT:** the
0/0 case of ARI returns 0 (sklearn warns and returns NaN); entropy's
`log(pi) - log(pi_sum)` grouping versus RAFT's divide-then-log (same
quantity, last bits differ, both within 1e-6 of each other in the gates).

**One integer bug of theirs not ported:** `mutual_info_kernel`'s
`a[i] * b[j]` is an `int` product (overflows past 46,340 samples in one
class pair); ours is Int64. Recorded in `mutual_info_score.mojo`.

## Sabotages (each reverted in the same session; a check that cannot fail is not a check)

| # | what was broken | check | failing output |
|---|---|---|---|
| (a) | contingency index written `(pd - off) * width + gt - off` (transposed) | check_contingency_matrix_exact | `cell 0,1 device 160 host 185 ... contingency matrix k=7 (SMEM arm): 38 cells differ` |
| (b) | the ported MI epilogue's (i, j) loop run DESCENDING | check_mutual_info_epilogue (IDENTICAL) | `BITWISE MISMATCH mutual_info_score got 0.4196613132953644 (0x3fdadbbb20000000) oracle 0.419661283493042 (0x3fdadbbb00000000)` |
| (c) | `ftz` dropped from the entropy epilogue's stored term | check_entropy_epilogue (IDENTICAL) | NO CHANGE on Apple: no term is subnormal on the label fixtures and the host flushes nothing either; recorded as the expected null. The FTZ seam that CAN be reached is Group B's (planted subnormal squares), where the sabotage must fail |
| (d) | the device sse/ssto chunk partition shifted by one value WITH WRAP (`i = (i0 + 1) % n`: same multiset, different chunks) | check_r2_matches_oracle (IDENTICAL) | `BITWISE MISMATCH ssto device 4113270.7 (0x4a7b0ddb) oracle 4113270.5 (0x4a7b0dda)` (sse happened to agree; r2 gated per part for this reason) |
| (e) | every `ftz` dropped from the ORACLE's sse path (term, slab load, tree, partial fold) | check_r2_ftz_seam_is_visible (IDENTICAL) | `BITWISE MISMATCH sse(all-subnormal) device 0.0 (0x00000000) oracle 5.479e-41 (0x000098bb)`; the same sabotage on check_r2_matches_oracle moved NOTHING (1e-42 into 1e6), which is why the second fixture exists |
| (f) | `ftz` dropped from the DEVICE sse term | check_r2_ftz_seam_is_visible (IDENTICAL) | NO CHANGE on Apple (the hardware flushes; the pin is inert on this column, as numerics.mojo says of every pin here); recorded as the expected null -- the pin's value is the denormal-honoring column |
| (g) | the silhouette row kernel's j partition shifted by one WITH WRAP (`j = (j0 + 1) % n_rows`) | check_silhouette_matches_oracle (IDENTICAL) | `samples row 1 device 0x3f11ad85 oracle 0x3f11ad84 ... samples: 347 of 1031 scores differ` |
| (h) | `identical_sqrt` -> `std.math.sqrt` in the device distance | check_silhouette_matches_oracle (IDENTICAL) | NO CHANGE on Apple (its sqrt is correctly rounded; DEVIATION 258 measured NVIDIA's approximate: 180,714 of 2^20 off by one ulp); recorded as the expected null -- the pin's value is the NVIDIA column |
| (i) | `ftz` dropped from the device distance's diff and accumulator | check_silhouette_matches_oracle (IDENTICAL) | NO CHANGE on Apple (hardware flush; no subnormal on this fixture); recorded as the expected null |
| (j) | the trustworthiness rank tie-break flipped to `j <= e` | check_trust_rank_sum_exact (both modes) | `rank sum device 293696 host 291291 ... rank sum differs` (every neighbor now counts itself); the planted-duplicate fixture separates the DISTINCT-row tie too: host(j < e) 189301 vs host(j <= e) 191658 |

## Deviations spent

| n | where | what |
|---|---|---|
| 650 | entropy.mojo (banner), mutual_info_score.mojo, adjusted_rand_index.mojo | the label metrics' float epilogue runs on the HOST, serially, from the INTEGER device product; readback is k or k^2 ints instead of one scalar |
| 651 | entropy.mojo (banner), mutual_info_score.mojo | IDENTICAL: Float32 through identical_log / identical_mul_add / ftz; FAST: Float64 host log (RAFT's precision) |
| 652 | rand_index.mojo | 64-bit atomic replaced by per-block Int32 partials summed on the host in Int64 (Apple has no 64-bit atomic; a 32-bit total overflows past 65,536 samples) |
| 655 | trustworthiness_score.mojo | ranks COUNTED per (row, embedded neighbor) with the stable-sort tie-break (same integer as their sort + lookup table, O(nk) per row); the embedded k-NN is neighbors' knn_search (expanded L2) with k+1; Int32 row partials summed in Int64; n_neighbors + 1 > 256 refused by name |
| 654 | batched/silhouette_score.mojo, mojo_only/pinned_distance.mojo | the float atomicAdd into a and b[i, c] replaced by one fixed tree per (row, cluster) over j (a pure function of n and the cluster membership), the distance tile by the one-thread unexpanded formula through identical_mul_add/ftz/identical_sqrt, min + SilOp in the row's thread; chunk is scheduling; metric != L2SqrtUnexpanded refused by name |
| 653 | mojo_only/pinned_sum.mojo, scores.mojo (r2), kl_divergence.mojo | the float sums over n as one fixed slab tree + ascending host fold, launch-invariant by construction; FAST arm is block.sum; r2's `powerScalar(x,2)` spelled `x*x`, `mean = sum * (1/n)` as mean.cuh |

## ROW TEXT FOR THE IDENTITY LANE

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| (next) | metrics: trustworthiness (Group D) | `brute_force_knn` on the embedding (L2SqrtUnexpanded), `pairwise_distance` + `sort_cols_per_row` (CUB segmented radix sort) + lookup table on X, `atomicAdd(double)` of INTEGER ranks (exact) | DEVIATION 655: ranks counted with the stable-sort tie-break on pinned unexpanded distances, k-NN through neighbors' knn_search, Int64 host sum; integer half gated EXACTLY, neighbor sets vs a Float64 host k-NN, t = sklearn's value on the fixture; sabotage (j) fails | Apple FAST+IDENTICAL green; the embedded k-NN rides on neighbors' row 24; no second vendor |
| (next) | metrics: silhouette_score / silhouette_samples (Group C, batched path) | cuVS `pairwise_distance` L2SqrtUnexpanded (a tiled contraction, vendor tile policy; cuBLAS TF32 on the expanded arms), float `atomicAdd` of `d/count` into a and b from every (i, j) thread of every chunk tile (arrival order, chunksize-dependent), CUB min reduce, thrust::reduce mean | DEVIATION 654: row-owned kernel, one slab tree per (row, cluster) over j ascending, the distance one thread per cell through identical_mul_add/ftz/identical_sqrt, min and SilOp in the same thread, the mean DEVIATION 653's tree; gated bitwise per sample vs a host model, launch-invariant over 8 launches x 1031 cells, chunk proven scheduling, ties planted; sabotage (g) fails, (h) (i) null on Apple | Apple FAST+IDENTICAL green; no second vendor |
| (next) | metrics: r2_score, kl_divergence (Group B) | `thrust::reduce` / `mapThenSumReduce`: a CUB block fold + float `atomicAdd` of block partials (arrival order, vendor lane width), vendor `powf`/`log` | DEVIATION 653: one PINNED_SUM_W slab tree per chunk whose width is a constant of the source (not the block), chunk totals folded ascending on the host, ftz at every stored seam, `identical_log` per term; gated bitwise vs a host model, launch-invariant across 8 launches (two block sizes, two grid shapes, two paddings), FTZ seam visible on an all-subnormal fixture; sabotages (d)(e) fail, (f) null on Apple | Apple FAST+IDENTICAL green; no second vendor |
| (next) | metrics: label metrics (accuracy, rand, ARI, entropy, MI, h/c/v) | contingency matrix and histograms are INTEGER atomics (order-free, identity-safe on every vendor); the float epilogue is a device double `atomicAdd` fold (arrival order) with the vendor's `log` | device product stays integer and is gated EXACTLY per cell; the float epilogue moves to the host, serial ascending, Float32 through identical_log/identical_mul_add/ftz under IDENTICAL (DEVIATIONS 650, 651); completeness is the transposed fold as theirs; gated bitwise vs an oracle and 1e-6 vs a Float64 sklearn-spelled reference; two sabotages fail, one null recorded | Apple FAST+IDENTICAL green; no second vendor |

## HAND-OFF (Python surface, for the bindings lane)

Names, mirroring `cuml.metrics`: `mojolearn.metrics.accuracy_score(y_true,
y_pred)`, `rand_score`, `adjusted_rand_score`, `mutual_info_score`,
`homogeneity_score`, `completeness_score`, `v_measure_score(beta=1.0)`,
`entropy(labels)` (cuML exposes `cuml.metrics.cluster.entropy`),
`r2_score(y_true, y_pred)` (float32; the Python side passes `n`), and
`kl_divergence(P, Q)` (float32; cuML's `cuml.metrics.kl_divergence`,
which does NOT normalize), `silhouette_score(X, labels, metric='euclidean',
chunksize=40000)` and `silhouette_samples(...)` (float32 row-major X; the
Python side maps labels to `0..n_labels-1` with `np.unique(...,
return_inverse=True)` exactly as cuML's `.pyx` does and passes
`n_labels`; `metric` other than 'euclidean'/'l2' and float64 X are
refused by name), and `trustworthiness(X, X_embedded, n_neighbors=5,
metric='euclidean', batch_size=512)` (float32 host arrays, the Mojo entry
takes `List[Float32]` because it calls knn_search's host-pointer
boundary; other metrics refused by name). Each label metric takes
int32 labels; the Python side computes `lower_class_range = min(y, y_hat)`
and `upper_class_range = max(...)` exactly as cuML's `.pyx` files do and
passes them to the Mojo entry; a label outside the range is a host-side
raise by name (RAFT writes out of bounds; we do not port that). Float64
inputs to the float metrics are refused by name (no Float64 on device).
pixi lines to register when pixi.toml is next edited by its owner:

    check-metrics-labels     = "mojo run -I . metrics/mojo_only/label_metrics_check.mojo"
    check-metrics-regression = "mojo run -I . metrics/mojo_only/regression_metrics_check.mojo"
    check-metrics-silhouette = "mojo run -I . metrics/mojo_only/silhouette_check.mojo"
    check-metrics-trust      = "mojo run -I . metrics/mojo_only/trustworthiness_check.mojo"
    metrics-card             = "mojo run -I . metrics/metrics_main.mojo"

## HAND-OFF TO THE IDENTITY LANE (and to neighbors/)

Nothing outside `metrics/` was edited. Three asks, none blocking:
0. neighbors/: a `knn_search` arm for `L2SqrtUnexpanded` (RAFT's
   `run_knn` metric for trustworthiness). Until then trustworthiness's
   embedded k-NN is the expanded arm (DEVIATION 655), which can order
   near-tied neighbors differently in the last bit.
1. `mojo_only/numerics.mojo` has `identical_exp64` but no `identical_log64`;
   a portable Float64 log (Cephes, the `portable_exp64` shape) would let
   DEVIATION 651 keep RAFT's double precision under IDENTICAL. Until then
   IDENTICAL is Float32 at these seams.
2. IDENTITY_PATHS.md: the row above.
