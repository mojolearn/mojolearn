# LANE kmeans-control-plane — 2026-08-19

Upstream read: `~/CascadeProjects/upstream/cuvs` @ `94c2819` (branch-25.08).

## VERDICT ON THE INVENTED CONVERGENCE KERNEL

**The brief was right on every point, and it is worse than stated.**

- `cpp/src/cluster/detail/kmeans.cuh:817-825` is the *signature and parameter
  list of `kmeans_fit`* plus the comment `// Check that parameters are valid`.
  It contains no convergence test.
- `detail/kmeans.cuh:920-930` — the other citation, in
  `kmeans_common.mojo`'s `check_convergence_kernel`, claimed to be "a
  `map_offset` over a single element" — is a `RAFT_LOG_DEBUG` for the `Array`
  init method plus the call to `kmeans_fit_main`. The only `map_offset` in
  `cpp/src/cluster/` is in `kmeans_balanced.cuh:785`.
- `kmeans_common.cuh:637-660`, the third citation, is the tail of
  `countSamplesInCluster`.
- **`grep -rn "check_convergence\|checkConvergence" cuvs/cpp/` returns ZERO
  matches.** It also returns zero in `raft/cpp/`. The only hits anywhere in the
  three checkouts are cuML's t-SNE and QN solver, unrelated.
- Their real rule is `detail/kmeans.cuh:461-497`: async D2H of one float at
  `:462`, ONE `raft::resource::sync_stream` at `:491`, host test
  `if (sqrdNormError < params.tol) done = true;` at `:492`, `break` at
  `:494-497` **in the same iteration**.
- `params.inertia_check` is `false` by default
  (`cpp/include/cuvs/cluster/kmeans.hpp:120`). The whole cost reduction and
  the `delta > 1 - tol` ratio test sit inside `if (params.inertia_check)` at
  `:468-489` and **do not run in a default cuVS fit**.

Additionally: the commit `2140532c` cited in the header of every `cluster/`
file is not a valid object in rapidsai/cuvs (`git cat-file -t` → *"Not a valid
object name"*). Two of the upstream files named in `PORTED_MAP.tsv`
(`minClusterDistanceCompute.cu`, `distance/unfused_distance_nn.cuh`) do not
exist in cuVS at all.

## 1. CONTROL PLANE, THEIRS vs OURS (per Lloyd iteration, default params)

`kmeans_fit_main`, `detail/kmeans.cuh:407-497`.

| | theirs (`inertia_check=false`, the default) | ours BEFORE | ours AFTER |
|---|---|---|---|
| stream syncs | **1** (`:491`) | 1 (top of next iter) | **1** (`:491`) |
| host round trips | **1** — 4-byte D2H of `sqrdNorm` (`:462`) | 1 — 4-byte D2H of an int32 done-flag | **1** — 4-byte D2H of the shift scalar |
| decided on the HOST | convergence (`:492`), break (`:494-497`) | nothing; host only read a flag | convergence, break, same iteration |
| decided on the DEVICE | nothing | convergence, in a `(1,1,1)/(1,1,1)` kernel | nothing |
| iterations run past convergence | **0** | **1** (flag read at the top of the next iteration) | **0** |
| kernel launches | **~9–11** (see below) | **14** | **11** |
| cluster-cost reduction | **not run** (gated off) | run every iteration, unconditionally | run only if `inertia_check` |
| inertia computed | once per restart, after the loop, over a FRESH assignment against the final centroids, WEIGHTED (`:500-537`) | reused the last iteration's in-loop cost (pre-swap centroids) | once per restart, after the loop, fresh + weighted |

Launch accounting, theirs (`inertia_check=false`):

| step | upstream | launches |
|---|---|---|
| centroid `rowNorm` | `kmeans_common.cuh:385-389` | 1 |
| `fusedDistanceNNMinReduce` | `kmeans_common.cuh:434-449` | 1 (+1 `thrust::fill` of the KVP output, `:407-410`) |
| `reduce_rows_by_key` | `kmeans.cuh:300-309` | 1 memset + 1 kernel |
| `reduce_cols_by_key` | `kmeans.cuh:312-318` | 1 memset + 1 kernel |
| `matrix_vector_op(div_checkzero)` | `kmeans.cuh:326-331` | 1 |
| `gather_if` (empty-cluster rule) | `kmeans.cuh:335-348` | 1 |
| `mapThenSumReduce(sqdiff_op)` | `kmeans.cuh:454-459` | 1–2 |
| D2D copy of the centroids | `kmeans.cuh:464-465` | 1 memcpy |
| **total** | | **~10–12 device ops, 1 sync, 1 D2H** |

Ours after the change: `zero`×2, centroid `rowNorm`, fused kernel,
`accumulate_centroid_sums`, `accumulate_weight_per_cluster`,
`finalize_centroids`, `sum_partials`+`finish_sum` (shift), `copy_f32`
= **11**, 1 sync, 1 D2H.

**Removed per iteration: 3 launches (2 cost-reduction stages + the
one-thread convergence kernel), and one entire Lloyd iteration per fit.**
Sync count is unchanged at 1, because theirs is also 1 — per the brief, that
one stays.

## 2. DIVERGENCES FOUND

| # | upstream (file:line) | what ours did | fixed? |
|---|---|---|---|
| 1 | `kmeans.cuh:491-497` — host test after the loop's one sync, break same iteration | invented an on-device `check_convergence_kernel` (`(1,1,1)` grid), read the flag one iteration late, **and still synced every iteration**; ran one extra Lloyd iteration on every fit | **FIXED** — host test at the same place theirs is; kernel deleted |
| 2 | `kmeans.hpp:120` — `inertia_check = false`; `kmeans.cuh:468-489` gated | computed the cluster cost every iteration unconditionally (2 launches + an `n_samples` read) and always applied the ratio test | **FIXED** — `inertia_check` added to `KMeansParams`, default `False`; block gated |
| 3 | `kmeans.cuh:470-476` — in-loop cost uses `raft::value_op`, **unweighted** | in-loop cost multiplied by the sample weights | **FIXED** — in-loop uses `SUM_MODE_PLAIN`, post-loop uses `SUM_MODE_PRODUCT` |
| 4 | `kmeans.cuh:500-537` — inertia is a FRESH assignment against the final centroids, then weighted `computeClusterCost` | reported the last iteration's cost, which belongs to the centroids from *before* the final update; `labels` were likewise one update stale | **FIXED** — post-loop assignment + weighted cost per restart |
| 5 | `kmeans.cuh:454-459` — `mapThenSumReduce(sqdiff_op)`, map inside the reduction | materialized `n_clusters × n_features` squared differences into a `shift_cells` buffer with a separate `centroid_shift_kernel`, then summed | **FIXED** — `sum_partials_kernel` gained a mode; `centroid_shift_kernel` and `shift_cells` deleted |
| 6 | `kmeans.cuh:877-883` — `n_init` forced to 1 when `init == Array` | ran `n_init` identical restarts | **FIXED** |
| 7 | `kmeans.cuh:938-943` — `n_iter[0] = n_current_iter`, no clamp; loop exits at `max_iter + 1` when it never converges | clamped with `min(n_current_iter, max_iter)` | **FIXED** (a non-converged fit now reports `max_iter + 1`, as theirs does; their own log at `:540` subtracts one) |
| 8 | `kmeans_common.cuh:378-379` — `is_fused` is `metric == L2Expanded \|\| L2SqrtExpanded`, full stop | a `use_fused(sm_major, m, n)` function with SM 8/9/10 codes and a 4096 threshold, cited to `kmeans_common.cuh:60-82` (which is `SamplingOp`). **No such rule exists anywhere in cuVS.** Never called by anything | **FIXED** — deleted, replaced by `is_fused(metric)` |
| 9 | `kmeans.hpp:28-121` — the `params` struct | had two invented fields, `init_size` and `device_buffer_samples`, under a docstring claiming "field for field, default for default". Zero grep hits in cuvs, cuml or raft | **FIXED** — removed, along with the python wrapper's keywords for them |
| 10 | `kmeans.cuh:825-835` — the `RAFT_EXPECTS` block does **not** restrict the metric | `validate()` raised on non-L2 metrics citing `:568` (= `initScalableKMeansPlusPlus`'s signature) as if it were theirs | **kept, relabeled** — the refusal is ours because only their fused arm is ported; now says so |
| 11 | `raft/linalg/detail/reduce_rows_by_key.cuh:270-288` — one thread per cell, early return | we grid-stride a capped grid, and cited `:292` (a template parameter list) claiming theirs grid-strides too | **comment fixed**, kernel left (same arithmetic, different launch shape); recorded as a deviation |
| 12 | `distance_ops/l2_exp.cuh:132-134` — the clamp has a SECOND factor zeroing the bit-equal-norm self-neighbor case | both our kernels clamp on sign alone | **NOT FIXED** — see §7. Near-unreachable for k-means; **the normal case for a self-join k-NN**, so the k-NN lane needs this |
| 13 | `kmeans_common.cuh:136-172` — `checkWeight` normalizes the sample weights to sum to `n_samples`, called at `kmeans.cuh:872` | absent | **NOT FIXED** — recorded in `UNPORTED.tsv`; theirs normalizes a private copy, ours would have to mutate the caller's buffer |
| 14 | no `minClusterDistanceCompute.cu` and no `distance/unfused_distance_nn.cuh` exist in cuVS | both named as `PORT OF` targets and in `PORTED_MAP.tsv` | **FIXED** — both remapped to `kmeans_common.cuh:360-493` |
| 15 | cuVS `94c2819` | every `cluster/` file said `2140532c`, which is not a valid object in the repo | **FIXED** |
| 16 | `fused_distance_nn/gemm.h:122`, `pairwise_distance_gemm.h:75` — `cutlass::arch::OpMultiplyAddFastF32`, i.e. **3xTF32** | `cluster/README.md` claimed plain TF32, "10 mantissa bits", citing `CUBLAS_COMPUTE_32F_FAST_TF32` in a file that does not exist. Their own commented-out alternative on the next line is the 1xTF32 one and they did **not** take it | **FIXED** in `cluster/README.md` |
| 17 | `kmeans.cuh:189-217` — k-means++ candidate step | our citations were off by 2–20 lines throughout (`:187`→`:189`, `:230-234`→`:249-257`, `:112`→`:109`, `:148`/`:794`→`:152-157`/`:885`,`:890`) | **FIXED** |

### Dispatch check (per the mid-lane rule change)

- **The Lloyd assignment step calls no vendor primitive.**
  `min_cluster_and_distance_compute` launches our transliteration of their
  SIMT fused kernel and writes no distance tile. This is the arm their
  dispatch takes for L2Expanded (`is_fused`, `kmeans_common.cuh:378-379`).
  Nothing device-wide was let into it.
- **k-means++ is clean too.** Ours calls `core/gemm.mojo::gemm_nt` (MAX
  `linalg.matmul`) for the candidate-cost matrix — but **their dispatch
  materializes that matrix as well** (`pairwise_distance_kmeans` into
  `pwd[n_trials × n_samples]`, `kmeans.cuh:195-196`), so no fusion is being
  frozen out. Ours is in fact one pass fewer: theirs is
  pairwise → `matrix_vector_op(min_op)` → `reduce` (3 device-wide passes over
  `n_trials × n_samples`), ours is GEMM → `candidate_cost_kernel` (2).
- The k-means++ greedy pick brings `n_trials` floats to the host and argmins
  there; theirs argmins on device (`cub::DeviceReduce::ArgMin`, `:224-240`)
  and brings back one int, then syncs (`:242-244`). Same one drain, same
  O(candidates) transfer. Left as-is, recorded.

## 3. WHAT CHANGED, FILE BY FILE

- **`cluster/gbdt/cluster/detail/kmeans.mojo`** — the lane's main change.
  Loop tail rewritten to `detail/kmeans.cuh:453-497`: `mapThenSumReduce`-shaped
  shift (`:454-459`), D2H of the scalar (`:462`), D2D copy-back (`:464-465`),
  gated cost block (`:468-489`) including their `ASSERT` at `:480-482`, the
  single sync (`:491`), the host test (`:492`), same-iteration break
  (`:494-497`). Post-loop inertia block added from `:500-537`. `n_init`
  collapse for `INIT_ARRAY` from `:877-883`. `n_iter` un-clamped per `:938-943`.
  `_sum_device` takes a mode instead of a bool. Buffers `shift_cells`,
  `d_prior_cost`, `d_done`, `h_done` removed; `h_shift` added. Module docstring
  rewritten.
- **`cluster/gbdt/cluster/detail/kmeans_common.mojo`** — `check_convergence_kernel`
  **deleted**. `use_fused` **deleted**, replaced with `is_fused(metric)` from
  `kmeans_common.cuh:378-379`. `check_convergence` rewritten to
  `kmeans.cuh:466-492` with an `inertia_check` argument. `std.gpu` import
  dropped (no kernels left in the file). Citations corrected.
- **`cluster/mojo_only/reduce_by_key.mojo`** — `centroid_shift_kernel`
  deleted; `sum_partials_kernel` gained `SUM_MODE_PLAIN/PRODUCT/SQDIFF`.
  `copy_f32_kernel` docstring corrected (theirs is `raft::copy` at `:464-465`,
  not a `std::swap` at `:907`, which is a log string).
- **`cluster/gbdt/cluster/kmeans_params.mojo`** — `inertia_check: Bool`
  added (default `False`, `kmeans.hpp:120`); `init_size` and
  `device_buffer_samples` removed; metric refusal relabeled as ours;
  citations corrected.
- **`cluster/gbdt/python/cluster/kmeans/kmeans.mojo`** — dropped the two
  invented keywords; documented their real keyword list (`kmeans.pyx:98-129`,
  which includes `hierarchical`/`hierarchical_n_iters` we do not have).
- **`cluster/gbdt/cluster/detail/min_cluster_distance_compute.mojo`**,
  **`cluster/gbdt/distance/unfused_distance_nn.mojo`**,
  **`cluster/gbdt/cluster/kmeans.mojo`**,
  **`cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo`** —
  headers and citations remapped to files that exist. `simt_kernel.mojo` was
  touched in **comments only** (commit hash, two `Reducer` citations); no
  structural change, per the note that another lane reads it as a model.
- **`cluster/mojo_only/kmeans_check.mojo`** — two clamp citations corrected.
  No behavioral change.
- **`cluster/README.md`**, **`cluster/PORTED_MAP.tsv`**,
  **`cluster/UNPORTED.tsv`** — rewritten where falsified.

## 4. PROPOSED `PORTED_MAP.tsv` / `UNPORTED.tsv` ROWS

Already applied to `cluster/PORTED_MAP.tsv` and `cluster/UNPORTED.tsv` (both
are mine). New rows added there:

```
gbdt/distance/fused_distance_nn/simt_kernel.mojo	cpp/src/distance/detail/fused_distance_nn/simt_kernel.cuh	partial	the SIMT fused kernel, no vendor call. Epilogue clamps on sign only; theirs also zeroes the bit-equal-norm self-neighbor case (distance_ops/l2_exp.cuh:132-134)
```
```
cpp/src/cluster/detail/kmeans_common.cuh::checkWeight	not ported	RAFT_EXPECTS that the sample weights sum to n_samples, and rescales them if they do not (kmeans.cuh:872). Ours takes the caller's weights as given
cpp/src/cluster/detail/kmeans.cuh::kmeans_transform	not ported	same as the .cu above, the header side
```

## 5. PROPOSED `archive/reference/PORTING.md` DEVIATION ENTRIES (renumber from 30)

**30. The k-means stopping rule is theirs, and the device-side one was never
theirs.** cuVS tests convergence on the host, on a scalar copied back after
the loop body's single `sync_stream`, and breaks in the same iteration
(`detail/kmeans.cuh:461-497`). `check_convergence` does not exist in cuVS,
cuML or RAFT. A previous version of this port ran the test in a one-thread
kernel, read the flag one iteration late, and cited
`detail/kmeans.cuh:817-825` (a function signature) and `:920-930` (a debug
log) as authority. Deleted, and replaced with theirs. Cost of the defect: one
extra full Lloyd iteration per fit, plus three launches per iteration, while
still paying the per-iteration sync.

**31. `inertia_check` is off by default and gates a whole step.** With
`params.inertia_check == false` (`cuvs/cluster/kmeans.hpp:120`) the cluster
cost is not reduced in the loop at all and the `delta > 1 - tol` ratio test
does not run (`detail/kmeans.cuh:468-489`). Inertia is computed once per
restart after the loop, over a fresh assignment against the final centroids
(`:500-537`). Our field name, default and gating now match.

**32. The cluster cost is unweighted inside the loop and weighted after it.**
`computeClusterCost` with `raft::value_op` reads `minClusterAndDistance`
directly at `:470-476`; only the post-loop path multiplies by the sample
weights first (`:516-535`). Two different numbers with the same name.

**33. `n_iter` is not clamped.** Their loop leaves `n_iter[0] == max_iter + 1`
when it never converges, which is why their own log line subtracts one at
`:540`. A non-converged fit here now reports `max_iter + 1`.

**34. `sum_partials_kernel` applies its map inside the reduction.** Both
reductions in the Lloyd loop are `map`-then-`sum` in cuVS
(`mapThenSumReduce` with `sqdiff_op` at `:454-459`; `computeClusterCost` with
`value_op` at `:470-476`) and neither materializes the mapped array. The
`centroid_shift_kernel` + `shift_cells` pair that used to do so is deleted.

**35. `is_fused` has no hardware term.** cuVS's assignment-arm selector is
`metric == L2Expanded || metric == L2SqrtExpanded` and nothing else
(`kmeans_common.cuh:378-379`). A `use_fused(sm_major, m, n)` function with
SM 8/9/10 codes and a 4096 threshold, cited to `kmeans_common.cuh:60-82`
(= `SamplingOp`), was invented and never called. Deleted. Consequence: our
"we port the unfused path because theirs sends new hardware there" story was
false; the fused SIMT kernel is the path their dispatch takes for k-means and
it is the one wired.

**36. cuVS's float32 distance GEMM is 3xTF32, not TF32.** Their CUTLASS
specializations select `cutlass::arch::OpMultiplyAddFastF32`
(`fused_distance_nn/gemm.h:122`, `pairwise_distance_gemm.h:75`), with the
1xTF32 `OpMultiplyAdd` alternative sitting commented out on the next line. The
earlier "10 mantissa bits, `CUBLAS_COMPUTE_32F_FAST_TF32`" claim named a
constant and a file that do not exist in cuVS and overstated the gap. The
architecture-dependence conclusion survives; the precision number does not.

## 6. FALSE DOC SENTENCES IN FILES I MAY NOT EDIT

- **`core/gemm.mojo:1-15`** — its opening paragraph justifies calling MAX's
  `linalg.matmul` by the now-deleted rule ("the rule for 'call a tuned vendor
  BLAS' is to call OURS"). The call itself is still right in `cluster/`
  (their dispatch materializes the same matrix), but the *reason* given is a
  retired rule. Whoever owns `core/` should restate it as: the closed-library
  exception, applied to a step their dispatch also materializes.
- **`UNWIRED.md`** — records `use_fused`'s SM/4096 rule as a "REACH fact".
  That rule does not exist in cuVS. The entry should be deleted, not amended.
- **`archive/reference/HOST_AND_DEVICE.md`** — cited twice by the old `kmeans.mojo` as authority
  for moving the convergence decision onto the device. If it says a
  host-decided per-iteration scalar is an anti-pattern, that is exactly what
  cuVS does at `detail/kmeans.cuh:462`/`:491`/`:492`, and the document needs
  the counter-example.
- **`archive/reference/PORTING.md 15`** — the old `kmeans.mojo` header pointed at it for "this
  port tests on the host, which is a deviation". The deviation was the other
  way round. Entry needs deleting or inverting.
- **`archive/reference/PORTING.md 17`** — cited by `HostRng` with upstream lines `:148`/`:794`
  that are a `rowNorm` call and a doxygen `@param`. The real lines are
  `detail/kmeans.cuh:152-157` and `:885`,`:890`. The *content* of the
  deviation (no `std::mt19937` counterpart) is correct.
- **`archive/reference/VENDOR_LIBRARIES.md` / `archive/reference/VENDOR_LIBS.md`** — cited by
  `reduce_by_key.mojo:227-231` as the reason `block_sum` replaced a
  hand-written tree reduction. That one survives the rule change (a block
  reduction is not device-wide and fuses fine), but the documents' framing of
  CUB as a substitution target does not.

## 7. WHAT I DID NOT DO, AND WHY

- **Did not remove the remaining per-iteration sync.** Theirs has it
  (`:491`), so per the brief it stays. If it is ever proposed for removal it
  needs a numbered DEVIATION BLOCK and Andrew's call, not a lane's.
  *Expected saving if removed:* the host would have to poll or double-buffer
  the shift scalar; at ~2.4 ms/iteration measured earlier this is the last
  serialization point in the loop, so it is worth pricing — but it is a
  deviation from their control plane and I am not taking it.
- **Did not fix divergence #12** (the self-neighbor factor in the l2_exp
  clamp). It changes epilogue arithmetic in `simt_kernel.mojo`, which another
  lane is reading as a model, and it is near-unreachable for k-means. **It is
  NOT near-unreachable for k-NN**: on a self-join every row meets itself with
  bit-equal norms, which is precisely the case the factor exists for. Flagged
  loudly for the k-NN lane.
- **Did not port `checkWeight`.** Theirs normalizes a private copy of the
  weights (`kmeans.cuh:862-872`); ours receives the caller's buffer as `mut`
  and normalizing it in place would mutate the caller's data. Recorded in
  `UNPORTED.tsv`.
- **Did not relax the metric refusal in `validate()`.** Opening it up would
  route Cosine into an arm that was never written for it. Relabeled as ours.
- **Ran no timing benchmarks**, per the brief.

## 8. BUILD AND CHECK EVIDENCE

```
tools/with_build_lock.sh pixi run --manifest-path .../mojotrees/pixi.toml \
  mojo build -I . cluster/kmeans_main.mojo -o /tmp/kmeans_final2
```
Builds clean (two pre-existing unused-variable warnings in
`min_cluster_distance_compute.mojo`). `bench/bench_main.mojo` also still
builds — it constructs `KMeansParams.default()`, so the removed fields do not
reach it.

**Baseline (before any change):**
```
check_reach_by_sabotage OK: centroid_norm moved 384/512 labels; x_norm moved 512 distances and 0 labels, which is the predicted shape
check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), 2 iterations
check_device_inclusive_scan OK: 20000 entries, worst relative error 0.0, past one block's worth
check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a permutation through the k-means++ path, inertia 170.703125, 2 iterations
check_fused_reduction_across_lanes OK: 512 rows x 40 clusters match a host argmin, winners spread over 10 lane groups
```

**After: byte-identical.** Same permutation, 0/512 misassigned, inertia
`170.703125`, **2 iterations**. Run **25×** after the change and **22×** after
the doc pass: `25 check_kmeans_fit OK: ... 2 iterations` / `22 ... 2
iterations` — one line, zero variance. No flakiness from the removed launches.

Why the iteration count did not move even though the semantics did: on this
fixture the assignment is already correct after iteration 1, so iteration 2
recomputes bit-identical fixed-point sums and the shift is exactly `0.0 < tol`.
The old one-iteration-late rule and the new same-iteration rule both stop at
2 here. On data that converges gradually the new count will be one lower, and
that is the correct one.

### Sabotage (reach, not digest)

| # | sabotage | prediction | result |
|---|---|---|---|
| S1 | host reads `1.0e30` instead of the transferred shift scalar | never converges → `max_iter + 1` = 51 iterations, inertia unchanged (stable fixed point) | **2 → 51 iterations, inertia 170.703125.** Host test and the un-clamped `max_iter+1` accounting are both live |
| S2 | `SUM_MODE_SQDIFF` drops its second operand (`d = v` instead of `v - b`) | shift becomes `sum(cur²)`, huge → never converges → 51 | **2 → 51 iterations.** The device map feeds the host decision |
| S3a | `raise` planted as the first statement inside `if params.inertia_check:` | default fit must still pass — the block must be dead by default | **passed, 2 iterations.** Gate is real |
| S3b | same `raise`, with `inertia_check` default flipped to `True` | must raise | **`Unhandled exception: SABOTAGE S3: inertia_check block reached`.** Both sides of the gate proven |
| S3c | `inertia_check = True`, no sabotage | must still fit correctly | **passed, 2 iterations, inertia 170.703125.** The gated arm is correct, not just reachable |
| S4 | post-loop assignment reads the stale `centroids` buffer (init, +30 off) instead of `cur_centroids` | labels must move | **`384 of 512 rows landed in the wrong cluster`.** The post-loop pass is what writes the returned labels — an fidelity gain from divergence #4 |

All sabotages reverted; the final binary is built from clean sources
(`grep -n SABOTAGE cluster/` → empty).
