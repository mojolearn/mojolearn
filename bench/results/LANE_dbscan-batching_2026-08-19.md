# LANE dbscan-batching — the audit, the knob, and the per-phase timer

**Verdict, one line.** `compute_batch_size` was one if-state away from
upstream — the `MAX_LABEL / n_rows` clamp ran unconditionally where
`dbscan.cuh:71` gates it off for RBC — and that if-state is now theirs;
`max_mbytes_per_batch` was already plumbed end-to-end and its name and
default are now VERIFIED against upstream rather than assumed; and the
runner prints per-phase, per-batch wall time behind an opt-in flag plus a
dedicated main, so the `weak_cc_batched` suspicion at 50,000 can be
confirmed or killed by one command. Along the way, arithmetic on already
recorded numbers falsified the previous lane's claim that the int32 clamp is
why 50,000 batches in two — **the clamp never bound any default-budget run
on this device; the 80% memory budget did** — and that sentence is corrected
at its source in this same commit.

No timing was run beyond a smoke fit at n = 4,000 to prove the
instrumentation prints. The 50,000 measurements are the orchestrator's; the
exact commands are in §5.

**CONCURRENT LANDING, read before trusting any single row below.** While this
lane worked, the rbc-maxk lane landed cuML's two-loop `max_k` dispatch INTO
`runner.mojo` in the same shared checkout, interleaved with this lane's
instrumentation (their loop-1 `maxklen` reduce deliberately sits inside this
lane's `mask.vertexdeg` window, as it sits inside cuML's nvtx VertexDeg
range). Both lanes were swept into one commit. Rows below that were written
against the three-walk RBC arm are marked; the PHASE format notes reflect
the MERGED state, which is the state the suite transcript in §4 and the
smoke in §3 were re-taken on.

---

## 1. TASK 1 — THE AUDIT, IF-STATE BY IF-STATE

`dbscan/gbdt/dbscan/dbscan.mojo` against
`upstream/cuml/cpp/src/dbscan/dbscan.cuh` at `00094f7`, then the batch's path
through `runner.mojo` against `runner.cuh`. Every row was read in their file,
not recalled.

### 1.1 `compute_batch_size` (`dbscan.cuh:34-98`)

| their line | what theirs does | ours | verdict |
|---|---|---|---|
| `:47` | `if (neigh_per_row <= 0) neigh_per_row = n_rows` | same | MATCHES |
| `:55` | `est_mem_per_row = n_rows * sizeof(bool) + (neigh_per_row + 2) * sizeof(Index_)` | `n_rows * 1 + (npr + 2) * 4` | MATCHES (bool = 1, Index_ = int32 = 4) |
| `:60` | `est_mem_fixed = n_rows * (sizeof(Index_) + sizeof(bool))` | `n_rows * (4 + 1)` | MATCHES |
| `:65` | `ASSERT(est_mem_per_row > 0, ...)` | raise on `<= 0` | MATCHES |
| `:66` | `batch_size = (max_mbytes_per_batch * 1000000 - est_mem_fixed) / est_mem_per_row` — **`size_t`, so a budget under the fixed cost WRAPS**, and `:69`'s `min` turns the wrap into a full batch; a budget between fixed and fixed+one-row yields batch 0, which divides by zero at `runner.cuh:131` | raise on both | **DEVIATION 37** (PORTING.md). The MB unit is decimal 1,000,000 in both, verified against their literal |
| `:69` | `min((size_t)n_owned_rows, batch_size)` | same | MATCHES |
| `:71` | **`if (eps_nn_method != EpsNnMethod::RBC)`** around the whole clamp block `:71-95` | **WAS MISSING: the clamp ran unconditionally.** The docstring knew and said "fixing it means threading eps_nn_method in" | **FIXED.** `eps_nn_method: Int` added in their parameter position (`:37`), clamp wrapped, threaded from `dbscan_fit_impl`. Reach on both sides in §4 |
| `:74-83` | clamp to `MAX_LABEL / n_rows` with an info log | same arithmetic, no log | MATCHES (the log is a CUML_LOG_INFO) |
| `:86-94` | "smaller index type would be sufficient" info, gated on `sizeof(Index_) > sizeof(int)` | absent | MATCHES BY DEADNESS: false for int32, the only index type here |
| `:96` | `estimated_memory` out-param | absent | DOCUMENTED, not ported: its only consumer is the debug log at `dbscan.cuh:171-173` |

### 1.2 `dbscanFitImpl` (`dbscan.cuh:101-207`)

| their line | theirs | ours | verdict |
|---|---|---|---|
| `:118-120` | `algo_vd = (Precomputed) ? 2 : 1`, `algo_adj = 1`, `algo_ccl = 2` | constants, L2 only | MATCHES (documented in the file header) |
| `:125` | `ASSERT(n_rows > 0, "No rows ...")` | raise, same message | MATCHES |
| `:127-139` | opg rank arithmetic | unported | in `PORTED_MAP.tsv` as such |
| `:147` | `if (max_mbytes_per_batch == 0)` → query device | same | MATCHES |
| `:151-153` | `dataset_memory` = `n*n*sizeof(T)` for Precomputed else `n*cols*sizeof(T)` | `n_rows * n_features * 4` | MATCHES for the ported metric; Precomputed goes with the metric-switch UNPORTED row |
| `:158` | `(80 * total_memory / 100 - dataset_memory) / 1e6` — double division floored into `size_t` | integer `// 1000000` | MATCHES for every non-negative value |
| `:168-169` | `compute_batch_size(estimated_memory, n_rows, n_owned_rows, eps_nn_method, max_mbytes_per_batch)` | **eps_nn_method now passed** | **FIXED** (was omitted) |
| `:194-207` | `run` twice: once null-workspace for the size, then allocate, then run | one allocation of the same buffers inside `dbscan_fit_impl` | MATCHES in effect, documented in the docstring |

### 1.3 How the batch feeds `runner.mojo` and each phase

| their line | theirs | ours | verdict |
|---|---|---|---|
| `runner.cuh:131` | `n_batches = ceildiv(n_owned_rows, batch_size)` | `(n_rows + batch - 1) // batch` | MATCHES (single node, `n_owned == n_rows`) |
| `:169-177` | workspace: adj `bool[N*batch]`, core `bool[N]`, m `bool[1]`, vd/ex_scan `Index[batch+1]`, row_counters `Index[batch]`, labels_temp + work_buffer `Index[N]` each | same set: `adj` uint8 `batch*n`, `core` n, `d_flag` 1, `vd`/`ex_scan` `batch+1`, `labels_temp`/`work_buffer` n, `block_sums` replacing `row_counters` + thrust scratch (documented) | MATCHES. The 256-byte `alignTo` is an rmm-suballocation artifact; ours are separate buffers. **The estimate is method-blind in theirs too**: `adj_size` is charged even though RBC never reads it (`:169`, and the dead-buffer measurement in LANE_rbc-build §0.3), and the RBC index lives OUTSIDE the workspace (`:233-240`), uncounted by either budget. Ours mirrors both |
| `:181-186` | `ASSERT(N * batch_size < MAX_LABEL)` — **unconditional** | **WAS MISSING ENTIRELY.** Now raised on the brute arm only | **ADDED, SCOPED.** Theirs cannot bind on their RBC path (RBC requires int64, `:143-150`); ours is int32 with RBC reachable (DEVIATION 35), where the honest bound is the EDGE count, already refused at the query (`runner.mojo`, the `nnz1 > MAX_LABEL` raise). Copying it unconditionally would re-impose the clamp `:71` just gated off. A fit that REQUESTS rbc but downgrades to brute (the `D > MAX_LABEL/N` guard) now dies exactly where theirs does |
| `:245-247` | loop 1 REVERSED, `n_points = min(n_owned - i*batch, batch)` | same | MATCHES |
| `:281-283` | `update_host(&curradjlen, vd + n_points, 1)` + sync per batch | same | MATCHES |
| `:288-293` | `maxklen[i] = thrust::reduce(vd, ..., maximum{})` per batch (rbc) | absent at audit time | PORTED by the concurrent rbc-maxk lane in the same commit (`rbc_max_reduce_kernel`, device-side, inside the `mask.vertexdeg` window as theirs is inside their VertexDeg range). The `UNPORTED.tsv` row this audit added lived for one working-tree afternoon and was deleted on landing |
| `:325` | loop 2 `if (n_points <= 0) break` | same | MATCHES |
| `:327-349` | vertexdeg `if (i > 0)` | at audit time: RBC arm counted + filled EVERY batch, batch 0 included — three walks (LANE_rbc-build E4) | RESOLVED by the concurrent rbc-maxk lane in the same commit: batch 0 reuses loop 1's resident CSR, batches > 0 take the one-pass `max_k` arm under `algo.cuh:119`'s spare guard (`rbc_take_one_pass`) or fall back to count + fill. Two walks, as theirs |
| `:355` | AdjGraph `if (!sparse_rbc_mode)` | same guard | MATCHES |
| `:374-386` | `weak_cc_batched` per batch, over ALL N, init inside | same (`csr.mojo:133`) | MATCHES — and it is why the 50,000 suspicion exists: every batch re-inits N labels and iterates to convergence with host syncs per pass |
| `:389-399` | MergeLabels `if (i > 0)` | same, now written as their separate `if` | MATCHES |

### 1.4 The knob's name and default, verified in upstream

No `DbscanParams` struct exists in cuML's C++; the knob is a parameter of the
public `fit` overloads. There, it is spelled **`max_bytes_per_batch`** while
the doc comment says **"the maximum number of megabytes"**
(`cuml/cpp/include/cuml/cluster/dbscan.hpp:54-56`), **default 0** (`:73`,
`:87`, `:102`, `:116`). The Python estimator spells it
**`max_mbytes_per_batch`**, default `None`, coerced to 0 at fit
(`python/cuml/cuml/cluster/dbscan.pyx:302`, `:316-317`). The internal name is
`max_mbytes_per_batch` (`dbscan.cuh:38`, `:111`). Ours keeps the internal /
Python spelling and the default 0 = the 80%-of-total estimate; all of this is
now cited in `dbscan_fit_impl`'s docstring instead of assumed.

### 1.5 A previous lane's claim, falsified by its own numbers

LANE_rbc-build E1 said the int32 clamp "binds above n = 46,341 ... and is why
50,000 is the first size with more than one batch." The clamp value at
50,000 is `2147483647 // 50000 = 42,949`; the batch that lane recorded at the
default budget is **40,669**. A clamp-bound batch would BE 42,949. The budget
bound it: one batch at 50,000 costs `50,000 x 250,008 B ≈ 12.5 GB` against a
~10.2 GB default budget, and a clamped batch would cost
`(MAX_LABEL/N)(5N + 8) ≈ 5 x MAX_LABEL ≈ 10.7 GB` at EVERY n — above that
budget — so the clamp never bound a default-budget run on this device at any
size. Corrected in `LANE_rbc-build_2026-08-19.md` E1 in this commit.
Consequence worth stating plainly: **the `:71` gate fix changes no
default-budget batch count on this box.** It is a dispatch-fidelity fix, and
it changes user-RAISED budgets (a user asking RBC for one big batch above
n = 46,341 now gets it, as cuML would give it).

---

## 2. TASK 2 — WHAT WAS EXPOSED

Nothing needed inventing: `max_mbytes_per_batch` already flowed
`dbscan_fit_impl -> compute_batch_size -> dbscan_fit(batch_size)` as one
number, with no scattered overrides. What this lane did:

- verified name + default against upstream (§1.4) and wrote the citations
  into the docstring;
- threaded `eps_nn_method` beside it into `compute_batch_size`, in their
  parameter order (`dbscan.cuh:37`);
- enumerated the callers: `bench/bench_main.mojo:252` (default 0),
  `bench/scaling_main.mojo:122,131,163` (explicit 0 = cuML's default),
  `dbscan/mojo_only/dbscan_check.mojo` (explicit values in the two new
  checks), `dbscan/phase_main.mojo` (argv). `dbscan_fit`'s own `batch_size`
  parameter is their `Dbscan::run(batch_size)` internal seam and is only fed
  by `dbscan_fit_impl` and by checks that mirror the null-workspace calling
  pattern.

---

## 3. TASK 3 — THE INSTRUMENTATION

`dbscan_fit` takes `phase_timing: Bool = False` (PORTING.md 38). It is the
port of what cuML hangs on these exact boundaries — one nvtx range per phase
(`runner.cuh:255`, `:299`, `:330`, `:355`, `:373`, `:397`, `:411`) and a
per-batch `CUML_LOG_DEBUG` gated by `verbosity` (`dbscan.cuh:114`) — printed
as wall-clock lines because Metal has no nvtx consumer. **No phase boundary
was invented**: every timestamp lands on a `ctx.synchronize()` the port
already performs, so the flag adds zero synchronization, and off (default)
nothing prints, so every existing check and benchmark line is untouched.

Format (documented on `dbscan_fit`):

    PHASE budget mbytes <mb> batch <b>                  (dbscan_fit_impl)
    PHASE plan n_rows <N> batch <b> n_batches <nb> method <rbc|brute>
    PHASE mask.vertexdeg batch <i>/<n> <ms>             loop 1, incl. the vd[n] readback, as their range :255-296
    PHASE mask.corepoints batch <i>/<n> <ms>
    PHASE label.vertexdeg batch <i>/<n> <ms>            loop 2, batches > 0 only (batch 0 is resident from loop 1); rbc = one max_k pass, or count + fill when the bound does not fit
    PHASE label.adjgraph batch <i>/<n> <ms>             brute arm only (:355)
    PHASE label.weak_cc batch <i>/<n> <ms> passes <p>   <p> = propagation passes, the number the 50k suspicion predicts grows with batch SIZE
    PHASE label.merge_labels batch <i>/<n> <ms>         batches > 0 (:389)
    PHASE final_relabel batch 1/1 <ms>

`<i>` is 1-based like their `"- Batch %d"`; loop 1 prints in its reversed
order. `dbscan/phase_main.mojo` is the dedicated main; it builds the DBSCAN
scaling fixture VERBATIM (`bench/scaling_main.mojo`: d = 8,
`_u01(i, f, 4) * 4.0`, eps 0.30, min_pts 5) and prints
`ARM dbscan_phase@<n> <total ms>` per repeat.

### The smoke run (n = 4,000, budget forced to 6 batches, rbc)

Taken TWICE: once before the rbc-maxk landing, once after, because that
landing rewrote the function under the instrumentation. The merged-state
transcript is the binding one; note batch 1 correctly prints NO
`label.vertexdeg` line — loop 2 reuses batch 0's resident CSR (`:327`):

    $ /tmp/phase_probe 4000 15 rbc 1
    PHASE budget mbytes 15 batch 748
    PHASE plan n_rows 4000 batch 748 n_batches 6 method rbc
    PHASE mask.vertexdeg batch 6/6 2.408
    PHASE mask.corepoints batch 6/6 0.478
    ... (batches 5..1) ...
    PHASE label.weak_cc batch 1/6 1.023 passes 1
    PHASE label.vertexdeg batch 2/6 2.113
    PHASE label.weak_cc batch 2/6 0.572 passes 1
    PHASE label.merge_labels batch 2/6 0.933
    ... (batches 3..6) ...
    PHASE final_relabel batch 1/1 1.139
    ARM dbscan_phase@4000 37.043

Every line type on the rbc arm printed. **The brute-arm prints
(`label.adjgraph`, brute `label.vertexdeg`) compiled but have NOT yet
printed** — the single smoke run went to the rbc arm because that is the
arm under suspicion; the orchestrator's first `brute` invocation exercises
them, and if one were malformed it fails loudly, not silently.

---

## 4. CHECKS

`check_dbscan_batching_agrees` untouched and green. Two new checks, both in
`dbscan/mojo_only/dbscan_check.mojo`, called from `dbscan/dbscan_main.mojo`:

- **`check_dbscan_max_mbytes_moves_the_batch`** — reach, host arithmetic:
  two budgets give different batch counts (1 MB -> 19 batches, 1000 MB -> 1
  at n = 1,932); the `:71` gate keeps rbc at 50,000 rows where brute clamps
  to 42,949 at a 100 GB budget; and BOTH methods agree exactly where the
  clamp cannot bind, so the gate leaks nothing into the estimate.
- **`check_dbscan_tiny_budget_agrees`** — end-to-end through
  `dbscan_fit_impl`, BOTH sides of the `algo.cuh:226` switch, on a scattered
  hashed fixture (six blobs x 320 hashed-jitter rows + 12 isolated noise
  points; every blob spans several 102-row batches): a forced
  `max_mbytes_per_batch = 1` yields labels IDENTICAL to one batch, ids
  canonical 0..5 and noise -1, and the returned pass count strictly rises
  (rbc 56 vs 2, brute 38 vs 2), which is the observable proof the budget
  reached the batch loop and not just the arithmetic.

Full transcript at the merged state (the `rbc_two_loop_arms` line is the
concurrent rbc-maxk lane's check, green in the same suite):

    check_fused_eps_agrees_with_materialized OK: 31950 cells identical to the materialized path AND to a float64 host oracle (15298 of them neighbours, so the pattern is irregular), 150 degrees identical, vd[m] = 15298
    check_dbscan OK: 3/3 blobs each one whole cluster with ids exactly {0, 1, 2}, 0 merges, 12/12 isolated points labelled -1 as scikit-learn does, converged in 2 propagation passes
    check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12 gives 1, which is impossible without the neighbourhood kernel running on real distances
    check_exclusive_scan_beyond_the_old_cap OK: 2000000 entries exact across 977 blocks, total 12000001
    check_dbscan_batching_agrees OK: 5 batches with 4 merge_labels folds give labels identical to one batch, in an adj buffer 4x smaller than the unbatched run needs
    check_dbscan_rbc_matches_brute OK: 612 points labelled identically by the ball-cover index and by brute force, 3 clusters
    check_dbscan_rbc_two_loop_arms OK: the one-pass arm (3 clusters) and the two-pass fallback (2 clusters) both label identically to unbatched brute force, and algo.cuh:119's guard provably routes each fixture to its arm
    check_dbscan_max_mbytes_moves_the_batch OK: 1 MB -> 19 batches and 1000 MB -> 1 batch at n = 1932; at n = 50000 the dbscan.cuh:71 gate keeps rbc at 50000 rows where brute clamps to 42949, and both agree at 309 rows when the clamp cannot bind
    check_dbscan_tiny_budget_agrees OK (rbc): max_mbytes_per_batch = 1 forced 19 batches (56 passes vs 2 for one batch) and all 1932 labels match one batch: 6 blobs whole with ids 0..5, 12 noise points at -1
    check_dbscan_tiny_budget_agrees OK (brute): max_mbytes_per_batch = 1 forced 19 batches (38 passes vs 2 for one batch) and all 1932 labels match one batch: 6 blobs whole with ids 0..5, 12 noise points at -1

---

## 5. WHAT THE ORCHESTRATOR SHOULD RUN FOR THE 50,000 DIP

Build once:

    cd /Users/andrewhendel/CascadeProjects/mojolearn
    tools/with_build_lock.sh pixi run \
      --manifest-path /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
      mojo build -I . dbscan/phase_main.mojo -o /tmp/phase_probe

Then, one thermal window, the three budgets LANE_rbc-build measured the
1.4-2.0x swing across (batch counts are unchanged by this lane's gate fix,
§1.5):

    /tmp/phase_probe 50000 0 rbc 3        # default: 2 batches of 40669 + 9331
    /tmp/phase_probe 50000 6251 rbc 3     # 2 batches of 25002 + 24998
    /tmp/phase_probe 50000 3126 rbc 3     # 4 batches of 12502
    /tmp/phase_probe 16000 0 rbc 3        # 1 batch, the other half of the table
    /tmp/phase_probe 16000 641 rbc 3      # 2 batches
    /tmp/phase_probe 16000 161 rbc 3      # 8 batches

How to read it: the suspicion (LANE_rbc-build §0.4) is that
`weak_cc_batched` re-inits all N labels per batch and needs MORE PASSES on a
LARGER batch. It is confirmed if, at fixed n, the summed
`PHASE label.weak_cc` milliseconds and their `passes` fields fall as the
batch shrinks while `label.vertexdeg` stays ~flat per row (the ball cover is
measured flat across these shapes); it is killed if `weak_cc`'s share is
flat and the movement sits in `merge_labels` or in the mask loop. Two
merged-state cautions: batch 0 emits no `label.vertexdeg` line at all now
(resident CSR), and `label.vertexdeg` for batches > 0 may be the one-pass
`max_k` arm rather than count + fill — the totals at these budgets therefore
sit BELOW the LANE_rbc-build §0.3 fit numbers, which were taken on the
three-walk runner. The decomposition WITHIN one process is the primary read
— cross-process totals drift 2-3x on this box. A brute run
(`/tmp/phase_probe 4000 15 brute 1`) also completes the print coverage noted
in §3.

---

## 6. FILES TOUCHED (this lane's only)

- `dbscan/gbdt/dbscan/dbscan.mojo` — the `:71` gate, DEVIATION 37 note,
  knob citations, `phase_timing` plumb-through, budget print.
- `dbscan/gbdt/dbscan/runner.mojo` — `phase_timing` + PHASE prints on
  cuML's nvtx boundaries; the scoped `runner.cuh:181` overflow raise; header
  now names the unported max_k dispatch (a peer had already removed the
  stale "ball-cover arm" entry from that header's NOT PORTED list).
- `dbscan/mojo_only/dbscan_check.mojo`, `dbscan/dbscan_main.mojo` — the two
  new checks.
- `dbscan/phase_main.mojo` — NEW, the dedicated measurement main.
- `dbscan/PORTED_MAP.tsv`, `dbscan/UNPORTED.tsv` — rows updated; the stale
  "cuvs neighbors/ball_cover not ported" row (false since the RBC default
  landed) was replaced by a two-loop max_k dispatch row, which the rbc-maxk
  lane then deleted the same day on porting exactly that dispatch.
- `PORTING.md` — deviations 37, 38.
- `bench/results/LANE_rbc-build_2026-08-19.md` E1,
  `bench/results/LANE_dispatch-audit_2026-08-19.md` rows 8/D2/F1 — falsified
  or completed statements corrected in this commit.
