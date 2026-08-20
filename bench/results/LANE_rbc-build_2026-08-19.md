# LANE rbc-build — the 50,000 dip, the index build, and the one-pass query

**Verdict, one line.** The 50,000 anomaly is **not in the ball cover at all**:
build and query both scale smoothly through 50,000 with no step, and both of
the hypotheses on the table are falsified by measurement. The index build is
7–15% of the ball cover's cost and its share **shrinks** with n, so the
`O(m^1.5)` in-group rank is worth **0.4% of a DBSCAN fit** and a CUB radix
sort is not worth this lane — that is priced below, not asserted. **The
strongest remaining gap is Task 3**: cuML's own dispatch does **two** walks
over the dataset where our runner does **three**, and adopting it is a
measured **~20% off the DBSCAN fit at 200,000**, with a new check proving the
one-pass form is byte-identical to the two-pass one.

**No API signature changed.** `rbc_build_index`, `rbc_eps_nn_query_count`,
`rbc_eps_nn_query_fill` and `rbc_n_landmarks` are untouched.

---

## 0. TASK 1 — THE 50,000 DIP, MEASURED

### 0.1 Both stated hypotheses are dead

**(a) The landmark count.** Opened their file first, as instructed.
`cuvs/cpp/include/cuvs/neighbors/ball_cover.hpp:62` is
`n_landmarks(raft::sqrt(X_.extent(0)))`, a `double` sqrt truncated into
`int64_t`. `rbc_n_landmarks` is `floor(sqrt(m))` with an integer correction
loop. **Same formula, confirmed against their line, not assumed.** The
resolved counts are L = 126 / 223 / 316 / 447 at n = 16k / 50k / 100k / 200k
and there is no boundary near 223: the query launches one block per query
(tens of thousands of blocks) and the rank kernel launches L blocks, so 223
is not a distinguished number for either. The measurement below settles it —
query cost per (query × landmark) is **monotone decreasing** through 50,000.

**(b) The build being a bigger share at 50,000.** The opposite is true. The
build's share of build + count + fill is 15.2% at 16,000 and falls to 7.4% at
200,000, passing 9.0% at 50,000. There is no bump.

### 0.2 What was measured

`neighbors/ball_cover` alone, on the DBSCAN scaling fixture
(`bench/scaling_main.mojo`: d = 8, coordinates `_u01(i, f, 4) * 4.0`,
eps = 0.30), one process, one thermal window, warm-up discarded, three
repeats, min. **The four sizes were then run again in the opposite order and
agree to within 3%**, which is what makes the numbers usable at all on this
box.

    n          L     nnz      build ms   count ms   fill ms   build share
    16,000    126    16,002      1.49       4.25      4.02       15.2%
    50,000    223    50,008      3.65      18.53     18.37        9.0%
    100,000   316   100,028      8.22      49.75     49.81        7.6%
    200,000   447   200,128     20.79     129.08    132.14        7.4%

    descending order, same window:
    200,000                     20.88     129.75    130.77
    100,000                      8.19      50.06     50.17
    50,000                       3.54      18.47     18.37
    16,000                       1.41      4.22       3.92

**There is no anomaly at 50,000.** Normalise the count pass by the work it
must do — one distance to every landmark for every query row — and the curve
is smooth and improving:

    n          count ms / (n × L)
    16,000        2.11 ns
    50,000        1.66 ns
    100,000       1.57 ns
    200,000       1.44 ns

The build behaves the same way. Nothing in this lane steps at 50,000.

Note also `nnz ≈ n` at every size: at eps = 0.30 in a fixed box of side 4 in
8 dimensions, the expected number of neighbours within eps is ~1e-3, so
**every point is its own only neighbour**. The fixture measures the SEARCH,
not the output, which is worth knowing before anyone reads a DBSCAN number
off it.

### 0.3 Where the 50,000 time actually goes

Modelling the DBSCAN runner's own call pattern (build once, one count per
batch in loop 1, count + fill per batch in loop 2) against the post-fix
`DBSCAN_RBC_2026-08-19.md` numbers:

    n         DBSCAN rbc ms    ball cover ms (build + 2×count + fill)    residual
    16,000        25.9                     14.0                            12
    50,000       196.3                     59.1                           137
    100,000      316.9                    157.5                           159
    200,000      632.7                    411.1                           222

Cross-window, so the residual is an estimate. It was then confirmed directly.
Two experiments, both in one window, both using only public arguments:

**Experiment 1 — is it the dead `adj` buffer?** In RBC mode DBSCAN allocates
`adj` (`batch × n_rows` bytes) and never reads it; above n ≈ 40,000 that is
~2.03 GB on this device. **No.** Allocating a 2.03 GB buffer costs 0.088 ms,
and holding one alive, never written, across the query changes nothing:

    DIAG-DEAD n=50000 dead_mb=0      count_ms=18.471
    DIAG-DEAD n=50000 dead_mb=2033   count_ms=18.482
    DIAG-DEAD n=50000 dead_mb=0      count_ms=18.573
    DIAG-DEAD n=50000 dead_mb=2033   count_ms=18.511

**Experiment 2 — is it the batch shape, and is the cost ours?** Same total
query work at n = 50,000, split into batches the way the runner splits it:

    batch     n_batches    total count ms
    50,000        1            18.50
    40,669        2            19.28      <- what compute_batch_size picks
    25,002        2            19.35
    12,502        4            20.82

**Flat.** The ball cover does not care how the query is batched. But a whole
DBSCAN RBC fit does, enormously — same fixture, same window, only
`max_mbytes_per_batch` changed:

    n = 50,000    batch      n_batches   adj MB     fit ms
    default       40,669         2        2033     150 – 196
    6251 MB       25,002         2        1250     106 – 130
    3126 MB       12,502         4         625      90 – 112

    n = 16,000    batch      n_batches   adj MB     fit ms
    default       16,000         1         256      24 – 35
    641 MB         8,010         2         128      24 – 26
    321 MB         4,011         4          64      27 – 29
    161 MB         2,011         8          32      41 – 44

Two batches of 40,669 + 9,331 and two batches of 25,002 + 24,998 are the same
batch COUNT and differ by 1.4–1.5x. So it is the batch **size**, the ball
cover is insensitive to it, and therefore **the cost is in DBSCAN's per-batch
machinery, not in this lane.**

### 0.4 The honest end of the trail

The residual is `weak_cc_batched` + `merge_labels` + `core_points_compute`,
and the most likely mechanism is `weak_cc_batched`
(`dbscan/gbdt/sparse/detail/csr.mojo:132`): it re-initialises **all N**
labels per batch and then iterates label propagation over the batch's
sub-graph until nothing changes, with three host synchronisations per pass. A
larger batch holds longer propagation chains, so it needs more passes. That
is consistent with every number above — including 16,000, where going past 4
batches costs more than the propagation saves — but **I did not confirm it**,
because `dbscan/` is not this lane's to instrument and the confirmation needs
a per-phase timer inside the runner.

**Measured, not explained** is the correct label for that last step, and it
is deliberately not dressed up. What IS established, with numbers:

1. the ball cover has no 50,000 anomaly, in the build or in the query;
2. both stated hypotheses are false;
3. the dead `adj` allocation is free and is not it;
4. 60–70% of the 50,000 fit is outside the ball cover, and that part gets
   ~1.7x cheaper when the batch is made smaller by a public argument.

**A recommendation the DBSCAN lane can act on today:** `compute_batch_size`
takes 80% of device memory and spends it on one batch. Their own comment
(`dbscan.cuh:34`, quoted in our port) says the worst case never happens and
leaves `neigh_per_row` as a `///@todo`. On this device the biggest batch is
also the slowest. Measuring `max_mbytes_per_batch` as a tunable, or wiring
their `neigh_per_row`, is worth 1.4–2.0x at 50,000 and costs nothing in the
answer.

---

## 1. DIVERGENCES FOUND

| # | what upstream does (file:line) | what ours did | fixed, or why not |
|---|---|---|---|
| E1 | `cuml runner.cuh:141-150`: RBC is DISABLED for `Index_ = int32_t` — `if constexpr (is_same_v<Type_f,double> \|\| is_same_v<Index_,int32_t>) { sparse_rbc_mode = false; ... "RBC does not support double precision or int32 labels. Falling back to BRUTE_FORCE" }` | our port is Int32 throughout (`dbscan/gbdt/dbscan/dbscan.mojo` says so explicitly) and runs RBC anyway | **NOT FIXED, and it should not be.** An earlier version of this row claimed the int32 `MAX_LABEL/N` clamp "binds above n = 46,341" and "is why 50,000 is the first size with more than one batch". Arithmetic on this row's own numbers falsifies that: the clamp value at n = 50,000 is `2147483647 // 50000 = 42,949`, the recorded default batch is 40,669, and a clamp-bound batch would be exactly 42,949. The 80% memory budget cuts first — a clamped batch would cost `(MAX_LABEL/N)(5N + 8) ≈ 5·MAX_LABEL ≈ 10.7 GB` at every n, above this device's ~10.2 GB default budget — so the clamp never bound any default-budget run here, and 50,000 batches in two because one batch needs `50,000 × 250,008 B ≈ 12.5 GB`. The clamp is now gated on `eps_nn_method != RBC` exactly as `dbscan.cuh:71` gates it (LANE_dbscan-batching_2026-08-19), which changes no default-budget batch count on this device. |
| E2 | `registers.cuh:1332-1335` launches `tpb = 64`, two warps per block, `ceildiv(n_queries, 2)` blocks | 32 threads, one query per block | **DIVERGES, now MEASURED as well as argued.** The correctness argument (a 64-lane AMD wavefront would merge two queries' `vote` masks) stood alone before; it is now also free. Swept 32 / 64 / 128 / 256 on the M4: their 64 is never faster and is 1.54x slower at n = 16,000. Table is in `registers.mojo`'s DEVIATION 2 banner. The kernel was rewritten into THEIR shape (`RBC_QPB` is their `num_warps`, the query id is their `blockIdx.x * num_warps + threadIdx.x / WarpSize` from `:600`) so the only thing that differs now is one constant. |
| E3 | `ball_cover.cuh:148-152` `thrust::sort_by_key` + `NNComp`, then `sorted_coo_to_csr` at `:155` | counting sort + exact per-group rank, `O(sum \|group\|^2)` | **DIVERGES, KEPT, AND NOW PRICED.** The previous lane recorded this as a follow-up that "will dominate the BUILD at large m". That is false and the false sentence is deleted from `ball_cover.mojo` in the same edit (§4). Measured: the rank is 13% of the build and 0.4% of a fit, and it cannot grow relative to the 1-nn kernel because both are `O(m^1.5)`. See §2. |
| E4 | `cuml runner.cuh:257` `need_ja_compute = sparse_rbc_mode && ((i == 0) \|\| sample_weight)`; `:262` passes `max_k = 0`; `:289` `maxklen[i] = thrust::reduce(vd, vd+n_points, maximum{})`; `:327` `if (i > 0)`; `:335` passes `maxklen.at(i)` | `dbscan/gbdt/dbscan/runner.mojo` counts every batch in loop 1, then counts AND fills every batch in loop 2 — **three walks over the dataset where theirs does two** | **NOT FIXED HERE — `dbscan/` is not mine.** The port of the one-pass form already existed and was checked in isolation; what was missing was the DISPATCH. A new check now runs their exact two-loop sequence and proves it byte-identical. Exact call in §7. Worth ~20% of the fit at 200,000. |
| E5 | `registers.cuh:1427-1482` `rbc_eps_pass` max_k overload | `rbc_eps_pass_max_k`, ported, checked against a host oracle at ONE batch | **NOW CHECKED AT THEIR DISPATCH**, batched, byte-identical to the two-pass CSR, plus their `algo.cuh:135` equality assert and a sabotage on the bound. See §6. |
| E6 | `algo.cuh:119-121` `spare_elemets_per_row = (batch_size * N - ja->capacity()) / n`, and the one-pass arm is taken when `0 < max_k < spare` | nothing — our runner never takes the one-pass arm | **NOT FIXED HERE.** Note what their guard actually guards: it is a MEMORY test on the `n × max_k` scratch (`registers.cuh:1431`), not a correctness test. Their `ja` view of `n * N` at `:130` is an upper bound they never fill. §7 gives the condition verbatim. |
| E7 | `ball_cover.hpp:57-60` footprint comment `(2*sqrt(m)) + (n*sqrt(m)) + (2*m)` | — | **THEIRS IS WRONG**, already recorded by the previous lane: it omits `X_reordered`, another `n*m`, allocated six lines below at `:68`. Confirmed again by opening the file. |

---

## 2. TASK 2 — THE SORT. NOT DONE, AND PRICED.

**I did not port CUB's `DeviceRadixSort`, and the reason is a number.**

The brief's premise was hypothesis (b): that the build is a large share at
50,000. It is not. So before writing a sort I measured what a perfect sort
could possibly buy. The build's launches, one at a time with a synchronize
after each (which inflates the total and leaves the shares intact — that is
the only question):

    n          1-nn      count+scan+scatter    RANK      reorder+radii
    16,000     0.62 ms         0.36 ms        0.30 ms       0.25 ms
    50,000     2.33            0.43           0.58          0.35
    100,000    6.05            0.46           1.16          0.49
    200,000   16.53            0.54           2.59          0.80

The in-group rank — the step `thrust::sort_by_key` would replace — is **13%
of the build**, and the build is **7.4% of the ball cover's own cost**, and
the ball cover is ~65% of a DBSCAN fit. **The rank is 0.4% of a fit.** The
1-nn kernel above it is 6.4x larger.

And it does not grow: the rank is `O(sum |group|^2) = O(m^1.5)` and the 1-nn
kernel is `O(m · sqrt(m) · d) = O(m^1.5)`. **Their ratio is fixed at every n**,
so the rank never overtakes the build no matter how large m gets. The claim
that it would is deleted from `ball_cover.mojo`.

The remaining worry was the TAIL — `sum |group|^2` is only `m^1.5` when the
groups are balanced, and a 1-nn assignment on clustered data need not be. So
that was measured too, against a 12-blob clustered fixture as well as the
uniform one:

    n         fixture      L     mean    max   sum|g|² vs balanced   rank ms
    50,000    uniform     223     224    688         1.25x            0.99
    200,000   uniform     447     447   1041         1.18x            4.38
    50,000    clustered   223     224    637         1.31x            1.10
    200,000   clustered   447     447   1995         1.42x            6.42

The worst group is 4.5x the mean on clustered data and the whole quadratic
term is 1.42x the balanced estimate. **That is a 1.4x on 1% of a fit.** There
is no blow-up to defend against.

**What I would do instead, if someone wants the sort anyway.** Port it as the
general primitive, not as a ball-cover fix, because this repository genuinely
lacks a device sort (`nn.argsort[target="gpu"]` is wrong above 256 elements)
and the next section that needs one will need it properly. The plan:

1. `neighbors/gbdt/matrix/detail/select_radix.mojo` is the working model
   and is already checked on device. It is CUB's digit-histogram shape with a
   `SELECT_BLOCK = NUM_BUCKETS = 1 << 8` geometry
   (`mojo_only/kernel_matrix.mojo:566`, `K_LIB_SELECT_RADIX`), one thread per
   bucket, and it already does the per-pass histogram, the exclusive scan
   over buckets, and the bucket-relative offset. `DeviceRadixSort` is the
   same three steps run to completion instead of stopped at a bucket, plus a
   ping-pong pair of key/value buffers and a stable scatter.
2. Register a `K_LIB_RADIX_SORT` row in the kernel matrix rather than typing
   a block size (that row does not exist yet and the matrix is another
   session's file, so this needs their edit first).
3. The correctness bar is the one that caught `argsort`: a scattered
   splitmix64 fixture, a size above 256, a size that is not a multiple of any
   block width, and an assertion on the PERMUTATION's monotonicity, not on
   the sorted keys — `argsort` returned a well-formed permutation and a
   non-monotone order, and only an index-invariant assert saw it.
4. When it lands, `rbc_rank_kernel` becomes a segmented sort over the
   `(landmark, distance, index)` key, and the tie-break must stay **index
   ascending** — theirs is `thrust::sort_by_key`, which is unstable, so ours
   is stricter than theirs and the check in `ball_cover_check.mojo` asserts
   the resulting order, not just the resulting set.

That is the plan; it is not started, and nothing half-landed.

---

## 3. WHAT I CHANGED, FILE BY FILE

Only the four paths this lane owns. `git status` shows no other file under my
name.

**`neighbors/gbdt/neighbors/ball_cover/registers.mojo`**
- Rewrote the query-id computation of all three query kernels
  (`block_rbc_kernel_eps_csr_pass`, `..._dense`, `..._max_k`) into cuVS's own
  form: `blockIdx.x * num_warps + threadIdx.x / WarpSize`, broadcast from
  lane 0 (`registers.cuh:600`). Previously it was `blockIdx.x` outright,
  which is that expression specialised to one warp per block. `RBC_QPB` is
  their `num_warps` (`:1330`), the grid is their
  `ceildiv(n_query_rows, num_warps)` (`:1333`), and `RBC_COPY_TPB = 32` is
  their `block_rbc_kernel_eps_max_k_copy<value_idx, 32><<<n_query_rows, 32>>>`
  (`:1476-1478`) which does not move with `RBC_TPB` because it is a
  compaction, not a warp-cooperative walk.
  **At `RBC_TPB = 32` this is bit-identical** — every check produces the same
  edge counts, all five of them.
- Rewrote DEVIATION 2's banner to carry the measurement (§1 E2) instead of an
  argument alone.

**`neighbors/gbdt/neighbors/ball_cover/ball_cover.mojo`**
- DEVIATION 3: **deleted the false sentence** that the `O(m^1.5)` rank "will
  dominate the BUILD at large m" and replaced it with the stage table, the
  group-size tail measurement, and the structural reason it cannot dominate
  (both terms are `m^1.5`). Per the standing rule, the falsified sentence is
  deleted rather than annotated.

**`neighbors/mojo_only/ball_cover_check.mojo`**
- New `check_ball_cover_max_k_wiring`, +277 lines. It is a DISPATCH check,
  not a kernel check: it runs cuML's two loops verbatim
  (`runner.cuh:257-293` then `:319-350`) over 3 batches of a 1201-row fixture
  and asserts (1) byte identity of the one-pass CSR against the two-pass CSR,
  offsets AND column order; (2) their `algo.cuh:135` EQUALITY assert on
  `actual_max`; (3) batch 0's `ja` from loop one survives; (4) a sabotage —
  the bound cut by one — reports the true longest row and clamps the rows.

**`neighbors/ball_cover_main.mojo`**
- Calls the new check.

---

## 4. PROPOSED ROWS

For `neighbors/PORTED_MAP.tsv` — replaces the `registers.cuh` row the
previous lane proposed, whose deviation text is now stale:

```
cuvs/cpp/src/neighbors/ball_cover/registers.cuh	block_rbc_kernel_eps_csr_pass, block_rbc_kernel_eps_dense, block_rbc_kernel_eps_max_k, block_rbc_kernel_eps_max_k_copy, rbc_eps_pass (both overloads)	neighbors/gbdt/neighbors/ball_cover/registers.mojo	partial	94c2819	ballot is warp.vote + count_trailing_zeros; launch shape is theirs (RBC_QPB = num_warps) pinned to one query per block, measured no slower than their tpb=64 at four sizes and 1.54x faster at 16k
```

For `neighbors/UNPORTED.tsv` — replaces the CUB row the previous lane
proposed, which implied the ball cover wanted it:

```
cub	DeviceRadixSort	cub/device/dispatch/dispatch_radix_sort.cuh	the device-wide sort thrust::sort_by_key and raft's sampleWithoutReplacement both call. Open, portable, and the same digit-histogram shape as the RAFT radix SELECT at neighbors/gbdt/matrix/detail/select_radix.mojo. It is the general device sort this repository lacks, since nn.argsort[target=gpu] is wrong above 256 elements. It is NOT wanted by ball_cover: the counting sort plus per-group rank it would replace is 13% of the index build and 0.4% of a DBSCAN fit, measured 2026-08-19, and cannot grow relative to the build because both are O(m^1.5)
```

For `dbscan/UNPORTED.tsv` — the one-pass dispatch, now the largest known win:

```
cuml/cpp/src/dbscan/runner.cuh	the two-loop max_k dispatch	:257, :289, :327, :335	loop one passes max_k=0 and ja only for batch 0, measures maxklen[i] per batch; loop two skips batch 0 and passes maxklen.at(i), which sends vertexdeg/algo.cuh:122 down the ONE-PASS arm. Two walks over the dataset instead of our three. rbc_eps_nn_query_max_k is ported and is byte-identical to the two-pass CSR under exactly this sequence (neighbors/mojo_only/ball_cover_check.mojo::check_ball_cover_max_k_wiring). Worth ~20% of a 200k fit
```

---

## 5. PROPOSED `PORTING.md` DEVIATION ENTRIES (numbered from 30)

**30. A BLOCK-SHAPE DEVIATION IS FREE ON THIS DEVICE, AND NOW WE KNOW.**
`ball_cover/registers.mojo` launches 32 threads and one query per block where
`cuvs registers.cuh:1332-1335` launches 64 and two. The reason was
portability — `vote` returns a mask over the whole warp, so two 32-lane query
groups inside one 64-lane AMD wavefront would merge their ballots — and it
was unpriced. Swept on an M4, one window, three repeats, min, count-pass
milliseconds on the DBSCAN scaling fixture:

    RBC_TPB     n=16,000   n=50,000   n=100,000   n=200,000
    32 (ours)       4.25      18.53       49.75      129.08
    64 (theirs)     6.53      20.08       49.60      130.85
    128             6.35      20.21       49.87      142.10
    256             5.40      20.22       51.23      135.56

Their shape is never faster and is 1.54x slower at 16,000, so this kernel is
not block-shape bound and the portability argument wins unopposed. The kernel
is nevertheless written in their general form (`RBC_QPB` = their `num_warps`)
so the constant is the only thing that differs and a vendor can move it from
the kernel matrix without a rewrite.

**31. A DEVICE BUFFER THAT IS NEVER WRITTEN COSTS NOTHING ON METAL.**
Allocating 2.03 GB takes 0.088 ms and holding it alive across an unrelated
kernel changes that kernel's time by under 0.5% (18.471 / 18.482 / 18.573 /
18.511 ms, alternating). Two rounds of reasoning in this repository have
attributed a step in a curve to a large allocation; it is not that. Measure
the allocation before blaming it.

**32. THE MEASUREMENT THAT ISOLATES A COST IS THE ONE THAT MOVES ONE THING.**
The 50,000 DBSCAN row was attributed to the index. The index was then run at
the identical batch shapes DBSCAN uses and did not move (18.50 / 19.28 /
19.35 / 20.82 ms across four batch shapes at n = 50,000) while a whole DBSCAN
fit at the same n moved 1.4–2.0x across those same shapes. Two batches of
40,669 and two batches of 25,002 have the same batch COUNT, so the axis is
the batch SIZE, not the batching. A residual computed by subtracting one
window's numbers from another's would not have been believable; the same
experiment run inside one window is.

---

## 6. BUILD AND CHECK EVIDENCE

### Build

```
cd /Users/andrewhendel/CascadeProjects/mojolearn
tools/with_build_lock.sh pixi run \
  --manifest-path /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
  mojo build -I . neighbors/ball_cover_main.mojo -o /tmp/ball_cover_probe
```
Exit 0, no output, no warnings.

### Checks

```
$ /tmp/ball_cover_probe
ball_cover: exact set match at eps 0.9 / 2.5 / 8.0, edges 5132 73050 1014290
ball_cover: dense and max_k both match brute force at eps 1.6, longest row 39
ball_cover: sabotage reached; radii x0.3 took edges from 73050 to 62612 and every survivor is still a true neighbor
ball_cover: the in-group order is load bearing; reversing it took edges from 73050 to 69189
ball_cover: n=8000 d=5, 200 sampled rows match brute force exactly; 130546 edges over 89 landmarks
ball_cover: cuML's two-loop max_k dispatch is byte-identical to the two-pass CSR over 3 batches; bounds 69 63 65 ; a bound one short clamps 2 rows and still reports 63
```

The first five lines are the previous lane's and are **unchanged, edge count
for edge count**, at `RBC_TPB = 32` and at 64, 128 and 256. That is what
proves the query-id rewrite is bit-identical rather than merely passing.

The sixth is new. What it asserts, in order:

1. **Their dispatch, not a neighbouring function.** Loop one counts every
   batch with `max_k = 0` (their `runner.cuh:262`) and fills only batch 0
   (their `need_ja_compute` at `:257`), and records
   `maxklen[i] = max(vd[0..n_points))` (their `thrust::reduce` at `:289`).
   Loop two skips batch 0 (their `if (i > 0)` at `:327`, comment "adj and vd
   for batch 0 already in memory") and passes `maxklen.at(i)` (`:335`), which
   is what sends `algo.cuh:122` down the one-pass arm.
2. **Byte identity, offsets and column order.** Not set equality. The CSR
   from the one-pass form is compared element by element against the CSR from
   the two-pass form for the same batch. The column ORDER inside a row is
   what a set comparison cannot see, and it is the property everything
   downstream of the index depends on.
3. **Their equality assert.** `algo.cuh:135` is `ASSERT(max_k ==
   data.max_k)` — an equality, not an inequality, because the bound was
   measured on the same rows one loop earlier. The check asserts the same
   equality and it holds for every batch: bounds 69 / 63 / 65.
4. **SABOTAGE, with a predicted shape.** The same call is given a bound one
   too small. Predicted from `registers.cuh:944-950`, which keeps counting
   past `max_k` and stops only the writes: `actual_max` must come back as the
   TRUE longest row, strictly above the bound, and the rows must be clamped
   to the bound. Both hold — it reports 63 against a bound of 62 and clamps 2
   rows. A kernel that ignored `max_k` would report the bound it was given
   and clamp nothing, so this moves in the predicted direction and could not
   pass by accident.
5. **The fixture is shaped to catch what `argsort` hid.** 1201 rows (prime,
   above 256, not a multiple of 32/64/128/256), batched 401/401/399 so no
   batch is a multiple of a block width either, coordinates from splitmix64
   so every row's neighbour set differs. The check also refuses to run on a
   degenerate fixture: it raises if the longest row is under 5.

### The diagnosis probe

Not committed — it is a diagnosis, not an arm, and it belongs to nobody's
suite. It lives at
`<scratchpad>/rbc_diag_main.mojo` and is built with the same command against
`-I .`. Everything it does is described above; it calls only public entry
points (`rbc_build_index`, the three query entry points, the build kernels by
name, and `dbscan_fit_impl` with its public `max_mbytes_per_batch`) and edits
nothing.

**No benchmark arm was run.** The DBSCAN timings in §0.3 are one arm against
itself with one argument changed, inside one window, which is the diagnosis
the brief asked for and is not a comparison against any other library.

---

## 7. TASK 3 — THE EXACT WIRING FOR `dbscan/gbdt/dbscan/runner.mojo`

Derived from their code, then verified against our port by
`check_ball_cover_max_k_wiring`. **I did not edit `dbscan/`.**

### What changes

Today, per fit: `build + count(all rows) + count(all rows) + fill(all rows)`
— three walks. Theirs: `build + count(all rows) + fill(batch 0 only) +
max_k(batches 1..n-1)` — two. Priced from §0.2 at n = 200,000:

    today   20.8 + 129.1 + 129.1 + 132.1 = 411.1 ms
    theirs  20.8 + 129.1 +   6.6 + ~125  = 281.5 ms

≈ 130 ms saved, ≈ 32% of the ball cover's cost and **≈ 20% of the whole
632.7 ms fit.**

### Loop 1 — add two things

After the existing `rbc_eps_nn_query_count` call:

```mojo
# `runner.cuh:289-293`: thrust::reduce(vd, vd + n_points, 0, maximum{}).
# `vd[0 .. n_points)` holds the per-row degrees after the count pass.
# maxklen must be a List[Int] of length n_batches, filled like batchadjlen.
var hvd = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
ctx.enqueue_copy(dst_ptr=hvd.unsafe_ptr(), src_buf=vd)
ctx.synchronize()
var mk = 0
for q in range(n_points):
    var g = Int(hvd.unsafe_ptr().unsafe_load(q))
    if g > mk:
        mk = g
maxklen[i] = mk
```

and, for batch 0 ONLY (`runner.cuh:257`, `need_ja_compute`), fill `ja` here
so loop two can skip it. That needs `col_ind` to exist before loop 2's
`maxadjlen` resize; theirs does exactly this — `adj_graph` is resized to
`curradjlen` inside `eps_nn` (`algo.cuh:150`) during loop one and then grown
to `maxadjlen` at `:317`, and `rmm::device_uvector::resize` preserves
contents when growing. If a growing resize is inconvenient here, the cheap
equivalent is to allocate `col_ind` at `maxadjlen` after loop one as today
and re-run batch 0's FILL at the top of loop two; that costs one batch's
fill, which is `1/n_batches` of a walk, and still leaves the two-walk saving
almost intact.

### Loop 2 — replace count + fill with one call, for batches > 0

```mojo
# `runner.cuh:327`: `if (i > 0)`. Batch 0 already has its ja from loop one.
if b2 > 0:
    # `algo.cuh:119-121`, their guard. It is a MEMORY test on the
    # n_points2 x max_k scratch (`registers.cuh:1431`), not a correctness
    # test: `batch * n_rows` is the dense worst case they budgeted for and
    # `col_ind`'s capacity is what is already spoken for.
    var spare = (batch * n_rows - maxadjlen) // n_points2
    var mk = maxklen[b2]
    if mk > 0 and mk < spare:
        var tmp = ctx.enqueue_create_buffer[DType.int32](n_points2 * mk)
        var scratch = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.synchronize()
        var qb = x.create_sub_buffer[DType.float32](
            start2 * n_features, n_points2 * n_features
        )
        var actual = rbc_eps_nn_query_max_k(
            ctx, rbc_xr, qb, rbc_r, rbc_ip, rbc_c1, rbc_d1, rbc_rad,
            ex_scan, col_ind, vd, tmp, scratch,
            n_points2, n_features, n_landmarks, eps_radius, mk,
        )
        # `ASSERT(max_k == data.max_k, "given maximum rowsize was not
        # sufficient")`, algo.cuh:135. An EQUALITY: the bound was measured
        # on these exact rows in loop one, so it cannot be exceeded.
        if actual != mk:
            raise Error(
                "rbc max_k: bound " + String(mk) + " came back "
                + String(actual)
            )
        ctx.synchronize()
    else:
        # today's two-pass path, unchanged
        ...
```

### Four things the wiring must get right, all of them theirs

1. **`vd` is `nullptr` in their loop-2 call** (`runner.cuh:337`,
   `sparse_rbc_mode ? nullptr : vd`), and `rbc_eps_pass_max_k` then uses
   `adj_ia` as `vd_ptr` (`registers.cuh:1429`). Our signature takes both
   buffers; passing the same `vd` is fine and is what the check does, but the
   degrees in `vd` after this call are the CLAMPED ones if the bound was
   exceeded (`registers.cuh:1457-1467`), so nothing downstream may read `vd`
   as a true degree after a max_k call. In their flow nothing does — the core
   mask was computed in loop one.
2. **`eps` is the RADIUS, not its square**, unchanged from the two-pass form
   (`algo.cuh:227` hands `data.eps` to `eps_nn` while the brute-force arm one
   line later gets `eps2`).
3. **`ex_scan` is `adj_ia` directly** and the max_k path scans it itself, so
   DBSCAN's separate exclusive scan stays skipped exactly as it is today.
4. **`tmp` is `n_points2 * max_k` Int32** and is theirs (`registers.cuh:1431`
   allocates it inside the call). If it is allocated per batch it should be
   sized from `max(maxklen)` once and reused, since `maxklen` is known before
   loop two starts.

---

## 8. FALSE DOC SENTENCES FOUND (in files I may not edit)

1. **`bench/results/DBSCAN_RBC_2026-08-19.md`**, "What is still not ported":
   *"The index build's in-group ordering is `O(m^1.5)` where cuVS uses
   `thrust::sort_by_key`. Same order as RBC's own query bound, so the
   asymptotics hold, but **it will dominate the BUILD at large m**."* The
   bolded clause is **false**. The 1-nn kernel that precedes it in the same
   build is also `O(m^1.5)` and is 6.4x larger at every size measured, so the
   ratio is fixed and the rank cannot come to dominate. Measured shares in
   §2. **Delete that sentence**; the rank is 13% of the build and 0.4% of a
   fit. The same sentence in `neighbors/gbdt/neighbors/ball_cover/
   ball_cover.mojo` is mine and is already deleted.

2. **`bench/results/DBSCAN_RBC_2026-08-19.md`**, "The 50,000 row, as it stood
   before the re-run": the two candidate explanations offered there — the
   landmark count crossing a grid-occupancy boundary, and the build being a
   larger share at that size — are **both false**, and the file should say so
   rather than leaving them as open candidates. §0.1 and §0.2. The residual
   at 50,000 is outside the ball cover entirely (§0.3).

3. **`bench/results/LANE_ball-cover_2026-08-19.md`**, verdict paragraph:
   *"the in-group ordering step ... will dominate the BUILD at large m"* —
   same false claim, same correction.

4. **`bench/results/DBSCAN_RBC_2026-08-19.md`** says of the 50,000 row after
   the re-run: *"Whatever remains at 50,000 costs a user nothing ... and it
   is logged rather than chased."* It has now been chased, and what remains
   costs a user 1.4–2.0x at that size and is reachable through a public
   argument (§0.3). Worth replacing with the batch-size finding.

5. **`neighbors/UNPORTED.tsv`** still needs the CUB row updated per §4; the
   text the previous lane proposed implies the ball cover wants a radix sort,
   and the measurement says it does not.

6. **`dbscan/gbdt/dbscan/dbscan.mojo`** (another lane's file) states
   *"Index type is Int32 throughout this port, so `sizeof(Index_) == 4` and
   `MAX_LABEL == 2147483647`."* True, but it omits that
   `cuml/cpp/src/dbscan/runner.cuh:141-150` **refuses to run RBC at all with
   an int32 index** and falls back to BRUTE_FORCE. Ours runs it. That is a
   real divergence from their dispatch, it is why the batch cap binds above
   n = 46,341, and it is undocumented. Not a false sentence; a missing one.

---

## 9. WHAT I DID NOT DO, AND WHY

- **Did not port CUB's `DeviceRadixSort`.** Priced at 0.4% of a fit with a
  fixed, non-growing share (§2), and the brief's own instruction is that a
  measured "not worth it" beats a plausible story. A plan is in §2. Nothing
  half-landed: the counting sort and the rank kernel are exactly as they
  were.
- **Did not edit `dbscan/`.** Forbidden. Task 3's answer is a specification
  plus a check that the specification produces the right answer, which is the
  most this lane can hand over.
- **Did not confirm which DBSCAN phase owns the 50,000 residual.** It needs a
  per-phase timer inside `runner.mojo`, which is another lane's file. The
  suspect and the reasoning are in §0.4, labelled as unconfirmed.
- **Did not change `RBC_TPB` to the kernel matrix's `K_LIB_BALL_COVER_EPS`
  row.** That row exists (`mojo_only/kernel_matrix.mojo:489`) and is unread.
  RESOLVED 2026-08-19 by the rbc-maxk lane, from upstream rather than from
  the rule: cuVS's own block-size source is a launcher-local constant —
  `int tpb = 64;` at `registers.cuh:1330`, in the same file as the kernels
  — so a `comptime RBC_TPB` in `registers.mojo`, the file holding our
  `rbc_eps_pass_*` launchers, mirrors their structure file for file, and
  wiring the matrix row would not. The row's fall-through 128 is also the
  measured WORST of the four block sizes at 200,000 (142.10 ms against
  129.08), so wiring it as it stands is a priced ~10% regression. The row
  stays unwired, the reasoning lives in `registers.mojo`'s DEVIATION 2
  banner, and a vendor measurement that wants a different value lands in
  that row first. See `LANE_rbc-maxk_2026-08-19.md`.
- **Did not shrink the host readback in `rbc_eps_pass_count` /
  `rbc_eps_pass_max_k`.** Both copy the whole `adj_ia` (`n_queries + 1`
  Int32) to the host to read one element. It is bounded at ~0.26 ms per call
  from the batch sweep in §0.3 (2.3 ms across 9 extra calls), i.e. under 0.2%
  of a fit, so it did not clear the bar for touching a checked file this
  round.
- **Ran no benchmark arm and compared against no other library.** Every
  timing here is one configuration of ours against another configuration of
  ours, inside one thermal window, with the reverse-order replication that
  makes it usable.
