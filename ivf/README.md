# ivf: IVF-FLAT, from cuVS

An inverted-file index over a coarse quantizer. `build` trains `n_lists`
centroids, assigns every index vector to its nearest centroid, and lays the
lists out CSR-shaped with the original row id carried beside every stored
vector. `search` scores a query against the `n_lists` centroids, takes the
`n_probe` nearest lists, brute-forces inside exactly those, and selects the
global top k. **COPY, DO NOT IMPROVE**, with twenty-five numbered
departures (DEVIATIONS 1780-1804, below).

## Status, honestly

**BUILT AND GATED ON ONE APPLE M4, BOTH MODES, 2026-08-25. NO SECOND
VENDOR HAS RUN THIS.**

`pixi run check-ivf` is green in FAST and green under
`tools/with_identical_mode.sh`, eleven checks in each mode, and five
sabotage arms were shown to fail the gate they target. The two headline
results of the IDENTICAL run:

    check_nprobe_equals_nlists_is_brute_force OK [IDENTICAL]: 120 slots,
      distances AND indices bit for bit against neighbors/ knn_search's
      TILED arm; every query scored all 320 candidates
    check_search_vs_oracle OK [IDENTICAL]: 360 slots over hashed /
      duplicate / signed-zero fixtures, every distance AND index bit for
      bit against the serial oracle

Under FAST both of those are REPORTS and both MOVE, 45 of 120 distances and
145 of 360 slots respectively, because the FAST distances come from MAX's
matmul and the FAST selector resolves a tie by atomic arrival. That
contrast is the evidence the pins are load bearing rather than decorative.

**No performance number exists for this lane and none is claimed.** Nothing
here has been timed against anything, on this vendor or any other. The
recall lines the check prints are REPORTS and the candidate-cell counts
beside them are not speedups.

`ROADMAP.md:299` positions IVF as worth building **"only if brute force
stops winning"**, and its own note says to *"measure where exact brute
force stops winning first."* **That measurement does not exist.** There is
no number in this repository establishing a crossover between exact
brute-force k-NN and an IVF index at any shape, on any device. So this lane
is a construction against that day and not evidence that the day has come,
and `ivf_main.mojo` prints the candidate cell count with a line saying in
those words that the ratio is not a speedup.

What a first run would have to establish, in order, is in
WHAT IS OWED at the bottom.

## The upstream pin

| upstream | checkout | commit | what this lane read |
|---|---|---|---|
| cuVS | `~/CascadeProjects/upstream/cuvs-v26.08.00` | **`6ba2ce2`** | `cpp/src/neighbors/ivf_flat/`, `cpp/src/neighbors/ivf_common.cuh` and `.cu`, `cpp/src/neighbors/ivf_flat_index.cpp`, `cpp/include/cuvs/neighbors/ivf_flat.hpp` |

**NOT `~/CascadeProjects/upstream/cuvs`** (`94c2819`, branch-25.08), which
`PORTING_RULES.md`'s table still names for `cluster/` and `neighbors/`.
This lane read the 26.08 tree because that is where the current
`ivf_flat.hpp`, `ivf_flat_interleaved_scan_jit.cuh` and `ivf_rabitq/` live,
and `[[mojolearn-upstream-pinned-trees]]` records what two lanes already
paid for auditing the wrong tree. The k-means and k-NN this lane CALLS were
ported against `94c2819`; that mismatch is named in WHAT IS OWED rather
than papered over.

Scope is **IVF-FLAT ONLY**. IVF-PQ, IVF-SQ, IVF-RaBitQ, CAGRA, ScaNN,
Vamana and NN-Descent are out of scope and each is in `ivf/NOT_IMPLEMENTED.tsv`.
HNSW is REFUSED PERMANENTLY and that is a different sentence from
unported, for the reason `PORTING_RULES.md` 0b-ii gives.

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Both halves of IVF already existed in this repository before this lane
opened. The coarse quantizer is k-means and the search inside the probed
lists is brute-force k-NN, and this tree has an identity-certified one of
each. **Nothing in the list below was written again here.**

| what | whose | the exact entry point this lane calls |
|---|---|---|
| the coarse quantizer (k-means fit) | `cluster/` | `cluster/impl/cluster/detail/kmeans.mojo::kmeans_fit_main_traced`, with `tag_prefix = "ivf.quantizer."` |
| the list assignment (argmin over centroids) | `cluster/` | `cluster/impl/cluster/kmeans.mojo::predict` |
| the fixed-point accumulator scale | `checks/` | `checks/fixed_point.mojo::choose_scale`, following `cluster/estimator.mojo`'s policy 2 |
| every distance, IDENTICAL arm | `neighbors/` | `neighbors/checks/pinned_distance_tile.mojo::pinned_distance_tile_kernel` (DEVIATION 505) |
| every distance, FAST arm | `core/` | `core/gemm.mojo::gemm_nt` then `core/expand_distances.mojo::expand_distances_kernel` |
| every top-k, IDENTICAL arm | `neighbors/` | `neighbors/checks/select_radix_identical.mojo::radix_topk_identical_kernel` (DEVIATIONS 500/501) |
| every top-k, FAST arm | `neighbors/` | `neighbors/impl/matrix/detail/select_radix.mojo::radix_topk_one_block_kernel` |
| the row norms | `core/` | `core/row_norms.mojo::row_norm_kernel` at `NORM_TPB` from the kernel matrix |
| the block fold inside those norms | `core/` | `core/pinned_reduce.mojo::pinned_block_sum` (DEVIATION 504), reached through the kernel |
| GEMM | `core/` | `core/gemm.mojo`, only on the FAST arm, only through `gemm_nt` |
| transcendentals and the flush | `checks/` | `checks/numerics.mojo` `ftz` / `identical_mul_add` / `identical_sqrt`, reached inside the kernels above |
| identity tracing | `core/` | `core/identity_trace.mojo::IdentityTrace` |
| the brute-force truth this lane gates against | `neighbors/` | `neighbors/estimator.mojo::knn_search` at `KNN_METHOD_TILED` |
| the hashed-fixture primitive | `kde/` | `kde/checks/kde_fixture.mojo::mix64` and `::bits_value` |
| the distance-tile sabotage arms | `hierarchy/` | `hierarchy/checks/sabotage_tile.mojo::sabotage_distance_tile_kernel` |

The k-means entry point deserves its own paragraph, because WHICH one is
the whole of DEVIATION 1795. `cluster/estimator.mojo::kmeans_fit` and
`cluster/impl/cluster/kmeans.mojo::fit` both construct their own
`IdentityTrace()` internally (`detail/kmeans.mojo:953`), which reads
`MOJOLEARN_IDENTITY_TRACE` and appends a second record numbered `seq 0` to
the file this lane is already writing.
`tools/identity_trace_diff.py` refuses a file whose sequence numbers
restart, so an IVF card built that way would be unreadable. That is the
defect DEVIATION 518 fixed for k-means|| and DEVIATION 544 for the k-NN
classifier, and `kmeans_fit_main_traced` is the sanctioned re-entry that
takes the caller's trace and prefixes every tag it writes.

**The identity status of the coarse quantizer is `cluster/`'s and is not
restated here.** `UNSUPERVISED_IDENTITY.md` is the file. In summary, and
only as a pointer to it: DEVIATION 503 pins the assignment's contraction
and flush, DEVIATION 529 gates each assignment arm against a Float64 oracle
with a magnitude-relative budget after an H100 leg found the unfused arm
was TF32, the Apple and MI300X columns produce bit-identical k-means cards,
and **NVIDIA is still owed**. Everything this lane computes is downstream
of that, so this lane's ceiling is that lane's ceiling.

**The k-NN arm this lane's search calls is the TILED one on every column.**
DEVIATION 509 pins `KNN_METHOD_AUTO` to tiled under `IDENTICAL` on every
vendor, because DEVIATION 502's earlier pin to the FUSED arm meant the
identical build did not merely lose its tie guarantee on a 64-wide
wavefront, it RAISED. The tiled arm is also the only one whose tie set is
NAMED (lowest index, by the composite key), which is what makes a reduction
gate expressible at all. This lane inherits that choice and does not get an
opinion about it.

## The identity table

The four hazards specific to IVF, each with its own row, then the
inherited pathways.

| # | hazard | what it is | what this lane does | gated by |
|---|---|---|---|---|
| **1** | **list assignment is an argmin and ties are common** | a point exactly equidistant from two centroids has no geometric reason to prefer either, and equidistance is not rare -- a symmetric dataset produces it, and so does any duplicate row sitting between two centres | **PINNED TO THE LOWER LIST ID**, on both sides. On the build side it is `raft::argmin_op`'s `(value, key)` total order, carried by `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:537` (`d < val[i] or (d == val[i] and col < key[i])`) and INHERITED, not re-implemented. On the query side it is the coarse selector's composite key, whose low half IS the list id. DEVIATIONS 1788, 1789 | `check_assignment_ties` -- an exactly equidistant fixture, both centroid orders, plus the probe-side sabotage |
| **2** | **list membership determines summation membership** | which vectors are in a list is which vectors the top-k sums over, so the assignment is a NUMERIC decision and not bookkeeping. The whole index therefore inherits every property the centroid set has and none it does not. A lane that gated its search and not its quantizer would be claiming reproducibility for a computation whose first stage it never looked at | **STATED AS A DEPENDENCY AND GATED BEFORE ANY SEARCH CLAIM.** `check_quantizer_is_reproducible` runs first among the claims that depend on it and compares centroids, centroid norms, labels and layout slots bit for bit. The CROSS-VENDOR half of the same claim is not this lane's to make -- it is `UNSUPERVISED_IDENTITY.md`'s, it is Apple and AMD only, and NVIDIA is owed there | `check_quantizer_is_reproducible`, and the `ivf.quantizer.*` / `ivf.centers` / `ivf.assign` card stages, which give a cross-vendor divergence an address inside the quantizer rather than inside IVF |
| **3** | **the list layout is a PERMUTATION** | a permutation changes nothing only if the selection is a total order over something the permutation preserves. The selector's key is `(distance, POSITION IN THE ROW)` and it takes no index array, so the position order IS the tie-break. **THE ORIGINAL INDEX MUST TRAVEL WITH THE VECTOR, NOT THE POSITION WITHIN THE LIST.** This is the classic IVF identity bug | the carry is `list_indices` (DEVIATION 1784, their `list_index[inlist_id] = source_ix` at `ivf_flat_build.cuh:135`), the within-list order is ASCENDING ORIGINAL INDEX instead of their atomic's arrival order (DEVIATION 1783), and the probe merge preserves that order across lists (DEVIATION 1786). Together those make the selector's key `(distance, original index)` restricted to the candidates, which is the order brute force uses | `check_list_layout_and_index_carry` (the layout is a permutation, the carry ascends within every list, `list_data[slot]` is `x[carry]` bit for bit, and `IVF_SAB_CARRY_POSITION` moves the answer) and `check_ivf_sabotages` |
| **4** | **`n_probe` is a NUMERIC parameter, not a tuning knob** | it changes which vectors are summed over, so it is part of the answer. Two runs at different `n_probe` are two different computations and must never be compared as if they were one | it is in the card's header, it has no default at the host boundary (`ivf/estimator.mojo` policy 1), and `n_probe > n_lists` RAISES where cuVS clamps at `ivf_flat_search.cuh:331` -- because a silent clamp means two callers who wrote different numbers get one answer and no card can say which computation ran. DEVIATIONS 1787, 1793 | `check_ivf_refusals` (the clamp is a refusal) and `check_recall_is_reported` (every REPORT line names its `n_probe` and its candidate count) |

Inherited pathways, named so this lane's ceiling is legible.

| pathway | `IDENTITY_PATHS` row | reached here through | status |
|---|---|---|---|
| the L2 distance accumulators | 19 | `pinned_distance_tile_kernel` | closed by DEVIATION 505, inherited |
| FMA contraction | 9 | `identical_mul_add` inside those kernels | pinned, and bit-inert on Apple |
| denormal flush | 10 | `ftz` at every seam of those kernels | pinned, bit-inert on Apple |
| `sqrt` | 10 | `identical_sqrt` on the `L2SqrtExpanded` metric | closed by DEVIATION 550 |
| k-NN tie handling and output placement | 11 | `select_radix_identical` | closed by DEVIATIONS 500/501 under IDENTICAL, OPEN under FAST |
| the vendor matmul's k-split | 24 | only the FAST arm reaches it | REFUSED under IDENTICAL, DEVIATION 505 |
| the vendor fp32 product's mantissa width (TF32 on NVIDIA) | 33 | only the FAST arm | inherited; the IDENTICAL arm calls no vendor product |
| the library block-size rows feeding float folds | 21 | `NORM_TPB` for `row_norm_kernel` | pinned at the comptime accessor by DEVIATION 508 |
| signed zero and the additive identity | 39 | the `+0.0` clamp inside the distance tile, and the signed-zero fixture | the clamp makes a `-0.0` distance unreachable; the fixture puts signed zeros in the DATA, where they are an exact duplicate pair only the index half of the key can order |

Two things the identity table deliberately does **not** claim.

- **No floating-point atomic reaches an output on any path here.** The only
  atomics in this lane's kernels are the `Int32` histogram and counter
  atomics inside the ported selector, which are exact and order-free in
  value. Their SLOT assignment is arrival order and DEVIATION 501's rank
  pass overwrites it.
- **No warp or lane primitive is on a numeric path**, which is why
  `ivfflat_interleaved_scan` is refused rather than ported. The selector's
  only block collective is a prefix sum over `Int32` histogram counts,
  exact at any tree shape.

## The DEVIATIONS

| # | what |
|---|---|
| **1780** | **the coarse quantizer is not their quantizer.** `ivf_flat_build.cuh:432-436` calls `kmeans_balanced`; this tree has no port of it, so the ported Lloyd k-means (`cluster/`) trains the centroids instead. A departure from `PORTING_RULES.md` 0b-i, taken because the alternative is porting a second quantizer inside a lane about an index. Consequences: list sizes are not balanced, and the centroids' identity status is `cluster/`'s |
| **1781** | `kmeans_trainset_fraction` PINNED to 1.0; anything else RAISES. Their trainset row count is a truncated float product (`:414-418`), `IDENTITY_PATHS` row 18's class, so near an integer boundary two hosts train on different rows |
| **1782** | the list layout is CSR, not their per-list allocations of interleaved groups of `kIndexGroupSize`. `conservative_memory_allocation` and `veclen` are refused with it, being that layout's parameters |
| **1783** | the within-list order is ASCENDING ORIGINAL INDEX (a stable counting sort), not `atomicAdd(list_sizes_ptr + list_id, 1)` arrival order (`:131`). The atomic's COUNT is exact; the slot it hands back is not |
| **1784** | the ORIGINAL index travels with the vector into the layout and into the SELECTION KEY. Theirs travels into the layout only, because their selector cannot use it |
| **1785** | `ivfflat_interleaved_scan` REFUSED and replaced by the tiled k-NN arm's two kernels. Three walls at once -- a warpsort queue at the hardware lane width, an occupancy-derived `grid_dim_x` that decides summation membership, and the interleaved layout |
| **1786** | the candidate row is the ASCENDING-ORIGINAL-INDEX MERGE of the probed lists, not their probe-concatenation. Without it the selector's positional key resolves an equidistant tie by which list was probed first, and `n_probe == n_lists` does not reduce to brute force |
| **1787** | `n_probe` is NUMERIC. In the card header, no default at the host boundary, and part of the answer |
| **1788** | the coarse selection breaks a tie toward the LOWER LIST ID, through the composite key's low half |
| **1789** | the list assignment breaks a tie toward the LOWER LIST ID, INHERITED from `raft::argmin_op` through `cluster/` and gated, not implemented |
| **1790** | `postprocess_neighbors` is a lookup through the carry, not their `find_chunk_ix` binary search over probe-concatenated chunk offsets |
| **1791** | `postprocess_distances` is shown to be the IDENTITY on the two ported metrics, with the citation, rather than ported as a no-op nobody can check |
| **1792** | `max_samples`, `is_local_topk_feasible` and their two-stage `select_k` merge are not ported; `k > SELECT_BLOCK` is REFUSED at the launcher, inheriting `UNSUPERVISED_IDENTITY.md`'s owed item 5 |
| **1793** | the refusals by name -- IVF-PQ, HNSW, CAGRA, IVF-SQ, IVF-RaBitQ, ScaNN, Vamana, NN-Descent, CosineExpanded, inner product, `n_probe > n_lists` (theirs clamps), `n_lists > n_rows`, `adaptive_centers`, `add_data_on_build=False`, non-finite input and the 2^63 magnitude bound |
| **1794** | a probed candidate set smaller than `k` RAISES. Their `kOutOfBoundsRecord` short fill (`ivf_common.cuh:106-108`) is not ported and the ported selector cannot take `k > len` at all |
| **1795** | the build and the search run under ONE `IdentityTrace`, with the coarse fit re-entered under the `ivf.quantizer.` tag prefix. A second `IdentityTrace()` restarts `seq` at 0 and the differ refuses the file |
| **1796** | recall is a REPORT and never an assertion. The single assertion in `check_recall_is_reported` is `recall == 1.0` at `n_probe == n_lists`, which is the reduction gate restated as a set |
| **1797** | signed zeros in the DATA are an exact duplicate pair (`+0.0 == -0.0`), ordered only by the index half of the key; a `-0.0` DISTANCE is unreachable because the distance tile clamps at `dist <= 0.0 -> +0.0`. `IDENTITY_PATHS` row 39 |
| **1798** | their query batching heuristic (`:343-353`, `get_workspace_free_bytes`) is not ported. Threads per block and grid shape are SCHEDULING and free; block count, fold shape, replication factor, tile size, accumulator width, the candidate merge order and `n_probe` are NUMERIC and pinned |
| **1799** | under FAST, the reduction gate and the oracle gates are REPORTS. The vendor matmul is called at `m = 1` here and at `m = n_queries` in `knn_search`, and a matmul is entitled to a different k-split per shape; the FAST selector resolves a tie by atomic arrival |
| **1800** | the CSR build and the probe merge run on the HOST. A departure from `PORTING_RULES.md` rule 2. The deterministic device spelling needs a segmented rank (a multi-block scan) and this lane has measured nothing; the host sort already produces the ordering the device one would, so it changes no bit and belongs behind a measurement |
| **1801** | the centroid norms come from `core/row_norms.mojo`, not RAFT's `linalg::norm` plus `utils::outer_add`; the coarse distance is the same expanded form their `alpha=-2, beta=1` arm computes |
| **1802** | `calc_chunk_indices` runs as one host integer scan rather than their `cub::BlockScan` per query block. Integer addition is associative, so the two are the same function and there is no float pathway to pin |
| **1803** | the candidate norms are taken over the PERMUTED matrix and indexed by slot. `row_norm_kernel` is one block per row reading only that row, so permuting the rows permutes the outputs and changes no float -- which is what lets the reduction gate compare against a `knn_search` whose norms were taken over the unpermuted matrix |
| **1804** | the index is HOST-RESIDENT between build and search; theirs is device-resident from `build` to the last `search`. Costs an upload per search call, changes no bit. Closure is a surface change (hold the `DeviceBuffer`s and give the estimator an explicit context lifetime) and is not taken without a measurement |

## WHAT THE ORCHESTRATOR MUST WIRE

Two task lines in `pixi.toml`, in the `[tasks]` block beside `check-kde`
and `check-linkage`.

```toml
check-ivf = "mojo run -I . ivf/checks/ivf_check.mojo"
ivf-card = "mojo run -I . ivf/ivf_main.mojo"
```

Both need the GPU, so both take the shared build lock. The three runs a
first pass owes, in this order.

```sh
tools/with_build_lock.sh     pixi run check-ivf
tools/with_identical_mode.sh pixi run check-ivf

MOJOLEARN_IDENTITY_TRACE=/tmp/mac.ivf.identical.card \
    tools/with_identical_mode.sh pixi run ivf-card
```

`ivf_main.mojo` takes `MOJOLEARN_IVF_N_LISTS` (default 8),
`MOJOLEARN_IVF_N_PROBES` (default 3), `MOJOLEARN_IVF_K` (default 8) and
`MOJOLEARN_IVF_SEED` (default 0). `ivf_check.mojo` takes nothing from the
environment on purpose and writes its cards to `/tmp`.

`tools/with_identical_mode.sh` rewrites `checks/numerics.mojo` for the
length of the command and holds `/tmp/cbsym-build.lock` for the whole
window (DEVIATION 514). The flip must never be committed.

If the orchestrator adds this lane to `check-column-invariance` or to
`check-unsupervised-identity`, note that `ivf_main.mojo` is ONE build plus
ONE search under ONE card, which is the shape those harnesses require, and
that its tags carry no machine property.

## SABOTAGES TO PERFORM

**THIS TABLE IS EMPTY AND STAYS EMPTY UNTIL SOMEONE RUNS IT.** A sabotage
table filled in from expectation instead of from output is precisely what
`[[reached-but-inert]]` is about, and this lane has run nothing. The arms
exist, are wired, and each has a check that raises if the arm fails to move
the thing it is supposed to move.

| arm | where | fixture | what MUST happen | observed |
|---|---|---|---|---|
| `IVF_SAB_CARRY_POSITION` | `sabotage_layout.mojo` | hashed, `n_probes = 2` | `check_list_layout_and_index_carry` sees the returned identities MOVE; the distances do not | *(not run)* |
| `IVF_SAB_LIST_ARRIVAL_ORDER` | `sabotage_layout.mojo` | duplicate, planted round-robin labels | `check_ivf_sabotages` sees the candidate order stop ascending in the carried index | *(not run)* |
| `IVF_SAB_MERGE_PROBE_ORDER` | `sabotage_layout.mojo` | duplicate, two interleaved lists | `check_ivf_sabotages` sees the candidate order stop ascending | *(not run)* |
| `IVF_SAB_PROBE_TIE_HIGH` | `sabotage_layout.mojo` | two lists at an exactly equal coarse distance | `check_assignment_ties` sees probe slot 0 flip from list 0 to list 1 | *(not run)* |
| `IVF_SAB_EMPTY_COUNTS_ONE` | `sabotage_layout.mojo` | planted empty list 2 | `check_empty_list` sees the chunk total overcount by exactly one | *(not run)* |
| `LINK_SAB_ROTATE_CONTRACTION` | `hierarchy/checks/sabotage_tile.mojo`, imported | hashed | the candidate distances MUST move under IDENTICAL (a per-block summation order where a pinned one is required); a REPORT under FAST | *(not run, and not yet wired into a named check -- see WHAT IS OWED)* |
| `LINK_SAB_STD_SQRT` | same file | hashed, `L2SqrtExpanded` | a REPORT. Apple's `sqrt` is correctly rounded so the bits may not move here; on NVIDIA they should (DEVIATION 258) | *(not run, not yet wired)* |
| the k-means entry point | manual | any | swap `kmeans_fit_main_traced` for `cluster/estimator.mojo::kmeans_fit` and the card must become UNREADABLE (two `seq 0` records), proving DEVIATION 1795 is load-bearing | *(not run)* |

## WHAT IS OWED

1. **A FIRST RUN.** Nothing here has executed. Both modes, both task lines,
   and the eleven checks. Expect the first run to find things -- every lane
   in this repository that wrote its gates before running them did.
2. **`check_nprobe_equals_nlists_is_brute_force` is the one to run first**,
   and if it fails under IDENTICAL the diagnosis order is (a) does
   `ivf.cand_idx` at `n_probe == n_lists` equal `0, 1, 2, ...`, which tests
   DEVIATION 1786 alone, (b) do the candidate norms match `knn_search`'s
   index norms, which tests DEVIATION 1803 alone, (c) only then the
   distances.
3. **The two distance-tile sabotage arms are imported but not yet driven by
   a named check.** `check_ivf_sabotages` covers the five layout arms;
   `LINK_SAB_ROTATE_CONTRACTION` and `LINK_SAB_STD_SQRT` need a check that
   swaps the kernel behind `_expanded_distances`, which means a
   `sabotage` parameter on that helper or a check-local copy of the
   dispatch. Named rather than done because getting it wrong puts a test
   knob in a production signature.
4. **The launch-invariance check has no CHOSEN poison.** Its padded arm is
   the candidate workspace being allocated at `n_rows` and used at
   `n_cand`, and its alone-versus-batch arm doubles as the poison arm
   because the workspace is reused across queries. Neither picks the
   poison value, so a benign leftover would not be caught. A `poison`
   parameter belongs on the search entry, which is a production signature,
   so it is named here.
5. **The upstream pin mismatch.** This lane read cuVS `6ba2ce2` (26.08);
   `cluster/` and `neighbors/`, which it calls, were ported against
   `94c2819` (25.08). Nothing observed suggests the k-means or brute-force
   dispatch moved between them, but nobody has diffed the two, and
   `[[mojolearn-upstream-pinned-trees]]` is the record of what assuming
   costs. `tools/check_upstream.sh` is the instrument.
6. **A crossover measurement, which is the whole reason to have this
   directory.** `ROADMAP.md:299` says IVF is worth building only where
   exact brute force stops winning, and no number establishes where that
   is. The measurement is `bench/`'s and it must be taken inside ONE
   process with the arms interleaved (`[[mojolearn-box-drifts]]`), it must
   report which arm ran beside the number (`PORTING_RULES.md` rule 8), and
   it must be taken at the SHIPPED size (`[[large-data-runs-default]]`).
   Until it exists this directory is a construction and this file says so
   at the top.
7. **`kmeans_balanced` (DEVIATION 1780).** Their actual quantizer. Porting
   it is a `cluster/` lane's work and would change every list size and
   every card stage below `ivf.centers`.
8. **The device CSR build and probe merge (DEVIATION 1800).** A segmented
   exclusive scan plus a rank-reading scatter. Changes no bit; belongs
   behind a measurement.

### Three changes this lane believes other directories need, and did NOT make

- **`cluster/estimator.mojo` wants a `plan_sum_scale` that takes a
  `List[Float32]`.** It currently takes a
  `MutPointer[Float32, MutUntrackedOrigin]`, so this lane holds a
  second copy of the same three-line loop in
  `ivf_flat_build.mojo::plan_quantizer_scale`. One exported overload
  deletes it.
- **`neighbors/estimator.mojo` wants its host sort exported.** The
  `(distance, index)` insertion sort at the end of `knn_search_traced` is
  the same sort `ivf_flat_search.mojo::sort_slots_by_distance_then_index`
  now spells a second time. It sorts a runtime host buffer there and a
  `List` here, so the export needs a small signature change.
- **`cluster/` wants a `kmeans_fit_traced` host entry**, shaped like
  `neighbors/estimator.mojo::knn_search_traced`. This lane reached past
  `cluster/estimator.mojo` into `cluster/impl/cluster/detail/kmeans.mojo`
  to get one card, which works and is the sanctioned re-entry, but it means
  the host-side policy that entry carries (the scale, the weight bound,
  `x_norm`, `fit_predict`'s fresh assignment) is spelled again in
  `ivf_flat_build.mojo`. That is the largest single piece of duplication
  this lane has and it is one signature away from disappearing.
