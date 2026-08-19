# LANE knn-fusion-close — 2026-08-19

**VERDICT, one line.** The placeholder is gone: `fusedL2kNN`'s selector is now
`faiss_select::WarpSelect`, it is **genuinely register-resident** (the emitted
Apple AIR carries both queues as loop-carried `phi` values of `<4 x float>` /
`<4 x i32>` / `<2 x float>` / `<2 x i32>` / `float` / `i64`, **zero `alloca`,
zero non-`air` calls**, and the kernel's only threadgroup memory is the two
GEMM pages), and **the answer is unchanged** — every pre-existing check passes
at the same bar, per slot and in order against the host Float64 oracle, at
k = 1, 8, 32, 64. Threadgroup memory per block fell 30,848 B → 18,496 B (-40%).
The strongest remaining gap is that the index is streamed `ceil(m/Mblk)` times
(125 times at the 400k benchmark shape), which is now the dominant term and is
a `launchConfigGenerator` / `Mblk` question, not a selector question.

---

## 0. THE ONE CORRECTION THAT CHANGES THE PICTURE

My brief said the `__ballot_sync` blocker was "the CROSS-BLOCK merge only" and
that "a single-x-block fused kernel is fully reachable with what is ported
now". **The first half is wrong and the second half is right for a different
reason.** I opened the file:

* `updateSortedWarpQ` (`fused_l2_knn.cuh:147-185`, `__ballot_sync` at `:160`,
  `__ffs` at `:165`) has exactly **two** appearances in the file: its
  definition and **one call site, `:437`, inside `epilog_lambda`** — which runs
  on every column tile after the first at **any** `gridDim.x`, including
  `gridDim.x == 1`. It is not cross-block code.
* Their actual cross-block merge, in `rowEpilog_lambda`, uses plain
  `heapArr[i]->add(...)` at `:284-300`. That is ported. What blocks the
  cross-block path is the mutex protocol (`atomicCAS`/`atomicExch`/
  `__threadfence`, `:241-281`, `:313-338`), **not** ballot.
* `__ballot_sync` also gates the candidate prefix-sum staging at `:405-412`.

So at `gridDim.x == 1` the kernel is still fully reachable, but only by taking
their **`else` arm** (`:454-477`) on *every* column tile and keeping the queues
alive in registers across the whole sweep, instead of their spill-to-`shDumpKV`
and rebuild. That is DEVIATION BLOCK 1 in the file, and it is now written down
accurately.

---

## 1. DIVERGENCES FOUND

| upstream (file:line) | what ours did | fixed? |
|---|---|---|
| `fused_l2_knn.cuh:222` — the selector is `WarpSelect<AccT, uint32_t, false, Comparator<AccT>, NumWarpQ, NumThreadQ, 32>`, register-resident | `fused_l2_knn.mojo` held a **shared-memory sorted list** with a serial one-thread-per-row merge and a 32-wide shared staging buffer, marked `SELECTOR SLOT` | **FIXED.** Replaced by the ported `WarpSelect`. Shared arrays `sh_val`, `sh_idx`, `cand_val`, `cand_idx`, `cand_cnt` all deleted; 12,352 B of threadgroup memory returned. |
| `fusedL2ExpKnnImpl:743-771` instantiates the **whole kernel** twice, `<NumWarpQ=32, NumThreadQ=2>` for `numOfNN <= 32` and `<64, 3>` for `<= 64`, dispatched on the host at `:765-771` | Ours had **one** kernel with a runtime `num_nn` and no queue parameters at all | **FIXED.** `fused_l2_knn_kernel[num_warp_q, num_thread_q]`, and `fused_l2_knn` picks between `[32,2]` and `[64,3]` at their two thresholds. |
| `fused_l2_knn.cuh:463` — an out-of-range column contributes `Pair{keyMax, identity}` and `add` is still called (`:468`), so every lane calls `add` exactly `AccColsPerTh` times | Ours `continue`d out of the epilogue for an out-of-range column | **FIXED, and it was load-bearing, not cosmetic.** `WarpSelect::checkThreadQ` (`Select.cuh:399-428`) holds a warp vote at `:404` and a shuffle merge; a lane that skips desynchronizes the warp. Their sentinel is the call contract. |
| `fused_l2_knn.cuh:147-185` `updateSortedWarpQ` reached at `:437` on **every column tile after the first**, at any `gridDim.x` | Comment in `fused_l2_knn.mojo` (and my brief) said this was cross-block-only | **FIXED in the docstring.** The port takes their `:454-477` arm on every tile instead; queues never spill. |
| `fused_l2_knn.cuh:284-300` — the cross-block merge in `rowEpilog_lambda` is plain `heapArr[i]->add` | Ours claimed it needed `updateSortedWarpQ` and cited `:290` | **FIXED.** `:290` is `otherKV.value = identity;`. The citation was fabricated. The blocker is the mutex protocol alone, and that is now stated as OPEN rather than as a wall. |
| `l2_exp.cuh` lives at `cuvs/cpp/src/distance/detail/distance_ops/l2_exp.cuh`; the epilog is `:114-146`, the two-clause clamp is `:132-134`, `get_clamp_precision` is `:30-39` with `case 4: 1e-6` at `:35` | Ours cited `distance_ops/l2_exp.cuh:120-129` and `:31-39` / `:34` — a path that does not resolve and line numbers that are not those lines | **FIXED.** The arithmetic was already right; only the citations were wrong. |
| `fused_l2_knn.cuh` — `constexpr bool sqrt = false` is `:977-979`; the `fusedL2Knn` entry point is `:947-1076`; its ASSERT block is `:963-975`; the `query, index` argument order is `:1003-1004`; `Policy2x8` is selected at `:724`; the Exp-arm `ASSERT(numOfNN <= 64)` is `:770` | Ours cited `:1017-1019`, `:996-1078`, `:1000-1015`, `:1042-1043`, `:722`, `:759` | **FIXED**, all seven. Every one was opened and confirmed. |
| `knn_brute_force.cuh:463-475` is the `unaryOp` post-processing block | Ours cited `:463-472` | **FIXED** (start was right, end was two lines short). |
| `storeWarpQGmem` is `fused_l2_knn.cuh:95-117`, `idx = j * warpSize + lid` at `:109`; the `else` arm is `:454-477`; `gmemRowId < m` is `:459`; `colId` is `:462`; `shDumpKV` is `:352`; `allWarpTopKs` is `:396` and its store `:427-429`; `epilog_lambda` is `:341-484` | Ours cited `:93-115`, `:107`, `:453-476`, `:458`, `:462` (for `otherKV`), `:351`, `:400`/`:424`, `:341-479` | **FIXED**, all. |
| `Select.cuh` — general `WarpSelect` is `:346-500`; ctor `:362-381`; `kLane` `:363`; `addThreadQ` `:383-397`; `checkThreadQ` `:399-428`; `__any_sync` `:404`; `warpKTop = shfl(...)` `:427`; `mergeWarpQ` `:433-443`; `add` `:447-451`; `reduce` `:453-458`; `writeOut` `:461-474`; fields `:477-499`; `NumWarpQ==1` specialization `:502-551`; `BlockSelect`+`FinalBlockMerge` `:18-344` | `select.mojo` cited `:340-500`, `:349-374`, `:351`, `:376-391`, `:393-419`, `:398`, `:418`, `:421-432`, `:436-440`, `:442-447`, `:450-463`, `:483`-`:505`, `:503-548`, `:20-338` — **systematically ~13 lines low, every one of them** | **FIXED**, all nineteen. The code was faithful; only the line numbers were a recollection. (`faiss_select/**` is my lane's, so I corrected them rather than reporting them.) |
| `MergeNetworkWarp.cuh` — `static_assert(WarpSize == 32)` is `:501`; `warpBitonicMergeLE16` `:84-137`; `BitonicMergeStep` generic `:140-219`, Low `:221-304`, High `:305-388`; `warpMergeAnyRegisters` `:390-442`; `BitonicSortStep`/`warpSortAnyRegisters` `:444-517`. `cuda_dev_essentials.cuh:83` defines `WarpSize`. `MergeNetworkUtils.cuh:13-24`. `Comparators.cuh:23-27` for the `half` specialization | `merge_network_warp.mojo` cited `:495`, `:83-136`, `:146-212`, `:218-303`, `:306-385`, `:391-436`, `:440-517`, `cuda_dev_essentials.cuh:74`, `MergeNetworkUtils.cuh:14-24`, `Comparators.cuh:22-27` | **FIXED**, all ten. |
| `contractions.cuh:203-206` `Policy2x8<float>` = `KernelPolicy<float, veclen, 16, 2, 8, 8, 32>`; `detail/contractions.cuh:99-100` `accrowid`/`acccolid`; `ldgXY` `:150-165`, `stsXY` `:166-175` (`stsX` `:261`, `stsY` `:270`), `ldsXY` `:176-180` (`ldsX` `:279-298`, `ldsY` `:299-317`) | Ours cited `:176-275` for ldgXY+stsXY and `:282-313` for ldsXY | **FIXED.** `Policy2x8` and `accrowid`/`acccolid` were already correct and are confirmed. |
| `_coord` in `knn_check.mojo` | Already fixed to splitmix64 by round 1 | **Not regressed.** New fixtures use `_fchk_coord` (= `_coord` + salt offset) unchanged. |

**Checked and cleared, not a divergence:** `ldd == n_index_rows`
(`fused_l2_knn.cuh:998`), so our `col < n` is their `colId < ldd`. And the warp
mapping is theirs: `threadIdx.x / AccThCols` is constant across a 32-lane warp
at `AccThCols == 32`, so their `gmemRowId < m` guard (`:459`) is warp-uniform
and it is safe to wrap `add` in it — which is the only reason their `else` arm
satisfies `WarpSelect`'s call contract in the first place.

---

## 2. WHAT I CHANGED, FILE BY FILE

### `neighbors/ported/neighbors/detail/fused_l2_knn.mojo` — rewritten selector

* Imports `WarpSelect` from `faiss_select/select.mojo`.
* `fused_l2_knn_kernel[num_warp_q, num_thread_q]` — their two whole-kernel
  instantiations (`fusedL2ExpKnnImpl:743-771`).
* Two `WarpSelect[nwq, ntq, False](identity, keyMax, num_nn)` constructed
  **before** the column loop — their `heapArr1`/`heapArr2` (`:359-361`), except
  theirs are re-declared per `epilog_lambda` call.
* Per column tile, their `else` arm (`:454-477`) verbatim: `regyn` loaded once
  (`:346`), then for each of the two rows, `AccColsPerTh` unconditional `add`s
  with `{keyMax, identity}` for `colId >= n` (`:462-468`).
* `l2_exp_epilog` factored out as an `@always_inline def` — `l2_exp.cuh:132-134`,
  both clamp clauses, unchanged arithmetic.
* At the end: `needSort = __any_sync(numVals > 0)` → `reduce()` (`:471-473`),
  then `writeOut` (`storeWarpQGmem`, `:95-117`).
* Deleted `sh_val`, `sh_idx`, `cand_val`, `cand_idx`, `cand_cnt` and the
  `Atomic` import.
* `@parameter for` over `AccColsPerTh` everywhere a `SIMD` lane is indexed, so
  the index is comptime and no dynamic `extractelement` can force the
  accumulators or the queue into memory.
* Docstring rewritten: DEVIATION BLOCK 1 is now the register-lifetime
  deviation and states the real `__ballot_sync` reach; BLOCK 2 is the
  cross-block merge with the mutex named as the actual blocker; BLOCK 3 is
  single-buffering (with the note that 18,496 B is now the whole footprint and
  the double buffer would fit NVIDIA/AMD — a `lib_smem_pages_for` row, left
  OPEN); BLOCK 4 (`sqrt`) unchanged. All citations re-verified against the
  checkout.

### `neighbors/mojo_only/knn_check.mojo` — three new checks

* `check_fused_edge_shapes()` — 17 queries × 1,013 index × 13 features, a
  partial tile on `Mblk`, `Nblk` **and** `Kblk` at once, at k = 1, 32, 33, 64.
  17 queries is the shape that puts a row tile with **one** live row in it, so
  seven of eight warps skip the queue entirely and the eighth has `heap0` live
  and `heap1` dead — the only shape in which a non-warp-uniform `:459` guard
  could be caught. 1,013 is not a multiple of 32 or of `Nblk`. k = 32 and 33
  straddle `:765-771`, so they select **different** kernel instantiations, and
  33 is the first k with `kNumWarpQRegisters == 2` and `kLane == 0`.
  Compared per slot and in order against a host Float64 direct-formula oracle.
* `check_fused_queue_reach_by_sabotage()` — a sabotage aimed at the selector
  rather than the epilogue. Plants a copy of one query's coordinates into index
  column 3,971 (deep enough that it arrives when the queue is already full) and
  requires four things: rank 0 is that column; its distance is **exactly** 0.0
  (which only `l2_exp.cuh:132-134`'s second clamp clause can produce); the
  victim row's slots 1..k-1 are its old slots 0..k-2 in index *and* distance;
  and every other row is bit-identical. The victim ROW is **chosen at run
  time** as the first query no other query is within its top-k of, verified in
  Float64 — a typed constant was wrong twice, because at 20 dimensions and 53
  queries some query pairs really are that close.
* `check_fused_k_ceiling()` — `fused_l2_knn` itself accepts k = 64 and refuses
  k = 65 with their ASSERT message (`:581`, `:770`). The existing
  `check_dispatch_takes_fused` only covered the `brute_force_knn_impl` side.
* `_oracle_at()` — the Float64 oracle at runtime shapes (the existing
  `_fchk_oracle` is nailed to the `FCHK_*` constants).

### `neighbors/knn_main.mojo`

Wires the three new checks. Docstring corrected (it said "the last three").

### `neighbors/ported/neighbors/detail/faiss_select/select.mojo`

Citation-only: nineteen `Select.cuh` line numbers corrected, plus the
`fused_l2_knn.cuh` cross-block reference (`:147-185` and `:284-300`, since the
cross-block merge is `add`, not `updateSortedWarpQ`). **No code change.**

### `neighbors/ported/neighbors/detail/faiss_select/merge_network_warp.mojo`

Citation-only: ten line numbers corrected. **No code change.**

### `neighbors/ported/neighbors/detail/knn_brute_force.mojo`

Untouched. Its dispatch already routes `k <= FKNN_MAX_NN && row_major` to
`fused_l2_knn`, and that is verified by `check_dispatch_takes_fused`.

---

## 3. PROPOSED ROWS

`neighbors/PORTED_MAP.tsv` — **replace** the `fused_l2_knn.mojo` row:

```
ported/neighbors/detail/fused_l2_knn.mojo	cuvs cpp/src/neighbors/detail/fused_l2_knn.cuh	partial	fusedL2kNN on Policy2x8 with faiss_select::WarpSelect in REGISTERS as the selector, at their two instantiations (:743-771). gridDim.x==1 only. Single-buffered: their SmemSize is 2 pages = 36,992 B against Metal's 32 KB
```

`neighbors/UNPORTED.tsv` — add:

```
cuvs neighbors/detail/fused_l2_knn.cuh::updateSortedWarpQ	not ported	:147-185 needs __ballot_sync + __ffs (:160, :165) -- WHICH lanes voted, then the first of them. Mojo 1.0 has warp shuffles and warp reductions but no ballot and no lane mask, so `activeLanes` cannot be formed. Its only call site is :437, the non-first-column-tile arm of epilog_lambda, at ANY gridDim.x -- NOT cross-block code. We take their :454-477 arm every tile instead and keep the queues in registers
cuvs neighbors/detail/fused_l2_knn.cuh::epilog_lambda candidate prefix sum	not ported	:396-451, the threshold pre-count + __ballot_sync warp prefix scan (:405-412) into allWarpTopKs. Same ballot blocker. Costs us one add (compare + warp vote) per accumulator element where theirs rejects a whole tile against warpKTop first
cuvs neighbors/detail/fused_l2_knn.cuh::rowEpilog_lambda	not ported	:224-339, the cross-block per-row merge. Its merge is plain heapArr[i]->add (:284-300), which IS ported; what blocks it is the mutex protocol -- atomicCAS/atomicExch spin plus __threadfence (:241-281, :313-338) -- which has not been established as sound on Metal. OPEN, not a wall
cuvs neighbors/detail/fused_l2_knn.cuh::launchConfigGenerator	not ported	the grid.x > 1 occupancy choice (:775-776). We pin grid.x == 1, which is their own configuration at :226 and :479, and costs parallelism when m is small and n is huge
cuvs neighbors/detail/fused_l2_knn.cuh::fusedL2UnexpKnn	not ported	:523-704, the L2Unexpanded / L2SqrtUnexpanded arm. Same kernel, unexpanded distance op; the dispatch at knn_brute_force.cuh:443 admits it and we raise instead
cuvs neighbors/detail/fused_l2_knn.cuh::usePrevTopKs	not ported	loadPrevTopKsGmemWarpQ (:120-145) and its :362-366 guard. A warm-start hook; every caller in this tree passes usePrevTopKs = false
```

---

## 4. PROPOSED PORTING.md DEVIATION ENTRIES (numbered from 30)

**30. A `WarpSelect`-style register queue survives a Mojo `while` loop as
`phi` values, and the way to prove it is `--emit=asm --target-accelerator`.**
`mojo build --emit=asm` alone emits only host assembly; the per-kernel Metal
`.ll` sidecars appear **only when `--target-accelerator` is also given** (e.g.
`--target-accelerator apple-m4`), written next to the `-o` path. That is the
only way in this toolchain to answer "is this in registers or in memory", and
the answer for `fused_l2_knn_kernel[64,3]` is: **zero `alloca`, one `define`,
and the two queues carried around the column loop as twelve `phi` values of
`<4 x float>` / `<4 x i32>` / `<2 x float>` / `<2 x i32>` / `float` / `i64`.**

**31. Index a `SIMD` with a `@parameter for` index, never a runtime one, in
any value you need in registers.** A runtime `acc[j]` is a dynamic
`extractelement` and is the most likely way a register tile silently becomes
memory. Every `SIMD` lane access in `fused_l2_knn.mojo` is comptime for this
reason; it is why the emitted AIR has no `alloca` at all.

**32. A warp-collective queue imposes a CALL-COUNT contract on its caller,
and upstream's "useless" sentinel branch is that contract.**
`WarpSelect::checkThreadQ` (`Select.cuh:399-428`) holds `__any_sync` at `:404`
and then a shuffle merge, so every lane of a warp must call `add` the same
number of times. `fusedL2kNN:463` builds `Pair{keyMax, identity}` for a column
past `n` and calls `add` anyway rather than skipping. Porting that as a
`continue` compiles, passes a fixture whose extents are multiples of the tile,
and desynchronizes the warp on any other fixture. **A guard around such a call
is only legal if it is warp-uniform**; theirs (`gmemRowId < m`, `:459`) is,
because `threadIdx.x / AccThCols` is constant across a warp at
`AccThCols == 32`. Check that before wrapping any queue method in an `if`.

**33. A fixture constant chosen by hand for an isolation property must be
chosen by the check instead.** `check_fused_queue_reach_by_sabotage` needs a
query no other query is within its top-k of. Two hand-picked row indices failed
in 20 dimensions, and each failure looked like a kernel bug for as long as it
took to read the message. The check now scans for a valid victim in Float64 and
raises only if none exists.

**34. `--emit=asm` sidecars: the shared-memory footprint of a kernel is in the
`addrspace(3) global` declarations and is worth reading on every kernel that
claims to be register-resident.** `fused_l2_knn_kernel` went from seven such
arrays (30,848 B) to two (18,496 B) when the selector moved into registers, and
that number — not a source-level count — is what Apple's 32 KB ceiling applies
to.

---

## 5. FALSE DOC SENTENCES FOUND, in files I may not edit

* `bench/results/LANE_knn-tiling_2026-08-19.md:9` — "the selector inside it is
  a placeholder, not their `faiss_select::WarpSelect`". **Now false.**
  Also `:29` (divergence D5e "NOT FIXED BY DESIGN"), `:54`, `:100` (the
  `PORTED_MAP.tsv` row it proposes says "selector is a placeholder"), and
  `:289` ("`SELECTOR SLOT` shared arrays (become register queues)" — that
  future-tense item is now done). These are a round-1 lane's record, so they
  are history rather than a live claim; the orchestrator should not paste
  `:100`'s row into `PORTED_MAP.tsv` — use §3 above instead.
* `bench/results/LANE_warpsort_2026-08-19.md:395` quotes
  `updateSortedWarpQ`'s ballot correctly but the surrounding §7 treats it as
  the cross-block merge. It is `epilog_lambda`'s per-column-tile merge; the
  cross-block merge is `add` (`:284-300`).
* `neighbors/README.md` — no false sentence found about the fused path; it
  describes the two warpsort implementations and is still accurate.

---

## 6. BUILD / CHECK EVIDENCE

### Build and correctness

```
$ tools/with_build_lock.sh pixi run --manifest-path .../mojotrees/pixi.toml \
    mojo build -I . neighbors/knn_main.mojo -o /tmp/knn_probe
$ /tmp/knn_probe
check_knn OK: 64 queries x k=8 over 4096 index points, every returned neighbor is in the exact true set
check_knn_reach_by_sabotage OK: index_norm moved 512/512 neighbors; query_norm offset moved 0 sets, which is the predicted shape
check_vendor_topk_matches_ported OK: nn.topk.top_k and the ported RAFT radix select agree on all 512 neighbours
check_fused_l2_knn OK: k = 1, 8, 32, 64 over 53 queries x 4093 index x 20 features, every slot matches the host Float64 oracle in order, and is_sqrt is exactly one square root away
check_fused_edge_shapes OK: k = 1, 32, 33, 64 over 17 queries x 1013 index x 13 features -- a partial tile on Mblk, Nblk and Kblk at once, and both WarpSelect instantiations
check_fused_reach_by_sabotage OK: index_norm ramp moved 424/424 neighbours and 424 distances; query_norm offset moved 0, which is the predicted shape
check_fused_queue_reach_by_sabotage OK: a co-located point planted at index column 3971 for query 0 came back at rank 0 at exactly 0.0, shifted that row's tail by exactly one, and moved no other row
check_fused_k_ceiling OK: k = 64 accepted, k = 65 refused with their ASSERT message
check_dispatch_takes_fused OK: k=8 wrote out_idx and left out_idx32 at its sentinel; k=65 did the opposite, so the `k <= 64` branch at knn_brute_force.cuh:443 is wired both ways

$ /tmp/ws_probe          # neighbors/warpsort_probe_main.mojo, unchanged behaviour
check_warpsort_matches_radix: OK
check_warpsort_reach_by_sabotage: OK
check_warpselect_matches_oracle: OK
check_warpselect_reach_by_sabotage: OK
```

`bench/scaling_main.mojo` and `bench/bench_main.mojo` both still build against
the new kernel signature (built, **not run** — no timing from this lane).

### REGISTER RESIDENCY — the measurement, not the argument

```
$ mojo build -I . --emit=asm --target-accelerator apple-m4 \
    neighbors/knn_main.mojo -o <dir>/knn.s
```

emits one Metal `.ll` per kernel next to `knn.s`. For the `[64, 3]`
instantiation:

```
defines = 1        (fully inlined; no callable device functions)
allocas = 0
addrspace(3) global [272 x float]        # sx, the GEMM X page
addrspace(3) global [4352 x float]       # sy, the GEMM Y page      -> 18,496 B total
stores to any address space other than (1) device and (3) threadgroup: NONE
 1638  air.simd_shuffle_xor.i32.i16
 1548  air.simd_shuffle_xor.f32.i16
   90  air.max.s.i32                     # the __any_sync votes: 16 adds x 5 + 2 finals x 5
   16  air.simd_shuffle.f32.i16          # shfl(warpK[last], kLane)
    2  air.wg.barrier
```

and the column-loop header carries both queues by value:

```
%152 = phi i64        [ 0, ... ]                            ; n0
%153 = phi i64        [ 0, ... ]                            ; heap0.num_vals
%154 = phi <4 x float>[ splat (float f0x7F7FFFFF), ... ]    ; heap0.thread_k (NumThreadQ 3, padded to 4)
%155 = phi <4 x i32>  [ splat (i32 -1), ... ]               ; heap0.thread_v
%156 = phi <2 x float>[ splat (float f0x7F7FFFFF), ... ]    ; heap0.warp_k   (kNumWarpQRegisters 2)
%157 = phi <2 x i32>  [ splat (i32 -1), ... ]               ; heap0.warp_v
%158..%162                                                  ; heap1, same five
%163 = phi float      [ f0x7F7FFFFF, ... ]                  ; heap0.warp_k_top
%164 = phi float      [ f0x7F7FFFFF, ... ]                  ; heap1.warp_k_top
```

`f0x7F7FFFFF` is `FLT_MAX`, their `identity`; `splat (i32 -1)` is `0xFFFFFFFF`,
their `keyMax`. **The queues are registers, across the whole column sweep.**

BEFORE, same command on the placeholder version:

```
defines = 1  allocas = 0
addrspace(3) global [272 x float]  [4352 x float]  [1024 x float]  [1024 x i32]
                   [512 x float]   [512 x i32]     [16 x i32]        -> 30,848 B total
```

### SABOTAGE — of the code path, not just of an input

Two, both run and both reverted:

1. **Input sabotage, kept in the tree.**
   `check_fused_queue_reach_by_sabotage` plants a co-located index point at
   column 3,971 and asserts a four-clause shape. The two intermediate failures
   it produced along the way are themselves evidence it bites: replacing the
   *query* instead of the *index point* correctly failed clause 3, and a
   hand-picked victim row correctly failed the fixture-isolation guard.

2. **Code sabotage, reverted.** `WarpSelect.add_thread_q`'s accept condition
   changed to `comp_lt(key, warp_k_top) and (val % 3) != 0`, so the queue silently
   drops every third candidate by payload. Predicted shape: the fused checks
   fail with neighbours that are real points but never the true nearest when
   the true nearest's column is divisible by 3, and the **tiled fallback is
   unaffected**. Observed, exactly:

   ```
   check_knn OK: ...                                    <- fallback, untouched
   check_vendor_topk_matches_ported OK: ...             <- fallback, untouched
   Unhandled exception: fused k-NN at k=1: 17 of 53 slots disagree with the host
     Float64 oracle [q=3 slot=0 got=523 d=0.766 want=2739 d=0.732]
     [q=5 slot=0 got=2426 d=0.899 want=861 d=0.580] ...
   ```

   This is the reach proof the digest cannot give: the ported `WarpSelect` is
   on the fused path and only on the fused path, and its **payload** is
   threaded through (the drop condition keyed on `val`).

   Reverted; the file is byte-identical to its pre-sabotage state and both
   probes are green again.

---

## 7. WHAT I EXPECT AT 400,000 INDEX POINTS, AND WHY

**Do not read this as a measurement. It is arithmetic, and the orchestrator's
interleaved arms decide.** Shape: n = 400,000, m = 2,000, d = 32, k = 10.
Baseline (unfused, materialized): **305.78 ms**. Stated compute floor: ~13 ms.

k = 10 selects the `[32, 2]` instantiation. Launch is
`grid = (1, ceil(2000/16) = 125, 1)`, `block = 256`.

| term | value | note |
|---|---|---|
| arithmetic | 51.2 GFLOP | `2 m n d`; unchanged by fusion |
| distance-matrix traffic | **0 B** | was ~23 GB. This is the deletion. |
| output traffic | 160 KB | `m k` pairs |
| **index re-reads** | **125 x 52.8 MB = 6.6 GB of requests** | the index is streamed once per query row tile, and `Mblk = 16` |
| queue cost | ~25,000 `add`s per warp | 8 per row per column tile x 2 rows x 1,563 tiles; each is a compare plus a 5-step `air.max.s.i32` vote, plus a merge when a lane's 2-deep thread queue fills |

The 6.6 GB is **requests, not necessarily DRAM traffic**: the 125 blocks all
walk the index in the same direction, and the instantaneous working set is one
`Nblk x Kblk` page per block (~16 KB), so if the resident blocks stay roughly
in lockstep the re-reads are served by the system cache and DRAM sees closer to
the 51.2 MB the index actually is.

* **If the re-reads hit cache**, the kernel is compute/issue bound: 51.2 GFLOP
  plus the vote shuffles, which on this part lands somewhere around **20-40 ms**
  — call it **8-15x**.
* **If they do not**, 6.6 GB at roughly 100 GB/s is **~66 ms** — **4.6x**.

So: **I expect 4.6x-15x, most likely 6-10x, and I do not expect the 13 ms
floor.** The single largest remaining term is the index re-read count, which is
exactly `ceil(m / Mblk) = 125`, and the levers on it are (a) their
`launchConfigGenerator`, which we pin to `grid.x == 1`, and (b) `Mblk`, which
Policy2x8 fixes at 16.

One more cap worth measuring rather than arguing: **only 125 threadgroups, at
18,496 B of threadgroup memory each.** If Apple's per-core threadgroup pool is
the same 32 KB as the per-threadgroup ceiling, a second block cannot be
resident (2 x 18,496 = 36,992 B), so the freed 12,352 B buys correctness and
traffic but **not** occupancy. The placeholder had the same cap at 30,848 B, so
this is not a regression — but it is the reason I would not promise the floor.

---

## 8. WHAT I DID NOT DO, AND WHY

* **No timing runs.** Per the brief and the context file. Both bench mains were
  built to prove they still compile; neither was executed.
* **No `git` anything.** Working tree only.
* **`updateSortedWarpQ` and the candidate prefix sum are not ported.** Both
  need `__ballot_sync`; see §1 and the `UNPORTED.tsv` rows. This is the one
  real algorithmic gap and it costs us their per-tile threshold pre-reject.
* **The cross-block path (`grid.x > 1`) is not ported.** Its merge *is*
  reachable now (`add`, `:284-300`); its mutex spin is not established on
  Metal. Left as an OPEN item with the blocker correctly named, which is a
  change from the previous claim that it needed ballot.
* **Double buffering not restored.** The kernel now uses 18,496 B, so one more
  page is 36,992 B — still over Apple's 32 KB. That is exactly the
  `lib_smem_pages_for[column, page_bytes]` case the round-2 addendum describes
  (NVIDIA's 48 KB and AMD's 64 KB both have room), but `mojo_only/kernel_matrix.mojo`
  is another session's file and I did not touch it. **Proposed for the next
  round**, and now cheap: the constant is no longer entangled with a selector.
* **`fusedL2UnexpKnn` not ported**, so `L2Unexpanded` / `L2SqrtUnexpanded`
  still raise even though `knn_brute_force.cuh:443` admits them.
* **`core/expand_distances.mojo` still lacks the second clamp clause.** Same
  finding as round 1, still true, still another lane's file. The fused path has
  both clauses and `check_fused_queue_reach_by_sabotage` clause 2 now asserts
  the exact zero it produces, so the two paths measurably disagree on a
  self-match.
* **`neighbors/PORTED_MAP.tsv` and `neighbors/UNPORTED.tsv` not edited** — not
  assigned to me. Rows are in §3, ready to paste.
