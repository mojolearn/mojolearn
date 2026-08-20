# LANE warpsort — 2026-08-19

**Verdict.** The compiler crash is FIXED, and it was not the bitonic network.
`select_warpsort.mojo` now builds, launches and returns RAFT's answer at every
capacity 1..256. On top of that, `faiss_select::WarpSelect` — the
register-resident queue `fusedL2Knn` is built on — is ported, instantiated at
both configurations the fused kernel uses, and checked. Four device checks
pass, and each was proved non-vacuous by a negative control.

    $ /tmp/warpsort_probe_main.bin
    check_warpsort_matches_radix: OK
    check_warpsort_reach_by_sabotage: OK
    check_warpselect_matches_oracle: OK
    check_warpselect_reach_by_sabotage: OK

**Strongest remaining gap:** nothing here is wired into a distance kernel. The
handoff signature for `fused_l2_knn.mojo` is at the end of this file.

---

## 0. THE MINIMAL REPRO OF THE CRASH, AND THE SECOND ONE UNDER IT

There were TWO independent compiler crashes, in two different compilers, and
the file could not be instantiated until both were fixed.

### Crash 1 — the Mojo compiler. Seven lines, no GPU.

`UNWIRED.md` (~line 662) names the recursive parametric bitonic network as the
prime suspect. **It is not the bitonic network.** The network instantiates
cleanly on its own at every size tried (`bitonic_merge` and `bitonic_sort` at
SIZE 1, 2, 4, 8 all build and launch). Bisected down to this, which has no GPU
in it, no SIMD, no warp primitive, no parametric anything:

```mojo
def main():
    var a = 8
    var b = 4
    while a > 1:
        a = b          # bare copy INTO the while-condition variable
        b = b - 1
    print(a)
```

    mojo: Please submit a bug report ... Stack dump: ...
    crash_report_exception_handler.cc:257 UniversalExceptionRaise: (os/kern) failure (5)

Characterized by four more variants:

| body | result |
|---|---|
| `a = b` / `b = b - 1`, condition on `a` | **CRASH** |
| `a = b` / `b = b - 1`, condition on `b` | builds |
| `a = b - 1` / `b = b - 1` | builds |
| `a = b` then `a = a - 1` | builds |
| `var t = b; a = t` | **CRASH** |
| `a = Int(b)` | **CRASH** |
| same update in a `for` loop instead of `while` | builds |

So the trigger is: **the `while`-condition variable's last write in the body is
a bare copy of another local.** Any arithmetic on top of the copy compiles. The
crash is a compiler bug, is deterministic, and reproduces on the host as well as
on the device.

Where it bit us: `block_sort_done` transliterates RAFT's C for-increment
`nwarps = split, split = (nwarps + 1) >> 1` (`select_warpsort.cuh:709-710`)
literally.

**Fix (HOW, not WHAT):** `split == (nwarps + 1) >> 1` holds on entry to every
iteration, by induction from their initializer and their increment, so `split`
becomes a loop-local and the induction step is written as the same arithmetic
instead of a copy:

```mojo
var shift_mask = ~Int(0)
while nwarps > 1:
    var split = (nwarps + 1) >> 1
    ...body, unchanged...
    nwarps = (nwarps + 1) >> 1
```

Same values, same order, same iteration count. No comparator was touched.

### Crash 2 — the APPLE METAL backend. Only above `capacity >= 64`.

With crash 1 fixed, `capacity` 1, 2, 8 and 32 built and ran; 64, 128 and 256
died with

    error: Metal Compiler failed to compile metallib. Please submit a bug report.
    mojo: error: failed to run the pass manager

`--emit=asm` succeeded, so the Mojo side was clean and the failure was in the
AIR compiler. The generated AIR showed the queue methods emitted as **callable
device functions**, not inlined — one of them containing `air.wg.barrier` and
`air.simd_shuffle_xor`.

**Fix, and it is a FIDELITY FIX, not a workaround.** Upstream marks every one
of these `_RAFT_DEVICE _RAFT_FORCEINLINE` (`bitonic_sort.cuh:27,35,97,125,144,
158,174,190,230`) and `__device__ inline`. Our port had dropped the annotation.
Restoring it as `@always_inline` on the free functions and the queue methods
makes every capacity 1..256 build and run.

**Recorded for the whole repo: on Metal, dropping upstream's `__forceinline__`
is not cosmetic. It turns a working kernel into a compiler crash with no
diagnostic.** The same annotation is applied pre-emptively to the new
`faiss_select` files.

---

## 1. DIVERGENCES FOUND

| upstream (file:line) | what ours did | fixed? |
|---|---|---|
| `bitonic_sort.cuh:27,35,97,125,144,158,174,190,230` — every function is `_RAFT_DEVICE _RAFT_FORCEINLINE`; `select_warpsort.cuh` marks all 32 members `_RAFT_DEVICE` | `select_warpsort.mojo` had **no inline annotation anywhere**. | **FIXED.** `@always_inline` on `is_ordered`, `dummy_key`, `twiddle_in`, `twiddle_out`, `bitonic_merge`, `bitonic_sort`, `block_sort_done`, `block_sort_store` and all seven `WarpSortImmediate` methods. This is what unblocked `capacity >= 64` on Metal. |
| `select_warpsort.cuh:709-710` — `for (... ; nwarps > 1; nwarps = split, split = (nwarps + 1) >> 1)` | Transliterated literally, which crashes the Mojo compiler (see §0). | **FIXED**, by restating the induction step as arithmetic. Values identical. |
| `select_warpsort.cuh` module docstring in ours claimed the port "COMPILES" | It compiled *alone* and crashed at every launch site, i.e. it was never at COMPILES by `VENDOR_LIBRARIES.md`'s four tiers. | **FIXED by the work**: it is now at RUNS ON DEVICE, checked against radix and against a host oracle. `UNWIRED.md` §"`select_warpsort`, 2026-08-19: COMPILES, NEVER RUN" is now false end to end — see §5. |
| `matrix/detail/select_warpsort.cuh` vs `neighbors/detail/faiss_select/Select.cuh` | This tree treated "warpsort" as ONE thing. | **Not a bug in code, a bug in the map.** They are two independent implementations of the FAISS design and neither subsumes the other; see §7 for the diff. Both are now ported, in separate directories, with the difference written into both file headers. |
| `select_warpsort.cuh:329` `set_k_th_` uses `__shfl_sync` with an explicit width and a deliberately out-of-range lane | Correctly recorded as the reason `warp_sort_filtered/_distributed/_distributed_ext` are unported. | **Still true, and now bounded.** That blocker does NOT reach `faiss_select`: its only indexed shuffle is `shfl(warpK[...], kLane)` with `kLane = (k-1) % WarpSize` (`Select.cuh:351`), already reduced, no width. So the fusion is reachable even if `warp_sort_filtered` never lands. |
| `fused_l2_knn.cuh:174-175` `__shfl_up_sync(mask, ..., 1)` and `raft::shfl` at `:141,:163` | Not previously assessed. | **Both expressible.** Mojo has `shuffle_up(value, offset)` and `shuffle_idx(value, offset)`; neither of those call sites uses a width, and `:141/:163` index by `kLane`/`srcLane`, both already in `[0,31]`. The one thing `updateSortedWarpQ` needs that Mojo lacks is `__ballot_sync` + `__ffs` — see §7, OPEN. |
| `Select.cuh:398` `__any_sync(0xffffffff, needSort)` | New code. | Ported as `warp.max(0/1) != 0`, a warp-wide OR with the identical value. Numbered deviation, in-register, no device-wide call. |
| `fused_l2_knn.cuh:749-760` instantiates `NumThreadQ = 3` | New code. | Mojo `SIMD` width must be a power of two, so the array is padded to 4 with element 3 never read or written. Numbered deviation; checked at `NumThreadQ=3` specifically. |

**Not a divergence, checked and cleared:** `warp_width` is a RUNTIME `int` in
`bitonic_sort.cuh:190,230` too, not a template parameter — our port is faithful
and the runtime loops are theirs. And the DEVIATION 2 claim in
`select_warpsort.mojo` (depth-first recursion emits the same network as their
level-by-level triple loop) holds: in-register compare-exchange only ever pairs
elements inside one subtree, so no element's cross-lane phase can observe
another subtree's later levels.

**Cleared by measurement, not by reasoning:** `stack_allocation` without an
address space is memory, not registers (`PORTING.md 26`). Neither file uses it
for a queue. The emitted AIR confirms it: `WarpSortImmediate[64,·]` lowers to
`{ i64, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i64 }` passed and returned
**by value** — the queues really are in registers. The only `stack_allocation`
in the file is the two SHARED-address-space halves, which is where RAFT's are.

---

## 2. WHAT I CHANGED, FILE BY FILE

### `neighbors/gbdt/matrix/detail/select_warpsort.mojo` (edited)
* `block_sort_done`: loop-update restated (`select_warpsort.cuh:709-710`), with
  the compiler-bug repro written into the file as a comment so the next reader
  does not "clean it up" back into a crash.
* `@always_inline` added to 15 definitions, mirroring
  `_RAFT_DEVICE _RAFT_FORCEINLINE` on `bitonic_sort.cuh:27,35,97,125,144,158,
  174,190,230` and `__device__ inline` throughout `select_warpsort.cuh`.
* Nothing else. No comparator, no network, no geometry.

### `neighbors/gbdt/neighbors/detail/faiss_select/merge_network_warp.mojo` (new)
Port of `Comparators.cuh` (float only), `MergeNetworkUtils.cuh` in full, and
`MergeNetworkWarp.cuh`'s power-of-two path in full:
`warpBitonicMergeLE16` (`:83-136`), `BitonicMergeStep` pow-2 (`:146-212`),
`warpMergeAnyRegisters` (`:391-436`), `BitonicSortStep` incl. the odd-`N`
generic case and the `N==1` five-call specialization (`:440-503`),
`warpSortAnyRegisters` (`:508-512`).

### `neighbors/gbdt/neighbors/detail/faiss_select/select.mojo` (new)
Port of `Select.cuh:346-500`, the general `WarpSelect`: `__init__`,
`add_thread_q`, `check_thread_q`, `merge_warp_q`, `add`, `reduce`, `write_out`.

### `neighbors/mojo_only/warpsort_check.mojo` (new) and `neighbors/warpsort_probe_main.mojo` (new)
Four device checks. Fixture is distinct hashed signed floats spread over
several binades, so a wrong reduction, a wrong lane map and a wrong payload are
all visible; a uniform fixture would verify the total and nothing about
placement.

---

## 3. PROPOSED ROWS

`neighbors/PORTED_MAP.tsv` (tab-separated):

```
raft	cpp/include/raft/neighbors/detail/faiss_select/Comparators.cuh	neighbors/gbdt/neighbors/detail/faiss_select/merge_network_warp.mojo	PARTIAL	float only; Comparator<half> not ported
raft	cpp/include/raft/neighbors/detail/faiss_select/MergeNetworkUtils.cuh	neighbors/gbdt/neighbors/detail/faiss_select/merge_network_warp.mojo	FULL	swap/assign, predicated
raft	cpp/include/raft/neighbors/detail/faiss_select/MergeNetworkWarp.cuh	neighbors/gbdt/neighbors/detail/faiss_select/merge_network_warp.mojo	PARTIAL	power-of-two BitonicMergeStep only; non-pow2 specializations unreachable for every fused_l2_knn instantiation
raft	cpp/include/raft/neighbors/detail/faiss_select/Select.cuh	neighbors/gbdt/neighbors/detail/faiss_select/select.mojo	PARTIAL	WarpSelect general specialization only
```

`neighbors/UNPORTED.tsv`:

```
raft	cpp/include/raft/neighbors/detail/faiss_select/Comparators.cuh	Comparator<half>	NOT NEEDED	no half path here
raft	cpp/include/raft/neighbors/detail/faiss_select/MergeNetworkWarp.cuh	BitonicMergeStep non-power-of-2 (:218,:306)	UNREACHABLE	fused_l2_knn.cuh:749-760 gives N in {1,2} only; comptime assert guards it
raft	cpp/include/raft/neighbors/detail/faiss_select/Select.cuh	WarpSelect<NumWarpQ=1> (:503-548)	UNREACHABLE	fused NumWarpQ is 32 or 64 for every k
raft	cpp/include/raft/neighbors/detail/faiss_select/Select.cuh	BlockSelect, FinalBlockMerge (:20-338)	OPEN	fused kernel uses the warp queue only
raft	cpp/include/raft/neighbors/detail/faiss_select/MergeNetworkBlock.cuh	all	OPEN	only BlockSelect needs it
raft	cpp/include/raft/neighbors/detail/faiss_select/key_value_block_select.cuh	all	OPEN	not on the fused path
```

`neighbors/UNPORTED.tsv` — **row to DELETE or amend**: any row asserting
`select_warpsort.cuh` is UNPORTABLE, or that it compiles but cannot be
instantiated. Both are now false.

---

## 4. PROPOSED `PORTING.md` DEVIATION ENTRIES (renumber from 30)

**30. A bare copy into a `while`-condition variable crashes the Mojo compiler.**
`while a > 1: a = b; b = b - 1` segfaults `mojo build`, on host and device
alike, with no diagnostic. Adding any arithmetic to the copy (`a = b - 1`, or
`a = b` followed by `a = a - 1`) compiles, and a `for` loop compiles. When a C
for-increment reads `x = y, y = f(x)`, restate `y` as a loop-local and write the
induction step as arithmetic. Found bisecting `block_sort_done`
(`select_warpsort.cuh:709-710`) down to seven lines.

**31. On Metal, dropping upstream's `__forceinline__` is a compiler crash, not
a performance choice.** `select_warpsort.mojo` at `capacity >= 64` failed with
`Metal Compiler failed to compile metallib` while `--emit=asm` was clean; the
AIR showed the queue methods as callable device functions carrying
`air.wg.barrier` and `air.simd_shuffle_xor`. Adding `@always_inline` — which is
what every one of those RAFT functions already carries — fixed all capacities.
Port the annotation, not just the body.

**32. Mojo `SIMD` width must be a power of two, so a register array of odd
length is padded.** `faiss_select::WarpSelect` is instantiated with
`NumThreadQ = 3` (`fused_l2_knn.cuh:759`). The array becomes width 4 and
element 3 is never read or written, because every loop runs `range(N)` with the
true `N`. Padding lanes carry the same sentinel as the rest.

**33. Mojo has no warp vote.** `__any_sync(mask, pred)` has no direct
counterpart; `warp.max(1 if pred else 0) != 0` is the same value, as a shuffle
tree rather than one instruction. `__ballot_sync` + `__ffs`, which return WHICH
lanes voted, have no such rewrite — see the OPEN item in this lane's report.

---

## 5. FALSE DOC SENTENCES FOUND (in files I may not edit)

**`UNWIRED.md`, section "`select_warpsort`, 2026-08-19: COMPILES, NEVER RUN".**
Every load-bearing sentence in it is now false and the section should be
DELETED, not annotated:

* "It compiles and it has never executed." — it executes; four checks pass.
* "It is instantiating `warpsort_topk_block_kernel[16, True, 8]` at a launch
  site that kills the compiler." — true then; `[16, True, 8]` and every other
  capacity now build and run.
* "The suspect is the recursive parametric bitonic network ... a `@parameter
  if` that fails to prune the dead branch recurses forever." — **wrong
  suspect.** The network was never implicated; the network alone instantiates
  fine at SIZE 1, 2, 4, 8. The Mojo crash was a two-variable loop update in
  `block_sort_done`, and the second crash was the Metal backend on a missing
  inline annotation.
* "So warpsort is not at COMPILES either ... it is at 'compiles alone, cannot
  be instantiated'." — it is at RUNS ON DEVICE.
* "Wiring it in is deliberately a separate step, and it must go in BESIDE those
  two rather than replacing either, so that `check_vendor_topk_matches_ported`
  can be extended to a three-way agreement." — the *rule* stands, but note the
  round's rule change: a vendor top-k can never do the fused kernel's job, so
  three-way agreement is a correctness check, not a design option.

**`neighbors/gbdt/matrix/detail/select_radix.mojo:38`** ends with "...and
warpsort IS now ported (`select_warpsort.mojo`) but cannot yet be instantiated
at a launch site without crashing the compiler; see UNWIRED.md." That clause is
false. I did not edit the file (not mine). Suggested replacement: "...and
warpsort IS now ported (`select_warpsort.mojo`), runs on device, and agrees
with this file per element on a scattered fixture for k = 1, 2, 8, 32, 64, 100,
256 and k > n."

**`neighbors/README.md` / `PORTING_RULES.md`** were not read for this, so no
claim about them.

---

## 6. BUILD / CHECK EVIDENCE

Build (every build in this lane went through the lock):

    tools/with_build_lock.sh pixi run --manifest-path \
      /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
      mojo build -I . neighbors/warpsort_probe_main.mojo -o /tmp/warpsort_probe_main.bin
    # exit 0

Run:

    $ /tmp/warpsort_probe_main.bin
    check_warpsort_matches_radix: OK
    check_warpsort_reach_by_sabotage: OK
    check_warpselect_matches_oracle: OK
    check_warpselect_reach_by_sabotage: OK

### What the checks actually assert

`check_warpsort_matches_radix` — 8 rows x 4096 columns of distinct hashed
signed floats, three-way per row: ported warpsort vs ported radix select vs a
Float64 host sort. Cases `(k, capacity, block_warps)`:
`(1,1,256) (2,2,128) (8,8,32) (32,32,8) (64,64,8) (100,128,8) (256,256,8)`,
plus `k > n` (length 5, k 8), plus a deliberately TIED fixture.
VALUES are compared strictly as a sorted multiset per row. INDICES are compared
exactly only on the distinct-value fixtures, where there is no tie to resolve;
on the tied fixture only values are compared, because `select_radix.mojo`
documents that RAFT's radix places tied outputs with `atomicAdd` and warpsort's
tie order is stable but different.

`check_warpselect_matches_oracle` — both fused instantiations and nothing else:
`NumWarpQ=32,NumThreadQ=2` at k = 1, 8, 32 and `NumWarpQ=64,NumThreadQ=3` at
k = 33, 50, 64, plus a row length of 1013 (not a multiple of 32) to exercise the
sentinel tail. Values against the host oracle; payload must name the cell the
value came from.

### Sabotage (reach, not digest)

Both sabotage checks plant `-9999.0` in ONE cell of ONE row and assert three
things at once: (a) that value comes back in that row's top-k **carrying its own
column as payload** — a no-op kernel fails this, and a kernel that ignores its
payload fails it too; (b) the victim row keeps exactly `k-1` of its previous
answers — the window is sized so the result must move by exactly one; (c) no
other row changes by a single value or payload — a selection leaking across
rows fails this.

### Negative controls (the checks are not vacuous)

Flipping `ascending` on the block kernel: **3960 disagreements**, check raises.
Flipping `dir` on `WarpSelect`: **4032 disagreements**, check raises.
Both reverted; the passing run above is the reverted state.

No timing was run.

---

## 7. THE TWO IMPLEMENTATIONS, DIFFED — AND THE HANDOFF

### They are different code, and neither subsumes the other

| | `matrix/detail/select_warpsort.cuh` | `neighbors/detail/faiss_select/Select.cuh` |
|---|---|---|
| network | `util/bitonic_sort.cuh` | `MergeNetworkWarp.cuh` |
| data layout | one element per lane per array slot, STRIDED across a subwarp of width `min(Capacity,32)` | `WarpSize * N` elements, `N` registers per lane, warp width pinned to 32 by `static_assert` (`MergeNetworkWarp.cuh:495`) |
| subwarp width | RUNTIME argument (`bitonic_sort.cuh:190,230`) | compile-time 32 |
| comparator | strict, one comparison, ties keep the incumbent | **both** `<` and `>` on the two sides of an exchange, on purpose, so the two lanes of a pair agree (`MergeNetworkWarp.cuh:38-63`) |
| shared memory | required — block-wide tree merge in smem | **none** |
| how it is fed | a kernel that reads a materialized distance row from global memory | a struct a kernel keeps in registers and feeds one value at a time |
| entry point | `block_kernel`, launched | `WarpSelect`, instantiated inside someone else's kernel |
| indexed shuffle | `set_k_th_` — `__shfl_sync` with a width and a deliberately out-of-range lane. **Untranslatable.** | `shfl(warpK[last], kLane)`, `kLane = (k-1)%32`, already reduced. **Translates.** |

The first cannot fuse: it reads its input from memory by construction. The
second is the only one that can. They are complementary, not redundant.

### HANDOFF: the exact signature `fused_l2_knn.mojo` must instantiate

```mojo
from neighbors.ported.neighbors.detail.faiss_select.select import WarpSelect

# fused_l2_knn.cuh:221-222 is
#   typedef WarpSelect<AccT, uint32_t, Dir, Comparator<AccT>,
#                      NumWarpQ, NumThreadQ, 32> myWarpSelect;
# with Dir = false. K = Float32 and V = UInt32 are baked in here.

var q = WarpSelect[num_warp_q, num_thread_q, /*dir=*/False](
    /*init_k_val=*/Float32(3.4028234663852886e38),   # numeric_limits<float>::max()
    /*init_v_val=*/UInt32(0xFFFFFFFF),               # keyMax, fused_l2_knn.cuh:219
    /*k=*/num_of_nn,
)

q.add(key: Float32, val: UInt32)      # add_thread_q + check_thread_q
q.add_thread_q(key, val)              # call these two separately only if you
q.check_thread_q()                    #   must; see the contract below
q.merge_warp_q()
q.reduce()
q.write_out(out_k: MutPointer[Float32, MutAnyOrigin],
            out_v: MutPointer[UInt32, MutAnyOrigin],
            k: Int)

# comptime, for anyone reaching into the queue the way
# fused_l2_knn.cuh:133-141 does:
WarpSelect[nwq, ntq, False].num_warp_q_registers   # == nwq // 32
# public fields, same names as theirs: warp_k, warp_v, warp_k_top, k_lane,
# thread_k, thread_v, num_vals, init_k, init_v
```

**The instantiation set is exactly two**, and it is theirs, not a limit of the
port (`fused_l2_knn.cuh:743-771`):

    num_of_nn <= 32  ->  WarpSelect[32, 2, False]
    num_of_nn <= 64  ->  WarpSelect[64, 3, False]
    num_of_nn  > 64  ->  their own ASSERT: "num of nearest neighbors must be <= 64"

Both are built and checked. `ThreadsPerBlock` is not a parameter here: the
general specialization uses it only for a `static_assert` (`Select.cuh:362`).

**CALL CONTRACT — this one bites.** Every lane of the warp must call `add` the
SAME number of times. `check_thread_q` contains a warp vote and a merge full of
shuffles; if some lanes exit the loop early the rest hang. Round the row length
up to a multiple of 32 and feed `init_k`/`init_v` for the tail. The probe kernel
in `neighbors/mojo_only/warpsort_check.mojo` does exactly this and the check
covers a length of 1013.

### OPEN, and it is the k-NN lane's decision, not mine

`fused_l2_knn.cuh:146-186`, `updateSortedWarpQ`, is the CROSS-BLOCK merge, and
it is the one piece of the fused path that Mojo cannot spell today:

```cpp
unsigned activeLanes = __ballot_sync(mask, KVPair.value < heapArr->warpK[i]);
const auto firstActiveLane = __ffs(activeLanes) - 1;
```

`__ballot_sync` returns WHICH lanes voted, and `__ffs` picks the lowest. Mojo
has the warp reductions and `shuffle_up`/`shuffle_idx` but no ballot. This is
NOT needed when `gridDim.x == 1` — their own `rowEpilog_lambda` returns
immediately in that case (`fused_l2_knn.cuh:225`) — so a single-x-block fused
kernel is fully reachable with what is ported here. A multi-x-block fused kernel
needs a decision on the ballot. Reported, not guessed at.

---

## 8. WHAT I DID NOT DO, AND WHY

* **Did not wire warpsort into `knn_brute_force.mojo`.** Told not to; another
  lane is rewriting its tiling. Reach is proved through my own probe main
  instead. The call the orchestrator (or that lane) must add is in §7.
* **Did not touch `fused_l2_knn.mojo`, `ball_cover/`, `select_radix.mojo`,
  the neighbors maps/README, or `core/`.** Rows and doc corrections are in §3
  and §5 instead.
* **Did not port `BlockSelect` / `MergeNetworkBlock.cuh` /
  `key_value_block_select.cuh`.** The fused path does not use them, and the
  round's priority is the fused path.
* **Did not port the non-power-of-two `BitonicMergeStep` specializations.**
  Unreachable for both fused instantiations; a `comptime assert` fails loudly
  if that ever stops being true, rather than silently producing a wrong network.
* **Did not run any timing.** Not mine.
* **Did not reconstruct `__shfl_sync`'s modulo by hand.** Still a guess at CUDA
  semantics; `warp_sort_filtered/_distributed/_distributed_ext` stay unported.
  §7 shows the fusion does not need them.

## 9. AN OPERATIONAL WARNING FOR THE ORCHESTRATOR

Three times during this session, `neighbors/gbdt/matrix/detail/select_warpsort.mojo`
**and** `select_radix.mojo` were DELETED from the working tree by something
outside this lane, and once my edited `select_warpsort.mojo` was reverted to
HEAD mid-bisect. That reversion produced a fake result — a build that had been
passing started "crashing" again — and cost real time before I noticed. I
restored both files with `git show HEAD:<path> > <path>` (no checkout, no
stash) and thereafter re-synced my owned files from a scratchpad copy before
every build. Worth finding what is doing it before the next parallel round.
