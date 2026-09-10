# kNN register-column ownership candidate

This opt-in candidate starts at main `d557b851`. It has not been built or run
by this lane. Root owns local builds and GPU execution. No default changes,
new opponent timings, or speed claims are included.

The last accepted NVIDIA distance profile was about 21.2 ms, selection 13.4 ms,
and merge 0.8 ms for 400k index rows, 4000 queries, 32 features, k=10. The
current register tile assigns four adjacent columns to each thread. A scalar
load/store instruction therefore addresses every fourth Float32 across a
warp. The candidate assigns `block*512 + lane + slot*128` instead. Each
instruction addresses consecutive columns across lanes. This may reduce
memory instruction transaction pressure; caches and existing compiler
vectorization may already hide the old cost, so measurement must decide.

The candidate preserves the 512-column block, eight NVIDIA query rows per
thread, register count, feature order, exact FMA/FTZ seam, epilogue, output
matrix layout, and selection. No floating reduction is introduced. Clamped
loads remain for partial tiles, and stores use the unclamped assigned column.
No thread whose first column is invalid can have a valid later column.

`MOJOLEARN_KNN_IDENTICAL_COALESCED_COLUMNS` enables the mapping only in
IDENTICAL. It is portable for Apple qualification but its performance target
is NVIDIA. FAST, DETERMINISTIC, and trees are untouched. This is distinct from
previously rejected deeper selector scans, partitioned selectors, feature
hoisting, and unsafe FTZ FMA.

`neighbors/checks/coalesced_distance_check.mojo` checks the ownership bijection
and 40 exact scalar/register distance and selection cases. Index widths
1/127/129/511/513 cross lane, slot, and block boundaries; query rows1/7/9/17
cross register-row boundaries. Four profiles cover ties, cancellation,
signed zeros/subnormals, and independent data; both squared and square-root
epilogues run. The existing public layout gate checks tile offsets and metrics.

Root can run, in its activated environment and granted exclusive slot:

```
bash tools/knn_coalesced_columns_probe.sh /fresh/absolute/evidence checks
bash tools/knn_coalesced_columns_probe.sh /fresh/absolute/evidence-gpu
```

The GPU driver runs forward/reversed five-round A/B prices at the large
k10/k15 requests and low-feature, small-query, and ragged controls. The new
optional `MOJOLEARN_KNN_REF_DUMP_FULL` artifact writes every selected index and
raw distance word outside timing. `cmp` requires all those bytes equal across
arms. Public gate comparisons likewise include every emitted cell. Reuse
existing matched opponent rows only if this candidate is accepted; do not
promote a component timing to an end-to-end or cross-hardware claim.

Independent address review: in a partial final block, slot zero is the minimum
column owned by a lane, so the early return cannot discard a later valid slot.
For each valid local column `j`, lane=`j % 128`, slot=`j // 128` is its unique
writer. Out-of-range loads clamp to the last column but stores recompute the
original column, preventing duplicate tail writes. There are no barriers or
warp collectives in this kernel that would be invalidated by partial exits.

At scalar instruction level, a full aligned warp requests four 32-byte sectors
for one candidate column slot versus sixteen for the old stride-four slot.
Across all four old slots those sectors overlap, so this is not a claim of
four times less DRAM traffic. Adjacent old columns may also be combined into
vector loads/stores by the compiler. No PTX/SASS inspection was performed;
compiled transaction behavior remains a hypothesis until disassembly or
profiling confirms it. The candidate also changes short-row utilization:
width65 activates 65 lanes rather than 17, with more discarded clamped chains.
That makes the short/tail controls necessary even if large requests improve.

`tools/knn_coalesced_columns_summary.py` validates both passes, all five shapes,
raw timing sample counts, device/request checks, component/public evidence,
and the full index/distance dumps. It emits raw-backed JSON with pooled
medians and per-pass samples. A pooled result should be read together with
both execution orders rather than treated as controlled statistical evidence.
