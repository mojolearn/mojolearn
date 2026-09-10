# Bounded 512-query batch candidate

The preceding coalesced-column prototype did not improve the target H100
requests and remains disabled. This separate opt-in candidate changes query
batching, not column ownership, index partitioning, selector rank order, or
floating arithmetic. It has not been built or run by this lane.

`MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512` makes the default query batch 512 in
IDENTICAL and budgets the distance tile using its actual index-column width.
The baseline still uses 256 and the historical full-index budget estimate.
Explicit requested batch values remain explicit; the candidate's more accurate
budget can avoid shrinking them unnecessarily. All dimensions still pass the
existing floor and query-count clamp. No behavior changes without the flag.

Why this is credible: one selector block handles one query. A 256-query batch
launches 256 blocks; 512 makes more independent work available while halving
the number of distance/select/merge launch groups for many-query requests.
This does not split any selector's row or introduce a new reduction/merge.
At 400k index rows, the old planner incorrectly estimates a 512-row distance
tile at about 781 MiB and halves it. The actual tiled IDENTICAL allocation is
128 MiB (512 x 65536 x 4). Radix fallback scratch is still allocated even when
the small-k selector runs; it increases from about 195 to 391 MiB across the
two buffers. The larger working set may hurt cache locality or allocation
cost, so the result must be measured, not assumed.

The throughput target is 1000–4000 queries against a large index. A 32-query
request clamps identically in both arms. This is a scheduling experiment,
not an assertion that larger batches are universally better or a new
workspace guarantee; the existing budget explicitly covers distance only.

Root-only validation in an activated environment:

```
bash tools/knn_query_batch_probe.sh /fresh/absolute/evidence checks
bash tools/knn_query_batch_probe.sh /fresh/absolute/evidence-gpu
```

The new gate compares every output word from explicit256 against the default
at 513 queries, including tied index rows, ragged feature dimensions, and an
index-column tile boundary. Existing public layout gates run in both arms.
The reference driver now uses the same default query tile for its request
and device regions. Its full-word dump is compared at five shapes in forward
and reversed order, five samples per phase per pass. No opponent rerun is
needed. Any accepted timing must reuse the existing matched opponent tuple
and retain raw samples and full-output equality evidence.

The candidate is now bounded to at most 400,000 index rows for its accurate
width budget. Above that bound, the historical full-index estimate applies.
Since 400001*512*4 exceeds 768 MiB, the first halving necessarily reaches
256, after which the baseline default's shrink/floor/clamp is identical.
Host-only planner assertions cover the boundary, one million and 100 million
index rows, and a one-query clamp without allocating those fixtures.

For accepted IDENTICAL k<=1024 and index<=400000, all known query-batch
scratch at batch512 is bounded by 548,012,032 bytes (about 522.6 MiB):
`512*(4*65536 + 16*max(400000//8,1024) + 8*1024)`. This includes distance,
both fallback radix buffers, and the partial distance/index pair. Fixed-size
index/query/norm/output and transposed-index allocations do not grow with
batch size. This is not a new total-memory guarantee; beyond this measured
scope the existing planner's minimum32 behavior is preserved verbatim.
If timing wins, production default admission should additionally require
NVIDIA IDENTICAL; portable Apple opt-in remains qualification only.
