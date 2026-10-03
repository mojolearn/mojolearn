# lane/apple-fast-vsearch: the IVF family (ivf, ivf-pq, ivf-sq, ivf-rabitq, ivf-refine, ivf-filter) under FAST on Apple

Every switch is a build define read with `is_defined` at module scope, default OFF, compiled only under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`; IDENTICAL compiles main's code unchanged.
The x_ann switches live in x_ann/vsearch_fast.mojo; the coarse build switches in
ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo and cluster/impl/detail/kmeans.mojo, which BOTH bindings compile
(x_ann's IVF-PQ/SQ/RaBitQ coarse quantizer is the IVF-Flat build). `-D MOJOLEARN_VSEARCH_ALL` turns on all six.
The per-fit and per-search profile they answer is docs/apple-fast/notes/vsearch.md. Request lines:
docs/apple-fast/ab/vsearch.txt (istella ivf-pq first, then the classical2 ivf row, then the rest). Recall is the
quality column: every arm must keep the board's recall vs exact equal to FAST's spread.

Not redone here (lane/apple-fast-ann, unmerged): the device PQ codebooks (the brief's PQ_SUBSPACE_BATCH idea, ~4.4 s
of Istella's 6.2 s ivf-pq), the device trainset and quantizer scale, the device list layout, the one-launch select.
The brief's IVF_KMEANS_FUSED idea is answered by LAZY_SHIFT (the waits, not the launch count, are the cost), its
IVF_SCAN_TOPK and PQ_LUT_SMEM ideas by PQ_SCAN_FUSED and PQ_LUT_TILED, and REFINE_FUSED by REFINE_TEAM (refine is a
separate user call, `mojolearn.refine`, so it cannot share the scan's launch; the filter is already tested inside
the score and select kernels, no separate pass exists to fuse).

**MOJOLEARN_PQ_LUT_TILED** (x_ann binding; ivf-pq, ivf-filter, ivf-refine). Mechanism: `pq_score_tiled_kernel`
stages the asymmetric lookup table in threadgroup memory in tiles of LUT_MAX entries (16 subspaces x 256 codes per
tile) and the query residual once per threadgroup. On Istella (55 x 256 = 14,080 entries) main's table does not
fit, so every one of ~50 M candidates re-forms the residual and re-evaluates 55 entries from device memory (220 FMAs
and ~660 loads each). Expected: the Istella score stage drops several fold; taxi (table already fits) about flat.
Risk: low; same fold on the same words, so bits and recall are unchanged; a threadgroup-memory fits gate guards the
tile.

**MOJOLEARN_PQ_SCAN_FUSED** (x_ann; ivf-pq, ivf-filter, ivf-refine's candidate pass). Mechanism:
`pq_scan_fused_kernel` does score and top-k in ONE launch per chunk of queries: one threadgroup per query walks its
probed lists with the tiled table, keeps a register top-k per thread and joins in threadgroup memory. No
mc x stride candidate-distance buffer (up to 64 MB), no 9 select launches per chunk, fewer live buffers. Only when
k <= SEL_KM and the table tiles fit; otherwise main's path. Takes precedence over LUT_TILED when both are on.
Expected: search time down on both datasets (larger on Istella). Risk: one threadgroup per query is less parallel
than one per (query, probe) at small query counts; bits unchanged (same words, NaN keeps the cell's order).

**MOJOLEARN_IVF_REFINE_TEAM** (x_ann; ivf-refine). Mechanism: the refine binding uploads the dataset straight from
the caller's array (no 352 MB host copy into a List) and `refine_team_kernel` runs one threadgroup per query, one
thread per candidate with the cell's fold, thread 0 inserting in slot order, instead of main's ONE THREAD PER QUERY
(4,000 threads walking 40 x 220 sequentially). Expected: refine stage down by a large factor on Istella; small on
taxi. Risk: low; same words.

**MOJOLEARN_IVF_KMEANS_LAZY_SHIFT** (both bindings; every IVF row, ivf-sq and ivf-rabitq most, since the coarse
k-means is their whole build). Mechanism: the Lloyd loop reads the centroid shift back and tests convergence every
4 iterations and at max_iter, not every iteration: 20 waits and host round trips per coarse fit (and per k-means||
recluster) become 5; the iterations between enqueue the same launches with no wait. Expected: a few ms per avoided
wait times the number of Lloyd loops. Risk: a fit that would have converged early runs up to 3 more Lloyd steps
(objective never rises); a fit that runs to max_iter (the board's 20) is bit-identical.

**MOJOLEARN_IVF_COARSE_RANDOM_INIT** (both bindings; every IVF row). Mechanism: the coarse quantizer starts from
n_lists distinct training rows drawn with the fit's seed (FAISS's own start) as INIT_ARRAY, so k-means|| does not
run: 8 rounds of every-row-to-candidate distances (~1.0 TFMA on Istella, about the cost of the 20 Lloyd iterations)
with ~30 waits and host readbacks, plus the candidates' own Lloyd loop. Expected: the coarse build roughly halves.
Risk: FAST bits MOVE (another start); recall must be checked in the pair; this is faiss-cpu's own initialization.

**MOJOLEARN_IVF_DEVICE_VALIDATE** (both bindings; every IVF row). Mechanism: the n x dim finiteness scan
(`ivf_validate_data`, a host pass over 352 MB on Istella) runs as `ivf_refused_words_kernel` on the device after
the upload; only a refusal reruns the host scan for its message. Expected: tens of ms on Istella, little on taxi.
Risk: none to bits; same refusal contract.

**MOJOLEARN_VSEARCH_ALL**: all six together (FUSED wins over TILED where both apply). The combination's FAST bits
move only through COARSE_RANDOM_INIT.
