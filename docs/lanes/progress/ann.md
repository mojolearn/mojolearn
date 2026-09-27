# ann: progress

Pass 1 design (all lanes): every Mojo computation is a per-cell function in
`x_ann/*_core.mojo`; the GPU binding runs one cell per thread
(`x_ann/*_device.mojo`), the CPU host binding runs the SAME function in a
loop (`x_ann/host/*_host.mojo`), so the two agree by construction. Integer
graph work (t-SNE symmetrization, CAGRA prune/reverse merge, CSR lists) is
the same host function in both drivers. NOT_IMPLEMENTED: `x_ann/NOT_IMPLEMENTED.tsv`.

Pod notes: dev pods with an NVIDIA driver < 580 need
`MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` at build and run
time (build_x_ann.sh then names the arch from nvidia-smi). A host build that
imports `ivf/host/ivf_host.mojo` next to new code hangs the Mojo compiler
(0 CPU, futex), so IVF-PQ carries its own fixed-order Lloyd.

| algorithm | lane | commit | AGREE (H100 pod, CPU == NVIDIA) | sanity |
|---|---|---|---|---|
| IVF-PQ (`IVFPQIndex`) | x-ann-ivf-pq | e7ccc453d | AGREE: compared batch 9, infer 9, train 9 | recall@10 0.639 vs faiss IndexIVFPQ 0.595 (64 lists, 8 probes, pq 8x8, 20000x32) |
| t-SNE (`TSNE`) | x-ann-tsne | "t-SNE: ..." (see git log) | AGREE: compared infer 9, train 9 (batch n/a: whole-set) | trustworthiness@10 0.9824 vs sklearn 0.9823; KL 1.021 vs 1.039 (1500x20 blobs, perplexity 30, 1000 steps) |
