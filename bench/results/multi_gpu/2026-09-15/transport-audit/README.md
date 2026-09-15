# Device-to-device transport audit on two MI300X, above 1 MiB

After `../peer-copy-mi300x/` showed that a kernel on an MI300X target can read
the previous contents of the target memory after a cross-device copy and a
drain of both contexts, the other multi-GPU drivers were asked the same
question with buffers above 1 MiB. Every leg is a RunPod pod with two GPUs
running `tools/identity_break.py` par-* lanes once with
`MOJOLEARN_PAR_DEVICES=0` and once with `0,1` on the same box, using the two
audit-only switches added for this: `MOJOLEARN_IDENTITY_N` (fixture rows) and
`MOJOLEARN_IDENTITY_WIDE=1` (wider neural models). Those columns record
`package.fixture_n` or `package.wide`, and `--diff` refuses to compare them
with default-size columns.

## Where the code copies between devices

`enqueue_copy_to` and `peer_clone` call sites (device copies, some of them
between two contexts of one device), by file, before the fix: `cholesky/multi_gpu.mojo` (3), `cluster/multi_gpu.mojo` (6,
KMeans), `core/gram_multi_gpu.mojo` (4), `core/gram_splitk.mojo` (3),
`core/householder_qr.mojo` (3), `dbscan/impl/multi_gpu.mojo` (12),
`gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo` (10),
`gbdt/methods/pointwise_multi_gpu.mojo` (11), `glm/impl/qn/multi_gpu.mojo` (3),
`hierarchy/impl/cluster/detail/multi_gpu.mojo` (5),
`isolation_forest/impl/isolation_forest.mojo` (1), `mixture/multi_gpu.mojo`
(10), `neighbors/impl/multi_gpu.mojo` (6), `resample/estimator.mojo` (3),
`solver/multi_gpu.mojo` (3), `svm/impl/distance/kernel_matrices.mojo` (5),
`training/byte_lm_layer_pool.mojo` (6), `training/byte_lm_parallel.mojo` (5),
and `gaussian_process/checks/kernels.mojo` (4, a check). The drivers run kernels
on the target of their cross-device copies, which is the pattern that failed.

## Results

| leg | lanes | switch | GPUs, pod, source | one vs two |
| --- | --- | --- | --- | --- |
| `neural-wide-mi300x-before-fix` | par-byte-lm, par-byte-lm-model-pool, par-samba, par-samba-clip; fixtures base, ties, denormal | WIDE (byte LM 2.1 million parameters) | 2x MI300X, `q0btiva1ejdemq`, `0565224c3` | **par-byte-lm MOVED on base and denormal, DIVERGENT on ties**; the other three lanes IDENTICAL |
| `neural-wide-h100-before-fix` | the same | WIDE | 2x H100, `y4y1ex89n3hzhw`, `0565224c3` | IDENTICAL=12 |
| `classical-n80000-mi300x` | par-kmeans, par-gram, par-forest-pool, par-boosting, par-logistic, par-scaler, par-boosting-pointwise; base, ties, denormal | N=80000 (X is 4.88 MiB) | 2x MI300X, `sz16ipj3agk1mk`, `0565224c3` | IDENTICAL=21; infer/model IDENTICAL=33, N/A=9; batch IDENTICAL=18, N/A=3 |
| `byte-lm-fix-mi300x` | par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload; base, ties, denormal, dupes, odd | WIDE | 2x MI300X, `qwk5oe9b2he1ar`, `e113ed529` | IDENTICAL=15 |
| `byte-lm-fix-h100` | the same | WIDE | 2x H100, `s5dzv0fwpghyhn`, `e113ed529` | IDENTICAL=15 |

`neural-wide-before-fix.four-columns.txt` names the column that stands alone:
for par-byte-lm the MI300X one-device column and both H100 columns hash
equal on every fixture (`c7e833dbba28573a` on base), and only the MI300X
two-device column moves between its own repeats or disagrees (`summary:
DIVERGENT=1, IDENTICAL=9, MOVED=2`).

## The fix, and its gate

`core/multi_gpu.mojo::transfer_bytes` copies between contexts; on an AMD
(HIP) build a cross-device copy reads the source into pinned host memory
through the source context and writes it through the target context. Other
vendors and same-device copies keep the device copy and the source drain the
call sites had. `training/byte_lm_parallel.mojo`'s five copies (parameter
broadcast, pooled and replicated gradient gathers, committed-gradient
scatters) use it.

- The unfixed tree was seen to fail (the first row above).
- At `e113ed529`, `byte-lm-fix-mi300x` and `byte-lm-fix-h100` read one against
  two devices IDENTICAL=15 each, and `byte-lm-fix-wide.four-columns.txt`
  (both vendors, both device counts, `--require-columns 4`) reads
  IDENTICAL=15. The fixed MI300X two-device par-byte-lm cells equal the
  unfixed MI300X one-device cells on base, ties and denormal.
- The default-size lanes on two devices after the fix (`default-two.json` on
  each vendor) equal the 166-lane record's two-device columns on all 27
  cells of the three byte-LM lanes (`byte-lm-fix-default.vs-166.txt`:
  IDENTICAL=27), so the default path's bits did not move.

## What this does not cover

- Only the lanes above ran above 1 MiB. The DBSCAN, neighbors, hierarchy
  (HDBSCAN), SVM kernel-row, Householder QR, split Gram, coordinate-descent
  and isolation-forest drivers, and the Cholesky trailing rows, have two-MI300X
  receipts only at the sizes of their existing gates, and those gates do not
  establish that their buffers passed 1 MiB. GaussianMixture and resampling
  were asked above 1 MiB on two MI300X on 2026-09-14/15 (`../large-buffers/`)
  and passed.
- A passing lane does not prove a driver immune: the failure depends on
  timing (the wide par-samba and par-samba-clip lanes, the layer-owned byte-LM
  pool and every classical lane passed on the same boxes).
- No throughput is measured, including the cost of host staging on AMD.
