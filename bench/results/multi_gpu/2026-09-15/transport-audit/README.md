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

Two drivers read wrong above 1 MiB on two MI300X and are fixed on this
branch: the byte-LM replica pools (below) and the Cholesky trailing-update
rows (further below).

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

## The drivers the lanes reach only at small shapes (`drivers-mi300x/`)

`tools/transport_audit_check.py` runs each driver on devices (0,) and (0, 1)
in one process and compares every output byte for byte, on two MI300X
(pod `wrn6t5ocqrlzo5`, source `3250fcb5b`): DBSCAN on 40000x8 (1.22 MiB),
SVM fit on 4096x16 and decision/predict on 1904 rows, ParallelQueries and
ReferenceShardedNeighbors over a 64000x16 reference (3.91 MiB), HDBSCAN on
12000x4, coordinate descent on 80000x16 (4.88 MiB), a Gaussian process on
1024 rows (a 4 MiB covariance), and the isolation forest on 160000x16
(9.77 MiB). All eight read EQUAL (`audit.txt`). The SVM and HDBSCAN inputs are
below 1 MiB; their kernel and distance rows are not. With
`MOJOLEARN_TRANSPORT_AUDIT_SABOTAGE=1` the cd and gp cases read DIFFERS and
the tool exits 1 (`audit-sabotage.txt`), so the comparison can fail.

## Full PCA, the wide Gram outputs and the Cholesky trailing rows

`drivers2-mi300x-before-fix` (pod `rm631tqz556fqi`, source `f88679bf7`): full
PCA on 40000x33 (the TSQR panels) and PCA over 8000x257 (the wide Gram
outputs) read EQUAL, and **fit_cholesky and solve_cholesky at n=4500 (a
77 MiB matrix) DIFFERED**: 235566 of 81000000 factor bytes and 16222 of
54000 solution bytes. `cholesky-h100-before-fix` (pod `7hwscm6ah030eu`, the
same source) read the Cholesky case EQUAL, so the MI300X two-device column
stands alone. The column solve was already host staged, so the moved bytes
come from the factor, whose trailing-update rows still moved by device copies
(with the pinned panel width 32 the packed operand stays under 1 MiB, but
each owner's output rows at the first panel are about 40 MiB).

`cholesky/multi_gpu.mojo::chol_trailing_rows` now moves each owner's left
rows, the packed right operand and each owner's output rows through
`transfer_bytes`. At `ce4634f4d`, on two MI300X (`cholesky-fix-mi300x`, pod
in `pod_id.txt`) and on two H100s (`cholesky-fix-h100`):

- the n=4500 Cholesky case reads EQUAL twice on each vendor (`audit.txt`);
- `training/checks/cholesky_parallel_check.mojo` passes (`PASS cholesky
  parallel gate`), and its 152 trace digests on each vendor equal, line for
  line, the two-H100 digests of `../kernel-methods-cholesky-final/h100/`;
- the par-cholesky and par-kernel-ridge lanes on two devices (`two.json`, all
  nine fixtures, STABLE) equal, cell for cell, the same vendor's two-device
  cells in `bench/results/identity_break/2026-09-15_par-lanes-new/`.

## What this does not cover

- GaussianMixture and resampling were asked above 1 MiB on two MI300X on
  2026-09-14/15 (`../large-buffers/`) and passed; the Householder QR used by
  full PCA and the wide Gram outputs passed here at one shape each.
- The drivers audit prints the input size, not the size of each device
  buffer a driver moves.
- A passing lane does not prove a driver immune: the failure depends on
  timing (the wide par-samba and par-samba-clip lanes, the layer-owned byte-LM
  pool and every classical lane passed on the same boxes).
- No throughput is measured, including the cost of host staging on AMD.
