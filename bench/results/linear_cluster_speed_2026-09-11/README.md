# linear-cluster-speed lane, 2026-09-11 (DEVIATIONS 2632, 2633)

RunPod pod `22up9vbhj3tbeg` (`linclu-speed-2026-09-11_202814`), NVIDIA H100 80GB
HBM3, driver 580.126.09, Intel Xeon Platinum 8480+ (224 threads visible, CFS quota
23.8 CPUs), image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
Mojo 1.0.0 (ed45d567), cuML 26.08 (cuml-cu12 26.8.0) in the same Python 3.11.
Ours IDENTICAL built on the pod with `MOJOLEARN_GPU_ARCHS=sm_90a` from main
36ca51fd (baseline) and from that plus this lane's changes (after). Harness
`tools/classical_two_datasets.py race`, 1 warm-up plus 5 interleaved rounds with
the order rotated every round, ms median (min..max), quality by the harness's
float64 NumPy functions. Taxi block: 4,000,000 x 11 (TAXI_NUMERIC), k-means k 64,
20 iterations from the shared init, tol 1e-7 on both arms (cuVS and our binding
both refuse tol 0); PCA 8 components (ours `covariance_eigh`, cuML
`svd_solver='full'`); OLS `fit_intercept=True` (cuML `algorithm='eig'`).
Logs: `~/mojolearn-evidence/linear-cluster-speed-2026-09-11/pod1/` (base0, ab1,
ab2 races, JSON per race with every round and digest, build logs, probes).

## Same-pod baseline, ours IDENTICAL (main 36ca51fd) against cuML FAST

| family | dataset | rows x features | cuML ms | ours ms | ours / cuML | cuML quality | ours quality |
|---|---|---|---|---|---|---|---|
| LinearRegression | taxi | 4,000,000 x 11 | 23.12 (21.92..26.83) | 263.5 (251.8..275.9) | 11.40x | R2 0.908836 | R2 0.908837 |
| PCA | taxi | 4,000,000 x 11 | 20.99 (20.08..21.73) | 28.69 (25.78..38.83) | 1.37x | EVR sum 0.99786 | EVR sum 0.997861 |
| KMeans | taxi | 4,000,000 x 11 | 128.2 (127.4..129.0) | 304.0 (279.6..368.6) | 2.37x | inertia 1.20192e8, 20 iter | inertia 1.20628e8, 21 (max_iter+1) |

`baseline_taxi_summary.tsv` is the first race (its cuML k-means arm died on the
tol 0 refusal); the k-means row is `ab1_taxi_summary.tsv`, whose `ours-base` arm is
the 36ca51fd tree in the same race. cuML k-means returned a different centroid
digest in every round (not reproducible on one GPU); ours held one digest.

Istella-S (2,043,304 x 220) was NOT measured: its 472 MB tarball took 980 s to
fetch on this pod and the text decode was still running when the lane was wound
down. RUN OWED below.

## Where the taxi time went (stage probe and host timings, same pod)

* LinearRegression, 4,000,000 x 11, warm: the device solve (`ols_fit_host`) is
  about 10 ms (upload 3.5, Gram 2.1, A^T b 3.6, Jacobi 0.5 ms at 5 sweeps). The
  public fit was 244 to 296 ms: host `column_mean_f64` 62 to 78 ms, y mean 11 ms,
  `center_columns_f32` 138 to 141 ms of which the zero-filled `empty()` allocation
  of the 176 MB centered copy alone is 101 to 121 ms, `_shift` 5 to 13 ms. All
  single-threaded host loops.
* KMeans, warm: `plan_sum_scale` (host, one thread, column-major strides) plus the
  fit. After the change the probe reads plan 46 to 50 ms and the fit given the
  scale 130 to 133 ms (upload, 21 fused assignments, accumulations).
* PCA: upload 3.2 ms, covariance 6.3 ms, Jacobi 0.4 ms (4 sweeps); the rest of the
  28.7 ms fit is context, allocation and host readback. No change made.

## The changes, and their before/after on the same pod (ab2, interleaved)

DEVIATION 2632 (`bindings/_mojolearn.mojo`, `python/mojolearn/linear_model.py`):
`column_mean_f64` runs column groups on the host pool, each column's float64 chain
in its defined row order with task-local totals; `center_columns_f32` and
`scale_rows_f32` run row chunks; the centered copies come from `_output_store`
(every byte written by the helper) instead of zero-filled `empty()`.
DEVIATION 2633 (`cluster/estimator.mojo`): `plan_sum_scale` runs feature groups
row-major with the same per-column order and the same sequential worst-column pick.

| family | dataset | before (36ca51fd tree) ms | after ms | after / before | cuML ms | after / cuML | digest before = after |
|---|---|---|---|---|---|---|---|
| LinearRegression | taxi | 264.7 (262.9..268.1) | 177.8 (146.0..180.0) | 0.67 | 23.39 (23.09..27.08) | 7.60x | fb86358654367fa0 both |
| KMeans | taxi | 304.6 (287.9..308.7) | 216.2 (196.2..222.7) | 0.71 | 128.9 (127.0..129.2) | 1.68x | 89520efe99a08d5f both |

Warm single-process timings (`helpers_cpu_after2.log`, 6 or 4 calls, median):
LinearRegression.fit 290.3 -> 152.3 ms, KMeans.fit 276.4 -> 201.2 ms,
`column_mean_f64` 76.2 -> 20.9 ms, centering into a touched buffer 36.6 -> 8.7 ms.
The race medians sit above these because each race worker also pays its first
pool start in the warm-up only, and the OLS worker's rounds alternate with two
other 4M-row arms.

## Identity evidence (H100)

* Race digests (sha256 of the saved outputs, every round): OLS coefficients
  `fb86358654367fa0` for `ours` and `ours-base` in ab1 and ab2; k-means centers,
  labels and n_iter `89520efe99a08d5f` for both in ab1 and ab2.
* `helpers_ident_after2.log`: byte hashes of `column_mean_f64`, the float32 means,
  `center_columns_f32`, `scale_rows_f32` and the y mean, before tree against after
  tree, equal on taxi 4,000,000 x 11, a planted 1,048,579 x 5 block whose first
  column's sequential float64 total depends on order (3e38, -3e38, then ones), a
  20,000 x 220 block with column scales 1e-3..1e6, and a 5 x 7 block (below the
  threading threshold).
* No device kernel changed, so no bit on any vendor is expected to move; the host
  arithmetic and each column's order are unchanged by construction.

## Flip verdict

None computed. ENGINEERING_RULES section 9 needs after/before on taxi AND
Istella-S; Istella-S is RUN OWED. Taxi after/before: 0.67 (OLS), 0.71 (k-means),
bits unchanged. `tools/flip_verdict.py` reads FSPEED logs, which this harness does
not write; the geomean is to be taken by hand from the two race JSONs.

## RUN OWED

1. Istella-S baseline and after on an H100 (same pod, both trees):
   `sh tools/classical_two_datasets_leg.sh` with `MOJOLEARN_CTD_PHASES="setup prep"
   MOJOLEARN_CTD_LANES=kmeans,pca,ols MOJOLEARN_CTD_DATASETS=istella
   MOJOLEARN_GPU_ARCHS=sm_90a`, then with a copy of 36ca51fd's built `python/` at
   `MOJOLEARN_CTD_BASE_PY`:
   `python3 tools/classical_two_datasets.py race --lane <kmeans|pca|ols> --dataset istella
   --data /root/ctd-data --out OUT --work /root/ctd-work --root /root/mojolearn --rounds 5
   --arms ours,ours-base,cuml-gpu --ours-python python3 --theirs-python python3`
   (`ab.sh` here does the three). The stage probe
   (`stage_probe_main.mojo.txt`, build `pixi run mojo build -I .
   -D MOJOLEARN_NUMERIC_IDENTICAL=1`) splits Istella's Gram (the v1 GEMM profile
   past 128 columns) from the 220-column device Jacobi, which is the expected
   Istella bottleneck for PCA and OLS.
2. Apple M4 (local, one deliberate run): `pixi run check-kmeans`,
   `tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo` (17/17),
   `MOJOLEARN_NUMERIC_MODE=identical python3 -m pytest -q
   python/mojolearn/tests/test_native_helpers.py` after `sh bindings/build.sh`, and
   `helpers_ident.py` against a 36ca51fd tree (hashes must equal the H100's above).
3. AMD (Hot Aisle MI300X): the same three checks, and the taxi race with
   `--arms ours,ours-base,torch-gpu`.
4. H100: `test_native_helpers.py` (pytest was not installed on the pod), and
   `tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo` plus
   `pixi run check-kmeans-identity` on the after tree.
