# HDBSCAN membership_vector and all_points_membership_vectors (2026-09-15)

Branch `lane/hdbscan-membership-vector`. `mojolearn.hdbscan.membership_vector` and
`all_points_membership_vectors`, cuML's soft clustering (`soft_clustering.cuh:385-627`), in float32
with pinned seams on every column (DEVIATION 1616, `hdbscan/impl/detail/soft_clustering.mojo`), on
the Metal identical binding, the CPU host binding and the NVIDIA identical binding. The lanes
`hdbscan` and `hdbscan-leaf` hash both calls in their infer cells (after approximate_predict) and ask
membership_vector alone and split in their batch part.

FIXTURE SCOPE. Every new column here runs the fixtures base, ties and dupes (duplicated rows and
distance ties), two repeats. The 166-lane record also carries denormal, denormal_ftz, hashed,
negative, odd and wide for these lanes; this lane did not run them, so in a `--require-columns 4`
diff those train rows read REQUIRE FAIL for a missing cell, not for a moved hash, and no claim is
made about them. The gate verdicts below are read on base, ties and dupes only.

## Apple M4, Metal (one core through the shared Mac slot)

| file | verdict |
|---|---|
| `apple-m4.json` | cells=6 stable=6, infer stable=6, batch stable=6 |
| `apple-m4.batch-sabotage.json` | `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`: every batch cell BATCH_MOVED (6 of 6), train and infer stable |
| `diff.record-train.txt` | against the 166-lane record's apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942 columns: every base, ties and dupes train row of both lanes IDENTICAL x4 (`summary: IDENTICAL=18`); the committed infer and batch cells read n/a:transductive |
| `diff.vs-approximate-predict-column.txt` | against `2026-09-15_hdbscan-predict/apple-m4.json` (approximate_predict only): train IDENTICAL=6, infer DIVERGENT=6, batch DIVERGENT=6, so the new parts are hashed |

`test_hdbscan_surface` GREEN, 60 checks, on Metal (the SOFT arm: shapes, finite and non-negative,
rows sum to at most 1, every exemplar's all-points row sums to 1, a clustered row's argmax is its
label, a far point's row is below 0.05 and below every clustered row, repeat and row-alone bytes,
batch_size changes no byte, duplicated rows finite, reversed query order reverses the bytes, the
refusals without prediction_data, batch_size 0, wrong features, NaN and min_samples=1, the (n, 0)
shape with no cluster, and float64 queries).

## CPU (RunPod CPU pod, AMD EPYC 9655P, x86-64-v3)

`tools/runpod_cpu_leg.sh --lane hdbscan-soft`, commit 82248f789 shipped (the merge of main into this
branch), pod qvc1lo41ub0ywv, $0.24/hr, 228 s billed, DELETE HTTP 204 then GET HTTP 404 and absent from
the listing (verified gone). Host bindings core and hdbscan, plus the hdbscan sabotage set
(`-D MOJOLEARN_HOST_SABOTAGE=1`).

| file | verdict |
|---|---|
| `surface-cpu-x86.log` | `test_hdbscan_surface` GREEN, 60 checks, on the CPU host binding |
| `cpu-x86.json` | cells=6 stable=6, infer stable=6, batch stable=6 |
| `diff.metal-vs-cpu.txt` | Metal against CPU: `summary: IDENTICAL=6`, `(infer/model): IDENTICAL=6, N/A=6`, `(batch): IDENTICAL=6` |
| `cpu-x86.sabotage.json`, `diff.metal-vs-cpu-sabotage.txt` | host sabotage: train DIVERGENT=6, infer DIVERGENT=6, batch DIVERGENT=6 |
| `diff.gate-owed.txt`, `owed_cells.json` | the three record columns, apple-m4 and cpu, `--require-columns 4 --owed-json`: train IDENTICAL=18; `summary (owed): OWED=12` (the infer and batch parts of both lanes on base, ties and dupes, hashed alike by Metal and CPU). The 12 REQUIRE FAIL lines are exactly the six fixtures this lane did not run (see FIXTURE SCOPE) |
| `owed-sabotage-check.txt` | `cpu_identity_gate_check.py owed`: 12 of 12 owed cell parts MOVED under the sabotage set |

## NVIDIA H100 and cuML's own membership_vector

`tools/gemm_remote_leg.sh nvidia --rent` with `MOJOLEARN_GEMM_LEG_EXTRA=tools/hdbscan_soft_nvidia_leg.sh`,
`NVIDIA H100 80GB HBM3`, driver 580.126.09, sm_90a, commit 55016a03c shipped (the soft clustering source
is unchanged since), pod v0pddryny9thce, lease armed before work, `VERIFIED: v0pddryny9thce is gone (HTTP 404)`,
and a later GET of both pods on the API returned 404. `nvidia-h100-leg/` holds `gate.txt`, `status.tsv` and
`cuml-reference.nvidia-h100-sm_90a.json` (cuml-cu12 26.8.0, cupy-cuda12x 14.2.0).

What did NOT run on the box: `test_hdbscan_surface` stopped at the PREDICT and SOFT arms (the leg built only
the hdbscan binding, and the float64 cast needs the base binding), and `identity_break` refused for want of a
commit witness (the body wrote no commit.txt). So the H100 infer and batch identity cells stay in
`owed_cells.json`.

Cross-vendor bytes on the reference fixtures (`tools/hdbscan_soft_cuml_reference.py`: blobs with uniform
noise, 2200 rows, and a half-unit grid with every row duplicated, 1200 rows; eom and leaf):
`soft-reference.apple-m4.json` (Metal) and `soft-reference.cpu-apple-m4.json` (CPU host binding) carry the
same sha256 as the H100 for labels_, membership_vector and all_points_membership_vectors on all 4 fits,
12 of 12.

Against cuML (`labels_equal` per fit): cuML's labels_ equal ours on dupes-grid/eom only (blobs/eom 1028
rows disagree, blobs/leaf 1064, dupes-grid/leaf 17 clusters against 11), so only that fit is compared.
There membership_vector cells differ by up to 0.303 (112 of 800 bitwise equal), all_points by up to 0.510
(284 of 4800 bitwise equal), row sums by up to 0.293, no argmax row moves, and neither side has a
non-finite cell.

The float32 seams alone (`vs_float64_transcription` in the Metal and CPU JSONs: cuML's formulas at their
precision in numpy, from OUR fitted prediction data): all_points row sums within 2.0e-7 on every fit and
single cells within 1.2e-2 (cells near 1e-6 that one side flushes or underflows); membership_vector within
5.1e-5 on blobs (that transcription recomputes the neighborhood in float64) and 1.2e-7 on the grid; no
argmax row moves on any fit. So the 0.29 row sum difference against cuML comes from the inputs (the fit's
tie resolution giving a different tree or prediction data under duplicates), not from DEVIATION 1616's
seams. Isolating it by feeding cuML's own tree through these bindings is not done.
