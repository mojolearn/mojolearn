# HDBSCAN approximate_predict, infer and batch cells (2026-09-15)

Branch `lane/hdbscan-approximate-predict`, commit 5938bac95 (every JSON records it).
`HDBSCAN(prediction_data=True)` and `mojolearn.hdbscan.approximate_predict`, mirroring cuML
v26.08.00 (`prediction_data.cu:92-239`, `predict.cuh:220-262`, `hdbscan.pyx:1264`), on the
Metal identical binding and the CPU host binding. The lanes `hdbscan` and `hdbscan-leaf` now fit
with `prediction_data=True`, hash `approximate_predict`'s labels and probabilities on 256 held-out
rows (infer) and ask the same call on 64 rows whole, alone and split (batch).

Where it ran: the Apple M4, one core (nice 19, one-thread knobs, `-j 1`), shared machine, one
process at a time. Fixtures base, ties and dupes (the lanes read `X[:6000, :4]`, which the dupes
rewrite does not reach, so base and dupes hash alike), two repeats (one for the sabotage runs).
The CPU column ran from a CPU-only copy of the package with `MOJOLEARN_HOST_DIR` holding the
hdbscan and core host bindings.

| file | verdict |
|---|---|
| `apple-m4.json` | cells=6 stable=6, infer stable=6, batch stable=6 |
| `cpu-apple-m4.json` | cells=6 stable=6, infer stable=6, batch stable=6 |
| `diff.metal-vs-cpu.txt` | `summary: IDENTICAL=6`, `summary (infer/model): IDENTICAL=6, N/A=6`, `summary (batch): IDENTICAL=6` |
| `diff.record-train.txt` | with the 166-lane record's apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942 columns: every base, ties and dupes train row of both lanes IDENTICAL x5 (`summary: IDENTICAL=18`), so prediction_data=True moved no committed train hash; the committed infer and batch cells read n/a:transductive |
| `apple-m4.batch-sabotage.json`, `diff.metal-vs-batch-sabotage.txt` | `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`: `summary (batch): BATCH_MOVED=6`, train and infer IDENTICAL=6 |
| `cpu-apple-m4.predict-sabotage.json`, `diff.metal-vs-cpu-predict-sabotage.txt` | `-D MOJOLEARN_HDBSCAN_PREDICT_SABOTAGE=1` (the lowest bit of every probability, prediction pass only): train IDENTICAL=6, infer DIVERGENT=6, batch DIVERGENT=6 |
| `cpu-apple-m4.sabotage.json`, `diff.metal-vs-cpu-sabotage.txt` | `-D MOJOLEARN_HOST_SABOTAGE=1` (fit and prediction): train, infer and batch DIVERGENT=6 |
| `apple-m4.par-hdbscan.json` | par-hdbscan on one device, base: stable, its batch hash d49a36bc7e6177b4 equals hdbscan/base's |

The held-out answers are not degenerate: on base the EOM fit (2 clusters) predicts 80 of 256
rows as noise with 29 probabilities strictly between 0 and 1; on ties the leaf fit predicts 61
distinct labels.

`test_hdbscan_surface` (fit, refusals, prediction on training rows, far noise points, a query at
an exemplar, float64 queries, rows alone, duplicated rows with reversed query order, refusal
without prediction_data, NaN, min_samples=1, membership_vector refusal) is GREEN, 38 checks, on
Metal and on the CPU host binding.

No box was rented (the release-only GPU rule). Owed to the release record: the NVIDIA and AMD
infer and batch cells for hdbscan and hdbscan-leaf, the two-device par-hdbscan cells, and a
comparison against cuML's own approximate_predict on the NVIDIA box. Not implemented:
membership_vector and all_points_membership_vectors (hdbscan/NOT_IMPLEMENTED.tsv).
