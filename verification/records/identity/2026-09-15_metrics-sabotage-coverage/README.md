# Metrics host sabotage coverage (2026-09-15)

Lane `lane/metrics-sabotage-coverage`, commit 8047f9d7b. One RunPod CPU pod
(8 vCPU, `tools/runpod_cpu_leg.sh`, pod cxooyhnah2xsjm, deleted and verified
gone with GET 404, `teardown.txt`). Lanes metrics, metrics-classification and
metrics-fowlkes-mallows (the control), all 9 fixtures. The production column
ran 2 repeats and each sabotage column ran 1.

The builds were core and metrics in production, plus two metrics sabotage
builds with `-D MOJOLEARN_HOST_SABOTAGE=1`. `new` is this lane's
`metrics/host/metrics_oracle.mojo`. `old` is origin/main's oracle, restored on
the pod from an embedded archive (`cmd.log`, `build_sabotage_old.log`). Both
sabotage directories hold only the metrics family sabotaged. Every other
family (core, which runs KMeans) is the production binary. The binaries are
in `so_sha256_all.txt`: the production, new sabotage and old sabotage metrics
binaries each have a different sha256.

## Production

`diff_prod_four_columns.txt` diffs the CPU column against the 166-lane Apple
M4, NVIDIA H100 and AMD MI325X columns
(`bench/results/identity_break/2026-09-14_166-lanes`). It exits 0 with train
IDENTICAL=18 and OWED=9. The 9 owed train cells are metrics-fowlkes-mallows,
a lane the record does not hold. The 9 owed batch parts are the
metrics-classification `silhouette_samples` chunk windows. The production
train hash of metrics and metrics-classification equals all three record
columns on 18 of 18 cells (`sabotage_moves.txt`).

## Sabotage, cells moved against production (`sabotage_moves.txt`)

| lane | old | new |
|---|---|---|
| metrics | 5/9 (hashed, wide, denormal and denormal_ftz unchanged) | 9/9 |
| metrics-classification (train and batch) | 0/9 | 9/9 |
| metrics-fowlkes-mallows (control) | 9/9 | 9/9 |

## Owed check

`tools/cpu_identity_gate_check.py owed owed_cells.json`:

- `owed_check_old.txt`: exits 1, FAIL, 9 of 18 moved. All 9
  metrics-classification batch parts DID NOT MOVE.
- `owed_check_new.txt`: exits 0, OK, 18 of 18 moved.

## The new arms

All of them are under `comptime if METRICS_ORACLE_HOST_SABOTAGE`, and the
chunk boundary shift is kept.

- Accuracy reads sample 0's prediction flipped.
- ARI, entropy and mutual information read label 0 as another label of the
  same array. Homogeneity, completeness and v-measure reach this through
  entropy and mutual information.
- r2 reads `y_hat + 1 + |y|`.
- The silhouette reads row 0 shifted by `1 + |x|` in every distance, the
  same at every chunksize.
