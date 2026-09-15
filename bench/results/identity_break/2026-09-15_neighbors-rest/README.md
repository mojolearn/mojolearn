# lane/neighbors-rest, the seven brute force k-NN metrics: Apple M4 column

What ran, on the M4 with one core and the Metal slot, at commit 08ab0b3d0
(`bindings/build.sh` and `bindings/build_core_host.sh` built from that tree).

- `apple-m4.json` -- `tools/identity_break.py --lanes knn-canberra,
  knn-braycurtis,knn-correlation,knn-jensenshannon,knn-hamming,
  knn-russellrao,knn-inner-product --repeats 2 --vendor apple-m4`:
  63 cells (7 lanes x 9 fixtures), verdict STABLE on all 63, train, infer,
  model and batch parts all stable, 0 refused. The same run was taken twice
  (once labeled with the default box label) and the two agree on every cell
  of every part, byte for byte.
- `check_metric.log` -- `pixi run check-metric-identical`:
  `check_metric_device_equals_oracle` reads 15 DistanceType values x 1961
  cells, 29,415 bit-equal, 0 differ, against the float32 host oracle;
  `check_metric_matches_float64_reference` prices each metric against a
  float64 reference computed a third way (worst relative error: canberra
  2.1e-07, correlation 3.4e-06, braycurtis 3.4e-07, jensenshannon 2.2e-06,
  inner product 1.7e-07, hamming 0, russellrao 0); `check_metric_refusals`
  resolves 19 metric names and takes 9 refusals by name or value.
- `tests.log` -- `python3 -m mojolearn.tests.test_knn_extended_metrics`,
  14 tests, OK: the name table, the refusals, each metric against its
  float64 definition, the neighbour sets, inner product's largest-first
  order, hamming's multiples of 1/n_features, a self query at distance
  zero, and a saved model reloading to the same bits.
- `diff_spot.txt` -- the scope rule's spot check: the existing knn, knn-cosine,
  knn-minkowski-p3, knn-clf, knn-reg, radius and kde lanes on the `base`
  fixture against the committed 166-lane Apple column: 0 DIVERGENT, 0 MOVED
  (IDENTICAL on every compared cell; the other fixtures were not run here and
  read ONE-COLUMN).
- The saved-model recording for the CPU gate is
  `bench/results/classical_host/2026-09-15-apple-m4-neighbors-rest`: 63 of 63
  fixtures RECORDED with `reload_equal` true and vendor `metal`.

## OWED, and why

THE CPU COLUMN AND THE SABOTAGE COLUMN ARE NOT HERE. No RunPod CPU pod was
rented for this lane: the Mac was about to restart to clear a Metal
command-queue leak (AGXCommandQueue 6754 against a limit of 512), and the
lane stopped rather than start work it could not finish. So this directory
proves the Apple column and the gates on it, and it does NOT yet prove:

1. Apple against x86 CPU IDENTICAL on all 63 cells (`--require-columns` and
   `--owed-json`),
2. that the sabotage host build moves every new cell,
3. that the recording above checks green on a box with no GPU, and fails
   under the sabotage host set on every fixture.

`docs/lanes/LANE_STATUS_lane-neighbors-rest.md` carries the exact commands.
The NVIDIA and AMD columns are owed to the next release record, as for every
lane landed since the 166-lane record.

A NOTE ON THE DEGRADED GPU. These numbers were taken while the M4's Metal
queue was leaking, so every fit ran roughly 20x slow. Slowness does not move
bits, and three independent checks here would have caught a corrupted stream
rather than a slow one: the two full runs agree byte for byte, the device
matrix agrees with a host oracle on 29,415 of 29,415 cells, and every
recorded model reloads to the same predictions. Read them as trustworthy on
that basis, not on the basis that the box was healthy.
