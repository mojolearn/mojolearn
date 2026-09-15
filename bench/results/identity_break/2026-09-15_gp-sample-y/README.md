# GaussianProcessRegressor.sample_y: the gp-sample-y identity lanes (2026-09-15)

Lane `lane/gp-sample-y`. The method is at 41dc1da24; the merge of origin/main
is 607d79794. The fixtures are all nine. The lanes are the four recorded gp
lanes (`gp`, `gp-matern12`, `gp-matern32`, `gp-matern52-ard`), to show that
their cells do not move. They also include `gp-normalize-y` and the two new
lanes, `gp-sample-y` and `gp-sample-y-normalize`.

A new lane fits its source lane's model, `gp` or `gp-normalize-y`. Its train
part hashes `sample_y` over 64 rows with 3 draws and `random_state=11`, held
to a second call bit for bit. Its infer part hashes 4 draws over 64 held-out
rows with `random_state=2**40 + 5`, a key whose high word is set. The batch
part is `n/a:jointly-correlated`: every row of one call is drawn from one
joint posterior, so splitting the rows changes the covariance by the
reference's contract. The model part is `n/a:no-save`. DEVIATION 2793 in
`gaussian_process/checks/sample_y.mojo` names the stream (tag "GPSY"), the
jitter (the Cholesky profile's pinned `2^-20`) and the solve order.

The draws are the same bits on every column. They are not scikit-learn's
bits.

## Columns

- `apple-m4.json` is the Metal identical column. It was built on the Apple M4
  at 41dc1da24 through the shared Mac slot (one thread), two repeats. The
  JSON's vendor field reads `arm64`. Train reads 63 of 63 STABLE, infer 63
  STABLE, and batch 45 STABLE with 18 n/a.
- `cpu-x86-epyc.json` is the CPU column. One RunPod CPU pod (AMD EPYC 9754,
  Linux x86-64, `tools/runpod_cpu_leg.sh`) built the gp, preprocessing and
  core host bindings at the merge 607d79794, two repeats. Train reads 63 of
  63 STABLE with 0 moved and 0 refused, infer 63 STABLE, and batch 45 STABLE
  with 18 n/a. Its timings, command, binding hashes and verified delete are
  in `x86-runpod/`.
- `cpu-x86-epyc.host-sabotage.json` comes from the same pod: the
  `-D MOJOLEARN_HOST_SABOTAGE=1` build of the same three families.

## Diffs

- `diff_metal_cpu.txt`: IDENTICAL=63 train, 63 infer and 45 batch over all
  seven lanes. The Metal column is from 41dc1da24 and the CPU column from
  607d79794 (`--allow-separate-builds`). No arithmetic either path compiles
  moved between the two commits. Main's side of the merge changed
  `gaussian_process/estimator.mojo` only in the text of `gpr_classify_host`,
  which always raises. It also added the classifier's entries beside the
  regressor's.
- `diff_record_metal_gp.txt`: the four recorded gp lanes against the 166-lane
  record's Apple M4, H100 and MI325X columns read IDENTICAL=36 train, 36 infer
  and 36 batch, `--require-columns 4` OK. No recorded cell moved.
- `diff_record_cpu_owed.txt`: the same record plus the CPU column over all
  seven lanes, `--require-columns 4 --owed-json`. Train reads IDENTICAL=36
  OWED=27, infer IDENTICAL=36 OWED=27, and batch IDENTICAL=36 OWED=9.
  `owed.json` lists 63 cell parts. Of those, 27 are `gp-normalize-y`'s, which
  no committed GPU record hashes yet (owed by its own lane). The other 36 are
  the train and infer parts of the two new lanes.
- `owed_check.txt`: `cpu_identity_gate_check.py owed` reads "owed verdict OK
  (63 of 63 owed cell part(s) moved, 0 failure(s))". Every owed part moves
  under the host sabotage build.
- `diff_cpu_cpuhsab.txt`: the new lanes read DIVERGENT=18 train and 18 infer
  under host sabotage. `diff_metal_cpuhsab.txt` reads the same against
  Metal.

## Tests

- `test_gp_sample_y.metal.log` is from the M4 on the merged binding, run
  before the device fault below. It reads GREEN with 21 checks: shape,
  repeat bitwise, the high word of the key, the draw not depending on
  `n_samples`, the moments against `predict`, the joint covariance and mean
  against scikit-learn 1.9.0's `predict(return_cov=True)` for both
  `normalize_y` arms, and the refusals.
- `test_gp_surface.metal.log` passes on the merged binding.
- On the pod (`x86-runpod/`), test_gp_sample_y is GREEN with 18 checks. The
  reference arm is skipped there because scikit-learn is not in that env.
  test_cpu_training_gp reads 6 passed and test_gp_normalize_y 3 passed, 1
  skipped.

## The Metal device fault during this lane (`metal-transient/`)

After the merge, three full Metal runs from the merged binding went bad on
the shared M4. That was between about 13:46 and 13:51 ET, while other agents
were compiling Metal bindings.

- One run had `gp-matern52-ard` DIVERGENT on all nine fixtures, with one hash
  (`96ec7127b1bf9354`) shared by different fixtures.
- Another run read many DIVERGENT and REFUSED cells, with NaN and zero batch
  outputs.
- A two-lane rerun from the same binding in between was clean. It read
  IDENTICAL=18 against CPU.

The probe (`probe.json`, `probe.log`) shows that the device was at fault,
not the code:

- `standard-scaler` REFUSED on all nine fixtures with "StandardScaler:
  invalid variance or scale". It goes through the preprocessing binding,
  which this lane never rebuilt and which ran clean earlier the same day.
- `gp` and `gp-matern52-ard` read DIVERGENT on 11 train cells and MOVED on 2
  against the clean pre-merge Metal column, although the merge did not
  change their arithmetic.
- The record has no `standard-scaler` hash, so that lane's diff against the
  record reads ONE-COLUMN.

The macOS log showed no GPU reset, hang or fault entry in that window. These
runs are kept as a record and are not evidence for or against any cell. A
Metal rerun of the seven lanes at the merge, on a healthy device, is owed.

## Owed

- The NVIDIA and AMD cells of `gp-sample-y` and `gp-sample-y-normalize`, to
  the next release record. No GPU pod was rented.
- A clean Metal column of the seven lanes at the merged commit, owed because
  of the fault above.
- Public CPU `sample_y` from a saved model belongs to
  lane/inference-neighbors-density. This lane adds only the GPU method and
  the internal verifier arm in the gp host binding.

The pods were `x62duunyatwvfx`, before the merge (its normalize lanes refused
because core was not built; `x86-runpod-premerge/`), and `x9id0vtzbc3q9o`, at
the merge. Both were deleted and verified gone. Together they billed 316 s,
$0.021.
