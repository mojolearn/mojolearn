# GaussianMixture.sample: the gmm-sample identity lanes (2026-09-15)

At 0c65514b9 on lane/inference-new-methods. Fixtures were base, ties, odd and
denormal. The lanes were gmm and gmm-random-init (to show their recorded cells
do not move), and the two new lanes gmm-sample and gmm-random-init-sample. A new
lane fits the gmm lane's model, hashes `sample(256)` as its train part (held to a
second call bit for bit) and `sample(1024)` as its infer part. Its batch part is
`n/a:no-batch-axis`, because `sample` reads no input rows. The draws are
position-mapped Philox keyed by `random_state` (DEVIATION 2791), and the normals
go through the fitted `precisions_cholesky_` (DEVIATION 2792); both are named in
`mixture/checks/sample.mojo`.

Two columns:

- `apple-m4.json`, the Metal identical column. It was built and run on the
  Apple M4 through the shared Mac slot (one thread), two repeats. Train is
  STABLE on 16 of 16 cells, infer is STABLE on 16, and batch is STABLE on 8 and
  n/a on 8.
- `cpu-x86-epyc.json`, the CPU column. It came from one RunPod CPU pod (AMD EPYC
  9654, Linux x86-64, `tools/runpod_cpu_leg.sh`), with the mixture host binding
  built on the pod, two repeats. It is STABLE on the same cells. The pod's
  timings, binding hashes and verified delete are in `x86-runpod/`.

Diffs and controls:

- `diff_metal_cpu.txt`: IDENTICAL=16 on train, 16 on infer and 8 on batch.
  Apple Metal and x86 CPU agree on every sample, count and label.
- `diff_record_metal_gmm.txt`: the gmm and gmm-random-init cells against the
  166-lane record's Apple, H100 and MI325X columns read IDENTICAL on train,
  infer and batch, so no recorded cell moved.
- `diff_record_owed.txt`: that record cut to these fixtures, with both columns,
  under `--require-columns 4 --owed-json`. Train reads IDENTICAL=8 and OWED=8,
  and infer reads IDENTICAL=8 and OWED=8. `owed.json` lists the 16 new sample
  cell parts the next release record owes.
- `apple-m4.batch-sabotage.json` and `cpu-x86-epyc.batch-sabotage.json`, run
  with `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`: BATCH_MOVED=8 on each, which is
  every batch cell the gmm lanes carry.
- `cpu-x86-epyc.host-sabotage.json`, the `-D MOJOLEARN_HOST_SABOTAGE=1` mixture
  host build (its readback reports `sabotage: True`): against Metal it reads
  DIVERGENT=16 on train, DIVERGENT=15 on infer and DIVERGENT=7 on batch
  (`diff_metal_cpuhsab.txt`). The arm moves the fit, and `sample_X` moves on
  every sample cell that inherits it.
- `test_metal.log` and `test_cpu_x86.log`: test_gmm_sample is GREEN with 19
  checks on both. It covers shape and grouping, repeatability, per-component
  counts, means and covariances against the model, the whitening
  `(X - mean) P ~ N(0, I)`, and the refusals.

Public CPU exposure of `sample` from a saved model is left to
lane/inference-neighbors-density. Still owed to the next release record: the
NVIDIA and AMD cells of both sample lanes. No GPU box was rented.
