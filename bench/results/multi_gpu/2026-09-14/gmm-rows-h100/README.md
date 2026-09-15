# GaussianMixture row-sharded E-step — two H100s

RunPod pod `5ygqny0p1yyhwj`, two NVIDIA H100 80GB HBM3 (sm_90a, driver
580.126.09), 2026-09-14 21:59-22:03Z. Source: the `git archive` of commit
`dc0a6ab3c` (`commit.txt`); the source SHA256 computed on the box equals the
one computed from the archive on the Mac (`source_sha256.txt`). Body
`body.sh`; GPU work ran serially. Design: `docs/multi_gpu/gaussian_mixture.md`.

## Result (`out/gate.txt`)

- `training/checks/gmm_parallel_check.mojo` (production build): 24 PASS lines,
  `PASS gmm parallel gate`. Seven E-step shapes (2x1x1 to 4097x17x2 and
  255x64x5) have every `mahal`, `wlp`, `rowmax`, `lse`, `logresp` and
  `meanll` bit equal between the original single-device `gmm_e_step` and the
  two-device dispatch. Sixteen full fits (six mixture fixtures and two blob
  sets, `init_params` kmeans and random) have equal identity traces, fitted
  state, `n_iter`/`converged`, `lower_bound`, `score_samples`,
  `predict_proba` and `predict` between one and two devices. The COLLAPSE
  fixture fits at both inits here. `out/trace_digests.txt` lists the SHA256 of
  each of the 32 trace files; each one-device trace file equals its
  two-device trace file byte for byte.
- The same gate built with `-D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1` (later
  owners read their rows one row early) FAILS at its first case:
  `E-step mahal n=2 d=1 k=1 bits differ at 1`.
- `tools/parallel_gmm_check.py`: `PASS 9 GaussianMixture configurations and 7
  refusals` (`out/public.json`): public `fit_gaussian_mixture` and
  `predict_gaussian_mixture` against one device, 37 to 4097 rows, 1 to 8
  features, 1 to 5 components, both inits, `max_iter` 0, 5, 50 and 100,
  `tol` 0; a collapse refuses identically and a failed distributed fit leaves
  the estimator unchanged.
- `mixture/checks/gmm_check.mojo`, the unchanged single-device mixture gate:
  `ALL PASSED`, including its 13 sabotage arms.

No speed or capacity claim is made.
