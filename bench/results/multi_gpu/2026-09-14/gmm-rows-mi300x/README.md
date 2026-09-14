# GaussianMixture row-sharded E-step — two MI300X

RunPod pod `gf9c66xxz3b45o`, two AMD Instinct MI300X (gfx942), image
`rocm/dev-ubuntu-22.04:6.4.1-complete`, 2026-09-14 22:26-22:30Z. Source:
the `git archive` of commit `959b8f9d5` (`commit.txt`), which differs from
the H100 leg's `dc0a6ab3c` only by that leg's evidence directory under
`bench/results/` (not shipped to the box); box and Mac source SHA256 agree.
Body `body.sh` (the same body as the H100 leg). Design:
`docs/multi_gpu/gaussian_mixture.md`.

## Result (`out/gate.txt`)

- `training/checks/gmm_parallel_check.mojo`: 24 PASS lines, `PASS gmm parallel
  gate`: every E-step output bit of seven shapes and sixteen full fits with
  equal one- and two-device identity traces, state and scoring outputs.
- The `-D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1` build FAILS at its first case:
  `E-step mahal n=2 d=1 k=1 bits differ at 1`.
- `tools/parallel_gmm_check.py`: `PASS 9 GaussianMixture configurations and 7
  refusals`.
- `mixture/checks/gmm_check.mojo`: `ALL PASSED`.

## Cross-vendor

`out/public.json` equals `../gmm-rows-h100/out/public.json` as JSON (every
model and output SHA256, `n_iter`, `converged`, and the identical collapse
refusal), and `out/trace_digests.txt` equals the H100 file line for line:
all 32 identity trace files (16 fits, one and two devices) have the same
SHA256 on two MI300X as on two H100s. No Apple column was run for this driver
(it needs two devices).

No speed or capacity claim is made.
