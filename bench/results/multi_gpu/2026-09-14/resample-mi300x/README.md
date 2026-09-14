# Multi-GPU resampling by global IDs — two MI300X

RunPod pod `bj93hauo7xk7y9`, two AMD Instinct MI300X (gfx942), 2026-09-14
22:34-22:37Z. Source: the `git archive` of commit `4b78e57fd` (`commit.txt`);
it differs from the H100 leg's `b9f219a9e` by that leg's evidence directory
and by `MOJOLEARN_RESAMPLE_CHECK_ONLY` in the native check (a family filter,
no production code). Box and Mac source SHA256 agree. Body `body.sh`.

## Result (`out/gate.txt`)

- `training/checks/resample_parallel_check.mojo`: 67 PASS lines, `PASS resample
  parallel gate` (36 bootstrap, 9 permutation, 21 Monte Carlo cases with equal
  one- and two-device traces and result bits).
- The sabotage build FAILS in each partition on its own:
  - bootstrap: `trace differs boot-n53-d2-s0-m0-a0-r2-f0: 2 resample.theta f32 2
    82fecf6e6cf037c4 VS 2 resample.theta f32 2 23e6ff649c2b542b` (the same two
    hashes the H100 sabotage run printed);
  - permutation: `trace differs perm-40-33-s4-a0-r2-f0: 1 resample.null f32 2
    80f203e112c5f400 VS 1 resample.null f32 2 471ddac3ce81f873`;
  - Monte Carlo, after 10 single-chunk cases pass as expected:
    `trace differs mc-f1-n257-i0: 2 resample.mc.partials f32 2
    9d9d798f9fe62795 VS 2 resample.mc.partials f32 2 0026544b9d62cafb`.
- `tools/parallel_resample_check.py`: `PASS 36 resample configurations and 5
  refusals`.
- `resample/checks/resample_check.mojo`: `resample_check mode=IDENTICAL ALL OK`.

## Cross-vendor

`out/public.json` equals `../resample-h100/out/public.json` as JSON: all 36
result fingerprints (distributions, sorted distributions, intervals, standard
errors, null distributions, p-values and counts, integrals and means) are the
same on two MI300X as on two H100s.

No speed claim.
