# Four CPU lanes promoted to public verification

2026-09-17. This work closes public CPU selection gaps for `spectral`,
`metrics-fowlkes-mallows`, `arima-exog`, and `arima-exog-seasonal`. Their CPU
implementations and harness lanes already existed; they were withheld as
`unwatched`. Removing that hold makes the default CPU verification command
select them. No production numerical implementation or reference hash changes.

The public wheel interface is `python -m mojolearn verify --all`; `--coverage`
shows lane availability and historical evidence. This is a local command in
the PyPI package, not a hosted verification service. These changes have not
been published to PyPI yet.

## Validation and evidence

- Installed CPU development wheel, outside the checkout with no host/harness
  overrides: all four lanes, all nine fixtures, two repetitions; 117 IDENTICAL,
  63 explicit N/A, zero OWED, REFUSED or DIVERGENT parts.
- Spectral and both ARIMA lanes check training, held-out inference, saved-model
  reload and batch invariance. Fowlkes-Mallows checks its numerical result;
  model, inference and batch properties do not apply to this standalone metric.
- Fresh-source Linux production and native sabotage builds on a two-vCPU RunPod
  instance, staged through R2; clean/sabotage columns detected all 36
  training negative controls; public replay matched 117 applicable parts and
  the mandatory OLS comparator self-test passed (exit 0).
- The installed updated wheel exposes all four lanes in default CPU
  selection and all nine native controls per lane in `--coverage`.

The updated installed wheel passed default-selection and coverage assertions;
CPU availability rises from 122 to 126 lanes, with 53 still withheld and 50
parallel exclusions. Historical native-control coverage rises from 48 to 52 of
246 appendix entries. This is not full release qualification.

Native pairs are retained under
`bench/results/identity_break/2026-09-17_cpu-public-promotion/`; public reports
and the wheel receipt are under `bench/results/cpu-public-promotion-probe/`.
Large JSON reports are losslessly gzip-compressed. The first Linux public gate
matched all lane results but failed its OLS self-test because the estimator
binding was omitted. Its failed report and teardown are retained; the runner
now checks that dependency before executing.

The development Mac wheel reuses the prior native CPU bindings. The Linux
control run builds the relevant bindings from source (or their content-addressed
R2 cache). Neither is final 0.8.7 wheel qualification. Comprehensive cross-GPU,
multi-GPU and release-artifact verification remains open.

261 focused tests and the generated-matrix consistency check passed. The
corrected Linux gate ran at `1bb58bc77f17` with no tracked source changes and
is retained under `2026-09-17-linux-comparator/`. R2 reused five bindings and
cached the newly built estimator dependency. Its 557-second staging delay
was source transport; builds took 22 seconds and execution 37 seconds.

Both RunPod instances were verified deleted. Total measured spend $0.0179.
Two cloud vCPUs and at most one local numerical worker kept this work within
the user's three-core limit. Fixtures are generated deterministically, so
these checks do not need large external datasets. No PyPI publication occurred.
