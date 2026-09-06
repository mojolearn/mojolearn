# Ordered RMSE, Mamba backward and kNN flags — September 6

Mamba certificate source: `395d9421adc40c37ed119895a353f67d37e8a20d`
(implementation `8e02b5b0`). Corrected kNN driver / continued-check source:
`6dd44ac53f178fa07fa91b13dadf749163df24f9` (implementation `6855d8cc`).

AMD MI325X: [complete raw evidence](../../e1/2026-09-06_071735-mojolearn-e2-amd/).
NVIDIA RTX 4090: [complete raw evidence](../../e1g/2026-09-06_033208-nvidia-mamba/remote/).
Both Mamba certificates pass and compare successfully: [54 baseline tensors](cross-device-mamba-baseline.log)
and [21 long-profile tensors](cross-device-mamba-long.log) match by bits.
The [Mamba3 diagnostic comparison](cross-device-mamba3-diagnostics.json)
also matches all 86 gradient tensors and nine forward operands.
The corrected continued-check source passes both GPUs. Final AMD continued
raw evidence: [matching-source run](../../e1/2026-09-06_080119-mojolearn-e2-amd/continued/).
The [cross-device comparison](cross-device-continued.json) matches 130 ordered
records and all 5,440 kNN selected distance/index pairs in all four arms.

The continued gate retains 130 ordered records and 5,440 kNN selected
index/distance records per arm. Baseline, selector, transpose and combined
arms match on both GPUs at `6dd44ac5`. Weighted CTR also passes on both. These are correctness runs;
there is no new performance measurement or external CatBoost parity claim.

Mamba3 L65 uses the explicitly documented
[compositional arithmetic contract](../../../../mamba/BACKWARD_CERTIFICATION.md).
All public gradients retain independent whole-forward float64 checks.
The thirteen sensitive intermediate outputs use independently checked
operands plus exact pinned float32 arithmetic, including per-token angle
modulo. The old direct-reference differences remain in the logs. All 76
diagnostics and nine forward operands are retained, not omitted.

The [native ordered RMSE API](../../../../gbdt/ORDERED_RMSE.md) supports a
numeric single-permutation path. Prefix isolation, persistent fold cursors,
independent replay, weighted multi-tree predictions, invalid permutations,
zero-mass prefixes and constant-tree prediction are covered. This does not
implement installed Python ordered boosting or full categorical parity.

## Recovery and provenance

The first AMD rental (598180320) was deleted with verified HTTP 404 after
a controller-resume failure. Editing the live Bash controller changed its
remaining input; the resumed controller encountered an unset variable before
fetching. Its later successful run logs/hashes are retained, but unfetched
raw tensors are not claimed as a certificate. Only the earlier explicitly
copied `amd-operand-capture` diagnostic survived that rental.

The replacement AMD rental (598187019) reran the final frozen source.
Its complete evidence was explicitly downloaded and locally validated before
the release marker was created. Deletion was verified by HTTP 404.
`tools/e2_remote_leg.sh` now freezes itself before provisioning; the NVIDIA
controller also uses its existing immutable snapshot mechanism.

Intermediate source JSON files and failed build logs describe the actual
implementation progression. Earlier attempts exposed Mojo model ownership
and GPU scalar argument typing issues; the final source fixes both.
The early operand capture uses a superseded seven-operand diagnostic policy
and is retained only for attribution, not final certification.

Only the main operator executed native tests and comparisons. The CatBoost
implementation lane performed implementation/static review only. GPU jobs
were serial, with four-core remote affinity, bounded compiler concurrency,
and one-thread BLAS/OpenMP. No heavy local model runs were performed.

Local small policy tests: 16 oracle, 9 arithmetic, 19 identity and 4
continued-comparator checks passed. Shell syntax and the no-cloud immutable
controller refusal/cleanup check passed.

## NVIDIA driver correction

The first kNN stress driver stalled on NVIDIA after the first case, in a
CPU futex wait with the GPU idle. The established driver uses an explicit
`with DeviceContext()` scope and synchronizes host-buffer allocation before
writing through pointers. Applying both conventions fixed the stress driver;
no kNN arithmetic or selector kernel changed. The original interrupted
`remote/continued` evidence is incomplete and is not a passing certificate.

All twelve build/check statuses pass under `remote/continued-lifetime` at
`6dd44ac5`: ordered RMSE, weighted CTR and all four kNN arms. The earlier
Mamba certificates remain at `395d9421`; only the continued native checks
were rerun. The driver correction does not alter the Mamba source contract.

The NVIDIA controller was paused during this bounded serial repair; the
pod's deletion guard remained armed. Complete Mamba and corrected continued
raw records were explicitly downloaded and validated before the controller
resumed its normal fetch/deletion. Pod `3raw0dqpzytezi` deletion was verified
by HTTP 404. The temporary local rsync attempt failed because the image lacked
rsync; tar transfer succeeded and the normal controller fetch also completed.

The final AMD rental (598192757) ran only setup and the continued checks at
`6dd44ac5`. All records were explicitly fetched and compared before release.
The NVIDIA controller now also requires the optional continued run's successful
exit marker, pinned source and complete record validation. Its earlier success
exit applied to Mamba and did not reject the interrupted continued run; that
reporting gap is fixed, with four no-cloud regression controls passing.
The separate corrected continued evidence, not that earlier controller exit,
is the basis for the final kNN/ordered claim.

All three DigitalOcean rental IDs and the NVIDIA pod return HTTP 404.
The final read-only [cloud inventory](final-cloud-inventory.json) contains no
droplets or pods. All paid resources for this continuation are closed.
