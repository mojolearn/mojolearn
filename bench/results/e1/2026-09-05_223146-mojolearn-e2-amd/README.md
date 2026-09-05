# AMD expanded UMAP: PASS; supplementary backward launch: INFRA_FAILURE

DigitalOcean AMD MI325X, source `6a3a2d30ab53e9207d327df38f4d22fb5111535e`.
All five baseline Mamba backward cases passed, retaining 54 native gradient
tensors. The certificate validates locally against its retained manifests and bytes.

All six expanded held-out UMAP cases passed in each of IDENTICAL, FAST and
DETERMINISTIC with unchanged thresholds and controls. The self-neighbor
regression, native broader/transform/sparse gates, public API suites in all
modes, both kNN dispatch checks and Mamba forward/state API passed.

The supplementary backward launcher failed before its numerical gates because
host `python` was absent from PATH. Classification: `INFRA_FAILURE`; no long
certificate exists. The follow-up now invokes it through the pinned pixi
environment. This result is not a complete passing campaign.

CatBoost's fold-axis wiring gate passed, including 15,360 exact histogram
cells and ordered partitions. Its weighted CTR slice failed because the
fixture did not produce the intended occupied zero-weight leaf. The preceding
weighted prediction comparisons had not failed; the missing coverage remains
a failing gate, not evidence for the zero-mass path. Raw logs are in `lanes/`.
End-to-end ordered boosting is still unimplemented.

The controller exited zero despite failed diagnostic/extra rows: interpret
`diag/status.txt`, `diag/followup/results.tsv` and the individual logs, not
that process exit alone. Droplet 598106166 was deleted and absence verified
by HTTP 404. See the [controller log](../../resume/2026-09-05-next-certification/amd-fixed-controller.log).
