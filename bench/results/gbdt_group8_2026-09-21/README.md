# Rejected H100 GBDT group8 receipts

These compact receipts cover the 2026-09-21 NVIDIA H100 trial of the default-off
`MOJOLEARN_GBDT_GROUP8=1` candidate against the shipped exact group4 path. The
experiment used IDENTICAL mode and exactly the Cloudflare R2 Taxi and Istella-S
objects recorded in `timings_identity_quality_compact.json`.

Each dataset/path/arm has three alternating processes. A process excluded one
warmup, then recorded five samples of five consecutive public API calls. The
JSON retains every per-call sample, process median, maximum/minimum spread,
paired process ratio, output hash, quality result, binding hash, prepared-file
hash, R2 object key/size/hash, device, driver, and source commits.

All prediction bytes and quality results matched exactly. The result rejects
default promotion because all four cells failed the predeclared 1.10 spread
gate in at least one process and Taxi `predict_proba` regressed by about 7.1%.
Full logs remain outside the repository at
`/Users/andrewhendel/mojolearn-evidence/2026-09-21_gbdt_group8/`.
