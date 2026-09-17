# CPU TSA negative controls

Source: `73a5af1734e24b872274c8830918c155e9534bd6`, clean tracked tree.
RunPod CPU leg: two vCPUs, two compiler workers, single-thread numerical libraries.
Production: core, TSA, forecast. Negative arm: TSA and forecast built with
`MOJOLEARN_HOST_SABOTAGE=1` and `MOJOLEARN_KPSS_DECISION_SABOTAGE=1`.
Both arms use the same production core binding. Native sabotage readback and
binding SHA-256 values are recorded in the columns. R2 reused four native
artifacts and cached the newly built core binding.

All four lanes, nine default fixtures, two repeats: 36 stable clean training
cells and 36 detected native negative controls. KPSS decision sabotage flips
the stationary flag so select-d observes the perturbation. This tests sensitivity
to the decision, not every internal operation. Holt-Winters uses the existing
arithmetic sabotage. These are source-harness CPU records, not installed final
wheel, GPU, batch, or multi-GPU release qualification.

The first attempt omitted the core transpose dependency: 18 Holt-Winters
controls passed and 18 KPSS/select-d cells refused. The gate failed correctly.
Its diagnostic summary and teardown receipts are retained under
`bench/results/reference-admission-probe/2026-09-17-missing-core/`.
Both pods were deleted and absence verified. Total measured spend: $0.0063.
