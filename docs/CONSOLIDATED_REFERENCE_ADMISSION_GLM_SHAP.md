# GLM and TreeSHAP reference repair admission

Six repaired lanes at source `9d64cb98b284afb3f0af94ffaf39aa127e19bbb6` agree on all nine fixtures across Metal and HIP: 162 numerical cross-vendor parts, zero mismatches or missing properties. Each GPU column independently agreed with its local CPU column. These are four execution columns (M4 Metal/Arm CPU and AMD HIP/x86 CPU), classified as three admission device classes (Apple, AMD, CPU); no current CUDA evidence is claimed.

Strict admission yielded 54 fixture cells and 432 referenced parts, each supported by all three classes, with zero conflicts. Original bytes of all 24 current records, source hashes, backend witnesses, and admission details are preserved in `bench/results/identity_break/2026-09-28_admitted-glm-shap/`. Its `candidate-lanes.json` is the scoped builder output; `admission.json` records SHA-256 hashes and the table transition.

The base already includes the earlier 124-lane admission. All 919 base record entries remain an unchanged prefix, every unselected table cell remains unchanged, and only these six resolved reference holds were removed. Other repair targets and physical-device scopes remain unresolved. Validation used saved records and source metadata; it ran no native fits.

Base table SHA-256: `ba6fe7fa5bea203e09b9ad20fdb93abe51d82744dff09b2651a17ac377392053`.

Promoted table SHA-256: `c3cec1e0a2926b9a331ff18e23ab6c164b899b80ef57dbfa306b686b5e30f0d4`.

Reference-admission and host-surface metadata checks: 285 passed. Both fixture-floor tests passed separately after the fragment-discovery fix.

## Admitted lanes

- `trees-dart-options`
- `trees-shap-tree`
- `x-glm-gamma`
- `x-glm-poisson`
- `x-glm-poisson-sw`
- `x-glm-tweedie`
