# CPU kernel variant source comparison (2026-09-18)

Source 3f778b7fe03d843c16a305fac7b3c3a3e9f43d79. Fresh kernel-family bindings,
with frozen-release support bindings as recorded in comparison-receipt.json.
Six lanes, nine fixtures, two repeats: 216 applicable numerical properties
matched CPU/Apple bit for bit. All 54 sabotage training cells changed.
Non-applicable properties are excluded from the numerical count.

This is source evidence, not installed-wheel or NVIDIA/AMD qualification.
The six lanes remain pending. Scripts retain the exact local paths used;
they are historical capture recipes, not portable public tools.

The initial runtime-tests.log has 15 passes and 16 failures: saved-model
host_model() routes through the estimators binding, which was an older
binary. That binding also needed degree transport updated for the new
oracle signatures. A fresh estimators build and runtime rerun are required.
These failures do not invalidate the separate direct-family comparison,
but that comparison does not qualify the public saved-model path.
