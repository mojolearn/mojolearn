# Corrected NVIDIA backward baseline: PASS; strict long profile: RED

RTX 4090, source `eebd7c9206cef7a5fbc32f7f0ca9f71bdc68ec63`.
This backward-only corrective run passes all five baseline cases and matches
all 54 native gradient tensors from AMD. Mamba3 passes its exact scale-chain
regression and the additional independent whole-forward float64 comparison
for all ten public leaves. This replaces the historical Mamba3 staged-reference
correctness claim that missed the scale/gamma chain-rule contribution.

Mamba1 L64 is GREEN. Mamba3 L65's public gradients pass both references, and
all 21 supplementary public gradient tensors match AMD by bytes. The strict
long profile remains RED on 13 intermediate comparisons at unchanged
rtol=1e-5 and atol=1e-6. Full native and oracle arrays are retained under
`remote/followup/mamba-long-cert/mamba3-l65/failed-*`; neither wider tolerances
nor omitted failing tensors were used. See the
[comparison](../../resume/2026-09-05-next-certification/corrected-backward-cross-device.json).

The run also passes the controller's small UMAP identity witness; it does not
repeat the separately completed expanded UMAP/API campaign. Source/native
certification does not qualify installed wheels or expose a Python backward API.

The controller collected all results, returned failure for the red long
profile, and deleted pod yj4navzu95jkbg with verified HTTP 404. Its workers
were restricted to CPU cores 0-3 and ran serially. See the
[controller log](../../resume/2026-09-05-next-certification/nvidia-chain-final-controller.log).
