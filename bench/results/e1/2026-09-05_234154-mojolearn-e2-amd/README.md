# Corrected Mamba3 backward: AMD baseline PASS; long intermediate gate RED

AMD MI325X, frozen source `eebd7c9206cef7a5fbc32f7f0ca9f71bdc68ec63`.
All five baseline cases pass (54 native gradient tensors). Mamba3's exact
first-token/shifted-beta scale-chain regression passes, and all ten public
leaves must additionally agree with the independently differentiated whole
float64 forward. The short Mamba3 gate passes all 86 named tensors.

Mamba1 L64 passes. Mamba3 L65's ten public gradients now pass both the staged
float32 and independent whole-forward float64 comparisons at the original
rtol=1e-5, atol=1e-6. The in_proj.weight maximum error against the independent
forward fell from 0.9516 before the chain-rule fix to 4.893e-7; x fell from
0.005726 to 4.003e-8. See the
[retained diagnostic analysis](../../resume/2026-09-05-next-certification/mamba3-corrected-long-diagnostics.json).

The complete long certificate remains RED: 13 intermediate float32 comparisons
still fail, with maximum discrepancies up to 4.578e-5. Their native and oracle
arrays are preserved in `diag/followup/mamba-long-cert/mamba3-l65/failed-*`.
These failures were neither removed nor hidden by wider tolerances. The
remaining work is to establish the arithmetic/operand contract for those
S14/S15/S16 reductions and joins, with independent semantic checks.

After the Mamba jobs stopped, the guarded operator window ran separately
pinned CatBoost checks, retained in their own timestamped directories.
Droplet 598115608 was then deleted with HTTP 404 verified. See the
[controller log](../../resume/2026-09-05-next-certification/amd-chain-repair-controller.log).
