# Native A1 result

Final FAST, DETERMINISTIC and IDENTICAL builds and executions passed on the
local Apple GPU. See native-driver.log, each mode's final .build.log/.log,
and native-binary-provenance.json for exact compile commands and binary hashes.
Executables are under build/regression-errors-a1/, not versioned evidence.
The retained fast.initial-pointer-failure.log is a failed preliminary build,
not a tested artifact; final mode logs supersede it.

Coverage: three distinct hand-computed metrics; independent Float64 reference;
separately spelled fixed-tree Float32 oracle in IDENTICAL; repeated runs;
block256/1D versus block64/2D launches;4099-row ragged tail and37 NaN-poisoned
padding cells; equal extreme inputs; residual, square and partial-sum
overflow; positive zero from signed-zero inputs; IDENTICAL operand/squared
residual FTZ seams; empty and nonfinite input rejection.

The production implementation performs residuals, chunk reductions, ascending
partial fold, division and optional square root on GPU. Host code validates
and transfers inputs and downloads the final scalar. The final fold uses one
GPU thread and is O(ceil(n/256)); performance is unmeasured. Float32 overflow
returns positive infinity, even if a higher-precision mathematical mean or
RMSE would be finite. Cross-vendor qualification and public binding validation
are separate work; these native runs alone establish neither.
