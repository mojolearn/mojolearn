# Native Gaussian process score composition

This tranche follows the completed Spectral producers and does not modify a
frozen integration build. Baseline source is `9d6fb29702`. Complete before
sources and route/ownership findings are saved under
`~/mojolearn-evidence/python-mojo-migration-20261110/gp-score-native-owner/`.

GPR score currently predicts into a private public-format Array, builds a
private prediction-argument tuple and metrics tuple, then constructs the
existing `_Reg` R² program. GPC likewise builds prediction arguments, decodes
public-format class values and passes them into accuracy. Required class
identity and actual boundary pins are excluded from removal counts.

## Additive native R² interface

`MetricR2Sums` exposes the existing Float64 numerator/denominator result.
`linear_r2_execute` executes the same `LinearR2Layout` stage words and output
ranges while accepting native input-range records. `linear_r2_sums` retains
the former linear connection checks, target copy and range switch, and
`linear_r2_finish` retains its existing denominator-zero policy unchanged.
No numerical stage or kernel has been replaced.

`core/gp_native_score.mojo` adds native prediction/scalar storage, the same
R² program arena/input ranges and GP's distinct policy: a constant target
scores1 only when its residual is exactly0, otherwise0. Its input preparation
preserves adjacent host-range coalescing and admits existing resident source
handles without reading the host words for those inputs. The target mean is
still rounded to Float32 by the existing `reg_epi` stage; sums remain Float64.

This additive chunk has not switched either public score implementation and
therefore claims zero removed Python roles. Public composition, native score
storage lifetime pins and preserving active arena resident scopes remain
required work. In particular, an unconditional host-copy substitute for the
resident scope would change transfers and is not an acceptable shortcut.

All compilation, native correctness, installed public tests, performance and
cross-device identity are pending and owned by the root orchestrator. No
lane build/test/timing/box action was performed.
