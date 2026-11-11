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
therefore claims zero removed Python roles. The initial generic-runner trace raised a possible resident-scope route.
The complete import/export trace subsequently established that metrics store
entries were already tombstoned and were never captured by the live metrics
connection. The correction is saved beside the original evidence in
`live-resident-path-correction.json`; no deleted export or cache mechanism is
restored. GP prediction resident/fold routes remain live and unchanged.

All compilation, native correctness, installed public tests, performance and
cross-device identity are pending and owned by the root orchestrator. No
lane build/test/timing/box action was performed.


## GPR public score composition

Mean prediction now returns a native storage/pin envelope shared by public
prediction and score. Public prediction keeps its existing exporter-backed
Array and standard-deviation/clamp publication. Score selects a native mean
buffer and calls that same prediction body directly, without a private Python
argument tuple or packed prediction Array.

After the same truth-shape and metrics-availability refusals, the target is
admitted/cast as before. Native score preparation performs the same forced
metric target and prediction snapshots, in that order, through the existing
parallel `admitted_convert` and finite scan. It then copies the same three
inputs into the same R² arena and executes the existing ranges/out descriptor.
The shared import-time metrics connection captures the already-exported native
program factory. It pins that factory's exact provider image through RUN.
No metric-store exports, cache owners or generic cache helpers were added.

Removed private roles include the GPR prediction-argument tuple, public-format
mean intermediate, metrics argument tuple, two metric snapshot Arrays, scalar
Array and completed arena Python store. Other shared metric builders keep their
existing paths; this is not a whole-metrics migration. GPC score composition is
still pending. Compile/runtime/performance evidence remains pending.
