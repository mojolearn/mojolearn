# Child Hessian feature verification, 2026-09-10

PASS on Apple M4 with Mojo 1.0.0 (ed45d567): native and rebuilt public bindings
in FAST, DETERMINISTIC, IDENTICAL. No NVIDIA or remote GPU execution occurred.
This feature makes no new performance or cross-library numerical-parity claim.

Native checks per mode:

- GPU comparison tests for zero, signed zero, smallest Float32 subnormals;
  a positive Float64 below Float32 range rounds upward to bit1.
- Candidate oracle: the highest-scoring candidate is rejected and the legal
  runner-up wins; both left and right constraints, equality, and a Float64
  threshold between adjacent Float32 values; all-invalid sentinel.
- 60 analytic cases across Depthwise/Lossguide, NewtonL2/NewtonCosine,
  RMSE/Logloss/CrossEntropy, and five threshold configurations. Weighted
  logistic first-iteration Hessians include Bayesian bootstrap at temperature0.
- Additional negative-zero fits, prepared-pool fits with one rejected sibling
  and another legal sibling, and all-rejected roots.
- Explicit disabled versus implicit-default model bytes match. Existing
  leafwise score host oracle and minimum split-gain analytic checks also pass.

Public checks per mode:

- Counted class weights, optional two-field ABI tail, simultaneous split-gain
  and Hessian controls, signed zero, equality and inter-Float32 threshold.
- Two boosting iterations yield leaf counts `[2, 1]`: the second tree uses
  recomputed logistic Hessians after the first tree's +/-2 logits.
- Nonconstant Bayesian sampling, model save/load prediction identity, compiled
  numeric-mode readback, and actual fits/predictions for RMSE, Logloss, MAE,
  MultiClass and MultiClassOneVsAll.

Python API/regression suite: 119 passed plus 13 subtests, including 43 new
Hessian-option tests. See `python-regressions.log` (root agent's run).

Reproduce native/public checks:

```sh
MOJOLEARN_PYTHON="$PWD/.pixi/envs/default/bin/python" tools/check_min_child_hessian.sh
```

The interpreter is configurable and must have NumPy. The helper sets the pixi
runtime lookup immediately before exec on macOS; protected shells otherwise
strip inherited DYLD variables. An initial copied-package build smoke failed
because an existing base extension could not locate its runtime library.
That diagnostic is retained compressed; no compiler-cache clearing was used.
The passing FAST build ran the builder's own multi-loss smoke. The builder
intentionally skips its smoke for DETERMINISTIC/IDENTICAL; the separate public
check explicitly ran all five loss families for every mode using the installed
artifacts. No skipped builder gate is counted as a passed test.

Native binaries remain in ignored `build/min_child_hessian/`; installed
extensions remain in their mode-specific package locations. `artifacts.json`
records sizes and SHA256 hashes. Compiler logs are compressed; `.run.log`
files are the final authoritative outputs.

Scope: optional GPU non-symmetric child eligibility for three scalar losses
with Newton scores. First-order scores and other objectives are refused when
enabled. Existing disabled behavior is preserved; no CPU learner is added.
