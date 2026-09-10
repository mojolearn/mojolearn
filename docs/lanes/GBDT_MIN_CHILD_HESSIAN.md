# Minimum child Hessian for GPU non-symmetric growth

`GradientBoosting(min_child_hessian=None)` preserves existing growth.
An enabled finite nonnegative threshold requires `Depthwise` or `Lossguide`,
`NewtonL2` or `NewtonCosine`, and `RMSE`, `Logloss`, or `CrossEntropy`.
The native `train`, `GbdtFitParams`, `fit_with_test`, and prepared numeric
pool APIs expose the same option with `-1` as the disabled sentinel.

Every candidate is rejected if either child has Hessian sum strictly below
the threshold, **before** scores, noise, and best-candidate selection. Equality
is permitted. A legal runner-up can therefore win when the original winner is
rejected. A leaf with no remaining candidate becomes terminal; in Lossguide
this also prevents rejected leaves reappearing beside the next two children.
The guard is independent of `min_split_gain`, which filters selected score
improvement, and `min_data_in_leaf`, which controls parent row count.

The statistic is the existing Newton search plane: the objective's Hessian
multiplied by row/class weights and the current tree's bootstrap weights.
For RMSE it is the Hessian of half-squared loss (one per unit-weight row).
For binary Logloss/CrossEntropy it is `w * p * (1-p)` at the current raw
prediction, before adding L2 regularization. It changes from tree to tree.
Cosine/L2 first-order scores store row weights in that plane and are refused;
those weights are not called Hessians. Other losses are deliberately outside
this option's audited support, including objectives with surrogate curvature.
No CPU training backend is added.

The public/native parameter is Float64, bounded by Float32.MAX_FINITE because
GPU score statistics are Float32. The host rounds the bound upward to the
next Float32 when necessary. Thus a bound between adjacent representable
statistics cannot accidentally become weaker by rounding down. The kernel compares nonnegative Float32 bit patterns as integers (treating
signed zero as zero), so a subnormal threshold is not silently flushed by
a device floating comparison.
Histogram accumulation and subtraction retain each compiled numeric mode's
existing arithmetic; this option does not claim exact-real Hessian sums or
numerical equivalence with another library's `min_child_weight`.

The ABI keeps default requests unchanged (`35 + n_class_weights` values).
An enabled Hessian uses the two-slot optional tail `min_split_gain` (or -1)
then `min_child_hessian`, after counted class weights. Existing gain-only
requests retain their one-slot tail. Old extensions reject the longer request
instead of accepting an option they cannot honor.

Checks: `checks/min_child_hessian_check.mojo` covers candidate runner-up and
both child bounds, equality and a Float64 bound between adjacent Float32
values, analytic RMSE/weighted logistic fits, all-invalid roots, explicit-off
model identity, and prepared pools with one rejected sibling and one legal
sibling. `checks/min_child_hessian_binding.py --mode <mode>` checks rebuilt
public bindings, class-weight/two-option tails, stochastic bootstrap, model
round trips, and compiled numeric-mode readback.

Example for the supported scalar GPU scope:

```python
from mojolearn.ensemble import GradientBoosting

model = GradientBoosting(
    loss="Logloss", grow_policy="Lossguide", score_function="NewtonL2",
    min_child_hessian=2.0, numeric_mode="identical",
).fit(X, y)
```

`2.0` is weighted curvature mass, not a row count. Choose it in relation to
weights and objective curvature; there is no universal cross-library value.

The capability reference is XGBoost's documented
[`min_child_weight`](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)
child-Hessian condition. This implementation uses MojoLearn's existing GPU
Newton statistics and split selection, rather than implementing XGBoost's trainer.
