# Serial cross-validation smoke — 2026-09-10

Base: `0c325f915b73a36081eb6f32410287721d7c3d0f`; Apple M4, existing
Mojo 1.0.0 ed45d567 bindings. Python 3.13.15, NumPy 2.5.2, sklearn 1.8.0.
No native source changes or rebuild required.

Command, serialized with the repository build lock:

```
PYTHONPATH=python tools/with_build_lock.sh /tmp/mojolearn-b1-sklearn/bin/python checks/model_selection_smoke.py --gpu
```

Exit 0; [output](smoke.log). The host-only test estimator is an index ownership
oracle, not a product CPU backend. Three folds prove heldout row isolation;
six malformed index pairs are rejected before even an earlier valid fold fits.
Callable scoring and preservation of the original unfitted estimator pass.

Six actual GPU fits: StandardScaler → two-tree RMSE GBDT, two folds for each
FAST/DETERMINISTIC/IDENTICAL. Fitted scalers see 12 training rows rather than
24 total rows; fitted tree mode is checked; heldout R² is 0.80246919 in each
cell. These repeated toy-data scores verify wiring, not generalization quality
or cross-device identity. No CoreAnalytics diagnostic appeared in this run;
previous diagnostics remain unresolved. No broad tests, timing, remote work,
or CUDA/HIP qualification was run.
