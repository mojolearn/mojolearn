# Mac M4 HIGGS benchmark — current suboptimized HEAD

## Provenance

- MojoLearn commit: `27ae6c82`
- Machine: Apple M4 (`arm64`), macOS 26.5.2 (25F84)
- MojoLearn device: Metal GPU
- Competitors: CPU only, 10 jobs
- Execution: algorithms run sequentially, not concurrently
- Dataset: NVIDIA `gbm-bench` HIGGS data; deterministic 80/20 train/test split
- Shared `gbm-bench` tree depth: 8
- Generated Python extension binaries were rebuilt from this commit immediately before measurement.
- This commit is known to contain a symmetric-tree performance regression. These results are retained as an experiment/baseline, not as a claim of optimized performance.

## Completed results

### 1,000,000 total rows (800,000 train), symmetric GBDT, 500 trees

| Estimator | Device | Train time (s) | AUC |
|---|---|---:|---:|
| MojoLearn symmetric GBDT | Metal GPU | 159.7614 | 0.828330 |
| CatBoost symmetric GBDT | CPU | 17.9477 | 0.828277 |

### 10,000,000 total rows (8,000,000 train), forests, 100 trees

| Estimator | Device | Train time (s) | AUC |
|---|---|---:|---:|
| MojoLearn RF | Metal GPU | 159.3338 | 0.774830 |
| sklearn RF | CPU | 379.0586 | 0.774758 |
| LightGBM randomized/RF mode | CPU | 31.2809 | 0.768616 |
| MojoLearn ExtraTrees | Metal GPU | 131.3677 | 0.702407 |
| sklearn ExtraTrees | CPU | 161.2403 | 0.703745 |
| LightGBM randomized/extra-trees mode | CPU | 36.0351 | 0.778408 |

### 10,000,000 total rows (8,000,000 train), symmetric GBDT, 500 trees

| Estimator | Device | Train time (s) | AUC |
|---|---|---:|---:|
| MojoLearn symmetric GBDT | Metal GPU | 586.3782 | 0.822907 |
| CatBoost symmetric GBDT | CPU | 270.2525 | 0.830035 |

## Interpretation constraints

- MojoLearn RF and ExtraTrees are directly comparable to the corresponding sklearn estimators closely enough to use sklearn as a semantic/reference baseline; their AUC values closely agree.
- LightGBM's `boosting_type=rf` and `extra_trees=true` modes use LightGBM's binned histogram/leaf-wise engine. They are not algorithmically equivalent to sklearn-style RF and ExtraTrees. They are retained as optimized-library targets and must not be presented as direct estimator-equivalence results.
- Fixed sequential order avoids simultaneous CPU/GPU contention but can still introduce thermal/order bias. Publication-quality comparisons should repeat with alternating or randomized order and report medians.
- The reported CatBoost log-loss is anomalously inconsistent with its AUC/accuracy and may reflect an adapter probability-shape/scoring issue; do not interpret cross-library log-loss until that path is audited.

## Raw artifacts

- `gbm_bench_higgs_1000000_500trees_mac_latest_27ae6c82.json`
- `gbm_bench_higgs_1000000_forests100_mac_latest_27ae6c82.json`
- `gbm_bench_higgs_10000000_forests100_mac_latest_27ae6c82.json`
- `gbm_bench_higgs_10000000_500trees_mac_latest_27ae6c82.json`
