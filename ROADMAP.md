# Roadmap

This is the only live project plan. Historical plans and handoffs are not
current instructions; git history and `archive/` retain their evidence.

## Now: release truth and artifact closure

1. Build a clean wheel and verify all 15 native extensions in `fast`,
   `deterministic`, and `identical` mode from an isolated installation.
2. Stamp native artifacts with source/build identity and refuse stale
   artifacts at import or release time.
3. Generate extension, public-surface, architecture, and certification tables
   from machine-readable registries instead of repeating those facts in prose.
4. Classify remote results as `PASS`, `EXPECTED_DIVERGENCE`, `INFRA_FAILURE`,
   or `ALGORITHM_FAILURE` before they enter summaries.

## Next: close claims already exposed

- Run the dedicated ARIMA optimizer/fit correctness gate. The Python fit
  surface has only been smoke-tested on one Apple M4; NVIDIA and AMD remain.
- Close current NVIDIA legs for Holt-Winters, spectral clustering, TSA,
  Mamba 3, transformer bindings, and other recently exposed surfaces.
- Localize and fix the recorded AMD `_mojolearn_mamba` binding memory fault
  before making an AMD Python-surface claim.
- Add independent numerical references where cards currently prove stable
  bits without validating the calculus or accuracy, especially transformer
  backward and training/embedding paths.
- Exercise shipped-scale and plan-invariance fixtures for newer neural and
  training operators.

## NVIDIA performance campaign

Use one guarded NVIDIA rental only after the artifact gates above are green.
Every method gets exactly three interleaved arms:

1. mojolearn `fast`;
2. mojolearn `identical`;
3. one external comparator appropriate to the method.

Use CatBoost GPU for GBDT, cuML for an equivalent classical estimator when it
exists, PyTorch CUDA for GEMM/training/transformer operations, mamba-ssm for
Mamba, and scikit-learn CPU only where no GPU equivalent exists. Record warmup,
at least seven samples, explicit synchronization, median and IQR, accuracy,
output hashes, versions, driver, device, architecture, and commit. Preserve
raw output; publish no ratio from separate rental sessions.

FAST is intended to be the fastest mojolearn tier. Treat a repeatable
`identical/fast < 1.0` ratio as a performance defect, not an interesting
anomaly. Require five interleaved rounds and three representative sizes
before changing a default; ratios whose ranges overlap remain inconclusive.
Current candidates are GBDT's row-index-only route and DBSCAN's scheduling
path. Do not weaken IDENTICAL to make the comparison green.

## Algorithmic work after closure

- Complete CatBoost ordered boosting: per-fold approximation cursors, weak
  targets, leaf estimation, and model averaging. Until then, comparisons pin
  CatBoost to plain boosting.
- Complete tree CTR/feature-combination wiring if categorical parity remains
  a product priority.
- Consider AutoARIMA/search only after the existing ARIMA fit is independently
  validated across vendors.
- Optimize only profiles that show a representative workload gap. Do not add
  estimators merely because an upstream implementation exists.

## Release rule

Evidence does not transfer automatically between source checks, bindings,
built artifacts, and installed wheels. A release claim requires the installed
wheel layer to pass. Cross-vendor identity is claimed only for a named profile,
fixture, commit, and recorded hardware column.
