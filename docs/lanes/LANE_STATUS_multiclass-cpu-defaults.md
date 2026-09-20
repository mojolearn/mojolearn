# Multiclass CPU public defaults, 2026-09-20

MultiClass and MultiClassOneVsAll now accept public Bayesian bootstrap and
random_strength=1 on CPU. Bernoulli and Poisson also work. The host bootstrap
scales every derivative plane and bounds histogram quantization by each row's
maximum derivative magnitude. The score variance sums every class, reconstructing
the pinned derivative for MultiClass, and consumes the same tree/level/feature
random streams as the GPU.

The existing two-plane helper callers retain their defaults and arithmetic.
The new `gbdt-multiclass-defaults` verifier lane trains both weighted objectives
with public defaults, and declares prediction/probability batch checks.

Measured with a freshly compiled host GBDT binding and the parent release
worktree's fresh Metal core/GBDT bindings (evaluation cursor fix included):

- All nine fixtures, two repeats, full verifier parts: CPU and Metal train,
  inference, saved model and batch hashes are identical (36 comparisons).
- Independent 16-case matrix: two losses × four bootstrap choices × score
  strengths 0 and 1. All model bytes and held-out labels/probabilities match
  CPU versus Metal. Each sampler produces a different model on this fixture.
- Numerical regression repeats all 16 CPU configurations and checks model,
  prediction and probability repeatability, ranges and multiclass normalization.

Records are under `evidence/multiclass-cpu-defaults-2026-09-20`. They were
collected from the implementation working tree based on 6552f2766, before this
commit; records include binary/source digests. These are CPU/Metal development
measurements, not NVIDIA/AMD evidence or final release wheel attestations.

Reproduce the verifier comparison after building CPU and GPU package trees:

```sh
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/identity_break.py \
  --lanes gbdt-multiclass-defaults --repeats 2 --require-cpu --fail-on-refused --json cpu.json
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/identity_break.py \
  --lanes gbdt-multiclass-defaults --repeats 2 --require-backend metal --fail-on-refused --json apple.json
python tools/identity_break.py --lanes gbdt-multiclass-defaults \
  --diff cpu.json apple.json --require-columns 2
PYTHONPATH=python python -m mojolearn.tests.test_gbdt_multiclass_stochastic
```
