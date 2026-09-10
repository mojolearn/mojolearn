# GPU log-loss implementation smoke — 2026-09-10

Base commit: `6567a6dc`. Local Apple M4; existing Mojo 1.0.0 toolchain.
Implementation and contract: [GPU_LOG_LOSS.md](../../../docs/lanes/GPU_LOG_LOSS.md).

Validation is deliberately restricted to builds and small smoke checks at the
user's request. No broad regression suite, timing run or remote GPU work was
performed. Cross-vendor identity and broader numerical qualification remain
pending; historical qualification of other metrics does not cover log loss.

Results: all three bindings compiled; FAST builder smoke passed (75 AIR blobs).
The separate public GPU smoke passed all 30 checks across the three modes.
Python compilation, shell syntax and `git diff --check` also passed. The initial
smoke launcher used an unsupported module import; it was corrected to
`from mojolearn import metrics`, with the failure log retained. No production
change was needed for that launcher error.

Reproduce the builds, serialized with `tools/with_build_lock.sh`, for each
`MOJOLEARN_NUMERIC_MODE=fast|deterministic|identical`:

```sh
MOJOLEARN_PYTHON="$PWD/checks/pipeline_python.sh" \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
MOJOLEARN_NUMERIC_MODE=fast \
tools/with_build_lock.sh pixi run sh bindings/build_metrics.sh
```

The FAST builder launches its existing metrics/spectral smoke plus log-loss
mean/sum checks. The other builders skip that internal smoke; the separate
public smoke covers all three artifacts:

```sh
PYTHONPATH=python \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
tools/with_build_lock.sh checks/pipeline_python.sh checks/log_loss_smoke.py
```

The public smoke has 30 cases spanning binary/multiclass, reversed explicit
string labels, zero-probability clipping, a 257-row tail, and mean/sum output.
It uses independent scalar logarithms for approximate expected values; it is
not a bitwise identity test.
