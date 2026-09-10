# Binary GPU ranking metrics — implementation smoke, 2026-09-10

Base commit: `f38ee93e`. Local Apple M4 and existing Mojo 1.0.0 toolchain.
Contract: [GPU_RANKING_METRICS.md](../../../docs/lanes/GPU_RANKING_METRICS.md).

Results: the final FAST, DETERMINISTIC and IDENTICAL bindings all compiled.
FAST's builder smoke passed (79 AIR blobs); the separate public smoke passed
all 90 checks across the three modes, exit 0. Python compilation, shell syntax
and `git diff --check` passed. The public smoke waited for another local job
to release the shared lock before running; that job was left undisturbed.

This slice adds binary ROC-AUC and precision-recall curves with a shared
stable score sort, integer prefix counts and parallel tied-group processing.
AUC uses exact Int64 pair contributions and a final Float32 ratio. No CPU
metric implementation or remote GPU work was added.

Validation is limited to builds and small smoke checks at the user's request.
It does not qualify cross-vendor identity, large-input scaling, memory
stability or speed. Existing radix sorting uses 32 one-bit passes and serial
block-total scans; AUC retains a serial integer contribution fold. The public
boundary still stages host inputs and downloads results.

The first implementation's successful build logs are retained as
`*.initial-serial.build.log`. The final implementation replaces its serial
full-row/group walk with parallel group compaction and per-group calculations.
Final FAST compilation began before a whitespace/comment-only source cleanup;
the cleanup did not change semantics. DETERMINISTIC and IDENTICAL compilation
used the formatted source. `provenance.json` records final source and artifact
hashes.

Build each mode under the shared lock:

```sh
MOJOLEARN_PYTHON="$PWD/checks/pipeline_python.sh" \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
MOJOLEARN_NUMERIC_MODE=fast \
tools/with_build_lock.sh pixi run sh bindings/build_metrics.sh
```

Repeat with `deterministic` and `identical`. The FAST builder runs existing
metric/spectral smoke calls plus the new AUC/PR calls; the other mode builders
skip that internal smoke. The separate public smoke exercises all modes:

```sh
PYTHONPATH=python \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
tools/with_build_lock.sh checks/pipeline_python.sh checks/binary_ranking_smoke.py
```

The independent oracle counts positive-negative score wins/ties pairwise for
AUC and selects rows directly at each threshold for PR. Fixtures include
ordinary and all-equal scores, negative scores, signed zero, smallest
subnormals, string/large-integer labels, 513 rows crossing scan blocks, and
single-class PR curves. It checks approximate ratio values and exact threshold
values; it is not a bitwise cross-device test.
