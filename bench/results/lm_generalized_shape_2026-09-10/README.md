# Generalized language-model shape qualification

Actual Metal execution of the generalized native trainer, September 10, 2026.
See `verdict.json`, per-step JSON and complete NPZ arrays: four configurations,
eight training steps, eight evaluations. CPU FP64 oracle tolerances unchanged.
One-layer V257 and three-layer V513 cases exercise IDs above the old byte bound,
all layer gradients, optimizer state and evaluation invariance. Wrong nonlinear
derivative controls are evaluated independently for every layer.

`byte-regression.json` compares all native array bytes in four prior retained
captures from `../byte_runtime_numerical_2026-09-10/`. All match, including both
steps' full gradients, loss and state. This is a recorded prior/new comparison,
not interleaved timing or cross-vendor qualification. No performance claim.

`host-tests.log`: 76 host checks, including fake-binding plumbing (not numerical
arithmetic), configured named registries, 12-layer/V50257 state routing, bounds,
checkpoint roundtrip and legacy shape restoration. `native-config.log` exercises
Mojo registry counts including 162147840 parameters. `binding-config.log` checks
six Python/native profiles and rejects invalid dimensions/null pointer spans
before device creation. Large configurations were host-admitted only.

Reproduction from the repo with a rebuilt IDENTICAL Metal byte-LM extension:

```
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/byte_lm_runtime_numerical_check.py --run --native-vendor metal --oracle-device cpu --out NEW_DIRECTORY
python tools/byte_lm_capture_regression.py bench/results/byte_runtime_numerical_2026-09-10 NEW_DIRECTORY
PYTHONPATH=python python tools/byte_lm_config_binding_check.py PATH_TO_BINDING
```

Build command and final direct source hashes are in `build.json`. Runtime metadata
identifies the actual binding SHA and source witnesses. No binding was installed
into the shared root; the isolated checkout used a local symlink. Existing core
extensions were reused for package imports. Numerical work used the new byte-LM
binding. No rental or opponent run was started.

An initial numerical run reached the three-layer wrong-derivative control and
stopped because the oracle still allowed only block indices 0/1. That test-only
restriction was generalized to the configured layer count; the complete final
run retained here passed. The first compile exposed an Int32/Int token-bound
comparison, fixed with an explicit Int32 conversion before the successful build.
