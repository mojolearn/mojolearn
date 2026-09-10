# Device-gradient training qualification

Metal, September 10, 2026. Four configurations, eight full training steps and
eight evaluation state-invariance checks pass the unchanged FP64 oracle gates.
All 88 native arrays in the eight retained captures match the previous
`../lm_generalized_shape_2026-09-10` captures byte for byte (see bitwise-regression.json).
These are correctness fixtures, not representative training-speed measurements.
No performance, large-model fit, cross-vendor or opponent-ratio claim.

76 host tests and the public Transformer surface suite passed. Both the LM
binding and Transformer binding were rebuilt from the source hashes in
build.json. Source/build witnesses, complete numerical arrays, logs and binding
hashes are retained. Only isolated-worktree library symlinks were changed.
The first compile found a stale last-use of the deleted host gradient variable;
that reference was removed before these successful builds and runs.

Reproduce with rebuilt IDENTICAL Apple bindings:

```
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/byte_lm_runtime_numerical_check.py --run --native-vendor metal --oracle-device cpu --out NEW_DIRECTORY
python tools/byte_lm_capture_regression.py bench/results/lm_generalized_shape_2026-09-10 NEW_DIRECTORY --cases default alternate_gqa one_layer_vocab257 three_layers_vocab513
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python -m mojolearn.tests.test_transformer_surface
PYTHONPATH=python python -m pytest python/mojolearn/tests/test_byte_lm_runtime_shape.py python/mojolearn/tests/test_byte_lm_surface.py tools/tests/test_byte_lm_source_inventory.py -q
```

No rentals or opponent runs. Cached opponent measurements unchanged.
