# Tests to Mojo, lane tests-to-mojo-neural (2026-11-15)

The 35 files of the `L-neural` rows of `TESTS_TO_MOJO_2026-11-15.md`, read function by function. Nothing was
compiled or run (lane rule). Verdicts as in `TESTS_TO_MOJO_tests-to-mojo-classical.md`: KEEP-BOUNDARY (admission,
exceptions, returned objects and owners, hooks, `audit_native_call` one-call audits, files and archives, docs,
framework interop, outside oracles); KEEP-NOPORT (native numerics whose inputs come from NumPy's random stream, or
whose relation runs through a door with no typed RUN entry); KEEP-VERIFY / KEEP-TOOLING (drives a verify module or a
tool); PORT (moved to a Mojo check, the Python asserts deleted).

Totals: 35 files; 0 deleted, 2 shrunk, 33 kept unchanged; 58 Python lines removed, 30 added (docstrings, public
shape and counter asserts). Two new checks, registered in one block of `checks/native_tests/registry.mojo`
(`tools/ops/native_checks_report.py plan --column cpu` lists 167 names, both included).

No SOURCE-TEXT function was found: the inventory's "drop 28 l source-text" (`test_import_onnx_native.py`) and
"drop 19 l" (`test_small_mlp_surface.py`) are writes and reads of ONNX and checkpoint files, not source text.

| file | lines | tests | result | removed | added | function verdicts and what blocks a move |
|---|---:|---:|---|---:|---:|---|
| `test_autograd_model.py` | 275 | 12 | kept | 0 | 0 | KEEP-BOUNDARY 12: input forms (reversed, float16, bool/int labels, ids) give the contiguous bits, refusals by name, layer listing; inputs from `default_rng` |
| `test_autograd_public.py` | 324 | 13 | kept | 0 | 0 | KEEP-BOUNDARY 13: runs the user guide's blocks and the docstring examples verbatim (public docs), torch reference, `verify-model` CLI |
| `test_autograd_usability.py` | 41 | 1 | kept | 0 | 0 | KEEP-BOUNDARY 1: the validation label vocabulary of the door's label sort; `training_autograd_usability` holds the rest and says so |
| `test_checkpoint_native_loading.py` | 64 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: safetensors files, Python callbacks forbidden, maps closed, shard index refusal |
| `test_generator_draw_native.py` | 230 | 6 | moved to `generator_draw`; shrunk | 33 | 22 | PORT 4 (the Philox oracle parts of the uniform, dropout-layout, chunk-offset and last-stream tests), KEEP-BOUNDARY 6 (shape spellings, counters, state_dict, layouts against the C-contiguous public result, refusals and stream consumption). The retained-driver comparison stays (public evaluation order; the driver imports `_typed_input` / `_bufcheck`, see open points) |
| `test_generator_state_native.py` | 45 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: Python big-int seeds and counters, exhaustion and refusal audits |
| `test_import_onnx_native.py` | 281 | 9 | kept | 0 | 0 | ORACLE 9: the Mojo ONNX reader against the Python reader and resolver oracles (`_onnx_reader_oracle`, `_resolve_oracle`) on written ONNX files |
| `test_layernorm_public_native.py` | 36 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: one-call audits, `call forward first` refusal, the `__float__` hook refusal; the gradient words are public method vs public functional |
| `test_linalg_fast_bf16.py` | 173 | 6 | kept | 0 | 0 | KEEP-NOPORT 6: Apple `fast` only (paused programme; the checks build `identical`) |
| `test_linalg_lowbit.py` | 138 | 4 | kept | 0 | 0 | KEEP-BOUNDARY 4: public input forms, packed outputs, the raw host-binding pointer/count ABI, named refusals; the numerics already are `bf16_conversion` / `bf16_linalg` / `int8_linalg` |
| `test_lowbit_weights.py` | 190 | 6 | kept | 0 | 0 | KEEP-BOUNDARY 6: public packed containers, repacking, model routing through public blocks; inputs from `default_rng`; conversion numerics already `lowbit_weights` |
| `test_model_config_native.py` | 65 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: `HFConfig` / `plan_for` from a mapping and a file, public `ModelPlan` fields, refusals |
| `test_neural_backward_native.py` | 111 | 4 | kept | 0 | 0 | KEEP-BOUNDARY 4: one-call audits against raw binding calls, refusals inside the boundary; `default_rng` inputs |
| `test_neural_block_native.py` | 70 | 2 | kept | 0 | 0 | KEEP-NOPORT 2: compares the native owners with `_mamba_block_reference` drivers on `default_rng` inputs and checkpoint fixtures; length refusals |
| `test_neural_inference.py` | 378 | 12 | kept | 0 | 0 | KEEP-NOPORT 12: `default_rng` inputs, checkpoint files, refusals; `training_neural_inference` holds the moved numerics |
| `test_neural_public_coverage_p1.py` | 331 | 21 | kept | 0 | 0 | KEEP-BOUNDARY 21: keyword-to-option mapping, refusals, public session counters and owned-weight round trips; numerics already moved (its docstring) |
| `test_neural_repeat.py` | 220 | 1 | kept | 0 | 0 | KEEP-TOOLING 1: drives `identity_break.py` over the neural identity lanes in subprocesses |
| `test_neural_repeat_calls.py` | 84 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: every public call returns its own array and never writes the caller's input (owner isolation) |
| `test_ops_graph.py` | 19 | 1 | kept | 0 | 0 | KEEP-VERIFY 1: drives `mojolearn._verify_graph` (protected) |
| `test_ops_program.py` | 308 | 11 | kept | 0 | 0 | KEEP-BOUNDARY 11: capture/run/save/load of public programs, ids inputs, refusals, direct-route boundary, the read-back of uninitialized result arrays; `default_rng` inputs |
| `test_ops_session.py` | 83 | 3 | kept | 0 | 0 | KEEP-BOUNDARY 3: refusals, one buffer view for strided/Fortran/wider inputs, the name-to-code dispatch table (its numerics are `core_ops_session`) |
| `test_optimizer_native_moment_assignment.py` | 132 | 3 | kept | 0 | 0 | KEEP-BOUNDARY 3: moment assignment from Python exports, publication before upload, borrowed owners |
| `test_optimizer_native_owner_copy.py` | 63 | 3 | kept | 0 | 0 | KEEP-BOUNDARY 3: copy / deepcopy / pickle / `__setstate__` / `get_params` owners |
| `test_optimizer_native_scalar_constructors.py` | 118 | 3 | kept | 0 | 0 | KEEP-BOUNDARY 3: replaced builtin constructors are never called |
| `test_program_constructor_native.py` | 106 | 5 | kept | 0 | 0 | KEEP-BOUNDARY 5: Program metadata admission and refusal |
| `test_sequence_optimizer_public_native.py` | 192 | 10 | kept | 0 | 0 | KEEP-BOUNDARY 10: lifecycle audits, hook refusal, strided/integer gradient admission, finalizers, tuple containers |
| `test_sequential_native_checkpoint.py` | 258 | 8 | kept | 0 | 0 | KEEP-BOUNDARY 8: torch layout and safetensors import, checkpoint files, corruption refusals, no Python helpers |
| `test_sequential_native_surface.py` | 119 | 5 | kept | 0 | 0 | KEEP-BOUNDARY 5: public description, compile options, keyword binding, refusals |
| `test_small_mlp_surface.py` | 410 | 21 | moved to `training_small_mlp_relations`; shrunk | 25 | 8 | PORT 3 (`test_predict_matches_the_logits_of_the_next_step` deleted; `test_train_steps_equals_consecutive_train_steps` deleted, its step and loss count are `test_neural_public_coverage_p1.py::test_small_mlp_train_steps_runs`; `test_train_step_equals_loss_and_grads_then_apply_gradients` becomes `test_apply_gradients_returns_the_step_a_train_step_returns`, the door's step counter only), KEEP-BOUNDARY 18 |
| `test_training_array_metadata_contract.py` | 73 | 2 | kept | 0 | 0 | KEEP-BOUNDARY 2: returned Array metadata and NumPy ownership |
| `test_training_scalar_buffer_native.py` | 293 | 9 | kept | 0 | 0 | KEEP-BOUNDARY 9: NumPy scalar types, byte order, conversion hooks and warning order at admission |
| `test_tsa_resident_state.py` | 147 | 5 | kept | 0 | 0 | KEEP-BOUNDARY 5: public owners, read-only fitted arrays, resident counts, finalizers (numerics already in the tsa state checks) |
| `test_vjp_rank_native.py` | 29 | 1 | kept | 0 | 0 | KEEP-BOUNDARY 1: the full-shape refusal and the session's recovery, audited |
| `test_weight_import.py` | 294 | 13 | kept | 0 | 0 | KEEP-BOUNDARY 13: safetensors and mapping views, names and prefixes, refusals; `default_rng` inputs |
| `tests/test_autograd_smoke.py` | 77 | 2 | kept | 0 | 0 | KEEP-TOOLING 2: the `tools/autograd_smoke.py` 5 s entry (writes digests for the vendor columns) |

## New checks

| check | moved from | calls |
|---|---|---|
| `generator_draw` (cpu, gpu) | `test_generator_draw_native.py`: the `philox_words` / `units` oracle comparisons of the uniform (30 words, seed 0xfedcba9876543210), dropout forward and backward (seed 13, stream 0x80000003, offset 17, p 0.3), chunk offsets and p 0 (seed 71), last stream 0xffffffff and explicit stream 5 tests | `neural_rng_host_into` (cpu), `native_primitive_operation` 6 = `neural_rng_host` on the neural training context (gpu); an independent Philox4x32-10 in the check |
| `training_small_mlp_relations` (cpu, gpu) | `test_small_mlp_surface.py`: predict equals the next step's logits; train step equals GRADS then APPLY (loss, parameters, moments, logits after); `train_steps(k=3)` equals three consecutive steps (losses, parameters, moments) | `small_mlp_program_open` / `small_mlp_program_run` (STEP with k, APPLY, DOWNLOAD) on HostExec / DeviceExec |

Both compare words of this build (bits may change between versions; nothing frozen). The SmallMLP inputs are the
Python test's dyadic values bit for bit.

## Native entries missing (why the KEEP-NOPORT and some boundary-adjacent numerics stay)

- `SmallMLPTrainer.predict_logits_sessions`: the door calls the training binding's `mlp_sessions` /
  `mlp_sessions_rows` through `py_call_typed` (`bindings/small_mlp_native_door.mojo` `_predict_sessions`), so
  `test_sessions_concatenate_the_per_batch_logits` stays; a typed RUN entry for the per-session forward
  (training/neural_session_mlp.mojo) would let it move.
- `Generator.normal` and the parameter rounding of `uniform(low, high)` (`_round_f32` of the span): no independent
  oracle (the Box-Muller words go through `identical_log`/`cos`), and the rounding is the door's; the retained
  driver comparison stays.
- The `default_rng` inputs of `test_neural_inference.py`, `test_neural_block_native.py`, `test_lowbit_weights.py`,
  `test_weight_import.py`, `test_ops_program.py`, `test_autograd_model.py`: move once a committed generator stores
  those inputs as fixture words (or the relations are restated on a counter-based fixture, as
  `training_small_mlp` did); their numeric relations are public-vs-public through blocks with no typed RUN entry
  named in the test.

## Removed

Python: 58 lines (the Philox oracle helpers `philox_words` / `units` and their comparisons, two SmallMLP tests and
the numeric half of a third). Nothing else identified, because every other function in these files is a boundary,
tooling, verify or RNG-bound test (table above).

## Open points

- `LegacyDraws` in `test_generator_draw_native.py` imports `mojolearn.tests._typed_input` and
  `mojolearn._bufcheck` (product internals); it is the oracle-public-only lane's pattern. It stays while the
  retained-driver comparison stays.
- `test_ops_graph.py` and `tests/test_autograd_smoke.py` couple product tests to verify and tool entries; unchanged.
- No product defect seen (nothing was run).
