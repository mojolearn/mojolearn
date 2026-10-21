# Test coverage of public endpoints changed since 0.8.55 (2026-10-21)

Release gate (Andrew, 2026-10-21): every public endpoint that is new or changed since the last published
release (`fc8d425fb`, 0.8.55) has a test that calls it on the real binding path, a fault-injection test (a
fake or sabotaged binding, or a poisoned input that must be refused by name) and a test of each new keyword.
Audit of `origin/main` 1aeca3001 on lane `lane/tests-coverage-audit`.

## How the list was built

- `python3 tools/changed_public_methods.py --endpoints fc8d425fb` parses every changed file under
  `python/mojolearn/` (tests excluded) at both revisions and reports each def whose AST changed. A def is a
  public endpoint when its owner is in the export walk of `tools/public_runtime_inventory.py` (the package
  `__all__`, the `_expansion_*` doors, `packaging/native_only/manifest.json`) or in the `__all__` of a public
  module (`ops`, `models.*`, `cross_vendor`, `tokenizer`, `neural_inference`, `parallel_training`, ...).
  A public operation method whose own body did not change but whose walked call path reaches a changed
  helper is listed too (`call path`).
- 339 endpoints: 301 changed (1 docstring only), 33 new, 5 through a changed call path. 124 native entries were
  added under `bindings/` (`def_function` / `def_py_c_function`); none was removed.
- Coverage is read statically from `python/mojolearn/tests/test_*.py` plus the package qualification modules
  that a test drives case by case (`_verify_graph.py` via `test_ops_graph.py`, `_verify_generation.py`,
  `_verify_model.py`, `_verify_par.py`, `_verify_partitioned_program.py`, `_verify_distributed_program.py`).
  A test function counts as **real path** when it calls or references the method (directly, through a
  module helper, a `getattr` name list or a parametrize list) without a fake binding or monkeypatch; it
  counts as **fault** when the same function expects a refusal (`pytest.raises`, `_raises`, `_refuses`,
  `native_failure`, an audited `.exception`). Every gap the scan reported was checked by hand.
- Files owned by the running migration lanes (`linear_model.py`, `_native_classical_*`, `_mlp_impl.py`,
  `_training_impl.py`, `_mamba_impl.py` and `test_linear*`, `test_mlp*`, `test_training*`, `test_mamba*`)
  were not edited; their endpoints are listed as owed. The classical native owner
  (`_native_classical_public._install`) is not installed by the package yet, so it exposes no public
  endpoint in this release.

## Result

| | endpoints |
|---|---|
| changed or new public endpoints | 339 |
| covered before this audit (real path + refusal) | 246 |
| newly covered by this lane | 67 |
| owed: migration-lane files (24 have real and refusal tests today, 2 lack a refusal test) | 26 |

New test files (run with the 2026-10-21 integration bindings, Apple M-series, `identical`):

| file | covers |
|---|---|
| `test_ops_session_coverage.py` | `ops.Session` operations and losses, metrics, scheduler and profile calls: NumPy checks, dead/freed/str handles, poisoned shapes and options, closed session, non-numeric scalars |
| `test_parallel_trainers_coverage.py` | split step and device fold of `ParallelByteLanguageModelTrainer` (bit-equal to `train_step`), pooled and offloaded trainers with checkpoint replay, closed-trainer refusals, `ordered_sum_gradients`, `fold_pair`, `drain_events`, `ParallelNeuralTrainer` context manager |
| `test_models_coverage.py` | `SafetensorsFile`/`Checkpoint` `info`/`close`, `TensorInfo` admission, `pattern_name`, `GenerationStream.read(max_tokens=)`, `ParallelCausalLM.parameters` after close, `Program.fit(huber_delta=)`, `Program.run_dynamic(optimize=, profile=)` |
| `test_neural_coverage.py` | `SambaStack.loss_and_grads` divisor and refusals, `save_checkpoint`, `SambaInference.parameters`/`allocate_state`, TSNE through the one-call PCA, `Sequential` `layers`/`sigmoid`/`avg_pool2d` and layer-stack refusals, integer counts of `CausalLM` and its blocks |

pytest (the four files): 122 tests, 101 passed, 21 failed. 20 failures are the refusals added by the native
fixes below and pass once the binding is rebuilt from this branch; 1
(`test_tsne_fit_transform_through_pca_and_refusals`) fails because the integration build predates main's
`pca_fit_owned` (c66483695) and passes on a build of main.

## Defects found and fixed

The fault tests found public entries that accepted poisoned input. Each is fixed in the native code. The
tests above fail until the binding is rebuilt:

| defect | fix |
|---|---|
| `ParallelByteLanguageModelTrainer.apply_gradient` refused the float32 bytes that `cross_vendor.ordered_fold` returns, the input its docstring names | `bindings/parallel_byte_native.mojo:786` admits raw float32 bytes the way `fold_reset`/`fold_add` do |
| `TensorInfo(...)` stored any fields: non-str name, negative dimensions, reversed offsets, a byte span that disagrees with dtype x shape | `bindings/causal_checkpoint.mojo:48` applies the header parser's rules |
| `GenerationStream.read("2")`, `CausalLM.generate(ids, "2")`, `stream(ids, True)`, `allocate_state("1", 8)` parsed the str or counted the bool | `bindings/causal_lm_public.mojo:254` `_public_int` (anything with `__index__`; str and bool refused as `TypeError`) |
| block `allocate_state("1", 8)` parsed the str | `bindings/neural_block_public.mojo:84` |
| `SambaInference.allocate_state("1", 8)` parsed the str | `bindings/samba_inference_native.mojo:52` |
| `SambaStack.loss_and_grads(num_items=0)` returned a loss; `"4"`/`True` were accepted | `bindings/samba_public_native.mojo:18` refuses at the public dispatcher; the train step's internal microbatch divisor is unchanged |
| `ops.Session` scalar arguments (`pow_scalar`, `scale`, `leaky_relu`, `elu`, `hard_sigmoid`, `full`, `clip`, `pad`, `dropout`, `huber`, `layer_norm`, `rotary`, `optimizer`) refused a str with a bare `Exception: must be real number, not str` | `autograd/ops_frontend_pyapi.mojo:31` `_float` raises `TypeError: <argument> must be a real number` |

Bindings to rebuild: the tokenizer host / merged native binding (`bindings/_mojolearn_tokenizer_host.mojo`
imports `parallel_byte_native`, `causal_checkpoint`, `causal_lm_public`, `neural_block_public`,
`samba_public_native`, `samba_inference_native`) and the sequence binding that carries the ops session
(`autograd/ops_frontend_pyapi.mojo`), on every vendor and in the host column. No numeric result changes:
the fixes only add admission checks.

Noted, not changed: `Program.run_dynamic(profile=...)` and the other session flags take Python truthiness
(`profile="yes"` profiles); `ParallelCausalLM.parameters()` after `close()` raises `ValueError` while
`forward()` raises `RuntimeError`; `Coordinator().drain_events()` on an object that was never constructed
reports the missing attribute rather than the class; `ops.Session._metric` is dead code that still reads
`self._b`.

## Keywords and options added since 0.8.55

| keyword | test |
|---|---|
| `numeric_mode=` on `Mamba1/2/3Block`, `TransformerBlock` | `test_mamba1_backward_surface`, `test_mamba23_backward_surface`, `test_neural_inference`, `test_samba_attention_wiring` |
| `Sequential.fit(validation_data=, restore_best_weights=)` | `test_autograd_usability`, `test_sequential_native_execution` |
| `Coordinator(log_path=)`, `Worker(log_path=, checkpoint_dir=, checkpoint_steps=, expected_commits=, retain_gradients=, hash_scheme=)` | `test_cross_vendor_worker_native`, `test_cross_vendor_admission_native` |
| `BatchFeed(start_step=)`, `LearningRateTable(start_step=)` | `test_cross_vendor_worker_native` |
| `CausalLM.generate/stream/generate_text/stream_text(seed=, temperature=, top_k=, top_p=, eos_token_id=, pad_token_id=, greedy=)` | `test_generation_native`, `test_parallel_causal_lm`, `_verify_generation` |
| `GenerationStream.read(max_tokens=)` | `test_models_coverage` (new) |
| `Program.run/Session.run(dynamic_shapes=, optimize=)`, `Program.inspect/profile(optimize=)` | `test_autograd_usability`, `test_ops_transformer_program`, `_verify_graph` |
| `Program.run_dynamic(optimize=, profile=)` | `test_models_coverage` (new) |
| `Program.fit(loss=, epochs=, batch_size=, shuffle=, seed=)`, `fit_if`, `while_loop/vjp_while(max_iterations=)`, `save_control` | `_verify_graph` (`test_ops_graph`), `test_autograd_model` |
| `Program.fit(huber_delta=)` | `test_models_coverage` (new) |
| `Program.partition(stages=, tensor_shards=, microbatches=)`, `Program.distribute(logical_shards=)` | `test_partitioned_program`, `test_distributed_program*`, `_verify_partitioned_program`, `_verify_distributed_program` |
| `Session.gradient_report(values=)` | `_verify_graph` (`gradient_diagnostics`) |
| `Session.rotary(offset=, theta=)`, `Session.rms_norm(eps=)` | `test_ops_transformer_program` |

## New native entries

All 124 added entries are reached from the endpoints above; the table shows which tests call each family.
Five have no Python call site by name and are reached from native setup or dispatch: `causal_lm_admit`,
`generation_initialize`, `native_json_parse`, `native_transport_call`, `vocab_fingerprint`.

| entries | reached from | tests |
|---|---|---|
| `ops_session_create/initialize/call`, `ops_program_*` (6) | `ops.Session`, `ops.Program` | `test_ops_*`, `test_ops_graph`, `test_ops_session_coverage` |
| `ops_partitioned_*` (11), `ops_distributed_*` (10) | `Program.partition/distribute`, `Session.load_partitioned/load_distributed` | `test_partitioned_program`, `test_distributed_program*` |
| `causal_lm_*` (16), `causal_model_*` (3), `causal_state_init`, `generation_initialize`, `model_config_*`, `model_plan_*`, `model_checkpoint_names`, `model_layer_weights` | `models.CausalLM`, `GenerationStream`, `HFConfig`, `ModelPlan`, `plan_for` | `test_generation_native`, `test_models_loader`, `test_model_config_native`, `test_causal_model_native_loading`, `test_neural_coverage` |
| `safetensors_*` (7), `checkpoint_*` (9), `tensor_info_*` (2) | `SafetensorsFile`, `Checkpoint`, `TensorInfo` | `test_checkpoint_native_loading`, `test_models_coverage` |
| `vocab_*` (10), `native_json_parse` | `models.Tokenizer`, `pretokenize`, `pattern_name` | `test_tokenizer_native_loading`, `test_models_loader`, `test_models_coverage` |
| `bpe_encode_batch`, `bpe_train`, `bpe_trained_*`, `bpe_fast_*`, `tokenizer_fast_numeric_mode` | `BpeTokenizer`, `BpeVocabularyTrainer` (fast entries: Apple FAST build only) | `test_tokenizer*` |
| `byte_lm_trainer_*` (6), `byte_lm_file_sha256`, `sha256_hex`, `optimizer_finalize` | `SmallByteLanguageModelTrainer`, byte-LM checkpoints | `test_byte_lm_*`, `test_byte_checkpoint_native`, `test_optimizer_native_lifecycle` |
| `parallel_byte_call`, `parallel_neural_call`, `parallel_mlp_call`, `parallel_causal_call`, `cross_vendor_call` | parallel trainers, `ParallelCausalLM`, `cross_vendor` | `test_parallel_*`, `test_cross_vendor_*`, `test_parallel_trainers_coverage` |
| `neural_block_*` (4), `neural_prefill_call`, `generator_draw` | Mamba/Transformer blocks, `Generator` | `test_neural_block_native`, `test_neural_prefill_native`, `test_generator_*` |
| `samba_native_call`, `samba_config_call`, `sequential_native_call` | `SambaStack`, `SambaConfig`, `SambaInference`, `Sequential` | `test_samba_*`, `test_sequential_native_*`, `test_neural_coverage` |
| `pca_fit_owned`, `pca_fit_randomized`, `pca_native_call` | `PCA.fit/fit_transform`, TSNE `init="pca"` | `test_pca_*`, `test_pca_exact_native_owner`, `test_neural_coverage` |
| `native_codec_call`, `native_process_call`, `native_transport_call` | process pools and wire transport of the parallel trainers | `test_native_codec`, `test_native_process_pool`, `test_native_transport` |
| `classical_native_call`, `classical_attribute_call`, `classical_attribute_descriptor` | classical native owner (not installed by the package yet) | `test_native_classical_public`, `test_classical_*_native_owner` |

## Owed

Migration-lane files (not edited here; the lanes that own them must keep or add these tests when they land):

- `Mamba2Block.step`: real path in `test_ragged_lengths`/`test_decode_sessions_cpu_route`, no refusal test.
- `Generator.dropout_backward`: real path in `test_generator_draw_native`, no refusal test.
- Covered today, owned by a migration lane: `Mamba1Block` (`__init__`, `allocate_state`, `backward`,
  `forward`, `step`), `Mamba2Block` (`__init__`, `allocate_state`, `backward`, `forward`), `Mamba3Block`
  (`__init__`, `backward`), `SmallMLPTrainer` (`from_checkpoint`, `save_checkpoint`), `Generator`
  (`__init__`, `dropout`, `kaiming_normal`, `kaiming_uniform`, `load_state_dict`, `next_stream`, `normal`,
  `state_dict`, `uniform`, `xavier_normal`, `xavier_uniform`).

Rebuild: the 20 refusal tests above pass only on a binding built from this branch.

## Per-endpoint table

Real path and fault columns name up to two test files (new files first); `+n` counts the others.

### `mojolearn.SmallByteLanguageModelTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `train_step` | call path | `test_byte_lm_checkpoint_binary`, `test_byte_lm_runtime_shape` +2 | `test_byte_lm_runtime_shape`, `test_byte_lm_session` +2 | covered |

### `mojolearn._expansion_ann.TSNE`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `fit` | call path | `test_neural_coverage`, `test_sklearn_compat` | `test_neural_coverage` | **new tests** |
| `fit_transform` | call path | `test_neural_coverage` | `test_neural_coverage` | **new tests** |

### `mojolearn._ivf_impl.IVFIndex`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `search` | doc | `test_cpu_training_embedding_ivf`, `test_distributed_ivf` +2 | `test_distributed_ivf`, `test_ivf_embedding_cpu_inference` | covered |

### `mojolearn._mamba_impl.Mamba1Block`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_decode_sessions_cpu_route`, `test_lowbit_weights` +1 | `test_decode_sessions_cpu_route`, `test_neural_backward_native` +1 | owed (migration lane) |
| `allocate_state` | changed | `test_decode_sessions_cpu_route`, `test_lowbit_weights` +2 | `test_decode_sessions_cpu_route`, `test_mamba1_backward_surface` +1 | owed (migration lane) |
| `backward` | changed | `test_mamba1_backward_surface`, `test_neural_backward_native` | `test_mamba1_backward_surface`, `test_neural_backward_native` | owed (migration lane) |
| `forward` | changed | `test_decode_sessions_cpu_route`, `test_lowbit_weights` +1 | `test_ragged_lengths` | owed (migration lane) |
| `step` | changed | `test_decode_sessions_cpu_route`, `test_lowbit_weights` | `test_decode_sessions_cpu_route` | owed (migration lane) |

### `mojolearn._mamba_impl.Mamba2Block`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_cpu_training_mamba`, `test_mamba23_backward_surface` +2 | `test_mamba23_backward_surface`, `test_neural_backward_native` +1 | owed (migration lane) |
| `allocate_state` | changed | `test_cpu_training_mamba`, `test_mamba23_decode_sessions` +1 | `test_ragged_lengths` | owed (migration lane) |
| `backward` | changed | `test_cpu_training_mamba`, `test_mamba23_backward_surface` +1 | `test_mamba23_backward_surface`, `test_neural_backward_native` | owed (migration lane) |
| `forward` | changed | `test_cpu_training_mamba`, `test_mamba23_backward_surface` +1 | `test_ragged_lengths` | owed (migration lane) |
| `step` | changed | `test_cpu_training_mamba`, `test_mamba23_decode_sessions` | none | owed (migration lane) |

### `mojolearn._mamba_impl.Mamba3Block`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_mamba23_backward_surface`, `test_mamba23_decode_sessions` +2 | `test_mamba23_backward_surface`, `test_mamba3_constructor_native` +1 | owed (migration lane) |
| `backward` | changed | `test_mamba23_backward_surface`, `test_neural_backward_native` | `test_mamba23_backward_surface`, `test_neural_backward_native` | owed (migration lane) |

### `mojolearn._mlp_impl.SmallMLPTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `from_checkpoint` | changed | `test_neural_inference`, `test_parallel_neural_admission` +1 | `test_neural_inference`, `test_parallel_neural_admission` +1 | owed (migration lane) |
| `save_checkpoint` | changed | `test_neural_inference`, `test_small_mlp_surface` | `test_small_mlp_surface` | owed (migration lane) |

### `mojolearn._samba_impl.SambaStack`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_neural_coverage`, `test_cpu_training_par_classical` +13 | `test_neural_coverage`, `test_numeric_profile` +7 | covered |
| `allocate_state` | changed | `test_neural_coverage`, `test_neural_inference` +4 | `test_neural_coverage`, `test_numeric_profile` +3 | covered |
| `forward` | changed | `test_cpu_training_samba`, `test_neural_inference` +5 | `test_neural_inference`, `test_numeric_profile` +4 | covered |
| `from_checkpoint` | changed | `test_neural_inference`, `test_neural_numpy_free_merge` +1 | `test_neural_inference`, `test_samba_checkpoint_native` | covered |
| `load_state_dict` | changed | `test_numeric_profile`, `test_samba_state_native` +1 | `test_numeric_profile`, `test_samba_state_native` +1 | covered |
| `load_weights` | changed | `test_torch_native_modes` | `test_samba_native_admission`, `test_torch_native_modes` | covered |
| `loss` | changed | `test_neural_coverage`, `test_cpu_training_samba` +3 | `test_neural_coverage`, `test_neural_inference` +1 | covered |
| `loss_and_grads` | changed | `test_neural_coverage`, `test_samba_attention_wiring` | `test_neural_coverage` | **new tests** |
| `parameters` | changed | `test_neural_coverage`, `test_cpu_training_samba` +2 | `test_neural_coverage`, `test_samba_state_native` | covered |
| `save_checkpoint` | changed | `test_neural_coverage`, `test_neural_inference` +2 | `test_neural_coverage`, `test_samba_checkpoint_native` | covered |
| `state_dict` | changed | `test_numeric_profile`, `test_samba_native_admission` +2 | `test_numeric_profile`, `test_samba_native_admission` +2 | covered |
| `step` | changed | `test_cpu_training_samba`, `test_neural_inference` +2 | `test_neural_inference`, `test_numeric_profile` +1 | covered |
| `train_step` | changed | `test_cpu_training_par_classical`, `test_cpu_training_samba` +5 | `test_neural_inference`, `test_numeric_profile` +3 | covered |

### `mojolearn._training_impl.Generator`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_cpu_training_par_classical`, `test_cpu_training_samba` +8 | `test_generator_draw_native`, `test_generator_initializer_native` +4 | owed (migration lane) |
| `dropout` | changed | `test_generator_draw_native` | `test_generator_draw_native` | owed (migration lane) |
| `dropout_backward` | changed | `test_generator_draw_native` | none | owed (migration lane) |
| `kaiming_normal` | changed | `test_generator_initializer_native` | `test_generator_initializer_native` | owed (migration lane) |
| `kaiming_uniform` | changed | `test_generator_initializer_native` | `test_generator_initializer_native` | owed (migration lane) |
| `load_state_dict` | changed | `test_generator_draw_native`, `test_generator_state_native` +1 | `test_generator_draw_native`, `test_generator_state_native` +1 | owed (migration lane) |
| `next_stream` | changed | `test_generator_state_native` | `test_generator_state_native` | owed (migration lane) |
| `normal` | changed | `test_generator_draw_native` | `test_generator_draw_native` | owed (migration lane) |
| `state_dict` | changed | `test_generator_draw_native`, `test_generator_initializer_native` +3 | `test_generator_draw_native`, `test_generator_state_native` +2 | owed (migration lane) |
| `uniform` | changed | `test_generator_draw_native`, `test_ragged_lengths` | `test_generator_draw_native`, `test_generator_initializer_native` +1 | owed (migration lane) |
| `xavier_normal` | changed | `test_generator_initializer_native` | `test_generator_initializer_native` | owed (migration lane) |
| `xavier_uniform` | changed | `test_generator_initializer_native` | `test_generator_initializer_native` | owed (migration lane) |

### `mojolearn._transformer_impl.TransformerBlock`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `_verify_graph`, `test_cpu_training_transformer` +9 | `_verify_graph`, `test_decode_sessions_cpu_route` +6 | covered |
| `allocate_state` | changed | `test_cpu_training_transformer`, `test_decode_sessions_cpu_route` +6 | `test_decode_sessions_cpu_route`, `test_models_loader` +3 | covered |
| `backward` | changed | `_verify_graph`, `test_cpu_training_transformer` +5 | `_verify_graph`, `test_neural_backward_native` +4 | covered |
| `forward` | changed | `_verify_graph`, `test_cpu_training_transformer` +7 | `_verify_graph`, `test_models_loader` +4 | covered |
| `step` | changed | `_verify_graph`, `test_cpu_training_transformer` +6 | `_verify_graph`, `test_decode_sessions_cpu_route` +3 | covered |

### `mojolearn._x_sequence_model.Sequential`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_neural_coverage`, `_verify_model` +16 | `test_neural_coverage`, `test_autograd_model` +11 | covered |
| `add` | changed | `test_neural_coverage`, `test_autograd_public` +1 | `test_neural_coverage`, `test_sequential_native_surface` | covered |
| `avg_pool2d` | changed | `test_neural_coverage`, `test_sequential_native_surface` | `test_neural_coverage` | **new tests** |
| `compile` | changed | `test_neural_coverage`, `test_autograd_model` +11 | `test_neural_coverage`, `test_autograd_model` +7 | covered |
| `conv2d` | changed | `test_autograd_model`, `test_sequential_native_surface` +1 | `test_autograd_model`, `test_sequential_native_surface` +1 | covered |
| `dropout` | changed | `test_neural_coverage`, `test_autograd_model` +1 | `test_neural_coverage`, `test_sequential_native_surface` | covered |
| `export` | changed | `test_autograd_save_resume`, `test_import_lower_native` +3 | `test_autograd_save_resume`, `test_import_torch` +1 | covered |
| `fit` | changed | `test_neural_coverage`, `test_autograd_model` +9 | `test_neural_coverage`, `test_autograd_model` +7 | covered |
| `flatten` | changed | `test_neural_coverage`, `test_autograd_model` +4 | `test_neural_coverage`, `test_autograd_model` +3 | covered |
| `from_checkpoint` | changed | `test_autograd_save_resume`, `test_sequential_native_checkpoint` | `test_autograd_save_resume` | covered |
| `layer_norm` | changed | `test_autograd_model`, `test_sequential_native_surface` +2 | `test_sequential_native_surface`, `test_torch_state_dict` +1 | covered |
| `layers` | changed | `test_neural_coverage`, `test_autograd_model` +1 | n/a: read-only property, no input to poison | covered |
| `linear` | changed | `test_neural_coverage`, `test_autograd_model` +12 | `test_neural_coverage`, `test_autograd_model` +9 | covered |
| `load` | changed | `_verify_model`, `test_autograd_save_resume` +3 | `test_autograd_save_resume`, `test_sequential_native_admission` +1 | covered |
| `load_state_dict` | changed | `test_autograd_public`, `test_sequential_native_admission` +4 | `test_sequential_native_admission`, `test_sequential_native_checkpoint` +3 | covered |
| `max_pool2d` | changed | `test_autograd_model`, `test_sequential_native_surface` | `test_autograd_model` | covered |
| `permute` | changed | `test_neural_coverage`, `test_autograd_model` +1 | `test_neural_coverage`, `test_sequential_native_surface` | covered |
| `predict` | changed | `test_neural_coverage`, `test_autograd_model` +8 | `test_autograd_save_resume`, `test_sequential_native_checkpoint` +1 | covered |
| `predict_proba` | changed | `test_autograd_model`, `test_autograd_save_resume` +3 | `test_autograd_model`, `test_autograd_save_resume` +2 | covered |
| `relu` | changed | `test_autograd_model`, `test_autograd_save_resume` +5 | `test_autograd_model`, `test_sequential_native_surface` +1 | covered |
| `reshape` | changed | `test_neural_coverage`, `test_autograd_model` +11 | `test_autograd_model`, `test_import_lower_native` +9 | covered |
| `residual` | changed | `test_autograd_model`, `test_sequential_native_surface` | `test_autograd_model` | covered |
| `save_checkpoint` | changed | `test_neural_coverage`, `test_autograd_save_resume` +3 | `test_neural_coverage`, `test_autograd_save_resume` +2 | covered |
| `sigmoid` | changed | `test_neural_coverage`, `test_import_lower_native` +1 | `test_neural_coverage` | **new tests** |
| `softmax` | changed | `test_autograd_model`, `test_sequential_native_surface` +1 | `test_autograd_model` | covered |
| `tanh` | changed | `test_autograd_model`, `test_autograd_usability` +4 | `test_autograd_usability`, `test_sequential_native_checkpoint` +1 | covered |

### `mojolearn.cross_vendor.BatchFeed`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__call__` | new | `test_cross_vendor_worker_native` | `test_cross_vendor_worker_native` | covered |
| `__init__` | new | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | covered |

### `mojolearn.cross_vendor.Coordinator`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_parallel_trainers_coverage`, `test_cross_vendor_admission_native` +1 | `test_parallel_trainers_coverage`, `test_cross_vendor` +2 | covered |
| `drain_events` | new | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `run` | changed | `test_cross_vendor_admission_native`, `test_cross_vendor_coordinator_native` | `test_cross_vendor`, `test_cross_vendor_admission_native` | covered |

### `mojolearn.cross_vendor.LearningRateTable`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__call__` | new | `test_cross_vendor_worker_native` | `test_cross_vendor_worker_native` | covered |
| `__init__` | new | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | covered |

### `mojolearn.cross_vendor.Worker`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_parallel_trainers_coverage`, `test_cross_vendor_admission_native` +1 | `test_parallel_trainers_coverage`, `test_cross_vendor_admission_native` +1 | covered |
| `drain_events` | new | `test_parallel_trainers_coverage`, `test_cross_vendor_worker_native` | `test_parallel_trainers_coverage`, `test_cross_vendor_worker_native` | covered |
| `run` | changed | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | `test_cross_vendor_admission_native`, `test_cross_vendor_worker_native` | covered |

### `mojolearn.cross_vendor`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `fold_pair` | changed | `test_parallel_trainers_coverage`, `test_cross_vendor_native` | `test_parallel_trainers_coverage` | **new tests** |
| `ordered_fold` | changed | `test_parallel_trainers_coverage`, `test_cross_vendor` +3 | `test_cross_vendor_admission_native`, `test_cross_vendor_native` +1 | covered |
| `state_hash` | changed | `test_cross_vendor_native`, `test_cross_vendor_worker_native` | `test_cross_vendor_worker_native`, `test_flat_bytes_without_buffer_protocol` | covered |

### `mojolearn.decomposition.PCA`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `fit` | changed | `test_pca_exact_native_owner`, `test_pca_full_surface` +1 | `test_pca_exact_native_owner`, `test_pca_full_surface` +1 | covered |
| `fit_transform` | changed | `test_pca_exact_native_owner` | `test_pca_exact_native_owner` | covered |
| `inverse_transform` | changed | `test_pca_exact_native_owner`, `test_pca_full_surface` +1 | `test_pca_exact_native_owner`, `test_pca_whitening_surface` | covered |
| `transform` | changed | `test_pca_exact_native_owner`, `test_pca_full_surface` +1 | `test_pca_exact_native_owner`, `test_pca_whitening_surface` | covered |

### `mojolearn`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `import_model` | call path | `test_import_lower`, `test_import_lower_native` +1 | `test_import_lower`, `test_import_lower_native` | covered |

### `mojolearn.model_pool_training.PooledByteLanguageModelTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `from_checkpoint` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `model_ownership` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |

### `mojolearn.models.causal_lm.CausalLM`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `_verify_generation`, `test_generation_native` +5 | `test_generation_native`, `test_models_loader` +4 | covered |
| `allocate_state` | changed | `test_generation_native`, `test_model_native_lifetime` +3 | `test_models_loader`, `test_numeric_profile` +1 | covered |
| `blocks` | changed | `test_generation_native`, `test_model_native_lifetime` +3 | `test_models_loader`, `test_neural_block_native` +2 | covered |
| `forward` | changed | `_verify_generation`, `test_generation_native` +5 | `test_models_loader`, `test_neural_block_native` +2 | covered |
| `generate` | changed | `_verify_generation`, `test_generation_native` +2 | `test_generation_native`, `test_models_loader` +2 | covered |
| `generate_text` | new | `test_generation_native` | `test_generation_native` | covered |
| `load` | changed | `_verify_generation`, `test_generation_native` +5 | `test_generation_native`, `test_models_loader` +3 | covered |
| `parameters` | changed | `test_generation_native`, `test_models_loader` | `test_models_loader`, `test_parallel_causal_lm` | covered |
| `reset_state` | changed | `test_generation_native`, `test_verify_causal_lm` | `test_parallel_causal_lm`, `test_verify_causal_lm` | covered |
| `resume_generation` | new | `_verify_generation`, `test_generation_native` | `test_generation_native`, `test_parallel_causal_lm` | covered |
| `step` | changed | `test_generation_native`, `test_models_loader` +2 | `test_models_loader`, `test_numeric_profile` +1 | covered |
| `stream` | new | `_verify_generation`, `test_generation_native` | `test_generation_native`, `test_parallel_causal_lm` | covered |
| `stream_text` | new | `test_generation_native` | `test_generation_native` | covered |

### `mojolearn.models.causal_lm.CausalLMState`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_verify_causal_lm` | `test_verify_causal_lm` | covered |

### `mojolearn.models.causal_lm.GenerationStream`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `info` | new | `test_models_coverage`, `test_cpu_training_samba` +1 | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `read` | new | `test_models_coverage`, `_verify_generation` +1 | `test_models_coverage`, `test_generation_native` +1 | covered |
| `state_dict` | new | `_verify_generation`, `test_generation_native` +2 | `test_generation_native`, `test_generator_draw_native` +2 | covered |

### `mojolearn.models.config.HFConfig`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__getitem__` | changed | `test_neural_coverage`, `test_byte_checkpoint_native` +26 | `test_neural_coverage`, `test_byte_lm_surface` +21 | covered |
| `__init__` | changed | `test_model_config_native`, `test_models_loader` +1 | `test_model_config_native`, `test_models_loader` | covered |
| `from_json` | changed | `test_model_config_native` | `test_model_config_native` | covered |
| `get` | changed | `test_model_config_native`, `test_models_loader` +2 | `test_model_config_native`, `test_models_loader` +1 | covered |
| `keys` | changed | `test_model_config_native`, `test_neural_numpy_free_merge` +1 | `test_model_config_native`, `test_samba_construct_native` | covered |
| `model_type` | changed | `test_model_config_native`, `test_models_loader` | `test_model_config_native`, `test_models_loader` | covered |
| `to_dict` | changed | `test_byte_checkpoint_native`, `test_model_config_native` +2 | `test_model_config_native`, `test_parallel_byte_native` | covered |

### `mojolearn.models.config.ModelPlan`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_model_config_native` | `test_model_config_native` | covered |
| `checkpoint_names` | changed | `test_model_config_native`, `test_models_loader` +1 | `test_models_loader` | covered |
| `layer_weights` | changed | `test_model_config_native`, `test_models_loader` +1 | `test_models_loader` | covered |

### `mojolearn.models.config`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `plan_for` | changed | `test_model_config_native`, `test_models_loader` +1 | `test_model_config_native`, `test_models_loader` | covered |

### `mojolearn.models.parallel_causal_lm.ParallelCausalLM`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_models_coverage`, `test_parallel_causal_lm` | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `__exit__` | changed | `test_models_coverage`, `test_parallel_causal_lm` | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `__init__` | changed | `test_models_coverage`, `test_parallel_causal_lm` | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `close` | changed | `test_models_coverage`, `test_parallel_causal_lm` | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `load` | changed | `test_models_coverage`, `test_parallel_causal_lm` | `test_models_coverage`, `test_parallel_causal_lm` | covered |
| `parameters` | changed | `test_models_coverage` | `test_models_coverage`, `test_parallel_causal_lm` | **new tests** |

### `mojolearn.models.safetensors.Checkpoint`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +1 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `__exit__` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +1 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `__init__` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +1 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `close` | changed | `test_models_coverage` | `test_models_coverage` | **new tests** |
| `info` | changed | `test_models_coverage` | `test_models_coverage` | **new tests** |
| `names` | changed | `test_checkpoint_native_loading`, `test_models_loader` | `test_checkpoint_native_loading`, `test_models_loader` | covered |
| `open` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +1 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `read` | changed | `test_models_coverage`, `test_models_loader` | `test_models_coverage`, `test_models_loader` | covered |

### `mojolearn.models.safetensors.SafetensorsFile`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +2 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `__exit__` | changed | `test_models_coverage`, `test_checkpoint_native_loading` +2 | `test_models_coverage`, `test_checkpoint_native_loading` +1 | covered |
| `__init__` | changed | `test_models_coverage`, `test_models_loader` +1 | `test_models_coverage`, `test_models_loader` | covered |
| `close` | changed | `test_models_coverage`, `test_safetensors_f16_py310` | `test_models_coverage` | **new tests** |
| `info` | changed | `test_models_coverage` | `test_models_coverage` | **new tests** |
| `names` | changed | `test_checkpoint_native_loading`, `test_models_loader` | `test_checkpoint_native_loading`, `test_models_loader` | covered |
| `read` | changed | `test_models_coverage`, `test_models_loader` +1 | `test_models_coverage`, `test_models_loader` | covered |

### `mojolearn.models.safetensors.TensorInfo`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_models_coverage`, `test_import_onnx` +1 | `test_models_coverage` | **new tests** |
| `nbytes` | changed | `test_models_coverage`, `test_import_onnx_native` +1 | `test_models_coverage` | **new tests** |
| `size` | changed | `test_models_coverage`, `test_host_model_kmeans` +2 | `test_models_coverage`, `test_byte_lm_surface` +1 | covered |

### `mojolearn.models.tokenizer.Tokenizer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_generation_native`, `test_models_loader` +1 | `test_generation_native`, `test_models_loader` +1 | covered |
| `decode` | changed | `test_models_loader` | `test_generation_native`, `test_models_loader` | covered |
| `decode_bytes` | changed | `test_models_loader` | `test_generation_native` | covered |
| `encode` | changed | `test_models_loader`, `test_tokenizer_native_loading` | `test_generation_native`, `test_models_loader` +1 | covered |
| `encode_bytes` | changed | `test_models_loader` | `test_generation_native`, `test_models_loader` | covered |
| `from_pretrained` | changed | `test_models_loader`, `test_tokenizer_native_loading` | `test_models_loader`, `test_tokenizer_native_loading` | covered |
| `pretokenize` | changed | `test_models_loader` | `test_models_loader` | covered |

### `mojolearn.models.tokenizer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `pattern_name` | changed | `test_models_coverage` | n/a: contract answers None for any unknown or non-str spelling | **new tests** |
| `pretokenize` | changed | `test_models_loader`, `test_tokenizer_surface` | `test_models_loader` | covered |

### `mojolearn.neural_inference.MLPInference`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `from_checkpoint` | changed | `test_neural_inference` | `test_neural_inference` | covered |

### `mojolearn.neural_inference.SambaInference`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_neural_coverage`, `test_neural_inference` +1 | `test_neural_coverage`, `test_neural_inference` +1 | covered |
| `allocate_state` | changed | `test_neural_coverage`, `test_neural_inference` | `test_neural_coverage` | **new tests** |
| `forward` | changed | `test_neural_inference` | `test_neural_inference` | covered |
| `from_checkpoint` | changed | `test_neural_inference` | `test_neural_inference` | covered |
| `loss` | changed | `test_neural_coverage`, `test_neural_inference` +1 | `test_neural_coverage`, `test_neural_inference` +1 | covered |
| `parameters` | changed | `test_neural_coverage` | `test_neural_coverage` | **new tests** |
| `step` | changed | `test_neural_inference` | `test_neural_inference` | covered |
| `train_step` | changed | `test_neural_inference`, `test_samba_cpu_inference_native` | `test_neural_inference`, `test_samba_cpu_inference_native` | covered |

### `mojolearn.offload_training.OffloadedByteLanguageModelTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `from_checkpoint` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |

### `mojolearn.ops.Program`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | changed | `test_autograd_public`, `test_program_constructor_native` | `test_program_constructor_native` | covered |
| `distribute` | new | `_verify_distributed_program`, `test_distributed_program` +1 | `test_distributed_program`, `test_distributed_program_boundaries` | covered |
| `fit` | new | `test_models_coverage`, `_verify_distributed_program` +7 | `test_models_coverage`, `_verify_graph` +4 | covered |
| `fit_if` | new | `_verify_graph` | `_verify_graph` | covered |
| `free` | changed | `_verify_graph`, `test_ops_program` | `_verify_graph`, `test_ops_program` | covered |
| `inspect` | new | `_verify_graph`, `test_ops_transformer_program` | `_verify_graph`, `test_ops_transformer_program` | covered |
| `partition` | new | `_verify_partitioned_program`, `test_distributed_program_boundaries` +1 | `test_distributed_program_boundaries`, `test_partitioned_program` | covered |
| `profile` | new | `test_models_coverage`, `_verify_generation` +3 | `test_models_coverage`, `_verify_graph` | covered |
| `run` | changed | `_verify_distributed_program`, `_verify_graph` +11 | `_verify_graph`, `test_autograd_usability` +3 | covered |
| `run_dynamic` | new | `test_models_coverage`, `_verify_graph` +1 | `test_models_coverage`, `_verify_graph` +1 | covered |
| `run_if` | new | `_verify_graph` | `_verify_graph` | covered |
| `save` | changed | `_verify_distributed_program`, `_verify_generation` +9 | `_verify_graph`, `test_distributed_program` +4 | covered |
| `save_control` | new | `_verify_graph` | `_verify_graph` | covered |
| `vjp_if` | new | `_verify_graph` | `_verify_graph` | covered |
| `vjp_while` | new | `_verify_graph` | `_verify_graph` | covered |
| `while_loop` | new | `_verify_graph` | `_verify_graph` | covered |

### `mojolearn.ops.Session`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_models_coverage`, `test_ops_session_coverage` +28 | `test_models_coverage`, `test_ops_session_coverage` +22 | covered |
| `__exit__` | changed | `test_models_coverage`, `test_ops_session_coverage` +28 | `test_models_coverage`, `test_ops_session_coverage` +22 | covered |
| `__init__` | changed | `test_models_coverage`, `test_ops_session_coverage` +27 | `test_models_coverage`, `test_ops_session_coverage` +19 | covered |
| `abs` | changed | `test_models_coverage`, `test_ops_session_coverage` +6 | `test_models_coverage`, `_verify_graph` | covered |
| `accuracy` | changed | `test_ops_session_coverage`, `test_autograd_public` +1 | `test_ops_session_coverage` | **new tests** |
| `add` | changed | `_verify_graph`, `test_autograd_public` +10 | `_verify_graph`, `test_import_lower` | covered |
| `all` | changed | `test_models_coverage`, `_verify_graph` +14 | `test_models_coverage`, `_verify_graph` +4 | covered |
| `any` | changed | `_verify_graph`, `test_autograd_public` +4 | `test_byte_lm_surface`, `test_ops_transformer_program` | covered |
| `argmax` | changed | `_verify_graph`, `test_import_lower` +2 | `_verify_graph`, `test_ops_program` | covered |
| `avg_pool2d` | changed | `test_ops_session_coverage`, `test_import_lower` +2 | `test_ops_session_coverage` | **new tests** |
| `backward` | changed | `_verify_graph`, `test_autograd_save_resume` +9 | `_verify_graph`, `test_ops_higher` +1 | covered |
| `bce_with_logits` | changed | `test_ops_session_coverage`, `test_autograd_model` | `test_ops_session_coverage` | **new tests** |
| `capture` | changed | `test_models_coverage`, `_verify_distributed_program` +11 | `test_models_coverage`, `_verify_graph` +6 | covered |
| `ceil` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `clip` | changed | `test_ops_session_coverage`, `test_import_lower` +1 | `test_ops_session_coverage`, `test_ops_import_kinds` | covered |
| `close` | changed | `test_models_coverage`, `test_ops_session_coverage` +5 | `test_models_coverage`, `test_ops_session_coverage` +2 | covered |
| `concat` | changed | `_verify_graph`, `test_import_lower` +2 | `_verify_graph` | covered |
| `conv2d` | changed | `test_autograd_model`, `test_import_lower` +3 | `test_autograd_model`, `test_ops_rankn` +1 | covered |
| `copy` | changed | `test_models_coverage`, `_verify_graph` +8 | `test_models_coverage`, `_verify_graph` +1 | covered |
| `cos` | changed | `_verify_graph`, `test_ops_import_kinds` +1 | `_verify_graph`, `test_ops_import_kinds` | covered |
| `cross_entropy` | changed | `test_autograd_model`, `test_autograd_usability` +3 | `test_autograd_model`, `test_autograd_usability` | covered |
| `cumsum` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph` | covered |
| `detach` | changed | `_verify_graph`, `test_distributed_program_boundaries` +2 | `_verify_graph`, `test_distributed_program_boundaries` +1 | covered |
| `div` | changed | `test_ops_session_coverage`, `test_import_lower` +1 | `test_ops_session_coverage` | **new tests** |
| `dropout` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +5 | `test_ops_session_coverage`, `_verify_graph` | covered |
| `elu` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage`, `test_ops_import_kinds` | covered |
| `embedding` | changed | `_verify_graph`, `test_autograd_model` +3 | `_verify_graph`, `test_autograd_model` +2 | covered |
| `equal` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +2 | `test_ops_session_coverage` | **new tests** |
| `erf` | changed | `test_ops_import_kinds` | `test_ops_import_kinds` | covered |
| `execution_profile_begin` | new | `_verify_graph` | `_verify_graph` | covered |
| `execution_profile_end` | new | `_verify_graph` | `_verify_graph` | covered |
| `exp` | changed | `test_ops_session_coverage`, `_verify_graph` +5 | `_verify_graph`, `test_higher_rank_native` | covered |
| `expand` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph` | covered |
| `flatten` | changed | `test_autograd_model`, `test_ops_program` +2 | `test_autograd_model`, `test_weight_import` | covered |
| `floor` | changed | `test_ops_import_kinds` | `test_import_lower` | covered |
| `free` | changed | `test_ops_session_coverage`, `_verify_graph` +2 | `test_ops_session_coverage`, `_verify_graph` +2 | covered |
| `full` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +12 | `test_ops_session_coverage`, `_verify_graph` +4 | covered |
| `gather_rows` | changed | `test_ops_session` | `test_ops_session` | covered |
| `gelu` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph`, `test_ops_import_kinds` | covered |
| `grad` | changed | `_verify_graph`, `test_autograd_save_resume` +8 | `_verify_graph`, `test_ops_higher` +1 | covered |
| `grad_tensor` | changed | `_verify_graph`, `test_higher_rank_native` +1 | `_verify_graph`, `test_ops_higher` | covered |
| `gradient_report` | new | `_verify_graph` | `_verify_graph` | covered |
| `greater` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph` | covered |
| `greater_equal` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `hard_sigmoid` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage`, `test_ops_import_kinds` | covered |
| `hard_swish` | changed | `test_ops_import_kinds` | `test_ops_import_kinds` | covered |
| `huber` | changed | `test_models_coverage`, `test_ops_session_coverage` +2 | `test_models_coverage`, `test_ops_session_coverage` | **new tests** |
| `hvp` | changed | `_verify_graph`, `test_higher_rank_native` +1 | `_verify_graph`, `test_higher_rank_native` | covered |
| `ids` | changed | `_verify_graph`, `test_autograd_model` +5 | `_verify_graph`, `test_autograd_model` +4 | covered |
| `index` | changed | `_verify_graph`, `test_ops_import_kinds` +1 | `_verify_graph`, `test_byte_lm_surface` +2 | covered |
| `jvp` | changed | `_verify_graph`, `test_higher_rank_native` +1 | `_verify_graph`, `test_higher_rank_native` | covered |
| `layer_norm` | changed | `_verify_graph`, `test_autograd_model` +5 | `_verify_graph`, `test_ops_higher` +3 | covered |
| `leaky_relu` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage`, `test_ops_import_kinds` | covered |
| `learning_rate` | changed | `test_ops_session_coverage`, `_verify_graph` +1 | `_verify_graph` | covered |
| `less` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph` | covered |
| `less_equal` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `linear` | changed | `_verify_distributed_program`, `_verify_graph` +14 | `_verify_graph`, `test_autograd_model` +5 | covered |
| `load_control` | new | `_verify_graph` | `_verify_graph` | covered |
| `load_distributed` | new | `_verify_distributed_program`, `test_distributed_program` +1 | `test_distributed_program`, `test_distributed_program_boundaries` | covered |
| `load_optimizer_state` | changed | `_verify_graph`, `test_ops_optimizer` | `_verify_graph` | covered |
| `load_partitioned` | new | `_verify_partitioned_program`, `test_distributed_program_boundaries` +1 | `test_distributed_program_boundaries`, `test_partitioned_program` | covered |
| `load_program` | changed | `_verify_graph`, `_verify_model` +3 | `_verify_graph`, `test_import_lower` +1 | covered |
| `load_state` | changed | `_verify_graph`, `test_autograd_save_resume` | `_verify_graph` | covered |
| `load_weights` | changed | `_verify_graph`, `test_autograd_public` +3 | `_verify_graph`, `test_torch_native_modes` +1 | covered |
| `log` | changed | `test_ops_session_coverage`, `test_autograd_usability` +3 | `test_autograd_usability` | covered |
| `log_loss` | changed | `test_ops_session_coverage`, `test_ops_session` | `test_ops_session_coverage` | **new tests** |
| `log_softmax` | changed | `test_ops_program`, `test_ops_rankn` | `test_ops_program` | covered |
| `logical_and` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `logical_not` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `logical_or` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `mae` | changed | `test_ops_session_coverage` | `test_ops_session_coverage` | **new tests** |
| `matmul` | changed | `test_models_coverage`, `_verify_distributed_program` +13 | `test_models_coverage`, `_verify_graph` +8 | covered |
| `max` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +13 | `test_ops_session_coverage`, `_verify_graph` +1 | covered |
| `max_pool2d` | changed | `test_autograd_model`, `test_import_lower` +2 | `test_autograd_model` | covered |
| `maximum` | changed | `test_models_coverage`, `test_ops_session_coverage` +5 | `test_models_coverage`, `_verify_graph` +1 | covered |
| `mean` | changed | `test_models_coverage`, `test_ops_session_coverage` +10 | `test_models_coverage`, `_verify_graph` +2 | covered |
| `minimum` | changed | `test_ops_session_coverage`, `test_import_lower` +1 | `test_ops_session_coverage` | **new tests** |
| `mse` | changed | `test_autograd_model`, `test_autograd_save_resume` +5 | `test_autograd_model`, `test_autograd_save_resume` +1 | covered |
| `mul` | changed | `test_models_coverage`, `_verify_graph` +10 | `test_models_coverage`, `_verify_graph` +1 | covered |
| `not_equal` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `numpy` | changed | `test_ops_session_coverage`, `_verify_graph` +19 | `_verify_graph`, `test_autograd_model` +10 | covered |
| `optimizer` | changed | `test_models_coverage`, `test_ops_session_coverage` +9 | `test_models_coverage`, `test_ops_session_coverage` +6 | covered |
| `optimizer_state` | changed | `_verify_graph`, `test_ops_optimizer` | `_verify_graph`, `test_ops_optimizer` | covered |
| `pad` | changed | `test_ops_session_coverage`, `test_import_lower` +2 | `test_ops_session_coverage`, `test_ops_import_kinds` | covered |
| `permutation` | changed | `test_ops_session_coverage`, `test_ops_session` | `test_ops_session_coverage` | **new tests** |
| `permute` | changed | `_verify_graph`, `test_autograd_model` +4 | `_verify_graph`, `test_ops_rankn` +1 | covered |
| `pow` | changed | `test_ops_session_coverage`, `test_import_lower` +1 | `test_ops_session_coverage` | **new tests** |
| `pow_scalar` | changed | `test_ops_session_coverage`, `test_import_lower` +1 | `test_ops_session_coverage` | **new tests** |
| `profile_report` | changed | `test_ops_session_coverage` | `test_ops_session_coverage` | **new tests** |
| `profile_reset` | changed | `test_ops_session_coverage` | `test_ops_session_coverage` | **new tests** |
| `reciprocal` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph`, `test_ops_import_kinds` | covered |
| `relu` | changed | `test_models_coverage`, `_verify_graph` +8 | `test_models_coverage`, `_verify_graph` +3 | covered |
| `reset` | changed | `_verify_graph`, `test_autograd_save_resume` +4 | `_verify_graph` | covered |
| `reshape` | changed | `test_models_coverage`, `test_ops_session_coverage` +20 | `test_models_coverage`, `_verify_graph` +9 | covered |
| `rms_norm` | new | `_verify_graph`, `test_ops_transformer_program` | `_verify_graph`, `test_ops_transformer_program` | covered |
| `roc_auc` | changed | `test_ops_session_coverage`, `test_ops_session` | `test_ops_session_coverage` | **new tests** |
| `rotary` | new | `_verify_graph`, `test_ops_transformer_program` | `_verify_graph`, `test_ops_transformer_program` | covered |
| `round` | changed | `test_models_coverage`, `test_ops_session_coverage` +3 | `test_models_coverage`, `test_ops_session_coverage` | **new tests** |
| `run` | changed | `_verify_distributed_program`, `_verify_graph` +14 | `_verify_graph`, `test_autograd_usability` +3 | covered |
| `save_state` | changed | `_verify_graph`, `test_autograd_save_resume` | `_verify_graph` | covered |
| `scalar` | changed | `test_models_coverage`, `test_ops_session_coverage` +1 | `test_ops_session_coverage`, `test_byte_lm_surface` | covered |
| `scale` | changed | `test_ops_session_coverage`, `_verify_graph` +5 | `test_ops_session_coverage`, `_verify_graph` +1 | covered |
| `scheduler_step` | changed | `test_ops_session_coverage`, `test_autograd_usability` +1 | `test_ops_session_coverage` | **new tests** |
| `shape` | changed | `test_models_coverage`, `test_ops_session_coverage` +21 | `test_models_coverage`, `_verify_graph` +10 | covered |
| `sigmoid` | changed | `test_ops_import_kinds` | `test_ops_import_kinds` | covered |
| `sign` | changed | `test_ops_session_coverage`, `test_ops_import_kinds` | `test_ops_session_coverage` | **new tests** |
| `silu` | changed | `_verify_graph`, `test_ops_import_kinds` | `_verify_graph`, `test_ops_import_kinds` | covered |
| `sin` | changed | `_verify_graph`, `test_ops_import_kinds` +1 | `_verify_graph`, `test_ops_import_kinds` | covered |
| `softmax` | changed | `_verify_graph`, `test_autograd_model` +6 | `_verify_graph`, `test_autograd_model` | covered |
| `softplus` | changed | `test_ops_import_kinds` | `test_ops_import_kinds` | covered |
| `split` | changed | `test_models_coverage`, `_verify_distributed_program` +9 | `test_models_coverage`, `_verify_graph` +2 | covered |
| `sqrt` | changed | `_verify_graph`, `test_import_lower` +4 | `_verify_graph` | covered |
| `squeeze` | changed | `_verify_graph`, `test_ops_rankn` | `_verify_graph`, `test_ops_rankn` | covered |
| `step` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +11 | `_verify_graph`, `test_distributed_program` +3 | covered |
| `sub` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +5 | `test_ops_session_coverage` | **new tests** |
| `sum` | changed | `test_ops_session_coverage`, `_verify_distributed_program` +17 | `_verify_graph`, `test_autograd_model` +4 | covered |
| `take` | changed | `_verify_graph`, `test_ops_import_kinds` +2 | `_verify_graph`, `test_ops_rankn` | covered |
| `tanh` | changed | `_verify_distributed_program`, `_verify_graph` +9 | `_verify_graph`, `test_autograd_usability` | covered |
| `tensor` | changed | `test_models_coverage`, `test_ops_session_coverage` +25 | `test_models_coverage`, `test_ops_session_coverage` +18 | covered |
| `transpose` | changed | `_verify_graph`, `test_ops_layernorm_view` +7 | `_verify_graph`, `test_ops_rankn` +3 | covered |
| `unsqueeze` | changed | `test_ops_session_coverage`, `test_ops_rankn` | `test_ops_session_coverage` | **new tests** |
| `vjp` | changed | `_verify_graph`, `test_ops_higher` +2 | `_verify_graph`, `test_ops_transformer_program` +1 | covered |
| `where` | changed | `test_models_coverage`, `test_ops_session_coverage` +4 | `test_models_coverage`, `_verify_graph` +2 | covered |
| `zero_grad` | changed | `_verify_graph`, `test_ops_optimizer` | `_verify_graph`, `test_ops_optimizer` | covered |

### `mojolearn.parallel_training.ParallelByteLanguageModelTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_parallel_trainers_coverage`, `_verify_par` | `test_parallel_trainers_coverage` | **new tests** |
| `__exit__` | changed | `test_parallel_trainers_coverage`, `_verify_par` | `test_parallel_trainers_coverage` | **new tests** |
| `__init__` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | covered |
| `apply_gradient` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `checkpoint` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `close` | changed | `test_parallel_trainers_coverage`, `test_cross_vendor_admission_native` +1 | `test_parallel_trainers_coverage`, `test_cross_vendor_admission_native` +1 | covered |
| `export_gradients` | changed | `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `export_raw` | changed | `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `fold_add` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `fold_export` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `fold_reset` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `from_checkpoint` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `has_device_fold` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `optimizer_ownership` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `set_lr` | changed | `test_parallel_byte_native` | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | covered |
| `shard_gradient` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `shard_gradient_fold` | changed | `test_parallel_trainers_coverage` | `test_parallel_trainers_coverage` | **new tests** |
| `state_dict` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage` | **new tests** |
| `step_` | changed | `test_parallel_byte_native` | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | covered |
| `train_step` | changed | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | `test_parallel_trainers_coverage`, `test_parallel_byte_native` | covered |

### `mojolearn.parallel_training.ParallelNeuralTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__enter__` | changed | `test_parallel_trainers_coverage`, `test_cpu_training_par_classical` | `test_parallel_trainers_coverage` | **new tests** |
| `__exit__` | changed | `test_parallel_trainers_coverage`, `test_cpu_training_par_classical` | `test_parallel_trainers_coverage` | **new tests** |
| `__init__` | changed | `test_parallel_trainers_coverage`, `test_cpu_training_par_classical` +2 | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | covered |
| `checkpoint` | changed | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | covered |
| `close` | changed | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | covered |
| `export_gradients` | changed | `test_cpu_training_par_classical`, `test_parallel_neural_admission` | `test_parallel_trainers_coverage`, `test_parallel_neural_native` | covered |
| `from_checkpoint` | changed | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | covered |
| `train_step` | changed | `test_parallel_trainers_coverage`, `test_cpu_training_par_classical` +2 | `test_parallel_trainers_coverage`, `test_parallel_neural_admission` +1 | covered |

### `mojolearn.parallel_training`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `ordered_sum_gradients` | changed | `test_parallel_trainers_coverage`, `test_parallel_mlp_native` | `test_parallel_trainers_coverage` | **new tests** |

### `mojolearn.tokenizer.BpeTokenizer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `__init__` | call path | `test_numpy_free_features`, `test_public_aliases` +2 | `test_numpy_free_features`, `test_tokenizer_manifest` +1 | covered |
| `encode_batch` | changed | `test_tokenizer_surface` | `test_tokenizer_surface` | covered |

### `mojolearn.tokenizer.BpeVocabularyTrainer`

| method | change | real path | fault / refusal | verdict |
|---|---|---|---|---|
| `train` | changed | `test_tokenizer_surface` | `test_tokenizer_surface` | covered |
