# Tests to Mojo, lane tests-to-mojo-models (2026-11-16)

The 17 files of the `L-models` rows of `TESTS_TO_MOJO_2026-11-15.md`, read function by function. Nothing was compiled
or run (lane rule). Verdicts as in `TESTS_TO_MOJO_tests-to-mojo-classical.md`: KEEP-BOUNDARY (admission, exceptions,
returned objects, hooks, `audit_native_call`, files and checkpoints, copies/ownership), KEEP-NOPORT (native numerics
whose inputs or oracle need Python: `np.random.default_rng`, a Python reference helper, the identity harness's hashed
stream, or an arm the native checks do not run), PORT (moved and deleted).

Totals: 17 files; 0 deleted, 1 shrunk, 16 kept unchanged; 13 Python lines removed, 7 added (routing asserts and a
pointer comment). One new check, `models_checkpoint_widen` (cpu and gpu), registered in
`checks/native_tests/registry.mojo` under a `# lane tests-to-mojo-models` comment;
`tools/ops/native_checks_report.py plan --column cpu` lists it (166 checks).

| file | lines | result | removed | function verdicts and the blocker of each kept native function |
|---|---:|---|---:|---|
| `test_byte_lm_checkpoint_binary.py` | 236 | kept | 0 | KEEP-BOUNDARY 11: file round trips through the public save/load, `state_dict` objects, and refusals of tampered files. The codec (`byte_checkpoint_save/load`) takes a Python dict and a path: no typed native entry to call from a check. |
| `test_byte_lm_host_native_provider.py` | 213 | kept | 0 | KEEP-BOUNDARY 6: each test compares the public call with the binding's own entry under `audit_native_call`, with hook refusal, transactional state and file errors. Parameters come from `default_rng(91)`. The native relations are already units of `byte_lm_host_column`. |
| `test_byte_lm_host_ragged_lifetime.py` | 29 | kept | 0 | KEEP-BOUNDARY 1: it guards the door's ownership of the ids Array through `neural_copy_row_prefixes` (bindings/byte_lm_host_trainer_native_door.mojo:479-491). A check that calls `byte_host_operation` directly skips that door. Weights come from the identity harness's hashed stream (`tools/identity_break.py::_byte_lm_params`). |
| `test_byte_lm_session.py` | 247 | kept | 0 | KEEP-BOUNDARY 12: session lifecycle, export copies, refusals, checkpoints, `run_metadata`. Parameters come from `default_rng(29)`. The full-vs-lean numerics are already in `byte_lm_trainer_session` unit 6. |
| `test_byte_lm_surface.py` | 327 | kept | 0 | KEEP-BOUNDARY 17: registry packing from a dict, copies, snapshot isolation, refusals, the checkpoint envelope, the thread lock, metadata. Parameters come from `default_rng(23)`. No numeric assertion is left that is not about copies or transactions. |
| `test_decode_n_tokens.py` | 188 | kept | 0 | KEEP-BOUNDARY 2 (the Mamba decode's shape, dtype and sync, which `mamba_decode_n_tokens` points to; refusals). KEEP-NOPORT 2: `test_transformer_decode_equals_the_steps` (`transformer_decode_door` holds the relation on the GPU column only, with other inputs and a different 8+4 split; the host arm is Python, `_transformer_impl.py`) and `test_samba_decode_equals_the_steps` (no typed native Samba stack construction from `Generator(1)` that a check can call). |
| `test_decode_sessions_cpu_route.py` | 244 | kept | 0 | KEEP-BOUNDARY 2 (ownership, stale buffers, `load_state`, weights copied at open). KEEP-NOPORT 2 (Transformer session step and forward equal the per-call step and forward): the Transformer host-arm session is Python, so a native check would compare the host binding with itself. |
| `test_mamba3_constructor_native.py` | 35 | kept | 0 | KEEP-BOUNDARY 2: `audit_native_call` on the constructor, metadata, copied weight bytes, the `dt_bias` refusal. |
| `test_mamba_native_public.py` | 221 | kept | 0 | KEEP-BOUNDARY 6, each under `audit_native_call`. Kept native parts: comparisons with `_mamba_block_reference` (a Python oracle) and session-vs-step on every arm. `mamba_decode_sessions` and `mamba3_prefix_step` are `gpu=False`, so the device-session arm is not covered natively. |
| `test_models_loader.py` | 664 | moved to `models_checkpoint_widen`; shrunk | 13 (+7) | PORT 1 (part): the F16 value oracle loop and the BF16 `<< 16` values of `test_safetensors_reader_dtypes_round_trip_and_widen_exactly`, now all 65,536 patterns of each through `core/causal_checkpoint_impl.mojo::checkpoint_widen_halves`. The test keeps the file, metadata, dtype/shape, a one-word routing assert per dtype, the `bits` path, ints and the refusals. KEEP-BOUNDARY 15 (sharded index, plans from configs, refusals, end-to-end loads and generate, tokenizer doors). |
| `test_resident_sampling_policy.py` | 150 | kept | 0 | KEEP-BOUNDARY 6: explicit GPU two-build comparisons (`_build_arms.two_build_compare`), generation `state_dict`/resume, lease release. No check column builds two define sets. |
| `test_samba_attention_wiring.py` | 94 | kept | 0 | KEEP-BOUNDARY 2 (the public backward equals the explicit binding ABI under `audit_native_call`, owned outputs, a missing entry is actionable). KEEP-NOPORT 1: the mixed-stack gradients come from `test_samba_gradients_native._reference`, a Python oracle. |
| `test_samba_cpu_inference_native.py` | 150 | kept | 0 | KEEP-BOUNDARY 4: `audit_native_call` with poisoned Python paths, foreign aliases, checkpoint load, the training refusal. Weights come from `default_rng(412)`. |
| `test_samba_decode_session.py` | 94 | kept | 0 | KEEP-NOPORT 1 (session equals the per-call step; same blocker as the Samba case of `test_decode_n_tokens`). KEEP-BOUNDARY 1 (refusals). |
| `test_samba_inference_native.py` | 269 | kept | 0 | KEEP-BOUNDARY 9: `audit_native_call`, uploading once, hook refusal, ragged padding. Weights come from `default_rng(428)` and the oracle is the Python `_reference`. `mamba3_prefill_session` unit 7 holds the host report rule; it is `gpu=False`. |
| `test_samba_step_native.py` | 209 | kept | 0 | KEEP-BOUNDARY 5: optimizer-owner preservation, refusals, no custom schedule. Inputs come from `default_rng(641/919)`. The oracle is the Python `_reference` and `_reference_update`. |
| `test_tokenizer_fingerprint.py` | 34 | kept | 0 | KEEP-BOUNDARY 2: what is under test is the door building the settings tail from Python fields (`unicode_version` is ignored without a normalizer and appended with NFC; bindings/tokenizer_public_native_door.mojo:1201-1235). `vocab_fingerprint_hex` alone cannot see that. |

## Native entries missing (blockers to fix so more can move)

1. A typed native Samba stack constructor and step/decode that a check can call without Python. Today `SambaStack(cfg,
   generator=Generator(1))` builds its weights in the door. This blocks the session-vs-step tests in
   `test_samba_decode_session` and `test_decode_n_tokens`.
2. A GPU column for `mamba_decode_sessions`, `mamba3_prefix_step` and `mamba3_prefill_session`. They are `gpu=False`,
   so the device decode-session arm is held only by the Python tests.
3. Committed fixture generators for the Python oracles (`_mamba_block_reference`, `test_samba_gradients_native._reference`,
   `test_samba_state_native._reference_update`) and for the `default_rng` parameter streams. Without them no
   bytes-for-bytes port keeps the same inputs.
4. A native Transformer host-arm decode session. While it is Python (`_transformer_impl.py`), session-vs-step on the
   host stays a Python test.
5. A typed entry for the byte-LM checkpoint codec (`byte_checkpoint_save/load` take a Python dict and a path).

## Product defects and doubts seen (reported, not fixed)

- `test_samba_decode_session.py::test_session_steps_and_state_equal_the_per_call_step` failed as PRODUCT in smoke 1
  (`SMOKE_TRIAGE_2026-11-15_tests.tsv:147`): the session's logits bytes differ from the per-call step's.
  Cause (lane samba-decode-logits): the host arm stepped the caller's state and `load_state` was a no-op, so the
  steps after `sync_state` were not discarded; the first differing logits were the first step after `load_state`.
  Fixed by da2ea61f61 (a session-owned state, bindings/samba_decode_native_unpack.mojo `_private`). The device arm
  runs the same `SambaStackProgram._body` ops for both routes (training/samba_program_entry.mojo `_forward` and
  `_decode_run`); no device difference was found statically.
- `test_byte_lm_host_native_provider.py` was PRODUCT in smoke 1 (`assert 1 == 2`). Smoke 2 found it stale (fixed by
  76f808f364).
- `checks/native_tests/registry.mojo` imports and appends `mamba_decode_sessions` twice (lines 110/113 and 302/305),
  so the check runs twice per column. It is not this lane's line to change (shared file).
- Two separate F16 widening implementations ship: `gemm/f16_widen.mojo::f16_bits_to_f32`, used by the checkpoint
  reader, and the linalg `from_f16`, used by `safetensors.widen_f16`. Each now has its own check
  (`models_checkpoint_widen`, `core_safetensors_f16` and `byte_lm_models_loader_f16`). Collapsing them to one is
  owed to a product lane.
- `byte_lm_entry` and `samba_entry_table` (suite checks with `main()`) are not in `registry.mojo`, so CI never runs
  them and they are no evidence of coverage.

Removed: 13 Python lines of value assertions in `test_models_loader.py`, covered by `models_checkpoint_widen`. Nothing
else was removed, because every other native function in these files is blocked as listed above. Verify is
untouched; nothing it imports changed.
