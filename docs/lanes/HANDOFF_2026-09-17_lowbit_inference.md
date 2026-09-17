# HANDOFF, 2026-09-17: low-bit inference, the four-lane fan-out, and what a new session does first

Written at the end of the session that built the bf16 and int8 bridge. Read
this before anything else in this area. The memory index entry is
`mojolearn-lowbit-inference-lane` (Claude's project memory); this file is the
copy that lives in the tree.

## 1. Where we are (main bd095ea1e, 2026-09-17, all pushed)

Everything below is ON MAIN and gated as stated. Nothing here is a three-vendor
claim unless it says three vendors.

| piece | where | gated how |
|---|---|---|
| Two low-bit GEMM profiles, `bf16f32.v1` (bf16 storage, fp32.v1 arithmetic, fused plan at the decode shape) and `int8i32.v1` (power-of-two row scales, exact Int32 accumulation, one exact dequant multiply) | `gemm/IDENTICAL_LOWBIT_CONTRACT.md`, `checks/numerics.mojo` (LOW-BIT STORAGE SEAMS), `gemm/host/gemm_lowbit_oracle.mojo`, `gemm/checks/gemm_lowbit.mojo`, `gemm/checks/gemm_lowbit_check.mojo` | `pixi run check-gemm-lowbit` (10 gates) and the two sabotage tasks that must FAIL; THREE VENDORS on the gate shapes: Apple M4 (`bench/results/identity_break/2026-09-17_lowbit-m4/`), H100 (`bench/results/lowbit/2026-09-17_h100-lowbit-mma/`), MI325X (`bench/results/lowbit/2026-09-17_mi325x-lowbit-mma/`) |
| int8 on the integer matrix units (NVIDIA IMMA, AMD MFMA), equal to the flat kernel and the oracle by construction; Apple and the CPU stay on the flat kernel | `gemm/checks/gemm_int8_mma.mojo`, `checks/kernel_matrix.mojo::lib_int8_matrix_unit_for`, `-D MOJOLEARN_INT8_FORCE_FLAT=1` | `check_int8_mma_matches_flat`, measured on the H100 and the MI325X above |
| The linalg surface: `matmul_bf16`, `matmul_int8`, `to_bf16`, `from_bf16`, `quantize_int8`, `dequantize_int8` on the GPU extension and the CPU host binding; `Array` learned uint16 and int8 | `python/mojolearn/linalg.py`, `_linalg_impl.py`, `bindings/_mojolearn_linalg*.mojo` | `python/mojolearn/tests/test_linalg_lowbit.py` (16, M4) |
| Low-bit WEIGHT STORAGE for every inference class: `mojolearn.lowbit.pack(weights, "bfloat16" or "int8")`; TransformerBlock, Mamba1/2/3Block, their `*Inference` classes, MLPInference, SambaInference, LanguageModelInference (dict form) materialize exactly and run the fp32 path; NumPy-free | `python/mojolearn/lowbit.py` | `test_lowbit_weights.py` (M4); the fourteen `-bf16w`/`-int8w` and `gemm-bf16`/`gemm-int8` lanes, M4 Metal == M4 CPU on every cell (`bench/results/identity_break/2026-09-17_lowbit-m4/`) |
| TransformerBlock option record: rope_theta, rope_scaling (linear, llama3), rope_dim, max_positions, qkv_bias, o_bias, norm (rmsnorm, layernorm, rmsnorm_offset), norm_eps, norm_bias, mlp (swiglu, gelu, gelu_tanh, geglu, geglu_tanh), mlp_bias, qk_norm, attn_softcap; the default record is the frozen profile bit for bit | `transformer/block_options.mojo`, `transformer/checks/transformer_options_check.mojo`, `test_transformer_options.py` | `pixi run check-transformer` unchanged (30/30, 17 cases), `pixi run check-transformer-options` PASS, 60 tests, all M4 only |
| The checkpoint loader `mojolearn.models`: `CausalLM.load(path, weight_format=...)`, safetensors reader, HF config option matrix (llama, mistral, qwen2, qwen3, gemma, gemma2, phi3, mamba, mamba2) refusing by name, greedy `generate`, `Tokenizer.from_pretrained` with the GPT-2, Llama 3 and Qwen 2 pre-tokenization patterns | `python/mojolearn/models/`, its README | `test_models_loader.py` (52, synthetic checkpoints only). NO REAL CHECKPOINT HAS COMPLETED A RUN YET |
| The model leg and timing harness | `bench/model/` (harness, torch twin, diff with the mandated wording), `tools/model_leg/` (RunPod, DigitalOcean, CPU, M4 runners) | `bench/model/tests/test_diff.py` (8); every results cell RUN OWED |
| SmolLM2-360M pinned in the R2 dataset store (`models/SmolLM2-360M`, six files) so a rented box stages it and never downloads (DEVIATION 2704) | `tools/dataset_store.sh`, `bench/results/dataset_store/manifest.tsv` | verified against the pins on push |

## 2. The lanes, and what each still owes

All five lane branches were merged into main and their worktrees removed.

| lane | done | still owed |
|---|---|---|
| lane/identical-lowbit-inference (me) | profiles, surface, packing, 14 lanes, M4 two columns | NVIDIA and AMD columns for the 14 lanes (run `tools/identity_three_columns_leg.sh` on both boxes and diff against the M4 JSONs in `bench/results/identity_break/2026-09-17_lowbit-m4/`) |
| A lane/int8-mma | matrix-unit int8, three vendors on the gate shapes | a CDNA2 (gfx90a) MFMA form; any timing (none is claimed) |
| B1 lane/block-options | all seven axes, default unchanged | NVIDIA and AMD columns for `transformer-options` (`pixi run check-transformer-options` on both boxes); backward under a non-default record (refused by name today) |
| B2 lane/model-loader | loader, matrix, tokenizer patterns, NumPy-free lowbit | a real checkpoint through it (see section 3); Mojo-side Llama 3 / Qwen 2 pre-tokenizer (today cut in Python); SentencePiece families |
| C lane/model-leg | harness, twin, diff, four runners, R2 staging | every column of the results table; the CPU column runner (`run_leg_cpu.sh`) has not been dry-run |

## 3. In flight at the moment of writing

The SECOND H100 model leg (the first refused its own staged model through a
`tr -c` newline bug, fixed in bd095ea1e) was launched from commit bd095ea1e's
parent line and its evidence lands under
`$HOME/mojolearn-evidence/model-leg/2026-09-17_181037-nvidia-h100/`
(`remote/model-leg/status.tsv`, `ours.<label>.json`, `torch.<label>.json`,
`ratio.txt`). Pod name `mojolearn-gemm-nvidia-2026-09-17_141037`, 60-minute
lease, self-terminating. A new session does this FIRST:

1. `cat $HOME/mojolearn-evidence/model-leg/2026-09-17_181037-nvidia-h100/remote/model-leg/status.tsv`
   and `gate.txt`. Every phase 0 except `model` means the loader ran a real
   model; read `ratio.txt` and the per-format hashes in the JSONs.
2. If the pod is somehow still live: `tools/runpod_guard.sh` / `tools/gemm_remote_leg.sh reap <pod-id>`.
3. If the harness or loader failed on the real checkpoint, fix it in the
   integration worktree, commit, and rerun `sh tools/model_leg/run_leg.sh --rent`
   (needs `MOJOLEARN_MODEL_LEG_LOCAL_CARD` pointing at an existing Apple gemm
   card, e.g. `$HOME/mojolearn-evidence/e1g/2026-09-17_172900-nvidia-h100-lowbit-mma/local/apple.card`).
4. When it passes: copy `remote/model-leg/*.json`, `ratio.txt`, `status.tsv`
   into `bench/results/model/2026-09-17_smollm2-360m-h100/` with a README in
   the style of `bench/results/lowbit/2026-09-17_h100-lowbit-mma/README.md`,
   fill the table in `bench/model/README.md`, and commit.

## 4. Where to work

- Integration worktree: `/Users/andrewhendel/mojolearn-wt/integrate`, branch
  `lane/integrate-lowbit-fanout` (== main at bd095ea1e). It carries BUILT
  identical binaries for `_mojolearn`, linalg, transformer, mamba, training
  under `python/mojolearn/identical/` and six host bindings under
  `python/mojolearn/host/`. The shared checkout
  `/Users/andrewhendel/CascadeProjects/mojolearn` is behind and dirty with
  other sessions' work; do not build or commit there.
- `bindings/build_host_family.sh` REFUSES to overwrite an existing `.so` and
  says so only in its full output: `rm` the old file before a host rebuild.
- Tests: `MOJOLEARN_NUMERIC_MODE=identical pixi run -e test python -m pytest ...`
  (pytest lives in the `test` environment only).
- Identity harness on this Mac: ONE Apple lane per invocation (the harness
  refuses a broad Apple matrix by name); the CPU column needs a package copy
  without `identical/` on `PYTHONPATH` and `MOJOLEARN_HOST_DIR` naming the host
  bindings; the CPU column needs `_mojolearn_core_host` for `all_finite_f32`.

## 5. Standing rules that bind this work (Andrew, 2026-09-17)

- Local testing gets AT MOST ONE CPU CORE AND ONE GPU CORE PER LANE
  (`MOJOLEARN_BUILD_JOBS=1`, `nice -n 19`, one process at a time). Prefer
  RunPod for anything heavier. Subagents never run anything.
- "Don't start new lanes": finish what is in flight; report follow-ups
  rather than opening lanes.
- Rented boxes stage data from the R2 store (DEVIATION 2704); never download
  on a box. RunPod balance was $152.93 before the two H100 legs (about $2.69
  per hour each, both terminated early); DigitalOcean MI325X one leg.
- Never write "faster"; the only number is our identical mode against the
  incumbent's fast default, as "X times the incumbent's time".

## 6. What comes after section 3, in order

1. The NVIDIA and AMD columns for the fourteen low-bit lanes and the
   transformer option gate (one `identity_three_columns_leg.sh` run per box;
   `gemm_remote_leg.sh nvidia --payload gemm --rent` with
   `MOJOLEARN_GEMM_LEG_EXTRA=tools/identity_three_columns_leg.sh`, then
   `do_extra_leg.sh amd`), diffed against the M4 JSONs.
2. The AMD and CPU columns of the model leg (`run_leg_amd.sh --rent`,
   `run_leg_cpu.sh --rent`), then `bench/model/diff.py --diff` across four
   columns and `--ratio` per box.
3. The paper (`~/CascadeProjects/mlsys`, main 71b05f8): Section 6.2 still
   claims the 246-variant run on three GPU vendors and the CPU as PLANNED
   (five red macros). That run has NOT been taken; the fourteen new lanes
   raise the count past 246 too. Take the run, fill
   `results/library-wide-verification-2026-09-17.json`, set its status RUN;
   and add the low-bit and real-model results as new records.
4. A bf16 projection INSIDE the blocks with device-resident weights (the
   bandwidth win; today the blocks materialize to fp32 and re-upload), then a
   0.8.7 release carrying the CHANGELOG bullets already written.
