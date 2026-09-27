# neural: progress

Family: transformer block, Mamba-1/2/3, Samba stack, MLP, embedding, byte LM,
tokenizer, training primitives (SGD/Adam/AdamW, cross_entropy,
clip_grad_norm_). Bindings `_mojolearn_{training,mamba,transformer,embedding}`
(+ `_mojolearn_byte_lm`, profile-only).

## Session 1 (2026-09-27): FAST tier + proof of the existing family. DONE

### What landed

- **FAST tier for training, mamba, transformer, embedding.** Build scripts
  take `MOJOLEARN_NUMERIC_MODE=fast` (flat `python/mojolearn/`) or
  `identical` (default); deterministic is refused by name. The PyInit aborts,
  the backward refusals, the MLP op refusal and the multi-GPU refusals now
  refuse DETERMINISTIC only (`GLOBAL_NUMERIC_MODE > NUMERIC_IDENTICAL`); the
  comptime selections (fresh prefill, caller transfer, weight copy, lean
  stages) take `<= NUMERIC_IDENTICAL`. Under IDENTICAL every condition has the
  same truth value as before, so the IDENTICAL code is unchanged by
  construction. `mamba/host/gen/` regenerated (`tools/mamba_host_gen.py`).
- Tier table: the four bindings joined `_backend._CLASSICAL_FAST` (the
  "fast + identical, never deterministic" set), and every packaging reader
  moved with it (build_sets.sh, build_release_wheel.sh, pack_wheel.py, both
  smokes, verify_linux_surface_qualification.py). `IDENTICAL_ONLY_*` lists
  are empty; `packaging/check_ext_lists.py` passes.
- Python: TransformerBlock / Mamba{1,2,3}Block backward and the fresh-prefill
  path accept FAST; SmallMLPTrainer runs in the process tier (IDENTICAL or
  FAST, its saved state records and checks the tier). SambaStack and the
  training primitives were already tier-generic.
- **Root fixes found on the way:**
  1. `_backend.load_set` initialized EVERY binding of a tier. A FAST process's
     buffer helpers load the IDENTICAL base binding through it, which also
     initialized IDENTICAL `_mojolearn_mamba` beside the FAST one; Mojo keys
     `add_type` by type id, so PyInit aborted. mamba, transformer and byte_lm
     (the bindings with `add_type`) now load on first use, and a second tier
     of one of them is refused by name (`_TYPE_REGISTERING`).
  2. FAST `identical_log1p` / `identical_softplus` were `log(1 + x)` /
     `log(exp(x) + 1)`: up to 5.5e-5 relative error near x = -7, which put
     the Mamba-3 `k_last` corpus arm outside the torch-FP32 tolerance under
     FAST. Both use `portable_log1pf` under FAST now (IDENTICAL arms
     untouched). This also moves FAST bits of arima, x_linear, x_prep and GPC
     users (accuracy only improves).
  3. `tools/samba_torch_reference.py` masked the Mamba-3 decay AFTER `exp`,
     so `inf * 0` went NaN at L = 128; it masks before `exp` now.
  4. `test_expose_d_manifest.py::test_packaging_lists_agree` failed on main
     (it did not count the expansion bindings the lists append at run time).
- **DECISION NOT OURS, recorded:** Mamba's former `pinned_mul` sites are
  `identical_mul` since the dedupe lane, i.e. a plain product under FAST.
  Kept as is.
- **Byte LM stays IDENTICAL-only this session.** Its trainer, checkpoint
  schema (`numeric_mode='identical'`) and profile-only packaging are the
  GPT-3 route-B path; a FAST byte LM is a later item (see Next).

### Proof

- **IDENTICAL bit-inert (NVIDIA A40, pod px7kqp3w26mli8):** `identity_break`
  GPU column over 43 neural lanes (every non-par lane the selector attributes
  to the four bindings plus the byte LM lanes) at the base (0ef87dec9) and at
  the branch: 342 lane/fixture cells IDENTICAL x2, 1422 parts IDENTICAL, 0
  differ; the 5 host-only lanes (byte-lm-host-*, language-model-config,
  saved-model-host-infer) REFUSE on a GPU column in both. Evidence:
  `~/mojolearn-evidence/neural/{before,after}.json`, `before_after_diff.txt`.
- **Lane check (NVIDIA pod, merged tree):** training-primitives,
  optim-adam-clip, transformer, mamba1, mamba2, mamba3, embedding, mlp, samba
  all AGREE clean (cuda vs cpu-intel-xeon-gold-6342, every part compared).
  Sabotage `~/mojolearn-evidence/neural/sab_adam_lerp.patch` (the device Adam
  moment in its lerp spelling; the host oracle is untouched): RESULT PASS on
  optim-adam-clip,mlp (AGREE, DISAGREE, AGREE). It does NOT reach the other
  seven lanes (Samba steps AdamW elsewhere), so a family-wide biting sabotage
  per lane is owed (see Next, item 1).
- **Final re-check after the numerics fix and the merge:** the same 43-lane
  GPU column against the base column: 342 cells IDENTICAL x2, 0 differ
  (`before_after_diff_final.txt`; the 5 host-only lanes now hash where the
  base column refused, so they read ONE-COLUMN, not different).
- **FAST surface tests (NVIDIA):** test_mamba_surface, test_transformer_surface
  (backward now held to the float64 oracle under FAST too),
  test_training_surface (ALLOW_FAST), test_embedding_surface,
  test_training_primitives_surface, test_mamba{1,23}_backward_surface GREEN.
- **Quality rule (`tools/neural_fast_quality.py`, NVIDIA A40):** FAST vs
  IDENTICAL from the same init and batches, torch eager fp32 (TF32 off) from
  the same init and batches, 5 seeds each:

  | task | dataset | IDENTICAL held-out | FAST held-out | torch | FAST-IDENT mean (se) | worst seed | verdict |
  |---|---|---|---|---|---|---|---|
  | Samba (mamba3+attention, 300 steps) | enwik8 | 2.37089 | 2.37103 | 2.37089 | +1.4e-4 (1.3e-4) | +0.027% | PASS |
  | Samba | pile_github | 2.83020 | 2.83028 | 2.83020 | +8.5e-5 (1.0e-4) | +0.018% | PASS |
  | SmallMLP 8-16-3 | wine | 0.07831 (acc 0.9593) | 0.07831 (0.9593) | 0.07831 | -5e-9 | 0% | PASS |
  | SmallMLP | digits 0-2 | 0.02457 (acc 0.9901) | 0.02457 (0.9901) | 0.02457 | -1e-9 | 0% | PASS |

  Blocks vs float64 references: all 47 Mamba-1/2/3 corpus cases (ref64 from
  the reference algorithms) and TransformerBlock vs a float64 LlamaDecoderLayer
  at 5 seeds x 2 shapes: FAST error within max(2 x IDENTICAL, torch fp32) on
  every case (all ~1e-7). Evidence: `~/mojolearn-evidence/neural/quality/`.
- **Stewards:** request 1790542263165-neural-aea8614db9 (m2pro, m3ultra,
  do-amd) on the nine lanes with the Adam sabotage, `--pass 1`, PENDING at
  merge (merge gate 0000b: stewards are post-merge release gates). Its RESULT
  will read FAIL for the seven lanes the sabotage cannot reach; read its
  CLEAN rows (Metal / gfx942 vs CPU AGREE) as the Apple/AMD proof and treat
  any CLEAN DISAGREE as a real fix.

## Next phases (one per session)

1. **Verification (charter phase 1) for the existing lanes:** a biting
   per-lane sabotage for every neural lane (transformer, mamba1/2/3,
   embedding, training-primitives, samba, byte-lm) and a `.checks` fragment
   so `--pass 2` runs; one steward request with a patch per lane group.
   Record the steward verdicts of request 1790542263165-neural-aea8614db9.
2. **Option parity vs torch/HF.** Attention variants (SDPA-style masks,
   ALiBi, MQA already via GQA, cross-attention, dropout), norms (LayerNorm
   w/o bias, Gemma offset done), activations (ReLU^2, SiLU MLP), dropout in
   the block and embedding, embedding max_norm/scale_grad_by_freq/sparse,
   optimizer options (Adam amsgrad, maximize, foreach/fused flags as no-ops,
   SGD nesterov/dampening already, weight_decay on Adam), LR schedulers,
   cross_entropy weight=. Start from each module's NOT_IMPLEMENTED.tsv.
   Also: FAST for the byte LM (trainer tier plumbing, checkpoint schema,
   profile-only packaging) without moving its IDENTICAL bits.
3. **FAST GPU speed** (NVIDIA, AMD, Apple), quality rule each change:
   `tools/neural_fast_quality.py` (samba / mlp / blocks + compare) is the
   harness; add the byte LM once it has FAST.
4. **IDENTICAL GPU speed**, same bits, re-proven on every column.
5. **CPU speed** of the host bindings, bits equal at every thread count.

## Pod notes (for a fresh agent)

- `/root/nf` on the pod is a second tree (the branch) whose `.pixi` is a
  symlink to `/root/mojolearn/.pixi`; FAST and IDENTICAL sets are built
  there. torch 2.4.1+cu124, scikit-learn and pytest are in the system
  python3. enwik8 and pile_github are staged from R2 in
  `/root/mojolearn/training/corpus/`.
- `tools/algos_lane_check.sh` must run WITHOUT `MOJOLEARN_GPU_ARCHS` set
  (build_byte_lm_host.sh refuses it); `build_byte_lm.sh` needs it.
- A FAST base binding build (`bindings/build.sh`) fails its own smoke unless
  `MOJOLEARN_SKIP_BUILD_GATE=1` (the smoke imports the identical set).
