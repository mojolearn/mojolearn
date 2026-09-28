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
  any CLEAN DISAGREE as a real fix. m2pro came back FAIL on an
  INFRASTRUCTURE error, not a verdict: every training-primitives batchgrad
  cell REFUSED with `Failed to create compute pipeline state ...
  XPC_ERROR_CONNECTION_INTERRUPTED` (the Metal compiler service on that
  Mac). m4-a, m3ultra and do-amd were still queued/working at merge;
  the next session reads them.

## Session 2 (2026-09-27): per-seam sabotage for the family. ON BRANCH, GATE OWED ON A POD

### What landed on lane/neural (pushed, NOT merged)

- **Seam drivers + arms** (`tools/identity_lanes/neural.checks`, `# lanes:` line
  names transformer, mamba1..3, embedding, samba, training-primitives,
  optim-adam-clip, optim-maximize, mlp, byte-lm): transformer, mamba1, mamba2,
  mamba3, embedding, optimizer, loss and maximize drivers, each with a patch
  that flips the DEVICE switch at its USE site. `tools/algos_lane_check.py`
  and `tools/lane_select.py` read the `# lanes:` line (families whose lanes
  live in identity_break itself).
- **maximize=** on SGD/Adam/AdamW (DEVIATION 6200, IDENTITY_PATHS row 202,
  lane `optim-maximize`, `training/maximize.mojo`, seam test
  `python/mojolearn/tests/test_optim_maximize_seam.py`).
- **Fix (6a5a8d67a):** the optim-maximize lane planted its zeros with the
  wrong nesting (`gs` is one tensor list per step), so every GPU arm REFUSED
  with `'list' object has no attribute 'reshape'`.

### Proof so far (NVIDIA A40 pod px7kqp3w26mli8, before its deletion)

- All eight seam drivers: PASS clean, FAIL under their patch, PASS after
  reversal (`/root/s2_family2.out`).
- CLEAN AGREE (cuda vs cpu-intel-xeon-gold-6342, every part compared):
  transformer, mamba1, mamba2, mamba3, embedding, samba, training-primitives,
  optim-adam-clip. optim-maximize REFUSED on the planting bug above (fixed).
- The family-wide sabotage `~/mojolearn-evidence/neural/session2/neural_family_e2e.patch`
  (transformer + mamba RMSNorm fold descending, mamba2 + mamba3 segsum
  descending, embedding backward fold descending, Adam moment lerp) was
  launched at 6a5a8d67a (`/root/s2_family3`); the pod was DELETED (RunPod
  balance negative) before it reported. No result exists.

### Stewards

- 1790542263165-neural-aea8614db9 (Adam-lerp patch, --pass 1): **m4-a**
  CLEAN AGREE on all nine lanes on Metal (its FAIL lists only the seven lanes
  that one patch cannot reach, as expected). **m2pro** FAIL = infrastructure
  (`XPC_ERROR_CONNECTION_INTERRUPTED`, Metal compiler service), not a verdict.
  do-amd and m3ultra had not reported.
- Resubmitted as ONE batched request **1790553937999-neural-6a5a8d67ae**
  (m2pro, m3ultra, m4pro-b, do-amd; --pass 2) at 6a5a8d67a with the family
  patch on transformer, mamba1..3, embedding, samba, training-primitives,
  optim-adam-clip, optim-maximize, mlp. Read it next session.

### OWED ON A POD (exact steps, NVIDIA)

1. `python3 tools/algos_lane_check.py transformer,mamba1,mamba2,mamba3,embedding,samba,training-primitives,optim-adam-clip,optim-maximize,mlp,byte-lm --sabotage ~/mojolearn-evidence/neural/session2/neural_family_e2e.patch --pass 2`
   must read CLEAN AGREE, SABOTAGED DISAGREE, RESTORED AGREE on every lane.
   byte-lm may not be reached by that patch (its binding does not import
   those kernels); if it AGREEs under it, add a byte-lm arm (a seam in
   training/byte_lm*.mojo) and rerun that lane.
2. `python3 -m pytest -x -q tools/test_lane_select.py` (lane_select.py
   changed; the earlier 1 failure was the stale 49-vs-51 count, main now
   says 51, merged in 588afc1c4).
3. `python3 -m pytest -q python/mojolearn/tests/test_host_surface.py`.
4. Existing bits unchanged: the 43-lane before/after column (session 1 tool
   `~/mojolearn-evidence/neural/bitcheck.py`).
5. Then merge lane/neural to main and push in the same command.

## Session 3 (2026-09-28): owed gate, one context per binding, steward findings

NVIDIA box: RunPod H100 80GB `mbbvzsw1kgzr7r` (key `neural`). AMD: the central
box (`tools/amd_central.sh`, slot per job, tree /root/mojolearn-neural).

### Landed

- **core/neural_context.mojo + the five bindings** (training, transformer,
  mamba, embedding, byte LM): ONE process-lifetime DeviceContext per binding
  and tier (`_Global`, x_cnn pattern), replacing 31 per-entry/per-session
  `DeviceContext()` (the byte LM's opt-in DEVIATION 2513 keeper is kept).
  Same kernels, same order, every entry still synchronizes: bit-inert.
  Test: `python/mojolearn/tests/test_neural_repeat.py` (all eleven lanes in
  ONE process per column, `--repeats 2`, GPU and CPU, then the diff).
- **tools/algos_lane_check.py `needed_bindings`**: builds every GPU binding a
  lane CALLS (`binding_use` exports) and its CPU route, not only the families
  it is declared for. samba alone REFUSED every cell (`_mojolearn_mamba.so`
  not built); in multi-lane runs other lanes built it. 0 lanes refuse after.
- **test_lane_select**: `neural_inference.py` pin 41 -> 42 (optim-maximize).
- **Family sabotage** `~/mojolearn-evidence/neural/session3/neural_family_e2e.patch`:
  session 2's patch with a new embedding arm (every contribution x (1+2^-23),
  carry-consistent). The descending-fold arm tripped the lane's own
  carried-vs-unsplit dW check, so every sabotaged embedding cell REFUSED and
  m4pro-b read NOTHING COMPARED.

### Proof (NVIDIA H100)

- `algos_lane_check.py` (11 lanes, `--pass 2`, the session 3 patch, with the
  context change in the tree): 8 seam drivers PASS/FAIL/PASS; CLEAN AGREE,
  SABOTAGED DISAGREE, RESTORED AGREE on transformer, mamba1-3, embedding,
  samba, training-primitives, optim-adam-clip, optim-maximize, mlp, byte-lm.
  RESULT: PASS.
- Existing bits, test_host_surface, test_lane_select, test_neural_repeat: see
  the gate line below.

### Steward findings (coordinator request)

- **do-amd hang, request 1790542263165 (aea8614db9):** stuck 3 h 40 min at
  `sabotaged samba/negative ragged`. gdb: main thread in `sched_yield`
  (libamdhip64) under `DeviceBuffer::~DeviceBuffer` in
  `fused_attention.device_absmax`. rocm-smi: CU occupancy 0, GFX activity flat,
  so no kernel was resident and the host waited on a completion that never came.
  A decomp request (1790542727482) hung in the SAME frame (`lanczos._dot` ->
  DeviceBuffer release) within the same minute (log mtimes 21:41 / 21:42Z).
  **Not reproduced**: the same lane + patch on the pre-fix tree (per-call
  contexts) on the central MI300X ran clean/sabotaged/restored through
  every fixture with no stall. Reading: a device/driver event on the do-amd
  MI325X droplet that took down two unrelated lanes at once, not a neural bug.
  Both hung processes still need killing by the steward owner.
- **m2pro / m3ultra `training-primitives batchgrad` REFUSED** with
  `XPC_ERROR_CONNECTION_INTERRUPTED` (Metal compiler service): one
  `linear_backward` per row, each on a fresh context. Addressed by the
  one-context change; to be confirmed by the next steward request.
- **m4-a on aea8614db9:** sabotage AGREE on the seven lanes the Adam-lerp patch
  cannot reach, as expected; CLEAN AGREE on all nine on Metal.
- **m4pro-b on 1790553937999:** every lane DISAGREEd under the family patch
  except embedding (NOTHING COMPARED, fixed above).

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
