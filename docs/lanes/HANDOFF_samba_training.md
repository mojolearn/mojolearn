# HANDOFF: Samba training lane (branch `lane/samba-training`)

Written 2026-09-09 at the orchestrator's wind-down order. Everything below
that says "built" or "ran" names a log path; everything else is NOT RUN.

## What landed (commits, `%h parent %p`)

- `92cd1e0d parent f3d76e8d` Samba training stack: schedules, clause 9.2
  accumulation, neural RNG, SambaStack (first draft, unbuilt at commit time)
- `2b992dbf parent 92cd1e0d` samba leg: ship the Tiny Shakespeare corpus
- the handoff commit (this file, `tools/samba_pod_bootstrap.sh`,
  `tools/samba_pod_run.sh`)

Files:

| file | what |
|---|---|
| `core/philox_neural.mojo` | NEW. Philox4x32-10 stream keyed by (seed, stream id, element index); uniform, normal (Box-Muller through the identical seams), dropout forward/backward; host entry `neural_rng_host` |
| `training/samba_ops.mojo` | NEW. Host-pointer transport for embedding fwd/bwd, RMSNorm fwd/bwd (llama kernel + `bwd_rms_norm`), LM-head GEMM fwd/bwd (OP_NT), balanced-tree accumulate with the clause 9.2 refusal |
| `bindings/_mojolearn_training.mojo` | nine new entries: `embedding_forward/backward`, `rms_norm_forward/backward`, `linear_forward/backward`, `accumulate`, `accumulation_is_aligned`, `neural_rng` |
| `python/mojolearn/_training_impl.py` | `ConstantLR`, `WarmupLinearLR`, `WarmupCosineLR` (exact rational, no libm), `accumulate_grads`, `accumulation_is_aligned`, `Generator` (init + dropout), op wrappers; optimizers take `lr_schedule=` and `accumulation_steps=`, gain `step_accumulated` and `lr_` |
| `python/mojolearn/_samba_impl.py` | NEW. `SambaConfig`, `SambaStack` (forward, loss_and_grads, train_step with accumulation, state_dict, JSON+hex+sha256 checkpoint) |
| `python/mojolearn/training.py`, `__init__.py` | exports |
| `python/mojolearn/tests/test_samba_surface.py` | NEW gate: SCHEDULE, ACCUM, RNG, STACK, STACK-ACCUM, ATTENTION arms |
| `tools/samba_train_run.py` | 64-step training run, checkpoint sha, ms/step |
| `tools/samba_torch_reference.py` | same-shape torch eager fp32 model, ms/step |
| `tools/samba_training_leg.sh` | rent / ship / ssh / extend / terminate one L40S |
| `tools/samba_pod_bootstrap.sh`, `tools/samba_pod_run.sh` | on-pod provisioning and the phased payload (cards, bytelm, surface, samba, torch) |
| `training/IDENTICAL_OPTIMIZER_CONTRACT.md` | clauses 9.3 (device accumulate) and 9.4 (schedule) |
| `training/CHECKPOINT_FORMAT.md` | section 11, the stack's JSON checkpoint and the RNG counter |

## Python API added (signatures)

    ConstantLR(peak_lr, warmup_steps=0)               .lr_at(t) .bits_at(t) .config()
    WarmupLinearLR(peak_lr, warmup_steps, total_steps, min_lr=0.0)
    WarmupCosineLR(peak_lr, warmup_steps, total_steps, min_lr=0.0)
    SGD/Adam/AdamW(params, ..., lr_schedule=None, accumulation_steps=1)
    Optimizer.step_accumulated(microbatch_grads, tokens, max_norm=None)
    accumulate_grads(microbatch_grads, tokens, numeric_mode=None)
    accumulation_is_aligned(tokens, accumulation_steps, numeric_mode=None)
    Generator(seed).uniform/normal/kaiming_uniform/xavier_uniform/kaiming_normal/
        xavier_normal, .dropout(x, p, offset=0, stream=None) -> (y, key),
        .dropout_backward(dy, key), .state_dict(), .load_state_dict(), .next_stream()
    embedding_forward(weight, ids), embedding_backward(dy, ids, vocab),
    rms_norm_forward(x, w, eps), rms_norm_backward(dy, x, w, eps),
    linear_forward(a, w), linear_backward(dc, a, w)
    SambaConfig(vocab, d_model, layers, n_heads=None, n_kv_heads=None, head_dim=None,
        intermediate=None, tie_embeddings=True, norm_eps=1e-5, dropout=0.0)
    SambaStack(config, weights=None, generator=None, lr=1e-3, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01, lr_schedule=None, max_norm=None,
        accumulation_steps=1, numeric_mode=None)
      .forward(inputs) .loss(inputs, targets) .loss_and_grads(inputs, targets, num_items=None)
      .train_step(inputs, targets) .state_dict() .load_state_dict(s)
      .save_checkpoint(path) -> sha256   SambaStack.from_checkpoint(path)

## Current attention wiring (2026-09-10)

`TransformerBlock.backward` is implemented and registered in the native
extension. `SambaStack.loss_and_grads` already calls it in reverse layer order;
no additional binding is needed. Its attention blocks use full causal,
zero-state prefill; backward returns the input gradient and all nine weight
gradients. This is IDENTICAL-only and does not supply carried-cache gradients.
The historical status below describes the original draft, not current support.

`python/mojolearn/tests/test_samba_attention_wiring.py` checks the native ABI,
owned output arrays, mixed attention/Mamba3 reverse routing, gradient registry,
and tied/untied embedding accumulation with CPU mocks. Those mocks do not
qualify numerical correctness. The existing `test_samba_surface` ATTENTION arm
runs the real hybrid backward and now requires it rather than treating its
absence as an acceptable refusal. Root must run these checks on the new source.

## Historical status per task

1. LR schedule: WRITTEN. Exact rational spelling (Fraction, Taylor cosine on a
   pi interval, single round-half-even to float32, ftz). Pins in
   `SCHEDULE_PINS` are EMPTY: record the bits the SCHEDULE arm prints and
   paste them in. NOT RUN anywhere.
2. Gradient accumulation: WRITTEN (device op + refusal + optimizer/stack
   plumbing). The ACCUM arm asserts the head and final-norm weight gradients
   at T=512, A=2/4, and REPORTS the embedding and Mamba-3 tensors. NOT RUN.
3. Philox init/dropout: WRITTEN. RNG counter is in the stack's state dict
   and checkpoint. NOT RUN.
4. SambaStack: WRITTEN. Attention layers forward only; `loss_and_grads`
   refuses by name while `TransformerBlock.backward` is absent (checked by
   `hasattr`, so it engages automatically once the transformer lane lands
   it). NOT RUN.
5. L40S: ONE THING RAN. On pod `uwe57f00b6d6tr` (NVIDIA L40S, driver
   580.126.09, torch 2.4.1+cu124, mojo per `/root/samba_out/mojo_version.txt`)
   at commit 2b992dbf, `MOJOLEARN_NUMERIC_MODE=identical bash
   bindings/build_training.sh` EXITED 0 with zero `error:` lines
   (`/root/samba_out/build_training.log`, pod-local, not fetched). The
   mamba / transformer / byte_lm identical builds were in flight at
   wind-down; their exit codes are appended at the bottom of this file if
   they finished inside the lease. No card, no byte-LM checkpoint, no
   surface test, no training run, no torch timing was executed. No
   checkpoint sha exists.

## Numbers actually produced

- Archive size of the shipped tree: 8,577,245 bytes gzipped (with the corpus).
- Apple reference md5s for the diff that is OWED: `training-loss.identical.card`
  a87615d9 (`bench/results/e1/2026-09-04_104052-MacBook-Air-1-terrabyte/lanes/`),
  `training-optimizer.identical.card` 97d160b0 (`.../2026-09-04_034529-...` and
  the NVIDIA `2026-09-03_091511-mojolearn-e2-nv`).
- Byte-LM three-vendor checkpoint sha to reproduce:
  a9858cd59b424b77b897d0171d5f225d63f3f161fcdee02049bd6c04cfd8a6dc.
- Nothing else. No loss, no ms/step, no fingerprint of any new op.

## What I need from the other lanes

- transformer lane: `TransformerBlock.backward(x, grad_output)` returning
  `{"x", <nine weight names>}` like `Mamba3Block.backward`; the stack picks
  it up by attribute.
- mamba lane: nothing new; `Mamba3Block.backward` is consumed as is
  (zero-state prefill, recomputes the forward).

## RUN OWED on the Apple M4 (orchestrator)

    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_mamba.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_training_surface
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_samba_surface
    # existing cards must not move (diff against the md5s above):
    MOJOLEARN_IDENTITY_TRACE=/tmp/tl.card pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/loss_check.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/to.card pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/optimizer_check.mojo
    md5 /tmp/tl.card /tmp/to.card       # want a87615d9..., 97d160b0...
    # STALE as of 2026-09-10: the training lane is IDENTICAL-only and
    # `bindings/build_training.sh` exits 2 for any other tier. The step this
    # replaced ("the FAST and DETERMINISTIC builds must still compile") no
    # longer has a subject. The identical build is the whole gate:
    bash bindings/build_training.sh

## Next commands for a fresh agent, in order

    git checkout lane/samba-training
    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key tools/samba_training_leg.sh rent --gpu "NVIDIA L40S"
    tools/samba_training_leg.sh ship $(git rev-parse HEAD)
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_bootstrap.sh'   # pixi + 4 identical builds
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_run.sh cards'    # md5 vs a87615d9 / 97d160b0
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_run.sh bytelm'   # sha vs a9858cd5...
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_run.sh surface'  # fix, sync, repeat; then paste SCHEDULE pins
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_run.sh samba --steps 64 --layers mamba3,mamba3 --d-model 64'
    tools/samba_training_leg.sh ssh 'bash /root/mojolearn/tools/samba_pod_run.sh torch --steps 64 --layers mamba3,mamba3 --d-model 64'
    tools/samba_training_leg.sh fetch /root/samba_out bench/results/samba/<stamp>-l40s
    tools/samba_training_leg.sh terminate

`tools/samba_training_leg.sh sync` rsyncs edited files without re-shipping
the archive. `training/samba_ops.mojo`, `core/philox_neural.mojo` and the
binding compiled once on the L40S; the surface test has never imported them,
so runtime refusals, wrong buffer sizes and Python-side mistakes are the
likely first findings.

## Bootstrap result on pod uwe57f00b6d6tr (L40S, Mojo 1.0.0 ed45d567, commit 2b992dbf)

    pixi_install_exit=0
    build_training_exit=0      21 s, 0 error lines   (the NEW binding, first compile)
    build_mamba_exit=0         42 s, 0 error lines
    build_transformer_exit=0   18 s, 0 error lines
    build_byte_lm_exit=2       6 ms: bindings/build_byte_lm.sh REFUSES an existing
                               python/mojolearn/identical (created by the training
                               build); run it FIRST. tools/samba_pod_bootstrap.sh
                               now orders it first.

Logs were pod-local (`/root/samba_out/build_*.log`) and the pod was
terminated at the wind-down order (DELETE 204, GET 404) without fetching
them; the exit codes and `grep -c error:` counts above were read over ssh
before termination. Nothing was imported or executed from the built
binaries.
