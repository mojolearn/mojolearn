# Two-block byte LM: authored implementation, not a qualification

`training/byte_lm.mojo` is new reusable orchestration source. It has not been compiled or executed by its author. All builds, tests, numerical validation and model work belong exclusively to root on the authorized remote NVIDIA/AMD devices. No Apple testing. No loss reduction, language quality, speed, independent gradient correctness or cross-vendor identity is claimed for this profile.

## Exact callable contract

```mojo
var trainer = ByteTrainer(ctx, initial_params, initial_m, initial_v,
                          momentum_flags, completed_steps, optimizer)
var heldout_loss = byte_eval_loss(ctx, trainer, heldout_token_ids)
var capture = byte_train_step(ctx, trainer, token_ids)
var ck = byte_checkpoint(capture, data_schedule_seed, steps_planned)
```

Construction accepts three host `List[Float32]` arrays of exactly 34,944 cells and `List[Bool]` of length 20. It admits finite state, nonnegative second moments, bounded completed steps, and legal positive-learning-rate AdamW configuration before the first allocation. The caller supplies every initial weight and all resume state. No initializer or data generator runs behind the API. The exact optimizer configuration is owned by the trainer; it must not be mutated between steps. Clipping and SGD options are refused in this first profile. Use the same `DeviceContext` throughout the trainer's lifetime; the module does not enforce cross-context pointer ownership.

Each step accepts a flat `List[Int32]` of exactly 66 integers in `[0,256)`, interpreted as row-major `[2,33]`. A byte maps directly to the same integer token: no UTF-8 character normalization, special tokens, padding, truncation or tokenizer vocabulary remapping. Inputs are the first 32 positions of each row; targets are the next 32. The scalar loss is the mean of 64 cross-entropies. Invalid tokens/configuration are rejected before device work. Existing device state must be downloaded for finite/shape validation; these reads precede numerical work and writes. An exception during numerical work poisons the trainer, so it cannot continue from potentially partial updates. Restore the last successful retained state into a new trainer.

`ByteStepCapture` owns the supplied IDs, pre-step parameters/moments/flags, all 20 **pre-update** gradients, post-step parameters/moments/flags, pre-update mean loss, new completed-step count, optimizer descriptor, fixed profile string, native numeric mode and compiled vendor. Every array is FP32 and must be retained as little-endian raw bits, not decimal float text. A step is intentionally synchronous and downloads full state; it is a correctness fixture, not a throughput path.

The architecture is B2/L32/DM32, four query heads, two KV heads, head width 8, intermediate width 64, vocabulary 256, **two decoder blocks**, RMS epsilon `1e-6`, RoPE theta `10000`, untied embedding and output head, FP32 IDENTICAL only. There is no final RMSNorm after block 1 and no biases/dropout. Each block gets a separate fresh causal KV cache per step, starting at position zero. Forward composition is embedding → block0 → block1 → head → CE. Backward uses the saved block1 input from block0's residual output, then passes block1's input cotangent into block0, then computes embedding gradients. All arithmetic calls existing operators; new orchestration introduces no numerical kernels.

`byte_eval_loss(ctx, trainer, token_ids)` uses the same `_byte_forward_loss` helper as training and invokes no backward or optimizer operation. It updates only scratch/weight views/temporary cache state; parameters, moments, flags and completed step are unchanged. Root must verify those arrays and counters before/after when qualifying evaluation. Each call evaluates the same 64-target batch shape; callers define and retain the ordered heldout batches and aggregation convention. This is the entrypoint for initial/final heldout loss, rather than simulating evaluation with a small learning rate.

## Registry and checkpoint state

`byte_names()`, `byte_offsets()`, `byte_param_count(j)` define the only registry. Flatten matrices in row-major `[out,in]` order.

| IDs | Names | Shapes/counts |
|---|---|---|
| 0 | `embed` | `[256,32]`, 8192 |
| 1–9 | `block0.norm1_w`, `block0.w_q`, `block0.w_k`, `block0.w_v`, `block0.w_o`, `block0.norm2_w`, `block0.w_gate`, `block0.w_up`, `block0.w_down` | `[32]`, `[32,32]`, `[16,32]`, `[16,32]`, `[32,32]`, `[32]`, `[64,32]`, `[64,32]`, `[32,64]` |
| 10–18 | Corresponding `block1.*` names | Same nine shapes |
| 19 | `lm_head` | `[256,32]`, 8192 |

The total is 20 tensors and 34,944 FP32 cells. `byte_checkpoint` fills the existing `Checkpoint` structure from the successful capture, including moments, flags, completed step, names/offsets and exact FP32 optimizer fields. `training/checkpoint.mojo` is unchanged. This profile's expected encoded size is `64+64+32*20+12*34944+16 = 420112` bytes. The codec has no architectural profile field; equal flattened counts alone cannot admit the intended model or schedule.

The adapter must require an adjacent versioned sidecar with the exact `BYTE_PROFILE`, complete dimensions and registry shapes, numeric mode, optimizer bits, completed step, planned steps, raw corpus SHA-256, actual token schedule SHA-256, next batch index/byte offsets, and checkpoint SHA-256. Source inventory, binary hash, runtime/device identity, retained command and successful guard exit evidence belong in run provenance. The sidecar is required before loading state into `ByteTrainer`; the checkpoint's descriptive seed is not sufficient to reconstruct a real-text cursor. Load the native file using `load_checkpoint(path, byte_offsets(), byte_names())`, compare the intended descriptor/sidecar, then pass decoded state into the constructor.

## Remaining integration and qualification

1. Root/API owner: add a bounded adapter or driver supplying actual raw text bytes and deterministic initialized weight arrays. Preserve distinct train/evaluation data and explicit batch ordering; do not claim training progress merely because one batch's loss changes. Serialize every returned capture and checkpoint to new exclusive paths, with the required sidecar and successful guard exit evidence.
2. Root/oracle owner: extend the independent FP64 mathematical graph to **this exact two-block profile** and all 20 gradient tensors, including the block-to-block cotangent. The existing `tools/transformer_training_gradient_oracle.py` admits only the historical one-block B2/L8/V64 profile and must not be silently reused against these arrays. Require effective sign and nonlinear derivative controls before certification.
3. Root only: build a consuming driver with `mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1` through the corresponding NVIDIA/AMD serial guard, with explicit deadline/RSS limits and the actual AMD accelerator target when applicable. No module entrypoint runs automatically. Run one bounded step first, then a short fixed actual-text schedule, retaining complete captures and comparing raw arrays independently of tolerance correctness.
4. Root only: validate same-device resume and opposite-vendor checkpoint transfer/continuation with missing-moments/restarted-step controls. The previous one-block 161008-byte resume harness deliberately refuses this 420112-byte profile and needs a separately versioned adapter. Cross-vendor agreement and language-model learning remain separate claims.

The historical one-block loop and training bindings were not modified by this slice. This module reuses their allocation/copy/download helpers as well as existing numerical operators. Compilation, public integration, file persistence, the larger independent oracle, actual-text runs, learning evaluation, and numerical certification remain unperformed by the implementation subagent.

## Standalone native binding (authored, not executed)

`bindings/_mojolearn_byte_lm.mojo` exports `byte_lm_run(addresses, params)` plus `byte_lm_numeric_mode()`, `byte_lm_vendor()` and `byte_lm_profile()`. It opens no device at import. The Python wrapper must validate all witnesses before numerical work, own every contiguous array for the call, and commit only complete successful outputs.

The address list is exactly `[in_param,in_m,in_v,in_flags_i32,in_ids_i32,out_param,out_m,out_v,out_grad,out_flags_i32,out_loss_f32]`. Parameters are exactly `[action,completed,kind,lr,beta1,beta2,eps,weight_decay,momentum,dampening,nesterov,max_norm]`. Kind 2 is AdamW. Action 1 trains and returns `completed+1`; action 0 evaluates, requires `out_grad=0`, and returns the unchanged completed step. The other output state buffers are required in both modes. Array counts are 34,944 FP32 for parameters/moments/gradients, 20 int32 flags, 66 int32 IDs, and one FP32 loss. Input flags must be exactly 0 or 1. Training admits `0 <= completed < 999999`; evaluation also admits the terminal completed step 999999.

The binding validates bounded address spans, four-byte alignment, integer addition overflow and nulls before dereferencing. Outputs must be disjoint from every input and every other output. All input data is copied into owned host lists and admitted before constructing `DeviceContext`. Arbitrary pointer validity cannot be established from an integer address; correctly allocated/sized arrays remain a required caller contract. Evaluation downloads the actual post-operation device state and refuses any bit, flag or counter mutation rather than simply echoing the input state. Every result is validated and the device scope completes before writes to the fresh caller outputs. No native pointer or device state is retained across calls.

`bindings/build_byte_lm.sh` is a Linux-only, IDENTICAL-only compiler script with exactly two compiler threads, an explicit single GPU architecture and x86-64-v3 host baseline. It never invokes a test, smoke run, import or model. It refuses existing output paths and publishes the new binary with an exclusive hard link. Root must run the script through its remote vendor guard, then separately read the mode/vendor/profile witnesses and retain successful guard exit evidence. For example, on NVIDIA with an actual compatible target:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90 \
MOJOLEARN_BYTE_LM_OUTDIR=/artifacts/byte-lm-native-new \
python3 tools/nvidia_serial_guard.py --seconds 900 --rss-gib 12 -- \
  sh bindings/build_byte_lm.sh
```

Use the corresponding AMD guard and actual `gfx` target for AMD. No Apple execution. This source slice does not register the extension in shared backend or packaging tables; that integration is a separate owner task.
