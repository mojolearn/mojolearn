# Lane afn-mlp: SmallMLPTrainer, Embedding and the CNN on Apple FAST (2026-10-03)

Branch `lane/apple-fast-neural-mlp`. Everything below compiles only under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and its
own `-D MOJOLEARN_AFN_<NAME>` define, default OFF. IDENTICAL and every other
vendor compile main's code unchanged. No bits change anywhere outside FAST on
Apple. Nothing here was run; the M3 manager measures (`docs/apple-fast/ab-neural/mlp.txt`).

## A. SmallMLPTrainer (board lane `mlp-train-step`, binding `training`)

Public call: `mojolearn.SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y)`, 8 -> 16 -> 3
ReLU, mean cross-entropy, AdamW, batch 1..256 rows, `python/mojolearn/_mlp_impl.py`.
Under FAST the step is one binding call, `mlp_train_step` -> `training/mlp_ops.mojo::mlp_train_step_host`.

### Profile of main's step (mode 2, train; counted from `mlp_ops.mojo` and its callees)

| phase | launches | waits | host<->device copies | allocations |
|---|---|---|---|---|
| transport in | 0 | 0 | 6 uploads (x, y, w1, b1, w2, b2) + 2 (m, v) | x, y, p (+4 views), ws, pre1, act, pre2, logits |
| forward | `_fast_vendor_gemm` x 2 (OP_NT, vendor kernel: 1+ launch each) + `_mlp_kernel` x 2 | 0 | 0 | 0 |
| logits refusal scan | `device_first_nonfinite`: 1 launch | 1 (its partials come back) | 1 readback | part, host buffer |
| logits down | 0 | 0 | 1 download | 0 |
| loss | `identical_ce_loss_resident`: ~6 launches (max, denom, logdenom, logp_target, nll, row/total) | 3 (`estimator.mojo:1292, 1341, 1347`) | 2 (row, loss) + the `ones` upload | ~11 (max_v, denom, logdenom, logp_target, nll, logp_sum, smooth, row, total, loss, ones, dlogits) |
| backward | dw2 OP_TN (pinned `identical_gemm_into`: no vendor TN, 1-3 launches), sum_rows, incoming OP_NN (vendor), relu_bwd, dw1 OP_TN (pinned), sum_rows, [dx OP_NN] | 0 | 4 gradient downloads (+dx) | g (+4 views), incoming, dhidden, dx |
| optimizer | `identical_optimizer_step` (no clip): ~3 launches | 1 (`optimizer.mojo:1657`) + its refusal readback | 2 uploads (m, v), 6 downloads (w1..b2, m, v) | m, v, denom_out, q_out, sumsq, norms, total_cell, out2, ws_opt, sab_partials |
| final | 0 | 1 | 0 | 0 |
| **total** | **~20 launches** | **7 waits** | **~24 copies** | **~40 allocations** |

The arithmetic is ~100k FMAs for a 256-row batch. On Metal a launch costs ~20 us host
plus ~0.25 us per live buffer, a wait with a readback ~180 us, so the step is transport.
The two OP_TN GEMMs (dw2, dw1) go to the pinned kernel because `_fast_vendor_gemm`
refuses TN (`gemm_identical.mojo:8720`), i.e. FAST runs the IDENTICAL fold for them.

### Candidates (training/mlp_fast.mojo, hooked at mlp_ops.mojo `mlp_train_step_host` and the binding)

1. **`-D MOJOLEARN_AFN_MLP_FUSED_STEP`** (`MLP_FUSED_STEP`). `mlp_fused_kernel`: one thread per
   row, 64 rows per block (grid <= 4), the 195 weights in threadgroup memory; forward, softmax
   cross-entropy (exact `exp`/`log`, max-subtracted), dlogits, ReLU mask, dhidden, dx in
   registers; the block stages x, act, dlogits, dhidden, nll in threadgroup memory and folds its
   partials (dw1, db1, dw2, db2, the loss share, each pre-scaled by 1/rows) into one 196-float
   slot. `mlp_fold_adam_kernel`: cell i sums the <= 4 block slots (free order) and applies AdamW
   (`p *= 1 - lr wd; m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps)`,
   the same seams as `optimizer.mojo`'s kernel, unpinned). Per step: **2 launches, 1 wait,
   2 allocations** (one float arena with sub-buffer views for x, p, g, m, v, logits, dx, loss,
   slots; one int32 buffer for y), 8 uploads, 11 downloads. The refusals are main's (raised by
   `mlp_train_step_host` before the hook) plus the host finiteness checks on the downloaded
   outputs. Modes 0/1/2 and `want_input_grad` are honored. No atomics: the block partials are
   folded by the second launch.
2. **`-D MOJOLEARN_AFN_MLP_RESIDENT`** (`MLP_RESIDENT`, implies the fused kernels). A session
   per trainer (`mlp_resident_open` .. `mlp_resident_close`): p, g, m, v, a shadow of p/g/m/v,
   the loss cells, the block slots and the batch staging (x, logits, dx for 256 x cap_k rows)
   in ONE arena, y in one int32 buffer. Per step: 1 copy launch (state -> shadow), 2 uploads
   (x, y), the 2 launches, 6 downloads (loss, logits, 4 gradients), **1 wait, 0 allocations**.
   The weights never travel per step; `state_dict()` downloads them (6 copies, 1 wait) and
   `load_state_dict` reopens. A nonfinite result restores p/m/v from the shadow on the device
   (1 launch, 1 wait) before raising, keeping the trainer's "publish only after success"
   contract. Python (`_mlp_impl.py`) takes this path only when the binding exposes
   `mlp_resident_open` (registered under the define only), so every other build is untouched.
   Deviation from the brief: the public `train_step(X, y)` hands a new batch per call, so X and y
   still upload per step; candidate 3 is where they upload once for k steps.
3. **`-D MOJOLEARN_AFN_MLP_MULTISTEP`** (`MLP_MULTISTEP`, implies RESIDENT). `train_steps(X, y, k)`
   on the trainer (X `(k*batch, 8)`, y `(k*batch,)`): x and y for k minibatches uploaded once,
   sliced per step on the device, 2k launches in one command stream, **1 wait per k steps**,
   per-step losses returned, the last step's logits and gradients. The board lane calls
   `train_step` once per round, so this define shows there only as RESIDENT does; the manager
   times it through the Python call in `ab-neural/mlp.md`.
4. **`-D MOJOLEARN_AFN_MLP_ALL`** = 1 + 2 + 3.

Quality: f32 throughout; the loss and gradients differ from main by reassociation only
(per-block serial fold over rows, then over blocks; the GEMV orders), the optimizer by the
removal of the ftz pins (FAST already removes them). The judge is `tools/neural_fast_quality.py mlp`.

## B. Embedding (binding `embedding`, no board lane)

Public: `mojolearn.Embedding(V, d).forward(ids)` / `.backward(ids, dy, grad=None)`,
`python/mojolearn/embedding.py` -> `bindings/_mojolearn_embedding.mojo` `_forward_run` /
`_backward_run` -> `embedding/checks/embedding_identical.mojo`.

### Profile of main (per call)

| call | launches | waits | copies | allocations |
|---|---|---|---|---|
| forward | W refusal scan (1) + `emb_gather_kernel` (1) | `_upload_f32_ptr` 1, scan 1, `_upload_i32` 1-2, `_forward_run` 2 = **5-6** | W up, ids up, y down (+ scan readback) | d_w, d_ids, d_y, scan part + host buffer |
| backward (PLAN_SCAN, fresh) | seed, counts, run_begin (ONE block of EMB_RUN_BEGIN_THREADS), perm, fold (one thread per (v, j) cell walking its run), pad row = **6** + the dY scan (1) + `emb_refuse_device_ids` | `_zero_f32` 1, `_upload_f32_ptr` 1, scan 1, `_upload_i32` x 4 (ids, counts, run_begin, perm: 1-2 each), `emb_refuse_device_ids` 2, final 1 = **~10** | dY up, ids up, 3 zero int32 lists up (host-allocated, V+V+1+T ints), the ids readback, dW down | d_dw, d_dy, d_ids, counts, run_begin, perm, scan part, host buffers |

The whole W table (V*d floats) uploads on every forward: that is the public API's shape
(the table is a host array) and is not changed here.

### Candidate (embedding/checks/embedding_fast_apple.mojo, hooked in the binding's two runs)

5. **`-D MOJOLEARN_AFN_EMB_ATOMIC_BWD`** (`EMB_ATOMIC_BWD`). Backward: seed (unless carried)
   + ONE `emb_scatter_add_kernel` launch over the T*d cells of dY with a relaxed f32
   `Atomic.fetch_add` into `dW[ids[t], j]` (free order), padding positions skipped, + the pad
   row store: **<= 3 launches**; no counts, run-begin, permutation or run scratch; the ids
   were refused by name on the host (`emb_refuse_ids` in the binding), so the device re-check
   and its two waits go; the uploads stop waiting (the host arrays outlive the call). **2 waits**
   (the dY scan, the final). Forward: uploads without waits, the W scan, a float4 gather
   (`emb_gather4_kernel`, d % 4 == 0) or main's gather, the download: **2 waits**. f32 atomics
   on device memory are the pattern `ensemble/checks/atomic_width_probe.mojo` records on Apple
   (threadgroup-memory float atomics do not compile on Metal; none are used).

## C. The CNN (binding `x_cnn`, no board lane)

Public: `mojolearn.Conv2d` (`forward`, `backward`), `mojolearn.CNNClassifier` (the conv block
path, `x_cnn_conv_block_forward[_r]` / `_backward[_r]`), `python/mojolearn/_expansion_cnn.py`
-> `bindings/_mojolearn_x_cnn.mojo` -> `x_cnn/device.mojo`. FAST builds per
`_backend._CLASSICAL_FAST` (every expansion lane).

### Profile of main's convolution (per layer forward, `_conv_relu_on_device`)

| shape | launches | buffers |
|---|---|---|
| k = C*KH*KW <= 32 and OC*k <= 2048 (the first 3x3 layer on a 3-channel image) | `direct_conv_kernel` (1; Apple only, `DIRECT_CONV`) | yconv (+ cols when saved) |
| every other layer | `im2col` (1) + `device_gemm` OP_NT (`identical_gemm_into[False]` or an `_apple_tuned_plan` candidate: 1-3 launches, the first call of a shape measures candidates with waits) + `conv_out_tiled_kernel` (1) | cols (rows x k), y2 (rows x OC), yconv |
| block forward, no pool | + `relu_fwd_at` (1) | pout |
| block forward, pool | + `relu_maxpool_fwd_at` (1) | pout, idx |

Each entry waits once (`conv2d_forward_m`, `conv_block_forward_into`); the workspace slots are
process-cached (`ws`), the resident path keeps x, w and the outputs on the device.

### Candidate (x_cnn/afn_direct.mojo, hooked in `_conv_relu_on_device` and the no-pool block forward)

6. **`-D MOJOLEARN_AFN_CNN_DIRECT`** (`AFN_CNN_DIRECT`). For the shapes im2col + GEMM served:
   ONE implicit-GEMM launch, `afn_conv_tiled_kernel`. A 256-thread block owns 64 output
   positions x 32 output channels; per 32-tap k chunk it stages the weight tile (32 x 32) and
   the input taps (64 x 32, gathered from NCHW x with the zero padding applied, the tap
   (c, kh, kw) decoded once per thread per chunk, the position's (n, h0, w0) decoded once per
   block) in threadgroup memory (13 KB), each thread accumulating 1 position x 8 channels in
   f32 registers. Epilogue: `conv_out_val` (bias, canon) into yconv and, in the conv block's
   no-pool forward, `relu(.)` into the block output in the same launch (the ReLU launch goes).
   The im2col words are still written to `cols` when the backward reads them (`save_cols`),
   from the staged values. The fold is per-chunk ascending then within the chunk (free order,
   FAST). The backward is unchanged (col2im path; `DEVIATION 5701` keeps the pinned fold for
   the weight gradient). Expected: 3-5 launches -> 1-2 per layer and no rows x k round trip.

## What was NOT done

- No simdgroup_matrix path for the MLP (the whole batch is 256 x 16; launch-bound, not MMA-bound).
- The MLP's optimizer is a tiny mlp-specific AdamW cell update in the fold launch (brief 1:
  "a second tiny launch ... call the existing one for now"); afn-optim's API can replace it.
- The Embedding's per-call W upload stays (public API: the table is a host array).
- No shared-memory float atomics anywhere (Metal refuses them).

## Compile proof

Compiled (no runs): FAST with each define on (one at a time) for the binding it touches, FAST
with every define off, IDENTICAL once per binding; results in the lane's final reply.

Results (head 5c48fd50e code, MOJOLEARN_COMPILE_JOBS=1, MOJOLEARN_SKIP_BUILD_GATE=1 so no kernel-launch
smoke ran; logs in ~/mojolearn-evidence/afn-mlp/build-*.log, summary results.tsv):

| build | mode | rc |
|---|---|---|
| training -D MOJOLEARN_AFN_MLP_FUSED_STEP | fast | 0 |
| training -D MOJOLEARN_AFN_MLP_RESIDENT | fast | 0 |
| training -D MOJOLEARN_AFN_MLP_MULTISTEP | fast | 0 |
| training -D MOJOLEARN_AFN_MLP_ALL | fast | 0 |
| training (no define) | fast | 0 |
| training | identical | 0 |
| embedding -D MOJOLEARN_AFN_EMB_ATOMIC_BWD | fast | 0 |
| embedding (no define) | fast | 0 |
| embedding | identical | 0 |
| x_cnn -D MOJOLEARN_AFN_CNN_DIRECT | fast | 0 |
| x_cnn (no define) | fast | 0 |
| x_cnn | identical | 0 |
