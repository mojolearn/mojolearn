# LM training-step memory inventory and probe, September 10, 2026 (DEVIATION 2495)

Source-analysis lane, IDENTICAL only. Nothing here changes arithmetic, fold
order, rounding or tie rules; no test, build or benchmark was run on the
Mac. Every byte count below is host arithmetic from the constructors and
launchers named by file and line; measured peaks are the probe's job
(RUN OWED at the end). The parent handoff is
[section 2 of the classical/AI handoff](HANDOFF_ai_classical_identical_next_2026-09-10.md).

Sources read: `training/byte_lm.mojo`, `training/byte_lm_config.mojo`,
`training/checks/loss.mojo`, `training/checks/optimizer.mojo`,
`training/checks/train_loop.mojo`, `embedding/checks/embedding_identical.mojo`,
`gemm/checks/gemm_identical.mojo`, `gemm/checks/gemm_backward.mojo`,
`gemm/checks/gemm_oracle.mojo`, `transformer/impl/llama/modeling_llama.mojo`,
`transformer/impl/llama/fused_attention.mojo`,
`transformer/checks/transformer_backward.mojo`,
`bindings/_mojolearn_byte_lm.mojo`, `python/mojolearn/_byte_lm_impl.py`,
`python/mojolearn/_byte_lm_config.py`, `tools/lm_training_capacity.py`,
`tools/byte_lm_session_bench.py`, `tools/gemm_remote_leg.sh`.

## 0. Shapes and the one number that must drive budgeting

| Shape | B | L | DM | H | KV | HD | FF | layers | V | parameters | M = B*L |
|---|---|---|---|---|---|---|---|---|---|---:|---:|
| control (Apple pilot) | 1 | 2048 | 384 | 6 | 6 | 64 | 1024 | 8 | 8192 | 20,453,376 | 2048 |
| target | 1 | 2048 | 768 | 12 | 12 | 64 | 2048 | 12 | 50257 | 162,147,840 | 2048 |

Both counts follow `ByteConfig._param_count` (`training/byte_lm_config.mojo:70-82`):
two untied `V*DM` tensors plus, per layer, `2*DM + 2*DM*DM + 2*DM*KV*HD + 3*DM*FF`.
The target count agrees with the handoff (162,147,840, not 125M). The source
confirms no final norm, no biases, RMSNorm/RoPE/SwiGLU.

The control shape is what the session handoff's second pilot ran
(`HANDOFF_lm_session_2026-09-10.md`, "20,453,376 parameters") and is the
probe's default. The control's batch is B1; no source names another batch.

## 1. Inventory of one training step

One step is `byte_train_step` (`training/byte_lm.mojo:465-486`) around
`_byte_step_admitted` (`:553-630`), reached from the binding's `_byte_lm_run`
(`bindings/_mojolearn_byte_lm.mojo:144-303`) and the Python `_run_impl`
(`python/mojolearn/_byte_lm_impl.py:419-481`). All device buffers except the
"transient" rows are allocated once per session by `ByteBuffers.__init__`
(`training/byte_lm.mojo:251-339`) and `ByteTrainer.__init__` (`:409-435`);
under `resident=True` they persist across calls, otherwise they are rebuilt
every call (same sizes, plus the upload cost).

Column key: bytes are for the TARGET shape; the control column follows in
section 2. "DUP" names a second copy of bytes that already exist on the
device; "MIRROR" names a host copy of device state. "up/down every step"
records the transfer that touches the buffer each step.

### 1.1 Device: flat state and optimizer scratch

| Buffer | Allocated at | Shape | dtype | Target MiB | Lifetime | Duplicate? | Traffic every step |
|---|---|---|---|---|---:|---|---|---|
| param | byte_lm.mojo:264 | n_total | f32 | 618.5 | session | authoritative | downloaded 2x (before/after validation, :478,:618) + 1x inside the optimizer refusal |
| grad | byte_lm.mojo:265 | n_total | f32 | 618.5 | session | authoritative | downloaded 1x (:609) + 1x optimizer refusal |
| m_state, v_state | byte_lm.mojo:266-267 | n_total each | f32 | 2 x 618.5 | session | authoritative | downloaded 2x each + 1x optimizer refusal |
| opt_ws | byte_lm.mojo:278 | max_j ws(1,1,count_j) = P(count) <= 1024 | f32 | 0.004 | session | | |
| denom_out, q_out, sumsq, norms, total_cell, out2, sab_partials | byte_lm.mojo:272-281 | 1,1,n_tensors,n_tensors,1,2,SAB_CHUNKS | f32 | < 0.01 | session | | OPT_RECORD_INTERMEDIATES off |

### 1.2 Device: duplicate parameter and gradient storage

| Buffer | Allocated at | Shape | Target MiB | Lifetime | Duplicate of | Why it exists / traffic |
|---|---|---|---:|---|---|---|
| emb_w | byte_lm.mojo:283 | V*DM | 147.2 | session | param[0 : V*DM] | `_copy_into` from param every forward (:513) |
| lm_w | byte_lm.mojo:284 | V*DM | 147.2 | session | param[lm_head slice] | `_copy_into` every forward (:514) |
| dw_emb | byte_lm.mojo:285 | V*DM | 147.2 | session | grad[0 : V*DM] | embedding backward writes it, then `_copy_into` grad (:605) |
| dw_lm | byte_lm.mojo:286 | V*DM | 147.2 | session | grad[lm_head slice] | head dB GEMM writes it, then `_copy_into` grad (:606) |
| weights[layer] x 12: norm1_w, w_q, w_k, w_v, w_o, norm2_w, w_gate, w_up, w_down | byte_lm.mojo:429 -> modeling_llama.mojo:1004-1012 | per-layer parameter count (7,079,424 each) | 324.1 total | session | param block slices | `_unpack_block` D2D copies all nine every forward (:512, :362-374) |
| backward[layer].dw_* x 12 (nine tensors) | transformer_backward.mojo:1820-1857 | per-layer parameter count | 324.1 total | session | grad block slices | `_pack_block` D2D copies into grad after the backward loop (:603-604, :377-389) |

The duplicate state is exactly two extra copies of the non-embedding
parameters (one weights, one gradients) plus two extra copies of both
`V*DM` tensors: 1,237 MiB at the target shape, and about 1.25 GB of
device-to-device copy traffic per step (`_unpack_block` plus the two
`V*DM` copies in, `_pack_block` plus two `V*DM` copies out).

### 1.3 Device: loss and logits (the five B*L*V buffers)

| Buffer | Allocated at | Shape | Target MiB | Lifetime | Written by | Last read by |
|---|---|---|---|---:|---|---|---|
| logits | byte_lm.mojo:292 | M*V | 392.6 | session | head GEMM (:538) | CE L1 max, L2/L3 shift+exp, L6/L7 nll (sabotage arms only), and `ce_refuse_device_inputs` (full host download) |
| ce_shift | byte_lm.mojo:296 | M*V | 392.6 | session | L2 `ce_shift_exp_kernel` | L6/L7 `ce_nll_kernel` (one cell per row: `shift[row*V + y]`) |
| ce_expo | byte_lm.mojo:297 | M*V | 392.6 | session | L3 `ce_shift_exp_kernel` | L4 denom GEMM (forward); L14 `ce_weights_kernel` (backward) |
| ce_weights | byte_lm.mojo:308 | M*V | 392.6 | session | L14 `ce_weights_kernel` (per cell from expo[cell], denom[row]) | L16 `ce_dlogits_kernel` (same cell) |
| ce_dlogits | byte_lm.mojo:309 | M*V | 392.6 | session | L16 `ce_dlogits_kernel` (per cell from weights[cell]) | head dA (:567) and head dB (:569) GEMMs |
| ce_max, ce_denom, ce_logdenom, ce_logp_target, ce_nll, ce_row | byte_lm.mojo:295-305 | M each | 0.05 | session | | |
| ce_logp, ce_logp_sum, ce_smooth, ce_total, ce_loss | byte_lm.mojo:302-307 | 1 each | 0 | session | smoothing placeholders (eps = 0 in `CeConfig.causal_lm`) | |
| ce_ones | byte_lm.mojo:310 | max(V, M) | 0.19 | session | | right operand of L4, left operand of L12 |
| ce_ws | byte_lm.mojo:311 | ws(M,1,V) = M*P(V): P(50257) = 393 | 3.07 | session | | plan SPLIT_16_1X1 for the L4 fold; L12 needs only P(M) = 16 |
| head_ws | byte_lm.mojo:315 | ws(M,V,DM) | 0 (1 float) | session | | (M,V,DM) is a TUNED plan: no workspace |
| head_bwd_ws | byte_lm.mojo:318 | max(ws(M,DM,V), ws(V,DM,M)) | 0 (1 float) | session | | both backward shapes are TUNED plans |

Total loss/logit storage: 5 * M*V*4 = 1,963 MiB (1.917 GiB) at the target,
320 MiB at the control. The handoff's "five B*L*V FP32 buffers" is exact.

### 1.4 Device: activations and attention workspaces (per layer, times layers)

Forward stages (`LlamaDeviceStages.__init__`, `modeling_llama.mojo:1224-1276`,
`lean=True` from `byte_lm.mojo:433`). qw = H*HD = DM and kw = KV*HD = DM at
both shapes (H = KV).

| Per-layer buffers | Shape | Target MiB per layer | x layers |
|---|---|---:|---:|
| norm1_out, o_proj, residual1, norm2_out, down_proj, residual2 | 6 x M*DM | 36.0 | 432.0 |
| q_proj, q_rope, ctxv | 3 x M*qw | 18.0 | 216.0 |
| k_proj, v_proj, k_rope | 3 x M*kw | 18.0 | 216.0 |
| k_cache, v_cache (per-layer packed cache the backward reads) | 2 x B*KV*L*HD | 12.0 | 144.0 |
| gate_proj, up_proj, silu_out, gated | 4 x M*FF | 64.0 | 768.0 |
| amax, denom | 2 x B*H*L | 0.19 | 2.25 |
| qbh, kbh (per-head gather scratch) | 2 x L*HD | 1.0 | 12.0 |
| norm1_sumsq, norm2_sumsq | 2 x M | 0.02 | 0.19 |
| scores, masked, aexp, weights, sbh (LEAN) | 1 float each | 0 | 0 |
| forward subtotal | | 149.2 | 1,790.4 |

Backward stages (`LlamaBackwardStages.__init__`,
`transformer_backward.mojo:1817-1870`, `lean=True` from `byte_lm.mojo:434`):

| Per-layer buffers | Shape | Target MiB per layer | x layers |
|---|---|---:|---:|
| in_d_residual2, d_down_proj_out, d_norm2_out, norm2_dx, d_residual1, d_o_proj_out, d_norm1_out, norm1_dx, d_x, dh, dprod | 11 x M*DM | 66.0 | 792.0 |
| d_attn_ctx, d_q_rope, d_q_proj_out | 3 x M*qw | 18.0 | 216.0 |
| d_k_rope, d_v_proj_out, d_k_proj_out | 3 x M*kw | 18.0 | 216.0 |
| d_mlp_gated, d_silu_out, d_up_proj_out, d_gate_proj_out | 4 x M*FF | 64.0 | 768.0 |
| tmp0, tmp1, tmp2 | 3 x M*max(DM,FF) | 48.0 | 576.0 |
| d_k_cache, d_v_cache | 2 x B*KV*L*HD | 12.0 | 144.0 |
| dw_norm1, dw_q, dw_k, dw_v, dw_o, dw_norm2, dw_gate, dw_up, dw_down | per-layer params | 27.0 | 324.1 (counted in 1.2) |
| norm1_dot, norm2_dot, rstd, dvcoef, ones | 5 x M | 0.04 | 0.47 |
| attn_zdot | B*H*L | 0.09 | 1.12 |
| head_a, head_b | 2 x L*HD | 1.0 | 12.0 |
| d_attn_weights, d_attn_masked, d_attn_scores, d_qk_cell, head_c (LEAN) | 1 float each | 0 | 0 |
| backward subtotal | | 254.1 | 3,049.7 |

Shared, not per layer: `x` and `d_h` (byte_lm.mojo:291,293, M*DM each, 6 MiB
each); `rope.inv_freq/cos/sin` (modeling_llama.mojo:1569-1572, HD/2 + 2*L*HD/2,
0.5 MiB); `prefill_cache.k/v` (modeling_llama.mojo:1128-1129, 2*B*KV*L*HD,
12 MiB, reused by every layer's forward as append scratch: `byte_lm.mojo:528`);
`ids`, `targets` (2*M int32); `emb_counts`, `emb_run_begin`, `emb_perm`
(V + V+1 + M int32, 0.39 MiB).

### 1.5 Device: eager and fallback allocations the memory fit must include

Lean allocation starts the eight quadratic arrays at one element per layer.
They GROW on demand and then stay for the session:

| Fallback | Trigger | Grown at | Per-layer size | Target total |
|---|---|---|---|---:|
| forward scores, masked, aexp, weights + sbh | fused forward returns anything but FUSED_RAN (regime refusal: `regime_product_ok(hd, qmax, kmax)` or `regime_finite(vmax)`, or a CORNER hit), or `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager` | `ensure_attention_stage_capacity` (modeling_llama.mojo:1277-1290, called at :2839) | 4*B*H*L*L + L*L floats = 784 MiB | 9,408 MiB (9.19 GiB) |
| backward d_attn_weights, d_attn_masked, d_attn_scores, d_qk_cell + head_c | fused backward refusal (`regime_product_ok` on q/k and on dctx/v) or eager path; the backward then also calls `ensure_attention_materialized`, which recomputes the FORWARD eager stages (grows the forward set too) | `ensure_backward_attention_capacity` (transformer_backward.mojo:1873-1885) | 784 MiB | 9,408 MiB (9.19 GiB) |

The regime check is data dependent (absmax of q_rope, k_cache, v_cache,
dctx per layer, `fused_attention.mojo:1417-1419, 1502-1507`), so a fit
assessment that admits the lean subtotal only is conditional on every layer
staying in regime for every step. A single refusal in one layer adds 784 MiB
(forward) or 1,568 MiB (forward plus backward) for the rest of the session.
With every layer falling back the device total is 28.4 GiB at the target.

Per-call transient device allocations (freed at return, small):
`identical_gemm` (synchronizing form, `gemm_identical.mojo:2843`) allocates
its own workspace on every layer GEMM. At these shapes every projection GEMM
is a TUNED plan (workspace 1 float); the only split plan is the RMSNorm
weight gradient `(1, DM, M)` OP_NN (`transformer_backward.mojo:2527`), plan
SPLIT_1X256, DM*P(M) = 768*16 = 12,288 floats (48 KiB), twice per layer.
`device_absmax` allocates up to ABSMAX_BLOCKS partials plus a pinned host
buffer per call (three calls per fused forward, four per fused backward),
and `_zero_flag` one float. `identical_gemm_into` callers in byte_lm.mojo
use the pre-sized `head_ws`, `head_bwd_ws`, `ce_ws`.

### 1.6 Host mirrors and readbacks (per step)

Native (`List[Float32]` plus a pinned staging buffer of the same size inside
each `download_f32`, `training/checks/train_loop.mojo:766-790`):

| Host copy | At | Size | Note |
|---|---|---|---|
| before_p, before_m, before_v | byte_lm.mojo:478-480 | 3n | MIRROR for `byte_validate_state`; every step, resident or not |
| ByteStepCapture before_* `.copy()` | byte_lm.mojo:627-628 | 3n | second copy made while the originals are alive |
| grads | byte_lm.mojo:609 | n | MIRROR of grad before the optimizer |
| `opt_refuse_device_inputs`: 4 pinned + 4 Lists | training/checks/optimizer.mojo:1127-1145 (called at :1228) | 8n | full param, grad, m, v download and host scan EVERY step; only `-D MOJOLEARN_OPT_TRUST_INPUTS=1` removes it, and that is a named profile downgrade |
| after_p, after_m, after_v | byte_lm.mojo:618-620 | 3n | MIRROR for post-step validation |
| `ce_refuse_device_inputs`: pinned logits + List copy | training/checks/loss.mojo:1263-1274 (called at :1328) | 2*M*V | full LOGITS download and host scan every step, train and eval (785 MiB at the target) |
| `emb_refuse_device_ids`, ids/targets staging | embedding_identical.mojo:442, byte_lm.mojo:500-501 | O(M) int32 | small |
| binding initial_p/m/v | _mojolearn_byte_lm.mojo:185-187 | 3n | copied out of the Python arrays before any GPU work |
| binding resident admission readbacks | _mojolearn_byte_lm.mojo:234-236 | 3n (sequential) | three `download_f32` + `_require_same_bits` per resident call |
| binding out_p/out_m/out_v/out_g | _mojolearn_byte_lm.mojo:244-251 | 4n | moved from the capture, written to Python at :293-297 |

Python (`mojolearn.Array` copies):

| Host copy | At | Size |
|---|---|---|
| `self._state` parameters, m, v | _byte_lm_impl.py:333-339 | 3n (trainer lifetime) |
| `working = _validate_state(self._state)` | :424 (copies at :246-248) | 3n |
| `before = tuple(tobytes())` | :429 | 3n |
| out_p, out_m, out_v, out_grad | :430-435 | 4n |
| `candidate = _validate_state(...)` | :467 | 3n |
| `gradients = _array(out_grad)` + per-tensor dict | :475-479 | 2n |

Sum of listed host allocations at the target: 17.7 GiB native + 10.9 GiB
Python; they are not all alive at once. A simultaneous-peak estimate during
the optimizer step is about 28n floats (Python 13n live around the native
call, native 15n: initial 3n, before 3n, grads n, optimizer refusal 8n),
17 GiB at the target and 2.2 GiB at the control. The control estimate is
consistent with the recorded two-trainer pilot RSS (4.8 GB for both arms).
The probe records VmHWM to replace this estimate.

## 2. Size evaluation at the two shapes

Group totals (bytes from the inventory; GiB = 2^30):

| Group | Control | Target | Eager or lazy |
|---|---:|---:|---|
| model + optimizer flat state (param, grad, m, v) | 312.1 MiB | 2,474.2 MiB (2.416 GiB) | eager, session |
| duplicate state (emb_w, lm_w, dw_emb, dw_lm, weights[layer]) | 102.0 MiB | 913.0 MiB | eager, session |
| duplicate gradients (backward dw_*) | 54.0 MiB | 324.1 MiB | eager, session |
| activations + attention workspaces (forward and backward stages, x, d_h, caches, rope) | 1,519.6 MiB | 4,528.0 MiB (4.42 GiB) | eager, session |
| loss/logit buffers (five M*V + row vectors, ones) | 320.1 MiB | 1,963.4 MiB (1.917 GiB) | eager, session |
| workspaces (ce_ws, head_ws, opt_ws, emb scratch, rope, prefill cache) | 7.1 MiB | 16.0 MiB | eager, session |
| DEVICE TOTAL, lean fused path | 2,315.0 MiB (2.26 GiB) | 10,218.8 MiB (9.98 GiB) | |
| eager-attention fallback (all layers, forward + backward) | +6,400.0 MiB | +18,816.0 MiB (18.4 GiB) | lazy: grows on refusal, then stays |
| DEVICE TOTAL with full fallback | 8,715.0 MiB (8.51 GiB) | 29,034.8 MiB (28.35 GiB) | |
| host native mirrors (sum of listed) | 2,312.7 MiB | 18,104.5 MiB (17.7 GiB) | per step, transient |
| host Python mirrors (sum of listed) | 1,404.4 MiB | 11,133.8 MiB (10.9 GiB) | per call |

Cross-check against the lean-attention handoff: its "fused allocation
subtotal" for B1/L2048 is 4.33 GiB = flat state 2.416 GiB + five vocabulary
arrays 1.917 GiB; its "materialized fallback" 22.33 GiB adds the eight
B*H*L*L arrays (18 GiB). This inventory reproduces both and adds the
activations (4.42 GiB), duplicate state (1.24 GiB) and the sbh/head_c
L*L arrays (0.38 GiB in fallback), which the handoff excluded by name.

What the memory-fit assessment must include, per the lean-attention handoff:

1. the lean device total (9.98 GiB at the target);
2. the fallback growth (up to 18.4 GiB more), data dependent per layer and
   per step, never released within a session;
3. the per-step host mirrors (about 17 GiB simultaneous at the target),
   which decide host RAM, not device fit, but can fail a step just as well;
4. allocator and context overhead, unknown until measured.

Consequence for the target on an 80 GB device: lean fits with a wide margin,
and even a full eager fallback (28.4 GiB) fits. The fit risk at B1 is the
host (17 GiB of transient mirrors per step) and time, not device memory. At
B8 the same arithmetic gives 5 * 8 * M*V = 15.3 GiB of loss buffers and
144 GiB of fallback, which is the capacity handoff's 161.75 GiB row.

## 3. Reuse and tiling candidates (fold preserved), ranked by target bytes saved

Contract facts the arguments rest on:

- GEMM: the leaf partition `(L, P)` is a function of `k` only
  (`contract_partition`, `gemm_identical.mojo:254`); one accumulator per
  output cell walks `p` ascending; the execution plan, `m`, `n` and buffer
  identity cannot reach the arithmetic (contract 6.1, 7.1). The gate
  `check_device_is_batch_invariant` asserts the same bits through a dirty
  shared workspace at four batch sizes.
- CE: "Nothing between L1 and L11 reads `n_rows` except as a bound, so a
  caller may split the rows into chunks of any size, call this per chunk,
  and concatenate. Only L12 folds over `n_rows`, and it folds `[N]` floats
  rather than `[N, V]`" (`identical_ce_forward_into` docstring,
  `training/checks/loss.mojo:1329-1336`). L14 and L16 are per cell.
- Backward per layer: reads only `fwd.norm1_out, norm2_out, gated, up_proj,
  silu_out, gate_proj, residual1, norm2_sumsq, norm1_sumsq, ctxv, q_rope,
  k_cache, v_cache, amax, denom` on the fused path (`fwd.weights` on the
  eager path), the previous layer's `residual2` as `x`, and its own
  `d_out`, which it copies at stage 0-1 before anything else
  (`transformer_backward.mojo:2694-2707`). `d_x` is written last.
- Argument exclusivity: the launchers take each buffer as a separate `mut`
  argument and the file says why (`transformer_backward.mojo:2477`). Any
  aliasing below is done by the CALLER handing sub-buffer views
  (`create_sub_buffer`) of one allocation, never by editing a kernel.

| Rank | Candidate | Target bytes saved | Fold preserved because | Cost / owner |
|---|---|---:|---|---|
| 1 | **Backward stage reuse across layers.** Allocate ONE `LlamaBackwardStages` and reuse it for every layer in the reverse loop, keeping per-layer `d_x` (ping-pong pair or 12 small buffers, 6 MiB each) and packing each layer's `dw_*` into `grad` inside the loop instead of after it (`byte_lm.mojo:603-604` moved into the loop at `:596`). | 2,432 MiB (2.38 GiB): 11 x (254.1 - 27.0 dw - 6.0 d_x) | Pure storage identity: every backward kernel of layer i reads `fwd[i]`, `weights[i]`, `d_out` and writes its own scratch; no kernel reads another layer's backward scratch (read set above). `d_out` is consumed at stage 0-1 by copy before any write, so even a shared `d_x` would be safe; the ping-pong removes the need to rely on that. `_pack_block` order moves from "after all layers" to "after each layer"; it is a copy, and the optimizer reads `grad` only after the loop. | byte_lm.mojo (other lane). The 88-array gate reads only the capture (ids, state, grad, loss), not per-layer structs, so it gates this unchanged. |
| 2 | **Row-tiled loss and head with the full `dlogits` retained.** Run the head GEMM, CE L1-L11 and L14-L16 per row chunk of R rows through sub-buffer views (logits/shift/expo/weights sized R*V; row vectors and `dlogits` addressed at the chunk offset), keep `ce_row` whole and fold L12/L13 once over all M rows, keep head dA per chunk or whole, keep head dB as today over the full `dlogits`. | 1,374 MiB (1.34 GiB) at R = 256 (4 x 392.6 - 4 x 49.1); 1,472 MiB (1.44 GiB) combined with rank 3 (two chunk buffers) | Head forward `(R, V, DM)` and dA `(R, DM, V)`: per-cell folds over k = DM and k = V, independent of m (contract 7.1, batch-invariance gate). CE L4 denom GEMM `(R, 1, V)`: fold over k = V per row, same bits at any R (the docstring's row independence). L14/L16 per cell. L12 stays one `(1, 1, M)` GEMM over the same `ce_row` vector, and L13 uses the same host `count = M`, so the loss bits are unchanged; the chunk backward passes the full count as `count`, so `ce_divisor` is identical. dB `(V, DM, M)` keeps k' = M by reading the retained full `dlogits`, so its fold is untouched. | Needs (a) sub-buffer views, (b) an L12/L13 tail entry in loss.mojo that is the existing code moved, not re-spelled (the per-chunk forward call would otherwise write a chunk-partial `total`/`loss` to be discarded), (c) `head_ws`/`head_bwd_ws` sized for the chunk shapes: at R = 128 the dA shape `(128, 768, 50257)` becomes a SPLIT_64_4X4 plan with a 154 MB workspace, at R >= 256 it stays TUNED. Also shrinks the `ce_refuse_device_inputs` host mirror to 2*R*V. Gate: bitwise loss, dlogits, d_h, dw_lm, grad against the untiled path. |
| 3 | **In-place aliasing of the CE chain.** Pass views so that `ce_weights` occupies `ce_expo` and `ce_dlogits` occupies `ce_weights` (level 1); additionally `ce_shift` occupies `logits` (level 2). | level 1: 785 MiB; level 2: 1,178 MiB (1.15 GiB) | `ce_weights_kernel` reads `expo[cell]`, `denom[row]` and writes `weights[cell]` in the same thread (loss.mojo:1000-1016); `ce_dlogits_kernel` reads `weights[cell]` and writes `dlogits[cell]` in the same thread (:1080-1094); `ce_shift_exp_kernel` reads `logits[cell]` and writes `shift[cell]`, `expo[cell]` (:612). No kernel reads a neighbor cell, so in-place per cell is a storage decision with the same operands. After L7 `logits` and `shift` are dead on the clean path (L6/L7 reads `shift[row*V+y]` only; `logits`/`expo`/`denom` there feed sabotage arms the trainer refuses at compile time, `byte_lm.mojo:155-161`); `expo` is dead after L14. | Trainer-side views only; zero kernel edits. Level 2 changes what the refused `SAB_NLL_*` arms would read in a sabotage build of the TRAINER; the loss gate fixtures pass distinct buffers and are unaffected. |
| 4 | **Parameter views instead of per-layer weight copies.** Build `LlamaDeviceWeights` and `emb_w`/`lm_w` as sub-buffer views into `param` at the registry offsets; delete `_unpack_block` and the two `_copy_into` at `byte_lm.mojo:512-514`. | 618.5 MiB (324.1 + 2 x 147.2) plus about 650 MB/step of D2D copies | No arithmetic: the kernels take `unsafe_ptr()` of the same bytes. The device refusal at construction (`_refuse_nonfinite_device`) runs on the views. | modeling_llama.mojo constructor variant plus byte_lm.mojo; the optimizer writes `param` in place after the step, which is what the copies were re-syncing every forward. |
| 5 | **Gradient views instead of dw copies.** Make backward `dw_*`, `dw_emb`, `dw_lm` views into `grad`; delete `_pack_block` and the two `_copy_into` at `:605-606`. | 618.5 MiB plus about 650 MB/step of D2D copies | Each dw GEMM writes its whole output (no accumulation) and the embedding backward seeds `dw` before scattering (`embedding_identical.mojo`, `emb_seed_kernel`); the download at `:609` reads `grad` after every write. | transformer_backward.mojo constructor variant plus byte_lm.mojo. Combined with rank 1 it removes the in-loop pack. |
| 6 | **Share the six forward stages the backward never reads** (`q_proj, k_proj, v_proj, k_rope, o_proj, down_proj`) as one cross-layer scratch. | 396 MiB (11 x 36) | DEVIATION 1421 in the backward docstring: RoPE backward reads only the table, so pre-rotation projections are off the backward path; `o_proj` and `down_proj` are inputs of adds whose backward is a copy. `residual2` is NOT in this set (it is the next layer's `x`). `ensure_attention_materialized` recomputes from `q_rope` and the caches, not from these. | modeling_llama.mojo stage struct split; the per-stage card (trace on) still needs them per layer, so this is for the trace-off trainer only. |
| 7 | Host: shrink `ce_refuse_device_inputs` to a device scan and `opt_refuse_device_inputs` to a device scan | host only: 785 MiB + 4.9 GiB transient per step at the target | Not a fold; both files already record the device scan as OWED with the requirement that it produce the SAME name, message and first offending index. | loss lane and optimizer lane; this is the largest single host item and outside device memory. |

REJECTED (would change a reduction order):

- **Row-tiled head dB (`dw_lm`) accumulated across tiles.** dB folds over
  k' = M (tokens): P(2048) = 16 leaves of 128 and a balanced tree. Summing
  per-tile dB partials in tile order is a different fold. Materializing all
  16 leaf partials to fold them with the contract tree costs V*DM*P*4 =
  2.47 GB, more than the 1.9 GB of logit buffers it would free, and an
  incremental tree fold needs a new leaf-kernel entry plus a pairwise fold
  kernel (new native code, a new gate), so it is not a reuse of an existing
  fold. Rank 2 avoids all of this by retaining the full `dlogits`.
- **Fusing CE L1-L3 into one row kernel that recomputes `shift`/`expo`.**
  A second spelling of the softmax; the loss file refuses exactly that
  ("the backward recomputes nothing", `loss.mojo:1522-1525`).
- **Sharing forward stages across layers beyond rank 6.** Every other
  forward stage is a saved tensor of the backward (read set above).
- **Sharing `prefill_cache` with the per-layer `k_cache`/`v_cache`.** The
  backward reads the per-layer copies (`fwd.k_cache`, `fwd.v_cache`), which
  hold different data per layer.
- **Reusing `ce_ws`/`head_ws` across the layer GEMMs.** They already cost
  3 MiB and 1 float; the layer GEMMs allocate 1-float or 48 KiB workspaces
  per call. Nothing to gain.

Sum of ranks 1-6 at the target: about 5.4 GiB of the 9.98 GiB lean device
total (rank 3 is counted inside rank 2's two-chunk-buffer figure).
None of them touches a kernel; ranks 1, 4, 5 also remove about 1.3 GB/step
of device-to-device copies. None of these is a measured gain.

## 4. Things in the source that the handoffs do not say, or say differently

1. `ce_refuse_device_inputs` downloads the FULL logits matrix to the host
   and copies it into a `List` on every step, train and eval
   (`loss.mojo:1263-1274`): 785 MiB of host allocation and 412 MB of
   device-to-host traffic per step at the target. No LM handoff names it;
   the loss file records the device scan as OWED.
2. `opt_refuse_device_inputs` downloads param, grad, m and v (4n floats,
   twice: pinned plus `List`) on every optimizer step
   (`optimizer.mojo:1127-1145`). Together with the trainer's own
   before/after downloads, every step downloads an n-float array 11 times
   (3 before, 1 grads, 4 inside the optimizer refusal, 3 after) at 648.6 MB
   each at the target: about 7.1 GB of device-to-host traffic per step
   before the binding's resident admission readbacks (3 more).
   The session handoff's "three state readbacks for host/device mirror
   admission" counts only the binding's admission readbacks.
3. `tools/lm_training_capacity.py` counts 8 lean attention floats per layer;
   the structs actually hold 10 (`sbh` and `head_c` are lean too). Its
   materialized row omits the L*L `sbh`/`head_c` arrays (0.38 GiB at the
   target for 12 layers). Neither changes a conclusion.
4. The per-layer weight and gradient copies (1.24 GiB at the target) are a
   second full parameter set plus a second full gradient set on the
   device. The capacity handoff lists "duplicate weights/gradients" as an
   omission; this brief sizes them.
5. Every layer forward GEMM goes through the synchronizing `identical_gemm`
   (allocate workspace, synchronize, free); DEVIATION 1428 states this as a
   discipline choice. It is not a memory item at these shapes (1 float or
   48 KiB), but it is a per-call allocation and two synchronizations per
   GEMM, 7 GEMMs per layer forward and 14 per layer backward.
6. The parameter count and architecture agree with the handoff exactly:
   162,147,840, RMSNorm/RoPE/SwiGLU, untied, no final norm, no biases.
7. The control shape's batch: the session pilot used B1 and nothing in the
   source names another; the probe defaults to B1.

## 5. Probe

`tools/lm_step_memory_probe.py` (new). Public API only:
`LanguageModelConfig`, `LanguageModelTrainer(..., resident=True)`,
`parameter_registry`, `train_step`, `state_dict`, `run_metadata`, `close`.
Per step it records wall seconds around `train_step` (the binding
synchronizes the context before publishing, `_mojolearn_byte_lm.mojo:280`;
no further device synchronization is reachable from Python without new
native code, and the record says so), tokens/s, host VmRSS/VmHWM/ru_maxrss,
device peak memory from a polled vendor tool (nvidia-smi per process and
device wide; rocm-smi device wide; else "unavailable"), and sha256 of the
loss bits, flat gradients, parameters, m, v and flags. Default shape is the
control; `--target` selects the 162,147,840-parameter shape. A worker
subprocess runs under `--budget-seconds` (default 300) plus a 30 s grace;
exceeding it, or any step crossing the deadline, is exit 2 with the
limitation written to `execution.json`, and no smaller model is substituted.
`--component-timing` runs one extra untimed step with
`MOJOLEARN_TRANSFORMER_TIMING=1` so the native phase printer's lines land in
the worker log (time fractions for the handoff; a timed step is not a sample).
In-process device peak is recorded as unavailable (it needs native code).

`tools/lm_step_memory_probe.sh` (new) is the leg extra body: builds ONLY
`bindings/build_byte_lm.sh` under IDENTICAL for the box's architecture
(MOJOLEARN_GPU_ARCHS from the leg or from `compute_cap`), runs the control
probe, then attempts the target, then one target component-timing step if
the target completed, writing to `/root/gemm_leg_out/lm-step-memory/`.

## RUN OWED (orchestrator, NVIDIA box)

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_step_memory_probe.sh \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 75 \
    --gpu "NVIDIA H100 80GB HBM3"
```

For an L40S or RTX 4090 use `MOJOLEARN_GPU_ARCHS=sm_89`. The extra body
also derives the architecture from `nvidia-smi` when the variable is unset.
Optional knobs on the driving host, passed through the leg environment:
`MOJOLEARN_LM_PROBE_BUDGET` (seconds per probe, default 300) and
`MOJOLEARN_LM_PROBE_STEPS` (default 3).

Expected under `<leg out>/remote/lm-step-memory/`: `status.txt` (build and
probe exit codes with seconds; probe exit 2 is a recorded budget limitation),
`build_byte_lm.log`, `control/` and `target/` each with `events.jsonl`,
`result.json`, `execution.json`, `worker.log`, plus `target-timing/` with
the phase lines in `target-timing.log` if the target completed, and
`gpu_before/between/after.txt`. Nothing under `bench/results` is written by
this lane; file the fetched directory there afterwards with its
`extra_body.sh` copy.

Local dry run of the argument parsing only (no GPU, no mojolearn import):
`python3 tools/lm_step_memory_probe.py --help`.

## Run 1 results (H100 sm_90a, main a51b6150, 2026-09-11 03:13Z to 03:19Z, `bench/results/e1g/2026-09-10_230820-nvidia/remote/lm-step-memory`)

Complete IDENTICAL training steps through the public trainer, timing at the
binding's synchronized boundary, three steps each, none budget-limited:

| shape | parameters | first call (setup) | steady step median | tokens/s | device peak (polled, device wide) | process RSS peak |
|---|---:|---:|---:|---:|---:|---:|
| control B1 L2048 V8192 (8 x DM384/FF1024) | 20,453,376 | 10.3 s | 5.53 s | 370 | 4.64 GB | 3.96 GB |
| target B1 L2048 V50257 (12 x DM768/FF2048) | 162,147,840 | 50.2 s | 44.97 s | 45.5 | 12.14 GB | 21.93 GB |

The device peak at the target (12.1 GB) sits between this brief's lean
estimate (9.98 GiB) and the eager fallback; the host RSS (21.9 GB) is the
mirror traffic this brief inventoried. The control step on the H100 (5.5 s)
is within 20 percent of the Apple pilot's 6.8 s: the step is not device-bound.

Component timing (MOJOLEARN_TRANSFORMER_TIMING=1, one target step, not a
timing sample): the twelve transformer blocks sum to 0.43 s, of which
attention core forward 82 ms, attention backward 262 ms, MLP and residuals
26 ms, projections and norms 19 ms. Everything else in the 45 s step, about
44.5 s or 99 percent, is outside the blocks: the head and loss over
2048 x 50257 logits, the optimizer, the per-step refusal downloads
(`ce_refuse_device_inputs`: full logits to pinned host and to a List;
`opt_refuse_device_inputs`: param, grad, m, v) and the host mirrors. None of
those phases is itemized by the existing timing switch, so the split among
them is NOT yet measured and must not be attributed from this brief's byte
counts. Next: itemize those phases with the same timing boundary, then remove
copies under the device-owned step API the handoff's step 1 calls for.

What this changes in the handoff order: GEMM and attention (step 3) are one
percent of the target step today; steps 1 and 2 (transfers and the loss/head
buffers) are the whole cost until they are gone.
