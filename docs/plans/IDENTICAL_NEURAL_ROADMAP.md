# IDENTICAL neural speed roadmap (NVIDIA + AMD), 2026-10-04

Status: **proposal from a read-only code review. Nothing here was compiled, run or measured.**
Every gain is inferred from code structure. Every file:line refers to
`integration/identical-all-20261004` as read on 2026-10-04 (about 464 non-merge commits ahead
of main); lines will drift as that branch moves.

Scope: the IDENTICAL tier on NVIDIA and AMD, neural families first (GEMM under the LM and
transformer rows, Mamba-1/2/3, Samba, x_cnn, sequence, optimizers, embedding, cross-entropy).
Apple is out of scope here except where a bits change must move every column together.
Trees and classical ML are in the companion `IDENTICAL_ML_TREES_ROADMAP.md`.

Rules that bind every item (see `CLAUDE.md`, `docs/plans/HOST_ROUTE_REMOVAL.md`):

- Same bits on NVIDIA, AMD, Apple and the host column within one version. Bits may change
  between versions when every column changes together.
- The GPU path is GPU only and parallel. No serial one-thread or one-block default.
- No dispatch rule fitted to an exact benchmark shape; prove gains on neighboring shapes.
- No reduced precision in place of fp32 without the owner's decision.
- Each item lands behind its own define, is off under `MOJOLEARN_IDN_ALL_OFF`, and is kept
  or dropped on one measured ON/OFF sample per GPU.

Items already in the fam/fam2 reports, the toggle inventory or the candidate audit
(`~/mojolearn-evidence/candidate-audit-2026-10-04.md`) are not repeated. Where an item builds
on a known one, the "overlap" note names it.

## 0. Where the time goes

Stored boards (code older than the integration wave):

| Row | NVIDIA L40S (0.8.34) | AMD |
|---|---|---|
| lm-train-step | 3.6x | 12.0x (MI325X, 0.8.25); 1.85x (MI300X partial, 0.8.34) |
| lm-forward | - | 4.15x |
| transformer-forward | 2.2x | 3.49x |
| samba-forward | 2.4x | - |
| mamba2-forward | - | 2.88x |
| gemm | - | 3.28x |

Single-op rows are far worse (L40S: embedding 96x, cross-entropy 69x, resnet-block 59x,
conv2d 29x; AMD optimizers 22-40x), mostly from host-in/host-out transfers the wave already
addresses.

The only step-time breakdown found (an older MI300X run at a larger shape, under
`bench/results/amd_step_time_2026-09-24/` in the amd-portable tree) has GEMM calls at about
85% of the LM train step and glue at about 2%. So for the LM rows the GEMM schedule matters
most, and launch/wait cleanups matter little on their own.

**No new fold contract.** The review found no throughput mechanism in interleaved accumulator
lanes or K>1 MFMA leaves on either vendor. The contract (leaf = 128 ascending
`ftz(fma_rn)` steps from +0.0, adjacent-pair balanced tree, odd tail carried:
`gemm/contract.mojo:106-160, 225-269`) is not what binds. What binds is which kernel body a
shape lands on, partial-plane traffic on the group path, and device fill at LM shapes.
MFMA K=1 equals scalar `fma_rn` (measured, `gemm_identical.mojo:2759-2770`); the K>1 internal
order is unknown and no probe exists.

## 1. Tier A: GEMM schedule (same bits)

`G` = `gemm/checks/gemm_identical.mojo`.

| # | Item | Evidence | Vendor | Risk |
|---|---|---|---|---|
| A1 | MFMA body at stepped-down tiles (one-wave 64x64 block). MFMA is reached only on the TUNED_128 plan and asserts 256 threads and a 128x128 tile; AMD's 1024-block floor steps a 2048x384x384 call down to 32x32 scalar tiles. | G:6435-6452, G:2877-2888, G:3372-3375; MFMA about 1.9x over scalar in `checks/kernel_matrix_gemm.mojo:173-191` | AMD | new staging geometry for a 64-thread block |
| A2 | kpack body for stepped-down calls, chosen by one fill rule: the largest kpack tile whose block count fills the device with no workspace; group only when it cannot. kpack runs only at TUNED_128 and is the only body with window admission. | G:6491-6522, G:7667, G:1326-1332, G:6855-6891 | NVIDIA | 16 cells per thread may lose to groups; needs neighbor shapes. New versus KPACK_RPT4/CPT4 and the slack arms: per-call selection and a zero-workspace option |
| A3 | Two MFMA blocks per CU: KS=8 with two pages, or the existing `MOJOLEARN_GEMM_ONE_PAGE`. | G:3142-3145, G:3226; `checks/kernel_matrix.mojo:462-463` | AMD | VGPR budget may already cap occupancy |
| A4 | Delete the leaf-band rule once A1 lands (it catches only calls that missed MFMA). | G:6480-6490, G:6532-6554 | AMD | depends on A1 |
| A5 | Multi-job launch (pointer arguments, grid.y = job) for GEMM sets that share operands or shapes: da/dw, gate/up, q/k/v. No concatenated weight. | `training/dev_tensors.mojo:180-181`; `transformer/impl/llama/modeling_llama.mojo:4884-4926` | both | jobs must share a kernel instantiation. Overlap: generalizes the known merged q/k/v |
| A6 | Samba linear ops allocate and wait per call; use `GemmWorkspace` and one end wait. | `training/dev_tensors.mojo:148-150, 174-182` | both | not confirmed that lm-train-step uses this path |
| A7 | Plan cache per (m, n, k, op) in `GemmWorkspace`. | G:9097-9123, G:8045, G:3338-3481 | both | small |
| A8 | Replace shape-fitted constants with a fill rule: S = 132 / 110 per vendor; NVIDIA short-k 512 / min-k 4096 band; `p_count >= 4`; `m % 128 == 0 and n % 128 == 0`; `m >= 4096 or (m >= 2048 and n >= 1024)`; the 16,384-cell fold switch. | `checks/kernel_matrix_gemm.mojo:194-201`; G:3387-3393, G:3587, G:3596, G:6554, G:2684 | both | a device-read S needs MAX to expose the unit count; if it does not, wait. Coordinate with main's `494c7103a` and the no-bench-tuning lane |
| A9 | kpack KS=32 at the narrow tile (the KS=32 loss was measured on the 128x128 tile only). | G:7169, G:1727-1733, G:7299-7305 | NVIDIA | needs two gathers per thread |

Not read: the outer-contiguous staging branch (G:1975). Transposed forms pay no copies (one
body with stride pairs, G:307-338).

## 2. Tier B: Mamba and Samba

Scan map: Mamba-2 SSD (Q=256) and Mamba-3 SISO (Q=64) are already chunk-parallel. Serial over
L in one thread: the Mamba-1 selective scan (forward and backward), the Mamba-1/2 depthwise
conv, the Mamba-3 angle chain and its reverse, and the Mamba-2 backward parameter folds.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| B1 | Mamba-2 SSD: shared-memory tiles and M = G⊙L written once. ydiag recomputes `cb_g*seg_l` for each of 64 p; cstate recomputes B·decay per p. | `mamba/impl/modules/ssd_minimal.mojo:412-513, 546-629, 760-819` | same | likely the main cost in mamba2-forward; follow Mamba-3's tile form (`mamba3_siso.mojo:1191, 1532, 1656`) |
| B2 | Conv1d: one thread per (b, l, channel). Four taps per cell, no recurrence. | `mamba/impl/modules/mamba2.mojo:506-555` (grid :1135); `modeling_mamba.mojo:1013-1041` (grid :1129) | same | regenerate the host column |
| B3 | Mamba-3 backward d_dt angle term is O(L²): each token folds its own suffix for 32 angles. Reuse one suffix sum (the carry `mamba3_theta_reverse_kernel` already computes), then chunk it. | `mamba/impl/modules/mamba3_backward.mojo:1267-1272, 1175-1187, 1204, 1283` | changes (d_dt only) | backward oracle and host backward column follow |
| B4 | Samba attention layers run their forward twice per train step ("forward is recomputed from a zero cache"). Retain stages in a prefill session keyed on x bytes. | `python/mojolearn/_transformer_impl.py:1244-1254`; `_samba_impl.py:613` | same | retained-stage memory |
| B5 | Mamba-1 selective scan: fixed-chunk parallel scan. h = a·h + b composes as (a, b) pairs; fixed chunk, serial chunk carry, re-walk. Chunk boundaries pinned to absolute position. | `mamba/impl/ops/selective_scan_interface.mojo:353, 570`; `selective_scan_backward.mojo:368, 567` | changes | all vendors, `mamba/host/gen`, oracle, backward checkpoint and decode step move together. FAST's chunk = ceil(L/32) is L-dependent and not reusable |
| B6 | Mamba-2 backward per-parameter serial folds. Per-cell parts (d_conv, d_in, d_dt) as per-cell launches keep bits; a fixed leaf and tree for the folds changes weight-gradient bits. | `mamba/impl/ops/mamba2_ssd_backward.mojo:491-531, 548, 736-762, 882, 953-957` | split | gradient oracle |
| B7 | Mamba-3 yintra tile is NVIDIA-only; AMD recomputes `qk_s*seg_l` per p. Thresholds (`b*nc*nh >= 128`) become an occupancy rule. | `mamba/impl/ops/mamba3_siso.mojo:2326, 1625-1633, 2249, 2327` | same | AMD A/B |
| B8 | exp(dacs) recomputed per p; compute once per (b, t, h). | `mamba3_siso.mojo:1688, 1747`; `ssd_minimal.mojo:812-816` | same | |
| B9 | `cb_g` computes the full Q×Q; only j ≤ i is read. | `ssd_minimal.mojo:384-401, 454` | outputs same; the recorded stage changes above the diagonal | trace cards; the backward reads `cb_g` (not checked) |
| B10 | Weight byte compare and recopy per layer per call; use an optimizer version stamp. | `bindings/_mojolearn_mamba.mojo:2106-2192` | same | subsumed by the single-binding Samba handoff |
| B11 | Row-norm folds: one thread per token row. | `modeling_mamba.mojo:876-890`; `mamba3.mojo:784, 1524` | changes | small gain at batch 1 |
| B12 | Launch geometry from constants (scan `block_size=64`, `MAMBA3_TPB = 128`, tile grids `*32`). | `selective_scan_interface.mojo:466`; `mamba3_siso.mojo:263, 2260` | same | sweep, then a rule |

Not covered: launch, wait and allocation counts beyond the two fam reports; the Samba
loss/head/optimizer path.

## 3. Tier C: LM / transformer step around the GEMM

Waits per train step on NVIDIA/AMD by code reading: about 5 per layer forward, about 4 per
layer backward, about 18 at step level (roughly 90 at 8 layers). One AdamW launch already
covers all parameters (`training/checks/optimizer.mojo:1708`).

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| C1 | Fold the state checks into the Adam kernel. The post-update validation is 4 scan launches with 4 waits, plus the pre-update gradient scan; the update kernel already holds p, g, m, v. Emit per-block first-index partials there. | `training/byte_lm.mojo:346-349`; `core/device_scan.mojo:338-349` | same | same refusal name and index in the same order. Overlap: `afn_step_finish` (FAST Apple); `OPT_SCAN_FUSED` covers entry scans only |
| C2 | Parameter and gradient views on NVIDIA/AMD (Apple-only today); NV/AMD copy every parameter and gradient each step. | `training/byte_lm_afn.mojo:57`; `byte_lm.mojo:694, 708-709, 742, 759, 1638-1639` | same | rebind after the handle swap; comment at :690 says untested on those columns |
| C3 | One-wait step via device status cells: every refusal, regime, corner, loss and state check writes an integer cell, read once at step end; on a hit, roll back and replay the synchronous path. | `byte_lm.mojo:1215, 1224, 1289, 1602, 1344, 1660`; `modeling_llama.mojo:3765, 4789`; `fused_attention.mojo:7931, 8161`; `transformer_backward.mojo:3264` | same | replay path. `ATTN_SPECULATIVE` alone measured 1.00 on the L40S, so low value alone |
| C4 | RMSNorm row kernels: one thread per token row walks d_model three times. (a) keep the S1 fold in the row thread, move scaling and residual add to a cell-parallel launch; (b) fixed-lane tree for S1 and the backward dot. | `modeling_llama.mojo:1961-2023, 2048-2062`; `transformer_backward.mojo:2946-2983` | (a) same, (b) changes | (b) rewrites contract S1 on every column. The gate at :2959-2963 (`m >= 32768 and dm >= 768`) is borderline under the exact-shape rule |
| C5 | Merged gate/up GEMMs (both read `norm2_out`; weights adjacent in the flat parameter). dA as one GEMM at k = 2·it keeps bits only if the balanced tree splits at `it` (check the contract). | `modeling_llama.mojo:4884-4926`; `transformer_backward.mojo:3394-3400, 3437-3451`; `byte_lm.mojo:736-737` | same (forward, dB) | A5 covers this without a layout change |
| C6 | Redundant id/target readbacks: embedding forward and backward each download the ids; CE downloads the targets; host ids were validated twice already and `*_prerefused_into` entries exist. | `embedding/checks/embedding_identical.mojo:727-734, 755, 772`; `training/checks/loss.mojo:1318-1320` | same | |
| C7 | SiLU and gate product in one launch; default `SWIGLU_FUSED` on for forward-only, add a two-output kernel for training. | `modeling_llama.mojo:3481, 4975, 5017` | same | |
| C8 | Dead KV work in training prefill (`kv_append2` plus two copies per layer; nothing reads that cache). | `modeling_llama.mojo:4691, 4757-4759`; `byte_lm.mojo:1251-1254` | same | decode callers need a flag |
| C9 | Backward entry copy (`d_out` to `in_d_residual2` per layer). | `transformer_backward.mojo:3291, 3523` | same | aliasing a caller buffer |
| C10 | CE elementwise fusion: weights + dlogits in one pass; shift-exp into the row-max block. | `training/checks/loss.mojo:1439-1474, 1656-1667`; `training/estimator.mojo:1532-1614` | same | the denominator GEMM fold is untouched |
| C11 | Retain the two pinned upload buffers per step. | `byte_lm.mojo:1211-1224` | same | subsumed by resident token batches |
| C12 | Hoist the per-layer `inv_freq` scan to once per session. | `modeling_llama.mojo:3761` | same | small |

Not reviewed in depth: the fused attention kernel bodies (launchers only).

## 4. Tier D: CNN, sequence, optimizers, embedding

Already fine, not candidates: the recurrent input projection is one GEMM over all timesteps
(`sequence/recurrent.mojo:309`); conv col2im is gather form (`x_cnn/ops.mojo:225`); the graph
CSR is radix-sorted and cached per graph.

| # | Item | Evidence | Bits | Notes |
|---|---|---|---|---|
| D1 | Tiled same-chain GEMM for the sequence family: `OP_GEMM` runs one thread per output cell with a K-long dot and no shared-memory staging. | `sequence/ops.mojo:399`; `sequence/exec_device.mojo:776`; callers `recurrent.mojo:309, 360, 370, 375, 446-451`, `mlp_fit.mojo:245, 255` | same | new kernel for strided operands. Overlap: MoE regtile (MoE only) |
| D2 | One-launch recurrent scan on NVIDIA/AMD. The scan kernel is documented same-bits but gated to FAST + Apple; IDENTICAL issues T launches per layer each way. | `sequence/recurrent_scan.mojo:39-40`; `recurrent.mojo:334-352, 410-438` | same | depends on the in-block barrier ordering device memory on CUDA/HIP |
| D3 | Blocked fold for recurrent weight and bias gradients (T·B-long chains; `op_sum` is one thread). | `recurrent.mojo:446-449, 116, 138`; `sequence/ops.mojo:689` | changes | device and host executor share the op |
| D4 | Optimizer refusal scan: have the update kernel emit non-finite partials for its outputs, so the next step scans only the gradient. Today: four full scans, two allocations, a readback and a wait before each update. | `training/checks/optimizer.mojo:1344-1376, 1602` | same | refusal for a bad state arrives one step later. Same idea as C1 |
| D5 | Embedding backward: radix sort on (id, t) keys and fold over touched rows only. SCAN is V threads each walking T; SORT is a bitonic network. | `embedding/checks/embedding_identical.mojo:265-288, 906, 974`; `embedding_sort.mojo:103-113`; `x_cnn/device.mojo:2670` | same | Overlap: `EMB_AUTO_SORT` keeps bitonic |
| D6 | Direct convolution on NVIDIA/AMD (`DIRECT_CONV`, k ≤ 32, is Apple-only). Larger k needs an implicit GEMM that replicates the pinned leaves and tree. | `x_cnn/device.mojo:269, 2390, 2405-2407` | same for one-leaf k | `afn_direct.mojo` uses a free fold and is not reusable |
| D7 | Conv2d layer backward recomputes im2col; `save_cols` exists only for the conv block. | `x_cnn/device.mojo:948`; `_expansion_cnn.py:582-600` | same | holds rows×ckk floats |
| D8 | Bias gradient as a dedicated column fold instead of an N=1 GEMM against ones. | `x_cnn/device.mojo:962, 2556` | same if the tree is replicated | small |
| D9 | Softmax rows compute exp twice. | `x_cnn/ops.mojo:660-663`; `sequence/ops.mojo:678, 683` | same | |
| D10 | LayerNorm forward and backward-x: one thread per row walks D three times. | `sequence/layernorm.mojo:38-65, 68-91` | split launch same; blocked stats changes | `LN_FOLD_BLOCK` appears to cover only dweight/dbias |
| D11 | Adafactor factor folds (R + C threads with long chains; `op_af_rmean` one thread). | `sequence/adafactor.mojo:141-152, 155` | tiled walk same; blocked fold changes | |
| D12 | SGD: one launch over an arena (Adam already is); use the batched clip inside the step. | `training/checks/optimizer.mojo:1669-1697, 1620, 1182, 1205` | same | |
| D13 | MLP fit epoch order on the device (host Fisher-Yates and a permutation upload per epoch). Reuse the Feistel `epoch_rows_at`. | `sequence/mlp_fit.mojo:172-174, 280-304` | changes | Overlap: `CNN_EPOCH_DEV` (CNN only) |

## 5. Proposed order

1. **Run the measured batch that is already owed first.** The audit counts 0 promoted
   candidates, 194 default-ON switches covered only by the all-OFF arm, and 34 opt-in arms
   never compiled. More code-only candidates add risk until that batch exists.
2. A1 and A2 (GEMM body selection). They touch every LM, transformer and Mamba row and keep
   bits. A8 rides with them.
3. B1, B2 (Mamba-2 forward), then B4 and B3 (Samba train step).
4. C1/D4 and C2 (optimizer scans and parameter views), then A5.
5. D2 and D1 (recurrent family).
6. The bits-changing set as one version step, every column and the host generator together:
   B5, B3, B6 folds, C4(b), D3, D10/D11 blocked forms, D13.
7. The rest by measured need.

## 6. Open decisions for the owner

- fixed15 as the neural training profile is the only route found to the 67 versus 20 TFLOPS
  gap on the L40S. It is outside the fp32 identity guarantee (`ROADMAP.md:56-57`) and was
  measured at 24.2 as a training op. Not proposed without a decision.
- Whether the bits-changing set (step 6) goes in one version.

## 7. Limits of this review

- Read-only. No build, no run, no timing. Gains are mechanisms, not numbers.
- The stored ratios predate the integration wave; several rows have changed code since.
- Not checked: whether integration's no-bench-tuning merge already covers main's `494c7103a`.
- Not opened: fused attention kernel bodies, the Samba head/loss/optimizer path, the GEMM
  outer-contiguous staging branch.
