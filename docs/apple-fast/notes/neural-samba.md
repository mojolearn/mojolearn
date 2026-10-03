# afn-samba: the Samba stack on Apple FAST (lane notes, 2026-10-03)

Board lanes: samba-forward, samba-train-step (full: batch 2, length 512, vocab 256, d_model 384,
layers mamba3/attention/mamba3/attention, 6 heads, intermediate 1024). Bindings: training (the Samba ops,
`bindings/_mojolearn_training.mojo`) and mamba (Mamba3Block forward/backward). Quality judge:
`tools/neural_fast_quality.py samba`.

## Profile (code reading, per train step at the board shape)

The step is a Python orchestration (`python/mojolearn/_samba_impl.py`, `SambaStack.loss_and_grads`) of
separate binding calls. Each Samba op call (training/samba_ops.mojo) does: host non-finite scan of every
input (a host loop over every element), fresh device buffers for every operand and scratch, upload,
kernels, readback, one `synchronize`.

| stage | binding calls | launches | host scans / allocations / waits |
|---|---|---|---|
| embedding forward | 1 | 1 gather | scan vocab*d; 3 buffers; 1 wait |
| per Mamba-3 layer forward | 1 (mamba) | Mamba-3 prefill forward (afn-mamba owns) | 1 wait |
| per attention layer forward | 1 (transformer) | LlamaEager.block (afn-attn owns) | 1 wait |
| final norm + head forward | 2 (fused head: 1) | norm, GEMM | scans; hn crosses the bus |
| head loss + head backward | 1 (`samba_head_loss`) | GEMM, CE, 2 backward GEMMs | logits read back for the host non-finite scan (M*V) |
| final norm backward | 1 | norm recompute + `bwd_rms_norm` | 3 scans, ~10 buffers, 1 wait |
| per Mamba-3 layer backward | 1 (mamba) | ~38 stage launches (`mamba3_prefill_backward_on`) | ~75 fresh zero-filled scratch buffers, 1 wait |
| embedding backward | 1 | run-sorted identical fold | scan, 6 buffers, 1 wait |
| tied pair add | 1 (`accumulate_grads`) | 1 | scans, 1 wait |
| AdamW step | 1 | afn-optim owns | |

Mamba-3 backward, angle stage (twice per layer backward, once per pass and once for the join):
`mamba3_theta_reverse_kernel` runs one thread per (batch, head, angle) chain walking all L=512 tokens
serially (768 chains); `mamba3_angle_dt_shared_kernel` folds each token's suffix from its own index
(L^2/2 dependent shared-memory adds per angle per block, 32 angles).

Apple costs that matter here: every live Metal buffer taxes every launch (~0.25 us each), each wait costs
~180 us, every host scan is a host loop over the operand, every readback ~21 ms per 64 MB.

## Candidates (each its own define, default off, FAST + Apple guard)

| define | binding | what changes | files |
|---|---|---|---|
| MOJOLEARN_AFN_SAMBA_FUSE | training | three fused entries: final norm + head forward (one call, hn stays on device); the train tail (norm fwd, head GEMM, CE with gradient, both head backward GEMMs, norm backward reusing the forward's row sums) in one call; the tied embedding gradient (embedding backward + pair add) in one call. Python uses them when registered. Cuts 4 calls (4 waits, 4 upload/readback rounds, the hn/dhn/logits bus crossings) per step and 1 per forward. | training/samba_afn.mojo, bindings/_mojolearn_training.mojo (samba block), python/mojolearn/_samba_impl.py |
| MOJOLEARN_AFN_SAMBA_ARENA | training | every per-op operand and scratch is a view of one arena chunk (core/device_arena.mojo) opened at the op's start (only when no arena is open) and released after its wait, instead of fresh Metal buffers. | training/samba_afn.mojo |
| MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT | training | the non-finite refusals become a device flag kernel per operand, the flags read back with the op's own wait (no host loop over the operands, no logits readback for the scan). Same refusals, same messages (plus "(device admit)"). | training/samba_afn.mojo |
| MOJOLEARN_AFN_SAMBA_EMB_ATOMIC | training | embedding backward by f32 atomic scatter (free order) instead of the run-sorted identical fold (3 int32 scratch buffers + sort passes); in the tied entry the head gradient is uploaded straight into the output and the scatter accumulates onto it (no pair-add launch). Out-of-range ids raise a device flag (contract 8, never clamped). | training/samba_afn.mojo |
| MOJOLEARN_AFN_MAMBA3_BWD_CHUNK | mamba | the angle stage's two reverse-time chains chunk-parallel: theta reverse in two launches over (chain, 64-token chunk) (chunk sums, then each chunk's walk seeded with the later chunks' sums); the d_dt suffix by a segment scan (each thread scans its own 16-token segment, publishes the total, adds the later segments' totals): O(L) per angle per block instead of O(L^2/2). Exact f32, only the fold order changes. | mamba/impl/modules/mamba3_backward.mojo, mamba3_prefill_backward.mojo |
| MOJOLEARN_AFN_MAMBA3_BWD_ARENA | mamba | the Mamba-3 backward pass's ~75 scratch buffers are zero-filled views of one arena the pass opens (only when no arena is open) and releases after its wait; the ten gradient outputs stay real buffers. | mamba/impl/modules/mamba3_prefill_backward.mojo |
| MOJOLEARN_AFN_SAMBA_ALL | both | all of the above (they compose). | |

Brief candidates not delivered as separate defines: RESIDENT (whole step resident across steps) needs
the Python driver to hold device handles across binding calls, which the binding API (host addresses in,
host addresses out) does not offer; FUSE + ARENA + DEVICE_ADMIT are the parts of it that fit the current
API. MAMBA2_BWD_MMA is not on the Samba path (Samba uses Mamba-3 only) and is left out.
GRAD_FOLD is delivered as EMB_ATOMIC (the embedding scatter); the cross-layer gradient tree belongs to
the optimizer (afn-optim).

## Safety notes

- Every changed line sits under `comptime if` on `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
  has_apple_gpu_accelerator()` and the define. The samba_ops early returns are comptime-false otherwise.
  In mamba3_prefill_backward.mojo the IDENTICAL arm of `_m3_scratch` is the `mamba_zeros[False]` call it
  replaces, and `_AfnM3Arena` is a plain -1 there.
- Arenas: an op or pass opens its own arena only when none is open, never takes views from someone
  else's arena, and the destructor ends and releases it if the op raises (no arena is left active).
- No smoke gate ran: the build scripts export MOJOLEARN_SKIP_BUILD_GATE=1; only the binary checks run.

## Compile results

See the final section (filled after the builds).
