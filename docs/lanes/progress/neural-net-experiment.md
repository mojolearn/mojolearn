# lane/neural-net-experiment (2026-09-30)

UNMEASURED. Written from the code and the saved 0.8.25 board records
(`bench/results/bench_board/recovered-2026-09-30/measurements.csv` on
`lane/board-resume-20260930`), in a container with no Mojo toolchain and no
GPU. Every Mojo change below is uncompiled. Build first, then measure; the
arithmetic is untouched by design (no kernel, launch order or operand
changed), so the identity gates are the check that this claim held.

## What the numbers said

Per-op milliseconds, ours / fastest fp32 opponent, from the CSV:

| op | L40S | MI325X | M3 Ultra | note |
|---|---:|---:|---:|---|
| transformer-forward (one block, B1 L2048) | 6.5 / 3.4 | 14.4 / 1.85 | 31.1 / 4.45 | the block's GPU work is well under 1 ms at this shape |
| lm-forward (8 blocks + head, 64 MiB logits) | 147 / 46.5 | 91 / 11 | 250 / 12.5 | 3x the training step on the L40S |
| lm-train-step | 46.5 / 11 | 124 / 10.3 | 220 / 43 | the repeated op of LM training |
| samba-forward (B2 L512, 4 blocks) | 12.0 / 5.0 | 26.7 / 4.4 | 61.9 / 42.8 | 7 host round trips per call |
| samba-train-step | 159 / 43.6 | 240 / 30.4 | 670 / 106 | forward + 4 recompute-backwards + optimizer |

Peak VRAM of our process (`peak_gpu_mb`): a fixed ~900 MB on the L40S and
~3.2 GB on the MI325X for every neural cell, then samba-forward at 8.9 GB on
the MI325X against 0.9 GB on the L40S for the same shape. Same code, so the
difference is the ROCm allocator holding what our per-call allocate/free
churn hands it.

## Root causes found in the code

1. **The block forward reallocates and re-uploads everything per call.**
   On the L40S the bench took `transformer_forward_fresh`
   (registered for NVIDIA builds only): 9 weight uploads, a KV cache, a
   RoPE table, 30 stage buffers each zero-filled, the input upload, the
   forward, the download, then every buffer freed -- about 45 device
   allocations and frees per call. On the MI325X and the Macs the entry
   is absent, so the bench took the session path
   (`transformer_session_forward`), which (a) re-uploaded the nine weights
   with a wait each, (b) uploaded and downloaded the 2 x 3 MB cache, and
   (c) dropped the whole workspace after every call because the retained
   budget was a fixed 64 MiB and the board's lean workspace at this shape
   is about 87 MiB. `hipMalloc`/`hipFree` and pageable copies are far more
   expensive than their CUDA counterparts, which is the AMD-specific 2x on
   top of the shared churn. `samba` runs four such blocks per forward and
   `TransformerBlock.backward` recomputes the forward with the same bill.
2. **LM logits returned through three host copies and a pinned
   allocation.** `_logits_forward` downloaded 64 MiB into a freshly pinned
   host buffer, appended it element by element into a `List[Float32]`,
   scanned it, and the binding copied it again into the caller's array.
   All of it after the GPU finished. `download_f32_into` (DEVIATION 3120)
   already existed for the exports and was not used here.

## What this lane changes (no bit moves)

- `bindings/_mojolearn_transformer.mojo`
  - `TransformerSession` retains the block's float32 weights on the device
    with an exact host copy of the uploaded bytes. Each call compares the
    caller's arrays bit for bit (SIMD-8 integer compare, ~7 MB at the
    board shape, well under a millisecond) and reuses the device copy when
    equal; changed bytes are copied into the same buffers (no realloc) and
    revalidated by the constructor's `_validate_finite`; a new shape or
    options record rebuilds. Options blocks keep the per-call upload.
  - New entry `transformer_session_forward_fresh(session, addrs, params)`:
    the stateless prefill on a session. Cache slots 10/11 are 0, the
    retained cache buffers are zero-filled on the device, nothing of the
    cache crosses the host boundary, `smax = L`, `cached_tokens = 0`.
  - Retained workspace budget is `MOJOLEARN_TRANSFORMER_RETAIN_MB`
    (default 512, was a fixed 64). `tools/transformer_session_check.py`'s
    budget group pins 64 so its eviction assertions still hold.
  - The session's three downloads share one wait (the fresh entry's
    pattern).
- `python/mojolearn/_transformer_impl.py`: a stateless `forward` (state=None)
  takes the session's fresh entry when the binding exports it, on every
  vendor; then the NVIDIA per-call fresh entry; then the state-carrying
  session path, as before. Samba's blocks inherit this.
- `training/byte_lm_logits.mojo`, `bindings/_mojolearn_byte_lm.mojo`: both
  logits entries download straight into the caller's array
  (`download_f32_into`) and refuse non-finite values with a SIMD-8 scan
  over that memory; same first offending index in the message, same
  sabotage bit. The List-returning functions stay for any other caller.

## What it does NOT change (next, and why not here)

- `TransformerBlock.backward` still recomputes the forward and re-uploads
  per call (Samba training). The fix is the same session treatment for the
  backward entry (retain weights, retain forward+backward stages, keep the
  forward's stages from the last forward instead of recomputing); left for a
  measured lane because it touches the backward's stage ownership.
- Mamba3's `mamba3_forward_fresh` has the same per-call shape; same fix
  applies.
- Samba's Python orchestration (embedding, per-block, norm, head) is still
  seven host round trips; a stack-level session is the larger follow-up.
- AMD kernel tuning. The identical GEMM (`core/gemm.mojo`) accumulates each
  output in a serial k-loop, so changing the 64x64 tile / 16x16 thread
  shape does not change any bit as long as split-K stays off; that is a
  safe sweep for a lane WITH a MI325X to measure on, not a blind edit.
- Fusing more kernels: `residual_next_norm_fusion_enabled` already fuses
  the residual into the next norm; anything further changes launch
  boundaries and needs the trace to certify.

## How to measure

```
bash bindings/build_transformer.sh && bash bindings/build_byte_lm.sh
python tools/transformer_session_check.py            # every group
python tools/transformer_fresh_prefill_check.py      # NVIDIA builds
MOJOLEARN_TRANSFORMER_TIMING=1 python - <<'PY'       # per-stage ticks
...one TransformerBlock.forward at B1 L2048 DM384, called 5 times...
PY
python tools/bench_board_neural.py --lanes transformer-forward,samba-forward,samba-train-step,lm-forward --arms ours,torch-compile-fp32 --rounds 5
```

Expected if the diagnosis is right: transformer-forward and samba-forward
drop by most of their gap on all three vendors (the second call onward
uploads nothing and allocates nothing); lm-forward drops to about the
training step's forward share; lm-train-step and samba-train-step move
little (their costs are in the backward and, for Samba, the recompute).
If transformer-forward does NOT move on the MI325X, the remaining cost is
in the kernels, and the GEMM tile sweep is the next lane.
