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

## Second pass (same day): the rest, still unmeasured

Each is its own commit, so a build failure reverts one at a time.

- **Transformer backward on the session** (`transformer_session_backward`):
  the twenty-one addresses and eight scalars of `transformer_backward` word
  for word; weights reused by exact bytes, the forward workspace retained at
  `smax = L`, the backward stages and grad_output buffer retained across
  calls (as the byte trainer retains `tr.backward[layer]` across steps).
  `TransformerBlock.backward` takes it when exported; Samba training calls
  it four times a step.
- **Mamba-3 prefill session** (`mamba3_prefill_session_*`): the fresh
  entry's 15 pointers on a session that retains the nine weights (exact
  compare, recopy in place with the block's non-finite refusal re-armed) and
  the x buffer. The zero state and the stages are still built per call:
  nothing resets them and a fresh construction is the certified zero state.
  So this removes the weight uploads, not the stage allocations; the rest
  needs a stage reset that has not been written or certified.
- **`MOJOLEARN_BYTE_LM_LAYER_SYNC=0`**: skips the sixteen per-layer host
  waits in the byte-LM step. Default unchanged. Try it on each vendor; if a
  Metal build misbehaves with it off, the enqueued-free assumption
  (DEVIATION 2520) does not hold there and the toggle stays off.
- **`tools/neural_stage_timing.py`**: the measurement the first pass asked
  for. Runs the board's shapes for transformer-forward, mamba3-forward,
  samba-forward, samba-train-step, lm-forward and lm-train-step through the
  same runners the board uses, N calls each, with the bindings' stage ticks
  on, and prints the LM session's `attention_stage_report` (which layers
  grew the quadratic stages, the answer to the 3.7 GB question).

Already true, so not written: the Samba optimizer is one flat binding call
over the packed registry (`_Optimizer.step`), not per tensor.

Not written, on purpose:

- **A Samba stack session.** The transformer and Mamba blocks live in two
  bindings with two contexts; a stack that keeps activations on the device
  across them needs the bindings merged or a shared-context protocol. With
  the two block sessions above, what remains per Samba forward is the
  embedding, the final norm and the head, three small round trips.
- **AMD GEMM tiles.** Blind edits are pointless: `gemm/checks/gemm_identical.mojo`
  already carries twenty plans and a runtime arm override,
  `MOJOLEARN_GEMM_ARM` (`shipped`, `lfold`, `half`, `quarter`, `head`,
  `ksplit`, `tuned128`, `kpack`, `kfoldv`, ... see `gemm_step_arm_parse`),
  and AMD-specific defaults already exist. The sweep is
  `for a in ...; do MOJOLEARN_GEMM_ARM=$a python tools/neural_stage_timing.py --lane lm-train-step; done`
  on the MI325X, and the identity gate on the winner. Every arm keeps the
  fixed fold tree, so the bits are the same by construction; the gate is
  the proof.
- **The 3.7 GB training-step peak**: diagnosed by the tool's report, not
  guessed at.

## Third pass: the L40S numbers, and the toggle round

The owner's L40S run of the first two passes (five rounds, medians, all six
output digests equal to before):

| cell | before | after | compiled torch |
|---|---:|---:|---:|
| transformer-forward | 4.130 | 4.359 | 1.794 |
| mamba3-forward | 3.858 | 4.248 | 8.402 |
| lm-forward | 105.836 | 50.425 | 37.663 |
| samba-forward | 10.408 | 8.647 | 3.934 |
| lm-train-step | 45.001 | 44.427 | 18.937 |
| samba-train-step | 141.844 | 137.481 | 40.199 |

Read: the LM-forward diagnosis was right (halved); the two block cells got
0.2 to 0.4 ms SLOWER, which is the exact byte compare of the weights on a
box where the upload it replaces was already cheap; the training steps did
not move, so their cost is the backward kernels, not orchestration. The
"before" for the block cells was already a newer wheel than the 0.8.25
record the first pass diagnosed from (4.1 ms, not 6.5).

So the third pass makes every candidate a RUNTIME TOGGLE, default off for
the new ones, and adds `tools/neural_experiments.py` to run them all on one
build and print one table with the digest check. `EXPERIMENTS.md` at the
repo root is the README: what each toggle does, where it should help, and
the order to try them per vendor. New in this pass:

- `MOJOLEARN_ATTN_SPECULATIVE=1`: the fused attention's regime scan behind
  the kernels, one host round trip per layer instead of two (plain and
  estash forwards). Same bits: a refused regime discards and reruns eager.
- `MOJOLEARN_SWIGLU_FUSED=1`: `swiglu_fused_kernel`, S20 and S21 in one
  launch, forward-only (`forward_only=True` threaded through
  `llama_decoder_layer_forward` from the block-forward entries and the
  LM logits path; the backward reads `silu_out`, so never where one
  follows; never with the trace on; never under the S20 sabotage).
- `MOJOLEARN_TRANSFORMER_STAGE_RESET=0`, `MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS=0`,
  `MOJOLEARN_MAMBA3_RETAIN_WEIGHTS=0`, `MOJOLEARN_TRANSFORMER_SESSION_FRESH=0`:
  the A/B arms of the first two passes.
- The timing tool prints `DIGEST` and `LOSSES` per lane so the sweep can
  flag a toggle that moved a bit.

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

## The classical pass (same branch, 2026-09-30)

Asked for after the L40S table: the board's worst GPU-versus-GPU cells.
Read from the code, unmeasured: lu-factor's pivot search and lu-solve were
one GPU thread each; sgd-* ran their sequential program on one GPU
thread; lars ran its 24,531 Gram chains on one block; the batched IVF scan
was gated to Apple. Four commits, each with an A/B env, each claiming the
same bits by construction (compares only; independent columns; the same
per-cell chain on another thread; the same kernel on another vendor). The
table and the run recipe are in EXPERIMENTS.md, "The classical pass".

Two things the identity gates must confirm before any of it ships: the
host SGD form equals the device form bit for bit (the tier's own claim, but
the route is new), and the IVF scan's results on a 64-lane AMD wavefront
equal the per-query path's (the merge now folds all 64 lanes).

## The priority-list pass (2026-09-30)

Ten commits `3f5dac18a`..`aa812f164` on the same branch, one per item of
the priority list handed over after the classical pass, all unmeasured;
the table with each commit, its change and its restore-env is in
EXPERIMENTS.md, "The priority-list pass". The reasoning in one line each:

* the Mamba-3 backward recomputed the forward it had just run: now it
  reuses the forward's stages on the session when x and the weights are
  byte for byte the forward's (a download-and-compare, paid by the
  backward only), else recomputes with the entry's own constructions;
* the LM step's last unrouted GEMM (the norm dW, `identical_gemm`, two
  waits per call) is on the retained workspace: 16 waits per step;
* the int8 unit plan read every fragment from device memory per unit
  step with nothing loaded ahead; a 32 x 32 warp tile halves the loads
  per unit step and prefetches the next step (order-free integer sums);
* eigh's only parallel route (round-robin Jacobi) changes the pinned
  order; it is exposed by env on every vendor as an explicit experiment,
  and the host route likewise; neither is on by default;
* svd's 95.8 s on istella was mostly a Python loop negating a million
  values per column; the kit's per-column multiply does the same flip;
* the x_decomp Cholesky ran on one thread; it is a 2n-launch column
  driver now, the same cells in the same order per cell;
* lr-warmup-cosine ran exact rational arithmetic per step; a binary64
  evaluation with a proven error bound decides the same float32 unless
  the value sits on a rounding boundary (12,927 values checked equal);
* adafactor folded 16.7M squares on one GPU thread twice a step; the same
  chain runs on the host from the caller's bytes and one download;
* clip_grad_norm_ packed and unpacked every gradient on the host; the
  tensors now go to their device slices directly;
* perceptron / PA / one-class SVM already take the classical pass's host
  SGD route (verified: all four fit through `_sgd_fit` -> `ALGO_SGD`).

What only a box can say: whether each row moves the cell it targets, and
whether every "same bits" row's digest stays. The compile risk is real
too: none of the Mojo here was built (no toolchain in this session); the
measuring agent's first build will find the syntax slips, as it did on
the first pass.
