# Handoff: Samba sliding-window attention lane (2026-09-09)

Branch `lane/samba-attention` (worktree `.claude/worktrees/agent-afd8cf1f28065c190`),
on top of `f3d76e8d`. Wound down by orchestrator order before every item of the
brief was verified; what ran and what did not is listed exactly below.

## Commits

- `cb0a7e43 parent f3d76e8d` transformer: sliding-window attention with a ring KV
  cache, backward on the Python surface
- `7be6a6cf parent cb0a7e43` transformer: the capacity refusal fires before the
  key-span refusal (fixes one surface-test arm; NOT re-run on a GPU)
- (this file and `bench/results/attnlane_2026-09-09/` follow)

Files changed: `transformer/impl/transformers/models/llama/modeling_llama.mojo`,
`transformer/checks/transformer_oracle.mojo`, `transformer_fixture.mojo`,
`transformer_check.mojo`, `transformer_backward.mojo`,
`transformer_backward_oracle.mojo`, `transformer_backward_check.mojo`,
`bindings/_mojolearn_transformer.mojo`, `python/mojolearn/_transformer_impl.py`,
`python/mojolearn/tests/test_transformer_surface.py`,
`transformer/bench_window_timing.py` (new), `transformer/NOT_IMPLEMENTED.tsv`,
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` (section 11 rows).

## Sliding window and the KV ring, status

- `LlamaKVCache(ctx, b, dims, s_max, window=0)` and
  `LlamaDeviceStages(ctx, b, l, s_max, dims, window=0)`; `window == 0` is the
  old linear cache and the old code path exactly. `window = W > 0`: query at
  position p sees keys `[max(0, p-W+1), p]`; the cache is a ring of W slots
  (`slot = position % W`); `kv_window_gather_kernel` packs the call's key span
  `[llama_key_lo, pos0+l)` out of the ring before `kv_ring_write_kernel` writes
  the call's tokens; `attn_mask_kernel` takes `key_lo` and `window`. Every
  softmax fold still walks the packed span ascending from +0.0, so decode and
  split prefill equal the whole prefill bit for bit (masked head and tail are
  exactly +0.0).
- Oracle mirrors it (`TransformerKVCache(..., window=0)`, `key_lo()`).
- Fixtures 15-18 (`win4_b1_l16_nrep2`, `win3_b2_l8_nrep1`, `win5_b1_l12_hd24`,
  `win20_b1_l16_nrep2`), window by name in `fixture_window`; clause (a) default
  set is now 17 cases; `MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1` also runs
  clause (d) and the new `clause_d_split` on three window cases.
- Python: `TransformerBlock(weights, *, n_heads, n_kv_heads=None,
  head_dim=None, window=0)`; `TransformerState` gains `window` and `capacity`;
  ring `keys()`/`values()` return the held positions in order (a copy).
  Binding params: forward 10 scalars (window last), decode step 9.

## Backward binding, status

- `transformer/checks/transformer_backward.mojo` reads `fwd.window`
  (`bwd_mask_grad_kernel` sabotage arm and `bwd_kv_slice_kernel` offset);
  backward oracle takes `window=0`.
- `bindings/_mojolearn_transformer.mojo::transformer_backward` (21 addresses,
  8 scalars, IDENTICAL tier only) and
  `TransformerBlock.backward(x, grad_output) -> dict` (x plus nine weight
  gradients, zero-state prefill, forward recomputed inside).
- `test_transformer_surface.py`: window arms, ring decode, split prefill,
  snapshot round trip, backward vs a float64 reverse-mode oracle (itself
  checked by central differences) at window 0 and 3, two-call bit repeat.

## Numbers actually produced (NVIDIA L40S, driver 570.124.06 with
`MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`, Mojo 1.0.0, commit
`cb0a7e43`; logs in `bench/results/attnlane_2026-09-09/`)

- `check-transformer` identical, clause (a): PASS, 17 cases, 30/30 stages,
  349206 cells; clause (d) window 0 PASS (4 steps, 11632 cells); window 4
  PASS (16 steps, 20600 cells) and split 5+11 PASS (16890 cells); window 5
  hd24 PASS (12 steps, 24868) and split 7+5 PASS (17770); window 20 PASS
  (16 steps, 23720) and split 6+10 PASS (19100). Log `check_transformer.log`.
- `transformer.identical.card` md5 `8ce661b469681b18fb5cf4d566ad78ff`,
  BYTE-EQUAL (cmp) to the three-vendor reference
  `bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/transformer.identical.card`.
- `check-transformer-backward` identical, clause (a): PASS, 17 cases, 37/37
  stages, 412172 cells. `transformer-backward.identical.card` md5
  `7eee8da90ecb4dba1d154991e2e67e30`, BYTE-EQUAL to
  `bench/results/apple_cards_2026-09-03/transformer-backward.identical.card`.
- Binding builds: identical, fast, deterministic all rc 0 (`build_*.log`).
- Surface test identical at `cb0a7e43`: 115 checks, 113 pass, 2 FAIL
  (`surface_identical.log`): (1) the capacity refusal arm saw the new key-span
  message first, fixed in `7be6a6cf`, unverified; (2) the corpus-debt arm,
  because the shipped archive excluded `transformer/corpus/` (an archive
  artifact, not code). Every window and backward arm passed.
- Timing (`window_timing.log`, d_model 1024, 16 heads, 4 kv heads, head_dim
  64, intermediate 4096, window 2048, seq 4096, batch 4, median of 3):
  ours identical forward 685.7 ms, forward+backward 1672.1 ms (through the
  Python surface, host copies included); torch eager fp32 SDPA with the
  sliding-window mask forward 33.6 ms, forward+backward 106.9 ms;
  torch.compile 34.7 ms / 93.6 ms. The harness's last step (agreement vs
  torch) died with a torch CUDA OOM after all timings were logged.
- NOT done: the window-0 byte comparison against the parent build
  (`dump_forward.py` failed on import because it was run by absolute path
  without `PYTHONPATH=<tree>/python`; nothing compared). Not done: the
  surface test at `7be6a6cf`; the Apple column of everything.

## RUN OWED on the Apple M4 (orchestrator runs these, one at a time)

    MOJOLEARN_IDENTITY_TRACE=/tmp/attn.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 \
        tools/with_identical_mode.sh pixi run check-transformer
    cmp /tmp/attn.card bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/transformer.identical.card
    MOJOLEARN_IDENTITY_TRACE=/tmp/attnb.card tools/with_identical_mode.sh pixi run check-transformer-backward
    cmp /tmp/attnb.card bench/results/apple_cards_2026-09-03/transformer-backward.identical.card
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_surface

Expected: both cards byte-equal, 115 checks 0 failed (the corpus-debt arm
passes in the full checkout).

## Next commands for a fresh agent, in order

1. Rent one L40S (`tools/gemm_remote_leg.sh` pattern; on a driver below 580
   export `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`), ship
   `git archive 7be6a6cf` plus `transformer/corpus/`.
2. Re-run the surface test at `7be6a6cf`; expect 115/115.
3. Window-0 byte comparison: build the parent (`f3d76e8d`) binding in a
   second tree, run a dump of forward/step/split outputs with
   `PYTHONPATH=<tree>/python` for each tree, compare bytes.
4. Optional: `check-transformer` with `MOJOLEARN_TRANSFORMER_CHECK_LONG=1`
   and the backward gate with `MOJOLEARN_TFB_CHECK_CLAUSE_D=1` on the win*
   cases (the backward chunk clause has not been run under a window).
5. Terminate the pod; merge after the Apple RUN OWED list above is green.
