# lane/apple-identical-neural (Sep 26 2026)

Budget: 4 CPU threads, one Metal job (tools/mac_slot.py). Worktree ~/mojolearn-wt/apple-identical-neural.

## Baseline (origin/main 06b4ec239)
- Byte LM step, shape 1x2048 d768 h12 ff2048 L=2 V50257, resident lean, component timing: 4.37 s/step.
  head_backward_db 955, head_backward_da 870, head_forward 644, blocks_backward 1053, blocks_forward 422,
  unpack_weights 279, attention bwd 363 (ms; stage timers inflate Metal). base_step_2L/.
- T3 shard GEMMs (bench/gemm_excp_ab_main.mojo, M=8192): 218-300 GF/s = 6-7% of M4 peak (~4 TF).
  head_fwd 2.8 s, head_dA 3.3 s, head_dB 3.5 s; proj 32-44 ms; gateup 95-142 ms; down 99-130 ms.
  ~28 s GEMM per shard, x64 shards per T3 step.
- Window admission (already on for Apple) is worth only ~8% (nowin build).
- Existing alt plans slower, same hashes: TUNED_128_8X8 170/157/154/145 GF/s; 64x64 K32 156/122/124/120 (vs 300/229/260/218).

## simdgroup_matrix exactness (gemm/checks/apple_simdgroup_probe.mojo)
- air.simdgroup_matrix_8x8_multiply_accumulate fp32 on M4 == ascending identical_mul_add chain seeded with C,
  0 mismatches on 7 kinds x 4.19M cells (normal, cancellation, signed zeros, wide spread, Inf/NaN payloads,
  overflow, subnormal step results). Desc / unfused / C+chain differ on 0.3-4.2M cells (probe discriminates).
- => on windows admitted by TUNED_WINDOW_ADMIT, MMA gives the contract bits. Non-admitted windows keep rtf_mul_add.
- MUST re-run the probe on any other Apple generation (cloud Mac) before trusting MMA there. M5 has matrix units.

## PLAN_APPLE_MMA (a0fbdd06e)
- 64/64 (call,kind) hashes equal the shipped kernel (6 kinds incl tiny/skew/sparse). Ordinary: proj 2.3-2.6x, head_fwd 3724->1622, head_dA 3494->1116 (ragged k via exact tail), head_dB 3735->1677 ms.
- gemm_device_check / backward / workspace green; sabotage MOJOLEARN_GEMM_SABOTAGE_APPLE_MMA fails 3 device gates (backward check stays green under it: its shapes do not reach the matrix plan -- gap).
- Byte LM step (2L, V50257), CONSECUTIVE steps, no exports between (stepnw.sh): base 3.097/3.090 s -> mma 1.735/1.661 s = 1.8x; per-step witnesses equal (step.sh, witness-every-step).
- CAUTION: step.sh (witness-every-step) inflates the first submission after each export (paging; swap 15.8 GB): unpack_weights 150-530 ms vs 1.4 ms consecutive. Use stepnw.sh for timing, step.sh for witnesses.
- Fanless Air drifts slower as it heats (1.69 -> 1.97 s over 4 runs): always alternate A/B.

## emb/head views (63a918619)
- Correct (witnesses equal, step_glue_check PASS incl noshadow swap arms + rollback) but NO step-time gain in consecutive steps (1.686 vs 1.711, 1.969 vs 1.992). Gain = 309 MB memory. Commit message overstated; corrected in code comment.

## Next (step ~1.7 s, component breakdown consecutive)
- head GEMMs ~250 ms each (fwd, dA, dB) = ~43%; blocks fwd ~255 ms, blocks bwd ~500 ms (attention bwd ~290, attn core ~140).
- GEMM kernel still ~400-450 GF/s vs 1287 ceiling: vector staging, double buffer, avoid per-element exp-min.
