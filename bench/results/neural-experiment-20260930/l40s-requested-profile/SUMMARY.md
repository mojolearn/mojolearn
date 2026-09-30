# L40S requested profile, 2026-09-30

NVIDIA L40S (sm_89), identical mode, mojolearn 0.8.31 wheel with candidate a84aadf2e bindings, 10 calls per lane, each lane run on its own.
Script: `neural_requested_profile.py`. Receipts and binding hashes: `complete.json`. The phase-timer runs synchronize, so their times are for diagnosis, not the race.

## Per-call time, ms (median of calls 1 to 9)

| lane | saved binding | phase timers on |
|---|---|---|
| transformer-forward | 2.68 | 2.90 |
| samba-train-step | 153.3 | 150.4 |
| lm-train-step | 45.5 | 51.8 |

## transformer-forward, one steady call (saved binding, ~2.7 ms)

- weights_up (weight compare) 0.23, cache_stages_x_up (input upload) 0.44, outputs_down 0.20
- surface.forward 1.69: attention total 1.08 (attn.core 0.74, qkv_proj 0.21, o_proj 0.08, rope 0.04), mlp_and_residuals 0.48, norm1 0.10
- With timers: attn.core 0.74 = fwd_r2_keep_kernel 0.69 + regime_scan 0.03 + corner_flag 0.02
- 2 syncs per block (the refuse call's two host reads), about 25 launches per block

## lm-train-step, one steady step with MOJOLEARN_STEP_PHASE_TIMERS=1 (51.8 ms)

- 558 launches, 162 synchronizes, 66 D2H copies, 16 D2D copies, 89 device allocs, 63 host allocs
- native call 51.5 = blocks_backward 31.3 + blocks_forward 15.1 + rest
- 8 layers: bwd.attention 15.3, bwd.mlp_through_oproj 10.3 (136 launches, 16 syncs), block.attention_total 9.2
- Scans per step: validate_after 327 MB, validate_grads 82 MB, ce_refuse 67 MB; weight unpack and grad pack 82 MB each
- Fused attention: FUSED_RAN on all 8 layers forward and backward, no corner or regime refusals

## samba-train-step (~150 ms)

- The transformer blocks account for about 10 ms of the step (surface.forward 3.4, backward 2.7, weights_up 2.8, outputs_down 2.3). The rest is outside the instrumented transformer surface, i.e. the Mamba side.
