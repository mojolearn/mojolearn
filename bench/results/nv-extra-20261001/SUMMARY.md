# L40S follow-up: stage timing, toggles, attention phases, transformer-forward wheel race (Oct 1)

Box: RunPod nvc1 (L40S). Raw files are in R2 at `measurements/2026-10-01/nv-extra.tar.gz`; summary-raw.txt holds the condensed lines.

## Transformer-forward: no regression
The board showed transformer-forward at 6.5 ms on 0.8.25 and 14.1 ms on 0.8.33. A direct race of the three wheels on one GPU gives:

| wheel | median ms | digest |
|---|---|---|
| 0.8.25 | 4.39 | d5a2b289afdb5709 |
| 0.8.32 | 3.98 | d5a2b289afdb5709 |
| 0.8.33 | 3.99 | d5a2b289afdb5709 |

The board's 14.1 ms was not a slowdown in the wheel itself.

## Toggles (ms: lm-forward / lm-train-step / transformer-forward)

| run | lm-forward | lm-train-step | transformer-forward |
|---|---|---|---|
| main | 49.7 | 41.1 | 2.39 |
| LAYER_SYNC=0 | 51.3 | 41.1 | 2.40 |
| SPECULATIVE=1 | 51.3 | 41.0 | 2.37 |
| ARENA=1 | 51.3 | 41.2 | 2.44 |

All runs had the same digests. None of the toggles is worth flipping.

## Attention phase ticks (ms per layer; lm-train-step + lm-forward)

| phase | ms |
|---|---|
| bwd.attention | 1.72 |
| block.attention_total | 1.00 |
| attn.core | 0.67 |
| attn.bwd_dq_tiled_pf | 0.63 |
| attn.fwd_r2_keep_kernel | 0.61 |
| attn.bwd_zdot_estash_dres_pf | 0.51 |
| attn.bwd_kvgrid_dkdv_pf | 0.50 |
| bwd.after_attention | 0.46 |
| block.mlp_and_residuals | 0.45 |
| attn.qkv_proj | 0.21 |
