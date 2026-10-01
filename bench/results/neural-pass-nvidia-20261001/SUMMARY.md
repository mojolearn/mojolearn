# NVIDIA L40S neural pass, PR #55 and PR #56 (Oct 1)

Box: RunPod nvc1 (L40S), released 0.8.33 overlay venv with the measure branch bindings rebuilt.
Raw: R2 `measurements/2026-10-01/neural-pass-nvidia.tar.gz`, `measurements/2026-10-01/pr56-nvidia.tar.gz` (see r2-index.tsv); condensed lines in summary-raw.txt.

Every run below gave lm-forward digest 4a8e781b0739a038 and the same loss trace, so the bits match the other vendors.

## PR #55 (GEMM tile-min-blocks knob; AMD default 1024, NVIDIA default unchanged)

| setting | lm-forward ms | lm-train-step ms |
|---|---|---|
| default | 51.1 | 40.7 |
| TILE=1024 | 53.0 | 49.0 |
| TILE=512 | 51.7 | 47.0 |
| TILE=256 | 51.6 | 47.0 |
| SPLIT=1M | 51.0 | 41.1 |

The large 4096³ GEMM race goes 54.1 to 47.1 ms with TILE=1024 (digest 535b4c27bd9313d1 both), but the LM stages get slower, so NVIDIA keeps its default. The device check and the 8 GEMM gates are green.

## Attention arm sweep (NVIDIA)

The default arm is the fastest. kvgrid_r64 ties it; bswz off and no_dres are slower on lm-forward. fgrid_r64, kvsplit, kvrecompute, zdefer and zlag are invalid compositions on this base (the binding refuses them: qres needs fgrid_r32; the z suffixes need a different base), so they were not timed.

## PR #56 (scratch pool)

| run | lm-forward ms | lm-train-step ms |
|---|---|---|
| pool 1 / 2 | 49.6 / 51.7 | 40.6 / 40.9 |
| no pool 1 / 2 | 53.1 / 50.4 | 41.1 / 40.9 |

On NVIDIA this is neutral (within noise); the attention phase timers show only sub-0.05 ms scan and flag phases.
