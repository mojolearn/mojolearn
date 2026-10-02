# regs2h as the AMD S16 default (PR #50, lane/mamba3-amd-regs2h, main merged in), 2026-10-01

MI325X: released 0.8.32 + this branch's Python and mamba/transformer/byte_lm/training bindings; default (regs2h)
vs the restore MOJOLEARN_MAMBA3_S16_QK_ARM=regs2. M3 Ultra: branch source tree, Metal default.

| | MI325X default (regs2h) | MI325X regs2 (old default) | M3 Ultra default |
|---|---:|---:|---:|
| strides_digest 2x512x384 md5 | 26abf7d3 | 26abf7d3 | 26abf7d3 |
| Mamba-3 backward stage sum | 57.1 ms | 71.0 ms | 169.9 ms (unchanged) |
| samba-train-step (median of 7 after the first) | 177.7 ms | 196.2 ms | |

Losses identical over 8 steps. NVIDIA default untouched (comptime AMD only; cross-compile run 36858932841 green).
Raw job directories: R2 measurements/2026-10-01/pr50-amd.tar.gz, pr50-metal.tar.gz (bench/results/r2-index.tsv).
