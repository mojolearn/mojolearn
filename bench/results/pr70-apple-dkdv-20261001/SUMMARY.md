# PR #70 dK/dV transposed staging off on Apple (lane/neural-pass66), M3 Ultra, 2026-10-01

main 3befccd28 (incl. #65) vs lane/neural-pass66, stage x2, timers runs, races. Same bits: 4a8e781b0739a038, same losses.
The box read ~1.6x slow in absolute terms during this run (main forward 111 ms vs ~70 earlier); both arms ran in the same window.

| row | main | #70 |
|---|---|---|
| lm-train-step stage (ms) | 371.7 / 368.6 | 326.8 / 325.9 |
| lm-train-step race (ms) | 355.0 | 309.0 |
| lm-forward stage / race (ms) | 111.4 / 110.4, 103.8 | 111.2 / 111.4, 103.6 |
| attn.bwd_kvgrid_dkdv_pf (ms/layer) | 9.72 | 4.94 |
| bwd.attention (ms/layer) | 19.44 | 14.61 |

MERGED: the gate is compiled out on NVIDIA and AMD (code unchanged there; writer's sm_89/gfx942 compile).
