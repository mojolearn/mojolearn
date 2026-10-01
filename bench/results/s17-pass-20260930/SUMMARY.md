# S17 tail pipe (PR #9, lane/neural-pass5) on NVIDIA L4, 2026-09-30: NOT MERGED (slower)

Evidence only; the code stays on lane/neural-pass5 (plus the compile fix `cur=nxt^` on lane/s17-measure-20260930:
an InlineArray is not implicitly copyable, the PR did not build).

| | shared (old default) | pipe (PR default) |
|---|---|---|
| Mamba-3 y + 10 gradients, both shapes | = strides/S16 digests | = strides/S16 digests |
| S17 tail kernel, board shape (B2 L512 d384) | 7.272 ms | 8.459 ms |
| S17 tail kernel, default shape (B8 L512 d768) | 41.125 ms | 45.997 ms |
| S17 operands kernel, board shape | 1.395 ms | 1.389 ms |
| samba-train-step (--set s17) | 104.9 ms | 113.2 ms |

Same bits; the pipe tail is 12-16% slower and the Samba step 8% slower. The per-kernel walls show the S17
stage is the tail chain (7.3 ms), not the operands kernel (1.4 ms).
