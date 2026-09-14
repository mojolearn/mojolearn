# Every identity_break lane on the GPUs, 118 lanes (2026-09-14, commit 71faae781): the two GPU columns

The first three-vendor run of the 118-lane harness (the 47 lanes of the morning plus the 71
census lanes merged as 71faae781, one per public constructor value that selects a numeric path).
Two columns are here; the Apple M4 column is retaken at the next commit because the process that
was writing it lost its estimators binding mid-run (an in-place overwrite of the shared checkout's
binary by the orchestrator, recorded in the handoff) and it is not trusted.

| column | box | commit | cells |
|---|---|---|---|
| nvidia-h100-sm_90a | RunPod H100 80GB HBM3 (`bench/results/e1g/2026-09-14_110207-nvidia-h100-identity-118-lanes`) | 71faae781 | 1062: stable 1053, MOVED 9, refused 0 |
| amd-mi300x-gfx942 | Hot Aisle MI300X, 22.04 ROCm container (`bench/results/e1g/2026-09-14_110207-amd-mi300x-hotaisle-identity-118-lanes`) | 71faae781 | 1062: stable 1052, MOVED 10, refused 0 |

`diff.nvidia-amd.txt`: `summary: DIVERGENT=8, IDENTICAL=1044, MOVED=10` and
`summary (infer/model): DIVERGENT=9, IDENTICAL=1440, N/A=675`. Two lanes carry every non-identical
cell; every other lane, including all 47 of the morning's and 69 of the 71 new ones, is identical
between the H100 and the MI300X on every fixture and column.

Open, found by this run:

- `byte-lm-resident` MOVED on all nine fixtures on BOTH boxes: NOT a deviation, lane hashing,
  fixed on main at 43f153247. Per part only `grads` moved; loss, params and logits were equal
  between the two fits and equal to the stateless byte-lm lane bit for bit on both GPUs. The lane
  hashed `np.asarray` of a nested gradient dict, a 0-d object array whose bytes are the dict's
  address. The fixed lane hashes the flat gradients and holds the resident export to the
  stateless step's gradient; its cells come from the next run.
- `mamba2-dtlimit` DIVERGENT between the two vendors on 8 of 9 fixtures (parts forward, prefill,
  step, backward) and MOVED once on the MI300X (`base`: 165b502a1c95f280 then 669a2db9147247f7,
  the second value being the H100's). One moved cell in nine on one vendor reads as a race on the
  AMD side of the dt clamp path, to be diagnosed the same way.

The Mac smoke of the 71 new lanes ran one repeat per cell and could not see either.
