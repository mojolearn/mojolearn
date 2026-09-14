# 120 lanes, three columns, at the DEVIATION 2712/2713 fix (2026-09-14)

The record the CPU identity gate diffs its CPU column against
(`python/mojolearn/host_surface.py` TRAINING_GPU_COLUMNS), taken at commit
9ade3e2cd (the m2_ydiag_kernel fix, every Mamba device buffer filled, the
poison-and-band gate), with every binding rebuilt from that commit on each box.
Nine hostile fixtures, two fits per cell, train, infer and model parts.

| column | box | how | cells |
|---|---|---|---|
| apple-m4 | Apple M4 (Metal), this Mac, one core for the harness | every binding rebuilt in the worktree, `tools/identity_break.py --vendor apple-m4` | 1080 stable, 0 moved, 0 refused |
| nvidia-h100-sm_90a | RunPod H100 | `tools/identity_three_columns_leg.sh` through tools/gemm_remote_leg.sh, leg `bench/results/e1g/2026-09-14_120658-nvidia` | 1080 stable, 0 moved, 0 refused; with apple-m4 and amd-mi325x-gfx942 IDENTICAL x3 on 1080 training and 1476 inference and model cells (`diff.apple-h100-mi325x.txt`) |
| amd-mi300x-gfx942 | Hot Aisle MI300X 8core, 22.04 ROCm container | the same body through tools/hotaisle_leg.sh, leg `bench/results/e1g/2026-09-14_161225-amd-mi300x-hotaisle-identity-120-lanes` | 1080 stable, 0 moved, 0 refused |
| amd-mi325x-gfx942 | DigitalOcean MI325X, 24.04 ROCm image; this image's FIRST full column (its earlier run died after mamba1, DEVIATION 2713) | the same body through tools/do_extra_leg.sh, leg `bench/results/e1g/2026-09-14_160236-amd-mi325x-do-identity-120-lanes` | 1080 stable, 0 moved, 0 refused; against apple-m4 IDENTICAL=1080 and (infer/model) IDENTICAL=1476, N/A=684 (`diff.apple-mi325x.txt`) |

The three gate columns together (`diff.three-columns.txt`, apple-m4, nvidia-h100-sm_90a,
amd-mi300x-gfx942, `--require-columns 3`): summary IDENTICAL=1080, (infer/model)
IDENTICAL=1476, N/A=684, require-columns 3 over all 120 lanes OK, no MOVED, no
DIVERGENT, no ONE-COLUMN cell. The MI325X column against apple-m4 and the H100
(`diff.apple-h100-mi325x.txt`): IDENTICAL x3 on the same 1080 and 1476. The
CPU gate's TRAINING_GPU_COLUMNS are taken once more at the 136-lane record
(the multi-GPU par-* lanes, 149707346) rather than here.

`diff.apple-vs-2711flip.txt` holds the new Apple column against the four-GPU
record `2026-09-14_120-lanes-2711flip`: IDENTICAL on every cell except that
record's own nine `byte-lm-resident` MOVED cells (a lane bug of that day,
fixed 43f153247) and the `tokenizer` lane its Apple column had refused. The
`mamba2` base cell is 5b05a3ecbd70248e on every column, the value every
record on main carries; the Apple column of `2026-09-14_120-lanes` that read
b09925d3d8b074a2 was DEVIATION 2712, closed by this commit (SUPPORT_MATRIX).

The Mamba poison-and-band gate at the same commit: `bench/results/mamba_poison_gate/2026-09-14/`.
