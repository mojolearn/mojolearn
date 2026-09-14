# The Mamba poison-and-band gate, 2026-09-14 (DEVIATIONS 2712 and 2713)

`pixi run check-mamba-poison` (`tools/mamba_poison_gate.sh`): the Mamba binding
built with `-D MOJOLEARN_MAMBA_POISON=1`, every scratch buffer filled with the
canonical quiet NaN and every Mamba allocation carrying a 4096-element NaN
guard band past its logical length, then `mamba1`, `mamba2`, `mamba2-dtlimit`
and `mamba3` cold through `tools/identity_break.py`, diffed against the three
vendor columns of `bench/results/identity_break/2026-09-14_120-lanes-2711flip`.
The verdict is positive: all 36 training rows IDENTICAL x4, or the gate fails.

Modes: 0 the gate; 1 the fix removed (`-D MOJOLEARN_MAMBA_2712_UNBOUNDED=1`,
`m2_ydiag_kernel` reads X_d rows past T again), must FAIL; 2 a planted read
past `silu_out`'s end with the band, must FAIL; 3 the same read without the
band, must PASS (the control that the band is what catches an over-read).

One directory per box: `mode<N>.json` is the poison column (mode 0 on every box;
the sabotage columns where kept), `mode<N>.verdict.txt` the gate's own lines.

| box | commit | mode 0 | mode 1 (fix removed) | mode 2 (over-read, band) | mode 3 (no band, control) | leg |
|---|---|---|---|---|---|---|
| Apple M4 (Metal), this Mac, 2 cores | 9ade3e2cd's tree before the rebase (a1b405898 content) | PASSED 36/36 | FAILED as required, 18/36 | FAILED as required, 18/36 (18 REFUSED: the NaN refusal) | control PASSED 36/36 | local, `tools/mamba_poison_gate.sh` |
| AMD MI325X (DigitalOcean, Ubuntu 24.04 ROCm image; every Mamba-2 launch faulted here this morning, DEVIATION 2713) | 9ade3e2cd | PASSED 36/36 | FAILED as required, 18/36 | FAILED as required, 18/36 | control PASSED 36/36 | `bench/results/e1g/2026-09-14_155249-amd-mi325x-do-poison-gate-c` |
| NVIDIA H100 (RunPod, sm_90a, runpod/pytorch:2.4.0-py3.11-cuda12.4.1) | 9ade3e2cd | PASSED 36/36 | FAILED as required, 18/36 | FAILED as required, 18/36 | control PASSED 36/36 | `bench/results/e1g/2026-09-14_115741-nvidia` |
| AMD MI300X (Hot Aisle 8core VM, rocm/dev-ubuntu-22.04:6.4.1 container, gfx942) | 9ade3e2cd | PASSED 36/36 | FAILED as required, 18/36 | FAILED as required, 18/36 | control PASSED 36/36 | `bench/results/e1g/2026-09-14_160010-amd-mi300x-hotaisle-poison-gate-d` |
