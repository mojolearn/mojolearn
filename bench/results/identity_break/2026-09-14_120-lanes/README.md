# 120 lanes on three GPUs at the harness fix (2026-09-14, commit 65ae7612f)

The record the claim-surface session asked for after its lane fix (byte-lm-resident hashes the
flat gradients, 43f153247) and the harness refusing object arrays by name (65ae7612f). Every
column is the 120-lane `tools/identity_break.py` at 65ae7612f, nine hostile fixtures, two fits per
cell, train, infer and model columns, the host set built as the CPU column beside the GPU set. No
GPU source changed between the 2711 flip (83380ca6d) and this commit, so the Apple bindings are the
flip's.

| column | box | cells |
|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal, two-core cap | 1080: stable 1080, moved 0, refused 0 |
| nvidia-h100-sm_90a | RunPod H100 80GB HBM3 (`bench/results/e1g/2026-09-14_123742-nvidia-h100-identity-120-lanes-clean`) | 1080: stable 1080, moved 0, refused 0 |
| amd-mi300x-gfx942 | Hot Aisle MI300X 8core, 22.04 ROCm container (`...2026-09-14_131953-amd-mi300x-hotaisle-identity-120-lanes-clean-b`) | 1080: stable 1080, moved 0, refused 0 |

`diff.apple-nvidia-amd.txt`: `summary: DIVERGENT=9, IDENTICAL=1071` and `summary (infer/model):
IDENTICAL=1476, N/A=684`. Every column is stable inside itself (no MOVED anywhere, the first
120-lane run with none), and the nine divergent cells are ONE lane:

- `mamba2` (the default dt_limit Mamba-2 lane) on all nine fixtures: the APPLE M4 column's `step`
  and `backward` parts differ from the H100 and MI300X columns, which agree with each other AND with
  every earlier record on all three vendors (base cell 5b05a3ecbd70248e, step 3fa40f29231409be,
  backward f252b3fc19f5f9f5; this Apple column has b09925d3d8b074a2 / 16508d9095dbaa6a /
  7c3fda003a3a0464). CORRECTED 2026-09-14 afternoon: this README first said the MI300X column
  differed; the per-column hashes say the Apple column did. Reproduced on the M4 with the harness's
  per-fit dump: after the 42 preceding lanes in one process the step output and every backward
  gradient are canonical NaN in every element, inputs and state equal to a cold run. DEVIATION
  2712, open: an uninitialized device read in the Mamba-2 step and backward, visible where the
  allocator hands back non-zero memory (`docs/lanes/BRIEF_resident_and_dtlimit_moved_2026-09-14.md`
  section 3.2).

A first attempt at the AMD column on a DigitalOcean MI325X (24.04 ROCm image) aborted with a GPU
memory access fault right after the mamba1 lane, before mamba2 (`amd-mi325x-gfx942.partial.*`,
102 rows, all stable; `bench/results/e1g/2026-09-14_130705-amd-mi325x-do-identity-120-lanes-clean`).

The CPU identity gate keeps diffing against the 47-lane record (every cell identical on three
vendors); this record becomes the gate's columns once 2712 is closed and the AMD column reads
identical on mamba2.
