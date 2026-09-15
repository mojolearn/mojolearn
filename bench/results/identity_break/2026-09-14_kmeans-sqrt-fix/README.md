# The kmeans lanes after the kmeans-sqrt fix, at 9fde8f5f7 (2026-09-14 night)

A follow-up to the 166-lane record at 1eea14f80
(`bench/results/identity_break/2026-09-14_166-lanes`), whose one DIVERGENT cell
was `kmeans-sqrt/wide`, part `inertia` only: Apple and AMD 52ea06cbbcc24144, the
H100 1a7e4ac5b8c0caaf. This directory carries the eight kmeans lanes (`kmeans`,
`kmeans-random`, `kmeans-array`, `kmeans-weighted`, `kmeans-sqrt`,
`kmeans-classic-pp`, `kmeans-cosine`, `par-kmeans`; 72 training cells) on the
three record columns plus a second NVIDIA architecture, built from the fix, and
the `kmeans-sqrt` lane built from the unfixed source on the same boxes.

## What was wrong, two defects

1. **DEVIATION 2715, the NVIDIA inertia.** Both min-reduce kernels
   (`cluster/impl/distance/fused_distance_nn/simt_kernel.mojo`,
   `cluster/impl/distance/unfused_distance_nn.mojo`) rooted the reduced
   L2SqrtExpanded distance with the stdlib `sqrt`, which on NVIDIA lowers to the
   approximate PTX square root (DEVIATION 258). The fit sums those roots into
   `inertia_`. Now `identical_sqrt`. The argmin runs on the squared value first,
   so centers and labels never saw it.
2. **DEVIATION 2716, wrong labels on every column.** `kmeans_fit`
   (`cluster/estimator.mojo`) computed the row norms for `fit_predict`'s final
   assignment with the root when the metric was L2SqrtExpanded, so that pass
   used `||x|| + ||c||^2 - 2 x.c`, clamped most rows to 0 and gave them the
   lowest key. cuVS takes squared norms for both L2 metrics
   (`detail/kmeans.cuh:141-144`, `:1082-1085`), and so does the fit's own norm.
   Every column agreed on the wrong answer, so the record read IDENTICAL on it.
   The host oracle (`cluster/host/kmeans_oracle.mojo`) mirrored the same
   mistake and is fixed the same way.

## The bits

`PROBE` lines from `legs/*.gate.txt` (each box built the core binding twice:
arm U with the four k-means source files of main e2d770ba8 swapped in, arm F
with the fix restored; the two binding sha256 differ on every box). Wide
fixture, `metric="l2_sqrt_expanded"`, `np.float64(m.inertia_).tobytes().hex()`:

| column | unfixed inertia | unfixed labels not the argmin | fixed inertia | fixed labels not the argmin |
|---|---|---|---|---|
| apple-m4 (one core, M4, shared machine) | 00000000e5f28d41 = 62807200.0 | 9675 of 20000 | 00000000e5f28d41 | 0 |
| nvidia-h100-sm_90a (RunPod) | **000000e0e4f28d41 = 62807196.0** | 9675 | 00000000e5f28d41 | 0 |
| nvidia-rtx-4090-sm_89 (RunPod) | **000000e0e4f28d41 = 62807196.0** | 9675 | 00000000e5f28d41 | 0 |
| amd-mi325x-gfx942 (DigitalOcean) | 00000000e5f28d41 = 62807200.0 | 9675 | 00000000e5f28d41 | 0 |

The unfixed NVIDIA value is one float32 ulp below Apple's and AMD's; both NVIDIA
architectures carry it, so the column that stood alone was NVIDIA, not the H100.
The inertia hashes: float64(62807200.0) is 52ea06cbbcc24144 and
float64(62807196.0) is 1a7e4ac5b8c0caaf, the record's two values. The `base`
fixture's inertia (00000060e4b6f140) and the L2Expanded inertia and labels are
the same on all columns and both arms.

## Verdicts

- `diff.three-columns.txt` (apple-m4, nvidia-h100-sm_90a, amd-mi325x-gfx942,
  `--require-columns 3`, exit 0): `summary: IDENTICAL=72`,
  `summary (infer/model): N/A=144`, `summary (batch): N/A=72`.
- `diff.four-columns.txt`, the same plus nvidia-rtx-4090-sm_89: IDENTICAL x4 on
  all 72.
- `diff.unfixed.txt` (`unfixed/`: the Apple column from the 1eea14f80 binding,
  the three boxes' arm U, `--require-columns 4`, exit 1):
  `summary: DIVERGENT=1, IDENTICAL=8`, the row
  `kmeans-sqrt/wide | DIVERGENT | parts differ: inertia; agree: centers,labels,scales`,
  both NVIDIA columns together against Apple and AMD. The unfixed H100 and MI325X
  `kmeans-sqrt` cells are IDENTICAL to the 166-lane record's cells of the same
  column on all nine fixtures, so arm U is the recorded code.
- `diff.166-vs-fixed.<column>.txt` (each column of the 166-lane record against
  the fixed build of the same column, the eight lanes): `summary: DIVERGENT=9,
  IDENTICAL=63` on all three. The nine are `kmeans-sqrt` on every fixture, part
  `labels` (DEVIATION 2716), plus `inertia` on nvidia `wide` (DEVIATION 2715);
  centers, scales and every cell of the other seven lanes are unchanged.

The CPU identity gate asserts the first two verdicts (the fixed three columns
exit 0 with `IDENTICAL=72`; the unfixed columns exit 1 with exactly that
DIVERGENT row), so a regression of either fix, or a fix that stops being seen,
fails a step.

## Legs

| column | leg | lease used |
|---|---|---|
| nvidia-h100-sm_90a | `bench/results/e1g/2026-09-15_004700-nvidia-h100-kmsqrt` | about 7 min |
| nvidia-rtx-4090-sm_89 | `bench/results/e1g/2026-09-15_004659-nvidia-rtx4090-kmsqrt` | about 8 min |
| amd-mi325x-gfx942 | `bench/results/e1g/2026-09-15_004701-amd-mi325x-kmsqrt` | about 4 min |
| apple-m4 | this Mac, one core, `bindings/build.sh` in the lane worktree | local |

`legs/body_nvidia-h100.sh` is the H100 body; the other two differ only in
`LABEL`. The raw leg archives stay untracked.
