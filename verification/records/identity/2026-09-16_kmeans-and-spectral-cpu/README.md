# The k-means NVIDIA column, and spectral's CPU column at 512 rows (2026-09-16)

`lane/classical-host-recordings`, one RunPod **NVIDIA A100-SXM4-80GB (sm_80)**,
pod `ujwcpglxvl4lo7`, 18:14 to 18:22 UTC, terminated and VERIFIED gone
(HTTP 404). Commit `8511c8c518088829c86cd052a1dee0125cd1939e`.

## Files

| file | what it is |
|---|---|
| `nvidia-a100-sm_80.json` | the k-means column, seven lanes, nine fixtures, two repeats |
| `nvidia-a100-sm_80.batch-sabotage.json` | the same with `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` |
| `nvidia-a100-sm_80.gpu-infer.json` | arm A, infer on the GPU, base fixture |
| `nvidia-a100-sm_80.host-infer.json` | arm B, the same fits through `mojolearn.host_model` on the CPU host binding |
| `nvidia-a100-sm_80.host-sabotage.json` | arm D, arm B against a sabotaged core host binding |
| `cpu-x86.spectral.json` | spectral and spectral-precomputed, x86 CPU column, nine fixtures, two repeats |
| `cpu-x86.spectral.host-sabotage.json` | its negative control, `MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE` |

## The k-means column

    cells=63 stable=63 moved=0 refused=0

Against the four training columns of `2026-09-14_kmeans-sqrt-fix` (Apple M4,
NVIDIA H100 sm_90a, AMD MI325X gfx942):

    summary: IDENTICAL=63          (diff.kmeans-four-columns.txt)

Its infer and batch cells against the two Apple columns of
`2026-09-15_kmeans-transform` (Metal and the M4's own CPU host):

    summary (infer/model): IDENTICAL=24, ONE-COLUMN=84
    summary (batch): IDENTICAL=24, ONE-COLUMN=30

Every cell those columns ran is IDENTICAL x3. The ONE-COLUMN cells are the
`model` part on every fixture and everything on the fixtures those columns did
not run. **`model` is ONE-COLUMN everywhere and that is not a divergence**: no
other column in the tree carries a real k-means `model` digest, because
`KMeans.save` is one day old (`lane/kmeans-save`, 2026-09-16). A strict scan
requiring `^[0-9a-f]{16}$` over every record on main says so lane by lane
(`~/mojolearn-evidence/classical-host-recordings/gap-scan-strict.txt`); a scan
that accepted any sixteen-character string would have called `n/a:no-save` a
hash, which is the false positive that hid the predict lanes' gap from
`lane/saved-model-reference-gaps`' first pass.

`kmeans-cosine` carries no digest in ANY column, here or anywhere, and must
not: its cells are the refusal sentence by design.

### Arms C and E, which lane/kmeans-save could not take

Arm C, the deliverable (GPU fit -> saved file -> `mojolearn.host_model` on the
CPU host binding -> predict and transform):

    train  IDENTICAL = 7
    infer  IDENTICAL = 12   model IDENTICAL = 12   (N/A = 2, kmeans-cosine)
    batch  IDENTICAL = 6                           (N/A = 1, fit-refused)

which is `lane/kmeans-save`'s L40S result reproduced on an A100.

Arm E, its negative control, which on the L40S leg did not exist because the
sabotage build exited 127 and every cell read REFUSED:

    summary (infer/model): DIVERGENT=6, N/A=2, RELOAD-MOVED=6
    summary (batch): IDENTICAL=6, N/A=1

**No REFUSED cell.** The six `model` cells read RELOAD-MOVED, which is the
harness saying the file reloaded through the sabotaged binding no longer
predicts what the model in memory predicted; that is a detection, not a
refusal. The batch cells are IDENTICAL because the batch protocol runs on the
GPU, which this arm does not touch; their own control is the batch sabotage
column, `BATCH_MOVED = 6 of 6`.

Arm F, the boundary in bytes, five fitted shapes: all five `fitted on cuda;
loaded as HostKMeans on cpu` IDENTICAL on predict, transform AND predict over
the training rows against `labels_`; the one-float32-ULP rewrite of
`centers[0]` fired on all five, moving 37 to 146 of 1280 or 1536 transform
cells and NAMING `predict` as unmoved each time. `cases=5 failed=0`. A single
ULP on one centroid need not change a label, because a row's nearest center is
decided by a margin and not by the last bit; reporting only the movers would
read as though it had.

## Spectral's CPU column, at the published size

    cells=18 stable=18 moved=0 refused=0

`lane/saved-model-reference-gaps` found that the spectral lane was shrunk from
2000 rows to 512 at `e2bb9e541`, which is NOT an ancestor of the 2026-09-15 CPU
column (`0a6957015`), so the diff tool's own `LANE_REVISIONS` excluded it and
read its cells as absent. That lane retook the Apple column and left the CPU
one owed. This is it, and against the two NVIDIA columns and the Metal column
of `2026-09-16_predict-nvidia`:

    summary (infer/model): IDENTICAL=36
    summary (batch): IDENTICAL=18

with no exclusion note, on all nine fixtures (`diff.spectral-four-columns.txt`;
the Metal column ran base and ties, so those two read IDENTICAL x4 and the
other seven IDENTICAL x3).

**How a CPU column came off a GPU box.** A package copy with `identical/`
removed routes every call to the host bindings, which are the same
`MOJOLEARN_TARGET_COLUMN=cpu` binaries a CPU-only pod builds; the GPU on the
box is simply not reachable from that PYTHONPATH. The leg asserts
`mojolearn.vendor() == 'cpu'` on the copy before the column is taken and
refuses otherwise, and the JSON records `host.column = cpu`. It saved a second
rental; it is the same arithmetic, not a shortcut around it. Its own negative
control, on the same copy, is `cpu-x86.spectral.host-sabotage.json`: DIVERGENT
on infer and batch for both lanes on base and ties.
