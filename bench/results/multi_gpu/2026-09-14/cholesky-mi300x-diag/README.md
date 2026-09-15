# Cholesky column solve on two MI300X: the divergence and the host-staged fix

All legs are RunPod 2x AMD Instinct MI300X (gfx942), 2026-09-14/15.

1. Two kernel-methods legs (`../kernel-methods-mi300x-failed/`, and the
   leg at `d24e39765`, which added two-context drains) failed
   `cholesky_parallel_check.mojo` at n=513 with two right-hand sides, with
   the same hashes both times. The one-device MI300X solve trace equals both
   H100 columns (`../kernel-methods-h100-drain/out/n513-r2/`); the MI300X
   two-device solve stands alone.
2. `sweep-gate.txt` (pod in `sweep-pod_id.txt`, commit `9750e0297`): n from
   300 to 512 at two and three right-hand sides pass; every n from 513 to 1024
   fails, and in each failing case every row of every right-hand side owned
   by device 1 differs while the column owned by device 0 is exact.
   512*512*4 bytes is exactly 1 MiB.
3. `transport-diag-gate.txt` (commit `8c9009364`, built with
   `-D MOJOLEARN_CHOLESKY_TRANSPORT_DIAG=1`): reading each owner's copies of
   the factor and the columns back to host before the solve showed no
   differing cell, and with those readbacks every sweep case passed.
4. `peer-copy-gate.txt` (commit `cda6bbfa0`): `peer_copy_check.mojo` copied
   4 KiB to 16 MiB patterns from device 0 to device 1 by `peer_clone` and by
   host staging, and a device-1 kernel read every cell correctly in both
   forms, twice. A bare peer copy does not reproduce the failure, so its cause
   is NOT identified.
5. `host-staged-gate.txt` (pod in `host-staged-pod_id.txt`, commit
   `bd1cac859`): the column solve now stages the factor and
   the right-hand sides through host memory and each owner's own context,
   with no device-to-device copy and no root gather kernel. The full gate
   passes 40 cases twice and the 300..1024 sweep passes all 26 cases.
   `host-staged-full-digests.txt` (152 trace files, one and two devices)
   equals the two-H100 `chol_trace_digests.txt` line for line.
