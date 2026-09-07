# Metal preparation: host checks passed; GPU launch not admitted

Latest common snapshot: **`d17c1aa1e977fb3831da2a8d2ca39de60c37b2c5`**.
It adds the optional Metal continuous-state comparator to the initial snapshot
below, without another numerical-inventory change.
[Latest snapshot record](snapshot-with-metal-comparator.json),
[latest source archive](common-source-with-metal-comparator.tar.gz).

Root prepared and archived common source
`59e35532433f3386cecb8c56e0ca0e6a5253d23d` in
`/tmp/mojolearn-byte-lm-metal-20260907`.
[Snapshot record](snapshot.json), [source archive](common-source.tar.gz),
[213-file inventory and exact delta](common-source-inventory.json).

Only four inventoried portability files differ from the retained two-vendor
source: native vendor admission, Darwin build flags, Python immutable-byte
checkpoint loading and capture platform/transfer handling. No mathematical
kernel file changed. Exact source equality remains required: this new source
needs fresh NVIDIA/AMD captures before underpinning a three-vendor result.
Earlier records remain unchanged.

Root's host-only checks passed **88 tests plus seven subtests**:
[passing log](host-portability-r2.log), [resource record](host-portability-r2.guard.json),
[tested file hashes](host-tested-source-sha256.json). The first run had three
fixture failures because Darwin temporary paths begin with the `/var`
symlink. Ordinary fixtures now use their physical directory; production
symlink refusal is unchanged and still tested. The initial failure is retained.

Tests used fake native bindings, bounded files and mocked supervisor events;
no compiler or GPU model ran. Root used two-thread environment limits,
a 60-second deadline and 1 GiB sampled RSS ceiling; the passing run's peak
observed RSS was 71,600 KiB. Darwin hard CPU affinity is not claimed.

The initial Mac telemetry and 06:28 UTC recheck both lacked the 4 GiB launch
reserve. The recheck saw about 116 MiB free/speculative, normal pressure and
no swap on a 16 GiB machine. **No native launch is admitted.**
[Initial observation](initial-preflight.json), [recheck](rechecked-preflight.json).
The user has been asked to close unused memory-heavy applications. Root does
not terminate unrelated applications or weaken the reserve to obtain a pass.

After memory readiness, root will validate actual guarded child supervision,
build/readback and one step, then full learning/resume evidence. The Metal
receipt policy admits only capture safety evidence; it does not pretend Metal
ran the CUDA/HIP FP64 oracle. Optional `--metal` compares all continuous states,
heldout bytes and checkpoints, requires a Metal guard receipt and the same
source inventory, and distinguishes that from the NVIDIA/AMD resume direction.
Root passed 38 focused comparator/policy checks plus seven subtests, then
re-admitted both old NVIDIA/AMD raw resume records with the new comparator.
[Focused test log](optional-metal-comparator.log),
[NVIDIA→AMD regression](nv-to-amd-comparator-regression.json),
[AMD→NVIDIA regression](amd-to-nv-comparator-regression.json).
Those are file-only regressions, not new GPU runs. Host tests and authored
Metal paths do not prove Apple numerical agreement.
