# Native autograd usability qualification

Numerical source: `7939b679a2079051f191d707067a487a9741fcce`.
This record covers source builds in `identical` mode. It is not a performance
measurement or an installed-wheel qualification.

| Executor | Passed | Skipped | Matching digest keys |
|---|---:|---:|---:|
| Apple M4 GPU | 106 | 0 | 96 |
| NVIDIA RTX 3090 GPU | 106 | 0 | 96 |
| AMD MI325X GPU | 106 | 0 | 96 |
| Apple CPU | 104 | 2 | 96 |

The CPU skips its two GPU-transfer cases. Both transfer directions pass on
each GPU. All 21 runtime-fence tests pass.

Coverage includes seeded Sequential dropout and resume, validation loss and
label mapping, patience and complete best-state restoration, native captured
graph minibatches/optimizers/custom objectives, frozen constants, variable
batch and sequence dimensions, failed-call recovery, and independently checked
LayerNorm Hessian-vector products. The nine suites also cover the existing
operation, optimizer, program, rank-N, public example and checkpoint APIs.

`summary.json` contains counts, every digest and native binary hashes. The
final GPU runs read back `metal`, `cuda` and `hip`, respectively, and numeric
mode `1` from the loaded sequence bindings. Earlier Linux runs are retained
but not admitted: an incomplete source-layout install selected the host
binding, the first AMD environment lacked Mojo's shared runtime, and initial
binding probes needed the native enum rather than a string. Repairs reused
the same frozen binaries.

Complete logs, failed attempts, source bytes, native binaries and model
artifacts are retained in
`~/mojolearn-evidence/autograd-usability-20261020/` and at the R2 object named
by `summary.json`. The source tar created on the patch-synced build box
initially omitted untracked additions; the retained sync patch, complete
source tar and immutable Git source archive preserve those files too.
NVIDIA model artifacts were retained by verified SHA-256 byte matches to
existing local files plus a fetched delta; all 49 files were verified.

See [implementation and remaining boundaries](../../../../docs/plans/AUTOGRAD_USABILITY.md).
