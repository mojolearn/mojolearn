# Root host-only portability and PCA dispatch checks

16 selected tests passed in 0.29 seconds: immutable checkpoint byte capture,
file delegation, existing checkpoint validation, and PCA full-export dispatch
sentinels. Root ran with two-thread environment limits, a 60-second deadline
and a 1 GiB sampled process-group RSS ceiling; peak observed RSS was 73,904 KiB.
No native model, Apple GPU kernel, build or performance benchmark ran.
The initial attempt used a Python without pytest and failed before collection;
the existing root-validation environment then passed. Both logs are retained.

The new immutable-bytes loader is groundwork for Metal portability; current
vendor allowlists remain CUDA/HIP. PCA full native export remains uncompiled
and numerically unqualified. Source hashes identify these host-test inputs;
none modifies the frozen ongoing remote checkpoint campaign or PyPI artifacts.

Nine macOS supervisor mock tests also passed under root's 30-second bound.
The cleanup warning lines are deliberately planted mocked cases, not a live
host cleanup event. Fixes coordinate the build lock, track observed escaped
descendants, enforce a separate timer and require verified cleanup. No actual
OS-telemetry supervision or Metal model was run; sampling/affinity and
supervisor-failure limitations remain explicitly documented.
