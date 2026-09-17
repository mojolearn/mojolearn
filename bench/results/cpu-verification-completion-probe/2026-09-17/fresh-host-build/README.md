# Fresh Mac host bindings, development artifact

All 32 host families built from frozen `7f5b786ae` with one compiler worker
in 244.7 seconds. The receipt records the source archive hash, toolchain,
commands and each original binary hash. Staging rewrites runtime paths and
signs the bindings; wheel receipts record the staged hashes separately.
All 32 installed host-export audits pass. Source archives and wheels are
retained externally under `~/mojolearn-evidence/cpu-verification-completion`.

The first dependency-free installed verifier invocation failed with an
unhandled missing-NumPy import; retained under `missing-numpy-before/`.
`b6132eec0` adds an actionable CANNOT RUN response. The rebuilt installed
wheel demonstrates that response before NumPy is installed. 56 verifier
tests pass. Its fresh native sources remain `7f5b786ae`; Python source is
`b6132eec0`. The all-150 available-CPU lane replay is IN PROGRESS, not passed.

Current-machine Python 3.14 CPU development evidence only. This does not
qualify other interpreters, older macOS versions, GPU or multi-GPU execution,
or a PyPI release. Compiler minimum-OS warnings are retained.
