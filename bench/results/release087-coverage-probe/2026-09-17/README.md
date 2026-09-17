# Development diagnostics, not release qualification

These records used modified, uncommitted Python harness code and prebuilt
native binaries. The recorded Git HEAD is the parent commit, not a claim
that it contains the new lane. The `probe` directory deliberately excludes
these files from reference-table admission. Native digests and source
signatures are retained in each record.

- CPU: all nine fixtures, two executions each, all stable.
- Metal: base fixture, two executions, stable; 3.38 seconds including scheduler.
- CPU sabotage: base fixture, two executions, stable but different from control.
- Base control CPU/Metal: `672e75f003fce299`.
- Base sabotage CPU: `fdd8000292e1a4ed`.

The checks cover the combined homogeneity/completeness/V-measure wrapper,
including order and beta 0.5, 1 and 2. They do not qualify other lanes, a
freshly built wheel, CUDA, HIP, or all metric parameter combinations.

GP optimizer diagnostics: both lanes on all nine fixtures, two executions,
18 stable cells; all 72 train/infer/model/batch hashes match the bundled
reference table. They remain candidates: the table has only Apple GPU
witnesses for these optimizers, and this was a source/prebuilt-host run.

Ordered-gradient sum: all nine CPU fixtures pass twice; Metal base passes
twice and matches CPU (`a742b6c812b94347`), in 0.55 seconds including queue.
The independent Float32 left-fold oracle checks summation order and input
immutability. Unit controls reject a balanced reduction and mutated input.
