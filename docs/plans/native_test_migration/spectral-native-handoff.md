# Spectral fit and embedding fixtures

CPU 7 and GPU 9 are authored and unrun. Dependencies are the pilot API
`f0201aa23b`, actual backends `764e2ad158`, and saved input/dense-fixture producer
`bcc32168f4`. Use `spectral_native_{cpu,gpu}_suite.run_checks` or the two
`check-native-spectral-*` tasks, from the repository root. Both need the
identical flag; CPU additionally needs the CPU-column flag.

The original 150-row embedding graphs, 80-row precomputed graph and 120-row
planted partition keep their dimensions, inputs, solver seeds, component
counts, neighbor counts and tolerances. Both providers execute their actual
dataset/COO routes; GPU also executes the existing dense route. Expected values
are the original exact column/diagonal relationships and the label-permutation
invariant planted partition. No solver was copied into an oracle and no fitted
reference hash was recorded. NaN-filled outputs, canaries and source word
checks cover complete writes and borrowed storage lifetime.

The CPU runs independent graphs across host cores. GPU graphs run in order on
the physical device, preserving kernel parallelism; the separate shared-context
concurrency failure is still open. No lane compiled or executed these fixtures.

Nothing removed: all Python tests stay pending native acceptance. Public output
identity/dtype/shape/metadata, option normalization/default aliases, refusals,
repeatability and interoperability remain Python seams. Prediction/retained
state/resident ownership and assignment routes are not claimed by this slice.
No product/kernel, algorithm, default, benchmark, timing or tolerance changes.
