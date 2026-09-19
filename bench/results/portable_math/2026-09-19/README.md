# Owned host math validation, 2026-09-19

Source change: `1e46c9ba1`. These are **private 0.8.8-labeled candidates**, not
new PyPI uploads. `validation.json` records their final SHA256, baseline hashes,
exact changed native files, and the three metadata-only changes after testing.
No numerical bytes changed during that final provenance stamp.

- Both wheel audits pass: 66 native files on macOS, 124 on Linux. Four bundled
  runtimes per wheel changed; all 61 macOS / 119 Linux existing model bindings
  remained byte-identical. One owned helper was added to each wheel.
- Seven installed host-math tests pass per platform, including 10,000 bit-pattern
  samples, full-range power-of-two log2 cases, special values, runtime C ABI,
  exact sum/scaling, GP kernel properties, and schedule rounding.
- The arithmetic digest files match exactly. Linux x86-64 ran under Docker
  emulation on Apple ARM64; this is not a physical NVIDIA/AMD execution claim.
- The existing compiled Mojo GP log matches FP64 bytes; GP theta parameters
  match the FP32-rounded exp outputs. Mac and Linux use independent binaries.
- Five guard tests pass, including rejection of a compiled foreign `sin` import,
  forbidden Python imports and a bundled libm filename.
- Eleven existing packaging tests plus four subtests pass. The API audit finds
  complete payloads. A direct PEP 517 wheel build finishes with the math guard.
- Real Apple Metal GP fit/predict/std and log2 random-forest fit/predict pass.

This validates the changed host arithmetic and packaging path. It does not
recertify all numerical lanes or claim new NVIDIA/AMD GPU runtime qualification.
The wheel/host-math claim excludes external Python, OS and driver dependencies.
Independent test oracles are outside the shipped product math implementation.
