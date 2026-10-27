# Lightweight verifier and endpoint coverage

Worktree: `mojolearn-light-verifier`, branch `fix/light-verifier-coverage`,
base `13bed8cc7`. No native numerical changes, measurements or builds.

`python -m mojolearn verify --light` runs eight tiny cases and compares 118
outputs bitwise against Apple/NVIDIA/AMD witnesses. Inputs have at most sixteen
elements; each optimizer takes one update. `--self-test` adds nine public-operation
faults. `--light-output DIRECTORY` retains raw output bytes and loaded binary hashes.
The package ships the injectors; repository runners use those same implementations.

All 118 outputs matched on Apple, RunPod NVIDIA L40S, and DigitalOcean AMD MI325X,
using isolated verifier overlays over installed 0.8.62 binaries. This does not
claim a rebuilt production wheel, all hardware, or PTX execution. Source bytes,
binary hashes, GPU/driver information, logs and raw output bytes are retained.
The first source fingerprint used Python AST dumps, which differ across Python
versions. Token fingerprints fixed that metadata failure; only the clean captures
were rerun. Native arithmetic and fixtures were unchanged.

Nine final numerical fault arms caught and restored locally. Four qualify on
NVIDIA and AMD too. The original fifth GPU arm changed both values and a dynamic
shape: it was detected, but is excluded from the stricter numerical-fault result.
Its replacement scales the native product, preserves shape, and was caught and
restored locally. The original records remain intact. New GPU runs were not
started after the box owners resumed their campaigns. These are public-operation
faults executing real native kernels, not mutations of kernel source. Existing
native build controls remain separate historical evidence.

The comparator's unit tests flip one bit in every consumed output, reject
incomplete captures, and distinguish shape/dtype failures from numerical changes.
The final comparator also rechecked the retained three-vendor captures and all
nine accepted local negative controls without repeating GPU work.

`verify --coverage` remains an inspection command. It now includes the complete
3,802-endpoint census: 1,607 mapped, 920 outside numerical identity, and 1,275
unmapped. Every numerical `mojolearn.ops` endpoint maps to a lane or a supplemental
check, including existing graph, distributed and partitioned checks. Mapping does
not establish execution/admission. Many remaining rows concern parameter access,
configuration, inherited methods and state protocols; they are not silently
credited through shared native exports. The full unresolved rows ship in the
coverage JSON. Whole-library endpoint qualification remains incomplete.

The report also fixes neural training's opt-in status, removes the stale
`public-set-params` hold (nine already-admitted fixtures), refreshes historical
evidence and the generated 571-lane matrix, and checks every harness fragment.
Lossless compression reduces the historical evidence module from a large Python
literal to about 352 KB without discarding witnesses. Source manifests detect
API and supplemental-check changes; the manual light-check workflow checks them.

Validation: 107 verifier/coverage regressions and 43 installed report tests pass.
Installed coverage and the light command pass. The packaging inventory reports
23 preexisting findings (29 at the base, zero new); its failure is retained,
not called a passing packaging audit. The work does not qualify the whole release.

Full evidence: `~/mojolearn-evidence/light-verifier-20261027/` and R2
`measurements/light-verifier/20261027/`. GPU archives have verified readback SHA256
receipts in `bench/results/r2-index.tsv`; the final archive contains local logs,
test results, raw captures and source changes. See `summary.json` for counts and
explicit limitations. Both borrowed GPU slots were returned without changing
their installed packages or tearing down their owners' machines.
