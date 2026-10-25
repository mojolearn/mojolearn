# Release receipt admission, 2026-10-25

The readiness ledger is a requirements/evidence index. It does not automatically
import or qualify every historical report. A receipt's claimed tool, counts,
PASS text and hashed log are insufficient proof of execution.

`tools/release_requirement_receipts.py` currently admits these actual producers:

| Requirement | Retained producer | What it proves |
| --- | --- | --- |
| `candidate-nvidia`, `candidate-amd` | `tools/gate_smoke_box.py` structured candidate gate report | Exact candidate wheel hashes/modes and completed installation, whole-plan smoke, autograd, adapters and measured plan pairs; counts come from the raw producer report. |
| `runtime-source` | `tools/public_runtime_inventory.py` full endpoint gate, schema 2 | Zero prohibited/unresolved endpoint source classifications, with actual endpoint rows/counts. Runtime proof remains required separately. |
| Expanded Linux native binding compile units | `tools/linux_surface_qualification.sh` complete release-build proof, original before-build input capture and ledger launch | Selected native binding/index bytes; exact declared compiler/toolchain/flag configuration. Development-cache passes are not admitted. See `RELEASE_COMPILE_RECEIPTS_20261025.md`. |

Each admitted producer also needs its retained `producer-inputs.json`, created
for the actual run or through explicit historical provenance reconciliation.
It must identify the begun source revision, input fingerprint, run ID and raw
report hash. The complete begun input closure must reproduce its fingerprint.
Both raw files are retained and hashed; every later validation reparses them,
derives the result/counts again and compares them with the receipt. Candidate
wheel hashes and modes must also match the begun input configuration.

No parser is currently connected for `coverage-source`, Apple compile, Linux runtime-library or C++ adapter compile units, runtime,
identity, adapter-only, physical multi-GPU, performance or published-wheel
requirements. Those producer schemas remain **unqualified / complete-provenance**
until an adapter validates their actual retained structured reports and derives
their required coverage. Implement the applicable adapters before qualifying
their runs or publishing. This is not a reason to rerun retained successful
work: reconcile and admit the original evidence first. Missing evidence bytes
require restoration; unsupported schemas require admission code.

The former generic typed-JSON acceptance was reproduced with a plaintext file
stating that no execution occurred and caller-supplied identity counts of 1/1;
the ledger incorrectly returned PASS. The retained reproduction is under
`~/mojolearn-evidence/release-ledger-review-20261025/arbitrary-receipt/`.
Production now rejects that receipt. Tests use real-shaped supported producer
reports and explicit refusal cases; ledger-mechanics fixtures narrowly patch
the producer-specific validator only within tests. There is no production
fixture bypass.
