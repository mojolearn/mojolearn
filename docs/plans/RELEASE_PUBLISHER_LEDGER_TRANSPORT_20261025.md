# Canonical ledger publication transport, 2026-10-25

Every invocation of `tools/release_linux_publish.sh` requires
`MOJOLEARN_REQUIREMENTS_LEDGER` naming the canonical ledger. The controller sets
this to `self.readiness().path`; there is no waiver flag. Even `publish=none`
creates GitHub release assets, so it passes the same local gate first.

The local publisher regenerates the complete requirements manifest from the
frozen source and recorded build configuration, then runs its pre-publication
gate. It verifies the wheel's version, source witness and hash against the
qualified artifact inputs. The file-free Linux `mojolearn` package receives a
separate metadata-only audit: no runtime files/entry points and exactly the
same-version core, NVIDIA and AMD dependencies. Its actual bytes are also bound
to the admission artifact.

`tools/release_publish_admission.py` packages the original ledger file without
rewriting it, the original producer evidence actually read by the successful
gate, source/artifact identities and admission-tool hashes. Evidence is stored
by SHA256, with a map from the original absolute paths. `release_evidence.py`
resolves only evidence reads; historical receipt JSON, paths and hashes stay
unchanged. Missing/corrupt bytes, duplicate archive members/JSON keys, links,
traversal paths and unreferenced objects refuse admission. Evidence is rehashed
before packing and the finished archive is checked again. The compact ledger
codec is supported through `RequirementsLedger.read()` and is itself a policy
dependency.

The bundle goes through the project's existing authenticated R2 client to
`release-readiness/v1/<sha256>.tar.gz`. It is not attached to a public GitHub
release or uploaded as a public workflow artifact. Only its digest and the
ordinary wheel/source metadata appear in the dispatch/alpha manifest. The
workflow uses the existing repository R2 secrets; no credential or presigned
URL is sent as a workflow input or printed.

The workflow's `ledger_admission` job downloads that exact private object,
fetches the frozen source through its full checkout, regenerates the canonical
manifest and reparses all required retained producer reports. The tool policy
and resulting requirements must reproduce. All eight PyPI/TestPyPI publisher
jobs directly require this job's success, including full-profile and prepared
alpha paths. Immediately before each upload, its actual `upload/*.whl` bytes
and source/metadata are checked against that same immutable bundle. A newly
built or patched wheel whose bytes were not qualified cannot use an older
bundle. Build-only workflows without a prepared candidate remain available.

Before local public upload, the staged wheel bytes are checked once more and
the canonical ledger must still match the admitted snapshot. A later ledger
update requires a new metadata admission; it does not authorize rerunning
completed compilation, correctness or performance work.

The old `qualification_reuse` fields remain only for compatibility with the
artifact-format verifier. They are not qualification evidence: neither the
local publisher nor any workflow publisher can proceed on those fields alone.

Validation uses actual source-producer receipts and ledger replay with a small
fixture source manifest; it removes original evidence paths to exercise the
off-machine resolver. Negative cases cover missing/corrupt objects, duplicate
IDs/keys, archive traversal/links, source/artifact/policy mismatches, shortened
requirements, changed ledgers, file-free metadata restrictions, and refusal
before any upload or dispatch. R2 is mocked; no network upload, publication,
build or rental was performed. Logs are retained under
`~/mojolearn-evidence/release-publisher-ledger-20261025/`.

This transport does not fill missing qualification evidence or implement the
unsupported producer parsers listed in
`RELEASE_RECEIPT_ADMISSION_20261025.md`. Those requirements remain unqualified
and block publication until their original reports can be admitted.
