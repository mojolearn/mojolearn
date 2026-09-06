# PyPI 0.6.0 alpha API release — published September 6

**Published and index-hash verified.** Workflow
[34066704839](https://github.com/mojolearn/mojolearn/actions/runs/34066704839)
published the exact macOS arm64 and AMD Linux x86-64 wheels in
`release-0.6.0-retry2/`. [Index verification](pypi-0.6.0-verification.json)
records both public URLs, timestamps and SHA256 values. NVIDIA Linux remains
source-build-only; the Linux artifact contains HIP/gfx942 binaries.

The API overlay preserves base native/runtime bytes and exposes current
Python modules. It does not manufacture missing native extensions or inherit
new numerical certification. The new byte-LM binding is not in these wheels.

The first0.6.0 publication attempt failed before upload because email
serialization folded the Summary header. The repaired assembler disables
header folding; all9 file-only fixtures passed, and both exact r2 wheels
passed artifact checks and packaging core-metadata validation. Original
candidates and failure logs remain retained. No published file was replaced.

## Earlier prerelease preparation (historical, not published)

Version `0.6.0a1` exposes ordinary `linalg`, `umap`, and `training` modules,
existing optimizer/loss primitives, Mamba/Transformer classes, and the fixed
MLP/byte-LM Python trainers. These are candidate artifacts, not evidence of
publication. Missing native extensions and symbols remain unavailable.

`combined/alpha-manifest.json` identifies both exact candidate wheels.
`combined-file-admission.json` records passing RECORD, metadata, native/Python
provenance and artifact inventory checks. The native/runtime bytes are copied
unchanged from the identified macOS and AMD 0.6.0 base wheels; current numerical
qualification is explicitly not inherited. NVIDIA binaries and the new
byte-LM native extension are not added by an API overlay.

Root alone assembled and checked these files. Six file-only overlay tests
passed, including corruption/traversal refusals and UTF-8 metadata handling.
The initial real-wheel assembly exposed an ASCII serialization bug; preserving
the original UTF-8 description bytes fixed it before candidate creation.

An isolated macOS import of the candidate resolved all `__all__` exports and
seven public modules. See `macos-import.log` and `macos-import-guard.json`:
exit zero, about 45 MB peak process-group RSS, 512 MB cap, 45-second deadline,
two-thread limits. No model or numerical operation ran in this check.

`stable-workflow-regression-fixed2.log` records 17 passing staging/publishing
plumbing fixtures. Earlier failures are retained in the preceding logs. The
fixtures use an explicit qualification-verifier test double with invocation
and refusal witnesses; they do not certify installed numerical behavior.
Stable workflow shell bodies were unchanged by the separate alpha route.

The final 0.6.0 publication and verification are recorded above; these earlier 0.6.0a1 artifacts were never uploaded to PyPI.
