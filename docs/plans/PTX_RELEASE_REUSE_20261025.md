# PTX release metadata admission

The release commit is no longer an input to the PTX compiled-set identity.
The set identity still includes every GPU and host binding's source closure,
flags, target, compiler/toolchain, builder and image identity, plus the runtime
closure. Changing those inputs invalidates reuse. A metadata-only refreeze can
reuse a completed leg after the release controller admits its unchanged set
identity and build tooling, successful exit, complete original proof and overlay.

`tools/release_reuse.py` stages the completed set's existing files, then
`packaging/linux/ptx_reuse.py` writes only `PTX_BASELINE.json`. Its top-level
`source_commit` binds the current core, as the existing loader requires. The
`release_admission` explicitly distinguishes that binding from build provenance:

- `original_manifest` retains the exact original manifest bytes and SHA256,
  including its original source commit, compiler version and PTX audit records.
- `original_build_proof` retains the complete original proof bytes and SHA256.
- `original_stage_manifest` retains the original runtime inventory bytes and
  SHA256; every staged runtime file must still match that inventory.
- `origin.original_set_identity` retains the original saved identity. Its stored
  digest is verified before removing the obsolete PTX source-commit salt. All
  remaining fields must equal the current admitted set identity.

GPU/host binaries, merged index bytes and members, runtime libraries and adapter
files must still match their original evidence. Assembly checks after replacing
any files from published wheels. The packer checks again and refuses a reused
PTX set if its admission was removed or its evidence changed. The fresh manifest
is not falsely listed among the byte-identical files in `reuse.json`.

This does not discover proof inside published wheels or infer build success from
source alone. Without an original completed-leg proof and identity, PTX reuse
remains unproved. The plan records that missing evidence explicitly, instead of
claiming that a new release commit requires compilation. Loader/compiler/runtime
code and numeric behavior are unchanged. Adapter runtime and installed-artifact
qualification remain separate requirements; this admission does not claim them.

## Release-controller integration

`tools/release.py` compares candidate identities using
`release_reuse.normalized_set_identity(candidate["set_identity"])`; `None`
invalidates reuse. It retains the raw candidate `original_set_identity` in the
admission and forwards that field through `leg_origin`. The existing build
tooling, exit, original proof and overlay requirements remain mandatory.

## Validation

CPU-only tests use inert payload bytes, without compilation or GPU execution.
They exercise commit-only reuse, merged and per-binding layouts, unchanged raw
provenance, original source/dirty state, changed source/toolchain/flags/runtime
identity, missing or stale proof, altered binary/index/host/runtime bytes,
missing tooling or saved identity, packer rechecking and provenance stripping.
Full logs and exit codes are retained in
`~/mojolearn-evidence/ptx-reuse-final-20261025/`.
