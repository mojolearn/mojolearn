# Fresh combined Linux 0.6.1 alpha publication

Source changes are authored, not executed or publication-ready evidence.
Published 0.6.0 files and the stable workflow route remain unchanged. Root owns
version changes, all tests/builds/runtime qualification, staging and publication.

A Linux 0.6.1 alpha wheel must carry a fresh
`mojolearn-0.6.1.dist-info/LINUX_PAYLOAD.json` from the packer's
`--profile release-0.6.1`. The alpha staging verifier refuses an inherited
native overlay for this Linux version and refuses attaching
`ALPHA_PROVENANCE.json` to the fresh combined payload. Existing macOS/older
alpha overlay contracts remain distinct; no inherited artifact receives new
numerical qualification through a version or metadata update.

The root-prepared alpha manifest extends its exact wheel-name/SHA mapping:

```json
{
  "schema": "mojolearn.alpha-release.v1",
  "version": "0.6.1",
  "release_profile": "alpha-api",
  "files": {"EXACT_WHEEL_FILENAME.whl": "EXACT_FINAL_WHEEL_SHA256"},
  "linux_qualification": {
    "file": "linux-qualification.tar.gz",
    "sha256": "EXACT_QUALIFICATION_ARCHIVE_SHA256",
    "wheel": "EXACT_WHEEL_FILENAME.whl"
  }
}
```

Place `alpha-manifest.json` and only its named wheels in the artifact directory.
Keep the separately downloaded qualification archive outside that directory.
The archive root must contain `build-proofs/` and complete retained directories
`cuda/sm_89/`, `cuda/sm_90/`, `hip/gfx942/`. Each includes the selected copied
`build-provenance.json`. The tar must contain only regular files/directories:
no symlinks, hardlinks, unsafe paths or duplicate names. Bounds: 512 MiB archive,
2 GiB expanded, 20,000 members, 128 MiB per member. Root must check real evidence
sizes before freezing; do not truncate evidence to satisfy these bounds.

Root-only file admission command (does not run models):

```sh
python3 packaging/verify_alpha_artifacts.py /absolute/WHEEL_DIRECTORY \
  --manifest-sha256 ROOT_PINNED_MANIFEST_SHA256 \
  --qualification-archive /absolute/linux-qualification.tar.gz \
  --source-root /absolute/EXACT_SOURCE_CHECKOUT
```

The gate validates full wheel RECORD/tag/Alpha metadata, SHA-bound archive
contents and calls `tools/check_linux_release_qualification.py`'s
`check_release061`. This recomputes all three architecture build proofs, source
and payload inventories, exact final wheel hashes in installed records, all
25 required jobs on each runtime architecture (the original 24 plus one IDENTICAL-only installed byte-LM forward/backward AdamW/checkpoint check), and scoped UMAP/Ordered identity
checks. A summary asserting PASS is insufficient without the underlying files.
Root must stage all three runtime results against the **same final wheel bytes**.
Any auditwheel/native/runtime/Python modification after qualification invalidates
that wheel hash and requires the corresponding proof/qualification updates.

The Trusted Publisher alpha route downloads `linux-qualification.tar.gz` from
its explicitly named candidate release only when the SHA-pinned manifest
requires it. The complete evidence archive is transferred as a separate Actions
artifact. Both staging and the immediate prepublish job repeat the full file
admission. Source hashes are compared against that workflow's checked-out source;
dispatch from the intended source revision. Missing archive or runtime evidence
fails closed. The archive is never uploaded to PyPI with the wheels.

Outstanding before release: root tests the authored alpha compatibility fixtures
and the real qualification checker fixtures, builds/assembles fresh architecture
sets, confirms metadata/version and runtime packaging, qualifies the exact
combined wheel on sm_89/sm_90/gfx942, then freezes its manifest/archive hashes.
The current payload explicitly excludes optional `_mojolearn_byte_lm`; the
separate research model results do not imply that binary is present in the
wheel. Do not call this all-features native coverage or universal certification.

The byte-LM payload adds one IDENTICAL extension per architecture: 46 per set,
138 total. FAST and DETERMINISTIC byte-LM remain unsupported. Its new installed
check retains full pre/post/restored checkpoint bytes and gradients for one
step only; it does not transfer the separate source-build 128-step or foreign
vendor resume evidence to the wheel. These added gates require root execution
before publication admission.
