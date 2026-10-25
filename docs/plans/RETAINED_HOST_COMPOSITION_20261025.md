# Retained common host assembly — 2026-10-25

The three c00c4d800 Linux builds completed successfully, but independently built
host binaries differ. The 46 host binding identities, source inventory,
compiler executable hash, installed compiler packages, host CPU flags, merged
index and runtime library bytes all match. sm89 and AMD even have the same host
cache key. The sm80 cache key differs only in the inherited project variable
`MOJOLEARN_CUDA_CODE_FORMAT=ptx`; the host builder does not consume that variable.
Their temporary generated-root paths differ. ELF text, symbol and relocation
sections also differ, so this is not classified as harmless metadata or erased.

`tools/release_host_composition.py` supports an explicit assembly choice: select
one already built host artifact, preserve its exact bytes, and retain all three
original build witnesses and variant digests. This is not a reproducibility
claim, a new build receipt, or numerical/runtime qualification. All original
leg directories, binaries, manifests and build proofs remain unchanged.

The helper requires complete original merged/identical build proofs, their
hash-linked before-build captures, all 46 resolved host identities, the declared
compiler byte hash, the actual host cache invocation, identical host indexes and
CPU readback coverage, and matching runtime dependencies. The sole cache-key
normalization is scoped to the inherited CUDA-format variable: raw frozen host
builder sources must be witnessed and must not consume it. Unknown or changed
compiler/source/host flags, incomplete records, altered original bytes and
changed dependencies fail closed. No ELF bytes are normalized or rewritten.

A new destination gets separate set trees with the selected common host. The
original per-leg stage manifests and GPU bytes remain intact. The composition
receipt records which host each leg originally built and which retained artifact
was selected. Existing pack host-byte agreement checks still run. Additionally,
a host artifact differing from its ordinary build proof now requires the
explicit composition receipt; copying common bytes alone is insufficient.

`pack_wheel.py --host-composition RECEIPT` revalidates the raw witnesses and
actual assembly files, then attributes the selected host only to its original
leg. `--source-root FROZEN_CHECKOUT` separates updated pack tooling from the
frozen source: source inventory, shipped Python, package metadata, identity
files and source commit checks all use that checkout. Pack tools can evolve
without falsely claiming that rebuilt product source was compiled.

## Commands and scope

From the tooling checkout, using the original retained build directories:

```sh
python3 tools/release_host_composition.py \
  --build "$LEGS/cuda-sm_89/cuda-sm_89/release-build/build" \
  --build "$LEGS/cuda-sm_80/cuda-sm_80/release-build/build" \
  --build "$LEGS/hip-gfx942/hip-gfx942/release-build/build" \
  --select cuda/sm_89 --compiler-sha256 "$DECLARED_COMPILER_SHA256" \
  --source-root "$FROZEN_SOURCE" --out "$NEW_ASSEMBLY"
```

Omit `--out` for read-only input validation. Existing output directories are
refused. Packing uses `$NEW_ASSEMBLY/sets/cuda` and `sets/hip`, the three original
`build-provenance.json` files, `--host-composition
$NEW_ASSEMBLY/host-composition.json`, `--source-root "$FROZEN_SOURCE"`, and the
normal release-split/adapters gates. The complete witnessed set collection must
be supplied. Existing release-controller routing is unchanged; the coordinator
chooses when to invoke this explicit composition route.

The source-only suite passes 116 tests and 68 subtests across host composition,
split-wheel packaging, release-split routing, merged-leg reuse and staged
admission. Two older publication fixtures were updated to declare a synthetic
qualification record, using the same test-only producer patches as severable
release tests. No production release/producer gate was weakened.

The real c00 inputs passed validation. A separate real assembly also passed the
packer's source/file/proof validation with all 46 host records attributed only
to cuda/sm_89; no wheel was produced. That regenerable 549 MiB copy was removed
at the coordinator's request to recover compile disk space, after all original
host hashes were verified. Its receipt and validation output remain retained.
The final additional host-format scope check also passed on the original c00
sources. Installed-wheel and numerical qualification remain pending.

Full comparison data, original/selected hashes, tests, retained receipts,
removal verification, and the exact regeneration script:
`~/mojolearn-evidence/host-merge-mismatch-20261025/`.
