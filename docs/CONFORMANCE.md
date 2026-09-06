# Conformance bundles

A conformance bundle packages a pinned IDENTICAL fixture so another
implementation can compare itself with mojolearn without running Mojo. It
contains frozen inputs, expected stage bytes, an identity-trace card, profile
and provenance metadata, and a SHA-256 manifest.

```sh
python -m mojolearn conformance export BUNDLE_DIR
python -m mojolearn conformance validate BUNDLE_DIR
python -m mojolearn conformance validate BUNDLE_DIR --report REPORT.json
python -m mojolearn conformance diff BUNDLE_DIR --report REPORT.json
python -m mojolearn conformance diff BUNDLE_DIR --self
```

`export` normally runs the pinned fixture and therefore needs an IDENTICAL
build and GPU. Maintainers can assemble a bundle from an existing card and
its adjacent raw stage dumps:

```sh
python -m mojolearn conformance export BUNDLE_DIR --from-card CANDIDATE.card
```

Use `--force` only when intentionally replacing contents in an existing
bundle directory.

## Integrity and comparison

The two hashes have different jobs:

| Hash | Purpose |
|---|---|
| SHA-256 | Integrity of bundle files and externally supplied raw stage bytes. |
| FNV-1a64 | Lightweight per-stage localization compatible with device traces. It is not collision-resistant. |

`validate` first checks the manifest and internal consistency. With
`--report`, it then checks the external implementation's declared profile and
per-stage results. `diff` routes comparison through
`tools/identity_trace_diff.py`, the same tag-aware comparator used elsewhere
in the repository. `--all` reports every divergence; the default stops at the
first.

Conformance commands use three outcomes:

| Exit | Meaning |
|---|---|
| 0 | Bundle or report passes the requested check. |
| 1 | A named integrity, structure, or numerical check failed. |
| 2 | The command could not judge, including invalid usage or unparseable input. |

## External implementation report

Generate the report against the exact bundle profile. Do not reinterpret or
normalize floating-point output before hashing it. The authoritative schema is
the exported bundle and the parser in `python/mojolearn/_conformance.py`; use
the CLI error messages to identify missing or malformed fields.

At minimum, a report identifies its format/profile and implementation, then
provides each expected stage in order with its tag, dtype, element count,
FNV-1a64 value, and SHA-256 digest of the raw bytes. Profile disagreement is a
refusal, not a numerical mismatch.

## Limits

A bundle certifies only its exact profile, fixture bytes, stages, and expected
outputs. Passing it does not certify other shapes or an entire implementation.
Conversely, a failed bundle reports a concrete divergence; it should not be
generalized to unrelated paths without further measurement.

See [verification](VERIFY.md) for the simpler installed-build check and
[support and certification](../SUPPORT_MATRIX.md) for accepted hardware
claims.
