# mojolearn 0.8.44

Source commit b6c49ce419c47b224b5d257440dadc6210d33151; release tooling at 284052f0f51409d1d928ffd49b269f9f49d53af9. Published 2026-10-13. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | cc450c2220077b5c00307b7c98b91782808017354595cfe848021ba01f7b3655 | PASSED, 0 jobs | pypi via alpha-api-0.8.44-macos-20261013 (source b6c49ce419c4, tooling 284052f0f514) |

Finish line: macOS: pip install mojolearn==0.8.44 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
