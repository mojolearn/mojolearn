# mojolearn 0.8.49

Source commit 62b015a2d291d29c9ebb69d3e6c11855fe889503; release tooling at 62b015a2d291d29c9ebb69d3e6c11855fe889503. Published 2026-10-16. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | b6941ec8e371a50a78bf5ccca9657102d3f453dcc672b2dc5f4ca11bac0e69b9 | preverified, 0 jobs | pypi via alpha-api-0.8.49-linux-20261016 (source 62b015a2d291, tooling 62b015a2d291) |
| mojolearn-nvidia manylinux_2_35_x86_64 | c4f535b8f6588b841438fda009fe1e120c66297f3d5a5f6743332760de53d0e8 | preverified, 0 jobs | pypi via alpha-api-0.8.49-nvidia-20261016 (source 62b015a2d291, tooling 62b015a2d291) |
| mojolearn-amd manylinux_2_35_x86_64 | 41e2e98efbeea6fcefbc7cdd66f0907b779d38efec6c395283133b45dcfbb574 | preverified, 0 jobs | pypi via alpha-api-0.8.49-amd-20261016 (source 62b015a2d291, tooling 62b015a2d291) |
| macosx_11_0_arm64 | f95ec5a5c5a0f5e3ffc4bfb26f02afa8a98ce15b1d2190f111df9a081dc9800c | PASSED, 0 jobs | pypi via alpha-api-0.8.49-macos-20261016 (source 62b015a2d291, tooling 62b015a2d291) |

Finish line: macOS: pip install mojolearn==0.8.49 on this Mac: OK; Linux: pip resolves mojolearn==0.8.49 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.49 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.49 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
