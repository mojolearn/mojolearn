# mojolearn 0.8.37

Source commit 8f9c7d407e2227e4b22b198778d155f4f5030d92; release tooling at d0f6d28bcfe3064ee940aa07efb292d59b5d2188. Published 2026-10-10. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 9841a1e4d2e48ac5f93775e82ed40916ffa22b587d42a98781eba7767cdd632c | PASSED, 12 jobs | not yet published |
| mojolearn-nvidia manylinux_2_35_x86_64 | 34126d91b1e6ba5bc0dc047d74a459e9c4f2290e278cc3341454b3ba3c90ba74 | PASSED, 12 jobs | not yet published |
| mojolearn-amd manylinux_2_35_x86_64 | 7d27fb1f23f54cf8fe12a215753122ed7b44fb8a0b8b7a41937db407349da1c2 | PASSED, 12 jobs | pypi via alpha-api-0.8.37-amd-20261010 (source 8f9c7d407e22, tooling d0f6d28bcfe3) |
| macosx_11_0_arm64 | efae74cb4f41e7231e898dff37d6093288647cbb0ae7e30e3460db08927033da | PASSED, 12 jobs | not yet published |

Finish line: mojolearn-amd: pip resolves mojolearn-amd==0.8.37 for manylinux_2_35_x86_64; bytes are the published wheel's; not yet published: macos, linux, nvidia.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
