# mojolearn 0.8.42

Source commit 0db362a62516faef0a52b112e93f7cba5e212a2b; release tooling at 12412eca4bf88b36d0e9e485566f1e74d9e2c62c. Published 2026-10-12. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 68ecab487a84c579dcd0f041ed38f8b23fe94987c70c6e6e76d1fe66d715c841 | PASSED, 12 jobs | pypi via alpha-api-0.8.42-linux-20261012 (source 0db362a62516, tooling 12412eca4bf8) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 2be5bafd7350e29f9e7fcf521ae6a7558b482a7eee487f5a7c42a5f238d8c7e8 | PASSED, 12 jobs | pypi via alpha-api-0.8.42-nvidia-20261012 (source 0db362a62516, tooling 12412eca4bf8) |
| mojolearn-amd manylinux_2_35_x86_64 | 87efcc0469b4ef2eb18018ba218a59b7263121668a2e68aba098503ca7cfe51b | PASSED, 12 jobs | pypi via alpha-api-0.8.42-amd-20261012 (source 0db362a62516, tooling 12412eca4bf8) |
| macosx_11_0_arm64 | e92c04494e679f51020108d24b6f431224ac23ea94bdec353a8f092c81f3ef7a | PASSED, 12 jobs | pypi via alpha-api-0.8.42-macos-20261012 (source 0db362a62516, tooling 9c7ad74132aa) |

Finish line: macOS: pip install mojolearn==0.8.42 on this Mac: OK; Linux: pip resolves mojolearn==0.8.42 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.42 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.42 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
