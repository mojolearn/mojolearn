# mojolearn 0.8.43

Source commit b99b0900e457fa5f2205384e3985f7433d704d5a; release tooling at 7b4d384fe5b82a8fd84befbe47e03023b158aee7. Published 2026-10-13. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | e7674768bf29bdc034184e818de0a89ffbcc1fb8561ebc4547a8771f83b8bb48 | PASSED, 12 jobs | pypi via alpha-api-0.8.43-linux-20261013 (source b99b0900e457, tooling 7b4d384fe5b8) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 214bcb97de6bde9d57dcb1d34682c8b3062a15ecdaf4efff4cf8886427ab95ba | PASSED, 12 jobs | pypi via alpha-api-0.8.43-nvidia-20261013 (source b99b0900e457, tooling 7b4d384fe5b8) |
| mojolearn-amd manylinux_2_35_x86_64 | fe958125c23ddd9c5adffa245c2cd4978d808b05e3c460bad3815f6c331007c1 | PASSED, 12 jobs | pypi via alpha-api-0.8.43-amd-20261013 (source b99b0900e457, tooling 7b4d384fe5b8) |
| macosx_11_0_arm64 | 56d10eb9d5a4356e6bfd6b092443c365496bb6a78bc9b12451280b1c0ca3e719 | PASSED, 12 jobs | pypi via alpha-api-0.8.43-macos-20261013 (source b99b0900e457, tooling 7b4d384fe5b8) |

Finish line: macOS: pip install mojolearn==0.8.43 on this Mac: OK; Linux: pip resolves mojolearn==0.8.43 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.43 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.43 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
