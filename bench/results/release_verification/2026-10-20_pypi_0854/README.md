# mojolearn 0.8.54

Source commit 486ac1cde0f59fa05eeceeebf2985846704e60e2; release tooling at ef7ba546c210e2176512152d1ba28efae61e1285. Published 2026-10-20. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 6f0ffeadda52d714bae6cbebe906cda1f8635794324f693ae535d1144cf51c8f | preverified, 0 jobs | pypi via alpha-api-0.8.54-linux-20261020 (source 486ac1cde0f5, tooling ef7ba546c210) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 224694df10f92a8ddb1ba749fda37d347880e0494e6996870c5dee1afebf104b | preverified, 0 jobs | pypi via alpha-api-0.8.54-nvidia-20261020 (source 486ac1cde0f5, tooling ef7ba546c210) |
| mojolearn-amd manylinux_2_35_x86_64 | 8a076430320257a552b1bbd3815d9a7c550216f05dafda1c2e3ae6191af8d708 | preverified, 0 jobs | pypi via alpha-api-0.8.54-amd-20261020 (source 486ac1cde0f5, tooling ef7ba546c210) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | 8b87d8736e14b7c4cd0bc19e0e79c076eef35aefdba64405c39ed1c30bd221f5 | preverified, 0 jobs | pypi via alpha-api-0.8.54-nvidia-ptx-20261020 (source 486ac1cde0f5, tooling ef7ba546c210) |

Finish line: Linux: pip resolves mojolearn==0.8.54 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.54 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.54 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.54 for manylinux_2_35_x86_64; bytes are the published wheel's; not yet published: macos.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
