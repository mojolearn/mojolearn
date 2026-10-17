# mojolearn 0.8.50

Source commit f2bfd69ee5819c90887382e133c98b181fd2e75b; release tooling at f2bfd69ee5819c90887382e133c98b181fd2e75b. Published 2026-10-17. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | a34efcfbc29c10180858cf0a6917471c1a476d48f59d71d108588707140622fd | preverified, 0 jobs | pypi via alpha-api-0.8.50-linux-20261017 (source f2bfd69ee581, tooling f2bfd69ee581) |
| mojolearn-nvidia manylinux_2_35_x86_64 | ac9aca2cfcf13a157c6e5e1f00af674cc1ff7bada1586568266d3e69da76da5b | preverified, 0 jobs | pypi via alpha-api-0.8.50-nvidia-20261017 (source f2bfd69ee581, tooling f2bfd69ee581) |
| mojolearn-amd manylinux_2_35_x86_64 | 71c7ee8f116760ff2ff9b674e8acafa78c1ade51fd3a4e7914926d349d76a7f1 | preverified, 0 jobs | pypi via alpha-api-0.8.50-amd-20261017 (source f2bfd69ee581, tooling f2bfd69ee581) |
| macosx_11_0_arm64 | 3a8a7fc3c386552cac5691a242f585f7babcb019e5d29a1e68bb568bc6e0088f | PASSED, 0 jobs | pypi via alpha-api-0.8.50-macos-20261017 (source f2bfd69ee581, tooling f2bfd69ee581) |

Finish line: macOS: pip install mojolearn==0.8.50 on this Mac: OK; Linux: pip resolves mojolearn==0.8.50 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.50 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.50 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
