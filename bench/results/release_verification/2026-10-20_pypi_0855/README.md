# mojolearn 0.8.55

Source commit a8255828597a3bd19ac7bb5f657a38d09b8c732f; release tooling at 813f9c4d79367f39d9727ba39b5768455cc07df4. Published 2026-10-20. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | ca912ce50c59efaeffd64d7b195215d4ea48c67e00c9dabd619475a6be4c1170 | preverified, 0 jobs | pypi via alpha-api-0.8.55-linux-20261020 (source a8255828597a, tooling 4e927c92495a) |
| mojolearn-nvidia manylinux_2_35_x86_64 | b37c82ba8438daab6ac278f280943ff441110f892a01d3ecb428fc4079901b14 | preverified, 0 jobs | pypi via alpha-api-0.8.55-nvidia-20261020 (source a8255828597a, tooling 4e927c92495a) |
| mojolearn-amd manylinux_2_35_x86_64 | 1402574cd2a22cbd525384478807ffb5fbecba10251b48114dd79d56b5f1acfc | preverified, 0 jobs | pypi via alpha-api-0.8.55-amd-20261020 (source a8255828597a, tooling 4e927c92495a) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | 4a3aa1b4633a7cbaf9e39f397298eeae41ae3e4ce4f1f79a3acb0bcc343b0d02 | preverified, 0 jobs | pypi via alpha-api-0.8.55-nvidia-ptx-20261020 (source a8255828597a, tooling 4e927c92495a) |
| macosx_11_0_arm64 | 32090799e1f757c8f7e3b01b0ff6475a2dcc43813ff84a3f43081deddd1f202a | PASSED, 0 jobs | pypi via alpha-api-0.8.55-macos-20261020 (source a8255828597a, tooling 813f9c4d7936) |

Finish line: macOS: pip install mojolearn==0.8.55 on this Mac: OK; Linux: pip resolves mojolearn==0.8.55 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.55 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.55 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.55 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.

## Custom models and imported inference

The [October 20 model-portability record](../../model_portability/2026-10-20/README.md)
retains the separate custom-training, six-model cross-vendor inference and
NVIDIA-to-Apple saved-program results. It includes raw records, provenance,
an evidence checker and a retained model artifact. These source-built checks
are separate from the wheel-publication checks above and carry no new
performance claim.
