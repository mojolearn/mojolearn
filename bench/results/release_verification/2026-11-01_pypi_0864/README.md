# mojolearn 0.8.64

Source commit f48ed127e6b094dbffb968d09bb91f1d93d4ba1a; release tooling at 3dc776d6fd602e870b72763a1b45e1ccc7f234fa. Published 2026-11-01. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| mojolearn-core manylinux_2_35_x86_64 | 1ee79fb26d9d536e1a7d4fa8c19f796146899469cbfa01ee97e6c302ba9d7cb1 | preverified, 0 jobs | pypi via alpha-api-0.8.64-linux-20261102 (source f48ed127e6b0, tooling 47fffd4e4a04) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 42585e66f3169b0b7c25df37ab1dab3dd4a71ca3ebddeb6e28dfc33f17db4fe6 | preverified, 0 jobs | pypi via alpha-api-0.8.64-nvidia-20261102 (source f48ed127e6b0, tooling 47fffd4e4a04) |
| mojolearn-amd manylinux_2_35_x86_64 | 2bcdcb15962583dc49912daa84d199372188c193f2cb93b5dc0ca89aabc51e52 | preverified, 0 jobs | pypi via alpha-api-0.8.64-amd-20261102 (source f48ed127e6b0, tooling 47fffd4e4a04) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | 7265feeacc122dc7345f97ef50ef0a9d8b2837c8120d5d09a9d683c0f4135334 | preverified, 0 jobs | pypi via alpha-api-0.8.64-nvidia-ptx-20261102 (source f48ed127e6b0, tooling 47fffd4e4a04) |
| mojolearn manylinux_2_35_x86_64 | 2a3491812353367a885ee3ac7d6c60a67bccf27e8be491785f35514e576be53a | preverified, 0 jobs | pypi via alpha-api-0.8.64-linux-meta-20261103 (source f48ed127e6b0, tooling 47fffd4e4a04) |
| macosx_26_0_arm64 | 4f5a05a5dfda4364166367db9d85c5d6276037231941d6c1ab807d1ced3fd924 | PASSED, 0 jobs | pypi via alpha-api-0.8.64-macos-20261103 (source f48ed127e6b0, tooling 47fffd4e4a04) |

Finish line: macOS: pip install mojolearn==0.8.64 on this Mac: OK; mojolearn-core: pip resolves mojolearn-core==0.8.64 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.64 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.64 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.64 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn: pip resolves mojolearn==0.8.64 for manylinux_2_35_x86_64; bytes are the published wheel's; pip install mojolearn-core==0.8.64 on manylinux_2_35_x86_64 resolves mojolearn-core; pip install mojolearn==0.8.64 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-amd, mojolearn-nvidia-ptx, mojolearn; pip install mojolearn-nvidia==0.8.64 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-nvidia-ptx; pip install mojolearn-amd==0.8.64 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
