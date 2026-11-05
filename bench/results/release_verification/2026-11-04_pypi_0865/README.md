# mojolearn 0.8.65

Source commit 6519a54c657930757a36a520a03724d78318140e; release tooling at 6519a54c657930757a36a520a03724d78318140e. Published 2026-11-04. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| mojolearn-core manylinux_2_35_x86_64 | f2fc3a14dad015c280e1ae844957b1e98b3ca3ec01f3fce19554717ad1a0d594 | preverified, 0 jobs | pypi via alpha-api-0.8.65-linux-20261104 (source 6519a54c6579, tooling 6519a54c6579) |
| mojolearn-nvidia manylinux_2_35_x86_64 | e9ce359d332dcd922602c5c66a66bba919910a792d0f96be1b2e3b6834af722c | preverified, 0 jobs | pypi via alpha-api-0.8.65-nvidia-20261105 (source 6519a54c6579, tooling 6519a54c6579) |
| mojolearn-amd manylinux_2_35_x86_64 | 69dadd47d7b590595fac7da06adb146dffe634d4ab79050128f4f916244c9abd | preverified, 0 jobs | pypi via alpha-api-0.8.65-amd-20261105 (source 6519a54c6579, tooling 6519a54c6579) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | a246a3f20caa3510f650bb5335fbbe20f93b5936b3b66d42c6233018f304fd57 | preverified, 0 jobs | pypi via alpha-api-0.8.65-nvidia-ptx-20261104 (source 6519a54c6579, tooling 6519a54c6579) |
| mojolearn manylinux_2_35_x86_64 | c25c447e5d5d163f8f56712960e7b847bb78e790f79e3f4ff7421ef35edb37b3 | preverified, 0 jobs | pypi via alpha-api-0.8.65-linux-meta-20261105 (source 6519a54c6579, tooling 6519a54c6579) |
| macosx_26_0_arm64 | ccd625528c846fb4b5af0a50ecae3c7a9db313e169430c91ccca6bdf1da3c001 | PASSED, 0 jobs | pypi via alpha-api-0.8.65-macos-20261104 (source 6519a54c6579, tooling 6519a54c6579) |

Finish line: macOS: pip install mojolearn==0.8.65 on this Mac: OK; mojolearn-core: pip resolves mojolearn-core==0.8.65 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.65 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.65 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.65 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn: pip resolves mojolearn==0.8.65 for manylinux_2_35_x86_64; bytes are the published wheel's; pip install mojolearn-core==0.8.65 on manylinux_2_35_x86_64 resolves mojolearn-core; pip install mojolearn==0.8.65 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-amd, mojolearn-nvidia-ptx, mojolearn; pip install mojolearn-nvidia==0.8.65 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-nvidia-ptx; pip install mojolearn-amd==0.8.65 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
