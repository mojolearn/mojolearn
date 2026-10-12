# mojolearn 0.8.39

Source commit 72f74b8d43ceb1283b62c3f92bf020145a6adb03; release tooling at 72f74b8d43ceb1283b62c3f92bf020145a6adb03. Published 2026-10-12. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | 30b7e22bac21287ecbdfc269564541e9f3416bced135936c62a393108d224c64 | PASSED, 12 jobs | pypi via alpha-api-0.8.39-macos-20261012 (source 72f74b8d43ce, tooling 72f74b8d43ce) |

Finish line: macOS: pip install mojolearn==0.8.39 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
