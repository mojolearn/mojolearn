# 0.8.8 verifier/reference patch

Status: published on [PyPI](https://pypi.org/project/mojolearn/0.8.8/) for macOS
and Linux on 2026-09-19 UTC. Both exact wheels passed all 13 light-smoke stages.
The [publisher run](https://github.com/mojolearn/mojolearn/actions/runs/35409439428)
succeeded, and PyPI reports the same wheel hashes.

The patch separates CPU replay of GPU-written CTR models from GPU model-byte
references. A recorded CPU N/A cannot erase a GPU numerical reference; different
GPU hashes remain a conflict, and an actual result mismatch remains DIVERGENT.

The targeted reference update is limited to 12 GPU parallel lanes with 148 missing numerical reference entries, nine existing
fixtures and two repeats on Apple, NVIDIA and AMD. Raw records and the strict
comparison accompany admission. Eleven lanes have three-GPU agreement; resampling
has complete Apple/AMD agreement and its partial NVIDIA capture is not admitted. Existing references outside those
lanes are preserved. These results do not add CPU routes, prove physical multi-GPU
execution, or establish the paper's broader every-variant four-device claim.

Native libraries are inherited unchanged from published 0.8.7. The new overlay
path checks unchanged native compile inputs and records package/native sources
separately. Each final platform wheel requires one bounded installed smoke;
unrelated algorithms and unchanged native builds are not repeated.

The admitted record is `bench/results/identity_break/2026-09-18_parallel-reference-gaps/`.
It fills exactly 148 missing numerical entries without changing any existing
numerical reference. All other lane cells remain unchanged. Focused validation:
121 verifier/coverage tests, 20 packaging tests, six light-policy tests, and the
generated documentation/version check. The prepared publisher defaults to light.

## Published artifacts

Package source: `d8047d4415b34327a28013dd9331969567ea02b7` on integrated main.
Native source: `4e1828f90384c4b2bb3e4bdc025f4931b7553dfb`, inherited from 0.8.7.
All 65 macOS and 123 Linux native/runtime files are unchanged.

| Wheel | SHA-256 |
|---|---|
| macOS ARM64 | `3774f120194b12fb71a4753866c370dfcf91989139a8a45d6fb80d4a4044fbec` |
| manylinux x86-64 | `3f0a7003d33c3de04349663350e37497c043f6a4b884950255dc5254295041e5` |

The Mac smoke took 30.58 seconds. The Linux smoke also passed all 13 stages.
Builds and full CPU certification were skipped by the explicitly selected light
publication profile. Three file-only table checks additionally confirmed that
every reference names a real source record and the admission policy stays honest.

All three owned rentals were deleted and confirmed absent by provider HTTP 404.
The [GitHub release](https://github.com/mojolearn/mojolearn/releases/tag/v0.8.8)
carries the wheels, manifest, light-smoke receipts and scoped evidence archive.
