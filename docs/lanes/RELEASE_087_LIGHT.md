# 0.8.7 release: full API, bounded installed smoke

The user authorized this release policy on 2026-09-18. Build from the latest
integrated main, then freeze one source commit for the candidate. Include the
implemented APIs; retain their existing experimental and certification limits.
Do not claim universal Apple identity or promote pending references merely to
publish. Existing boundary discrepancies remain unresolved outside certified
workloads.

The `light` workflow profile admits prepared alpha API artifacts only. Its default combined batch
requires the normal wheel content/RECORD/native-inventory admission and two
SHA-bound successful `qualify_verifier_wheel.py` receipts: one exact macOS wheel
on Metal and one exact Linux wheel on CUDA. Both must contain the frozen source
commit. The checks include installation, dependency/import origin, new APIs and
native entry points, bundled CPU models, CPU/GPU whole-loaded-model comparison,
small batch checks and a live negative control. Each qualifier stage has a
180-second limit. This replaces the full CPU architecture sweep for this
explicit publication profile; the default full workflow remains unchanged.

The light receipt does not certify additional Linux architectures, physical
multi-GPU execution, all extended neural properties or arbitrary inputs. R2
may distribute exact candidate bytes; the small synthetic fixtures are generated
locally by the pinned harness. No large corpus, benchmark, or multi-GPU campaign
is part of the smoke test.

Fresh native build and packaging remain required. Build time is separate from
the test budget. Preserve any failed attempt, stop bounded jobs on failure,
and terminate owned rentals after fetching results. Publish exactly the wheel
digests admitted by the prepared manifest, then verify PyPI's file hashes.

Main reconciliation restored the existing GEMM temporary-workspace allocation
that a partial SIMD revert had replaced with a refusal. This preserves the
previous main behavior for valid plans requiring more workspace than the
transpose buffer; arithmetic is unchanged.

## Independent platform publication

For this alpha release, ready platforms may publish independently. The default
light batch still requires both wheels; an explicit `light_platform=macos` or
`linux` requires exactly that platform's wheel and its complete successful smoke
receipt. Linux is added to the same version once ready; an existing filename is
never overwritten or silently skipped. This changes scheduling, not the smoke
required of each published wheel.

`artifact_source_commit` pins the already-built source when publisher-only tools
change after the freeze. It must match the SHA-bound receipt and embedded wheel
witness. For 0.8.7 the native/Python inputs remain the frozen `4e1828f90` tree;
publisher automation changes do not justify rebuilding those identical inputs.
