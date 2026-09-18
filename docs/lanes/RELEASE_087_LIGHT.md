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

## Published 2026-09-18

Version 0.8.7 is live on [PyPI](https://pypi.org/project/mojolearn/0.8.7/) and is
the [latest GitHub release](https://github.com/mojolearn/mojolearn/releases/tag/v0.8.7).
Both wheels contain frozen source `4e1828f90384c4b2bb3e4bdc025f4931b7553dfb`.
Publisher-only commit `18b1f08720647ab57f9c56615fddd1ce8f902ad9` added independent
platform scheduling without changing native or Python package inputs.

Fresh builds completed on Apple Silicon, NVIDIA sm_89/sm_90a and AMD gfx942.
The final installed macOS/Metal and Linux/CUDA wheels each passed all 13 light
smoke stages. The Apple smoke took 31.37 seconds. All 32 Linux CPU-binding hashes
matched across the three build hosts. This is the scoped alpha admission above;
`release_qualified` remains false. No broader Apple identity claim is added.

Published PyPI SHA-256 values, verified against the smoke-tested wheel bytes:

- `mojolearn-0.8.7-py3-none-macosx_11_0_arm64.whl`: `25c728aa4321b011c14b801b123905aed04f48f6139aef8ea81c33abb50f1d97`
- `mojolearn-0.8.7-py3-none-manylinux_2_35_x86_64.whl`: `5b1c2db8d4350aff672f155c804d309148311b0cc4a4e36efadef998ebbae470`

Publication workflows: [macOS](https://github.com/mojolearn/mojolearn/actions/runs/35398595805)
and [Linux](https://github.com/mojolearn/mojolearn/actions/runs/35402174014).
The GitHub release attaches both wheel receipts, their manifest and
`release-evidence.tar.gz` (SHA-256
`4f3be04af5c1f816579bfaf1cb29eed95b128ad3cff0264da040776f159db4e2`).
The archive includes the Linux build proofs and the actual orchestration copies
used to give the successful sm_89 build a bounded 50-minute budget and three
telemetry-query attempts; inventoried sources and compiler flags were unchanged.
Failed/timeout builds were not used. Every owned build/smoke rental was terminated
and deletion verified by the provider API.

The full external run record is
`~/mojolearn-evidence/release-087-light-2026-09-18`.
The [post-release process review](RELEASE_PROCESS_ALPHA.md) separates the changes
already made from the next implementation work. Do not restart this release to
close the separate numerical-certification backlog.
