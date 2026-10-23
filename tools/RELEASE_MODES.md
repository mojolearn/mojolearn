# Release tier selection

The release command defaults to `identical`. The explicit equivalent is:

```sh
MOJOLEARN_RELEASE_MODES=identical pixi run release 0.8.56 --build-backend cpu-box
```

Add `--publish pypi` only as part of the authorized release. The normal
preverified flow reuses the completed pre-merge numerical qualification and
still checks the final wheel artifacts. `--gpu-smoke` is a separate existing
qualification route.

The selection reaches the Mac build, the cross-compile workflow, the Linux
CPU-box build, the packer, and the release reuse plan. Linux proofs and payloads
record `release_modes`; installed qualification reads that metadata. The packer
requires exactly those tiers and all required host bindings, and refuses stray
unselected artifacts. Byte LM still appears exactly once in identical.

Explicit `MOJOLEARN_RELEASE_MODES='fast identical'` retains the historical
layout for reproduction. Old proofs without the new field retain that historical
meaning regardless of the current environment. Lower-level helpers retain their
historical default when called independently without a selection. Nonlegacy
selection is supported by the CPU-box release backend; the older optional remote
backends refuse it because they do not forward that choice.

A release state directory cannot change tier selection after work is recorded.
Use a new `--state-dir` for a different selection; the old artifacts and evidence
remain intact. An existing Mac source checkout containing an unselected tier is
refused before compiling; use a clean source checkout with `--source-checkout`.

Existing development partitions are unrelated to this option: releases retain
the compact merged layout. Reuse still requires exact source/toolchain/flags and
artifact provenance. No receipt or missing binary is synthesized.
