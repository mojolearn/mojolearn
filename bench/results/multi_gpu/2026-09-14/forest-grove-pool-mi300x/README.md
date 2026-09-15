# Resident RF/ExtraTrees grove pooling — two MI300X

RunPod pod `r266nfvh24fwyp`, two AMD Instinct MI300X (gfx942, SR-IOV), image
`rocm/dev-ubuntu-22.04:6.4.1-complete`, Mojo 1.0.0, 2026-09-14 21:52-21:58Z.
Source: the `git archive` of commit `81120c4ed` (`commit.txt`; the source
SHA256 computed on the box equals the one computed from the archive on the
Mac, `source_sha256.txt`). That commit is the squash of
`codex/forest-grove-pool` at `e64799c9c` whose H100 receipt is
`../forest-grove-pool-h100/`; the implementation and gate files are the same
bytes. The body is `body.sh`. Every GPU process ran serially.

## Result

`out/gate.txt`:

- Bindings built for gfx942 with `MOJOLEARN_TARGET_COLUMN=amd`, separate and
  packed layouts (`out/binaries.sha256`), plus the base binding the public
  check imports.
- `training/checks/forest_pool_check.mojo`, production and fault builds, both
  layouts: 76 `PASS forest pool` lines each (74 fixtures plus the two
  same-grove cancellation witnesses). Every sampled grove total and every
  prediction bit of the two-device pool equals the single-device resident
  forest on the same box, and injected owner failures publish nothing.
- `checks/forest_inference_model.mojo`, both layouts: layout, workspace and
  resident lifecycle gates pass.
- `tools/parallel_forest_pool_check.py`, both layouts: `PASS 16 RF/ET
  configurations, persistent snapshots and refusals`; every two-device
  prediction equals the one-device `parallel_groves` prediction in the same
  process. `public-separate.json == public-packed.json`.

## Cross-vendor

`out/public-separate.json` from this run is equal as JSON, every output shape,
dtype and SHA256, to `../forest-grove-pool-h100/out/public-separate.json`
from two H100s. The public RF/ET pooled predictions of these 16
configurations therefore have the same bits on two MI300X and two H100. No
Apple column was run for this driver (it needs two devices).

## The failed first leg

`failed-first-leg/` is the first attempt at the same commit (pod
`0itrsyexjs499c`). Its body ran four GPU processes at once; the packed native
and fault checks died with `hipErrorOutOfMemory` after 14 and 3 fixtures, and
the public check could not import because the base binding was not built. No
fixture that ran reported a bit difference. The second body builds the base
binding and runs GPU work serially.

No speed or beyond-one-device capacity claim is made.
