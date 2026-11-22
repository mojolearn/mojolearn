# The NVIDIA and AMD columns for the last five single-device vendor-class gaps

`docs/VERIFICATION_MATRIX.md` read **"GPU column on fewer than three classes:
5"** before these two files, and those five were the whole list:
`gbdt-ordered`, `gbdt-ordered-bayesian-noise`, `gbdt-border-types`,
`gbdt-bfa-quantile`, `gbdt-catboost-defaults`. Each carried `apple` alone.
Each is gbdt-family, and `host_surface.FAMILIES` routes all five through the
same pair -- `build_gbdt` and `build_gbdt_host` -- which the leg body derived
for itself on each box rather than being handed a list.

Two leases, run side by side, same body (`tools/gap_column_leg.sh`), same
wrapper, same commit `11275a323cc0e7a80fd787ea52b0dcb36ff3a222`, both built
from source on the box.

| file | box | pod | `package.par_devices` | `admit` | cells |
|---|---|---|---|---|---|
| `nvidia-nvidia-geforce-rtx-4090-sm_89.gbdt-class-gaps.json` | RTX 4090, driver 580.159.04, sm_89 | `rnjpdre1fkwe6v` | `0` | ADMISSIBLE | 45 STABLE, 0 moved, 0 refused |
| `amd-amd-instinct-mi300x-gfx942.gbdt-class-gaps.json` | MI300X, amdgpu 6.10.5, gfx942 | `l8d51stfb45sou` | `0` | ADMISSIBLE | 44 STABLE, 0 moved, **1 refused** |

45 = 5 lanes x 9 fixtures, `--repeats 2`, default fixture set at the default
size, one device, no `--fixtures` cap and no sabotage switch set anywhere in
either run. `admit` was asked on the box about THIS path, before the reap, and
again here at the committed path; it answers ADMISSIBLE both times for both.

## What the three columns say to each other

Run at home against the columns already on main, `identity_break.py --diff`:

    apple-m4 (2026-09-19_catboost-parity) vs nvidia   IDENTICAL=45   (infer/model 90, batch 45)
    nvidia vs amd                                     IDENTICAL=44   (infer/model 88, batch 44)

Not one cell MOVED or DIVERGED on any pair. The single cell that is not
IDENTICAL in the second row is not a disagreement about bits: it is the one
AMD cell that never produced any.

## THE ONE CELL THAT IS NOT CLEAN: gbdt-ordered/base on the MI300X

    REFUSED gbdt-ordered/base: Exception: At
      max/mojo/max/gpu/host/device_context.mojo:3825:17:
      HIP call failed: hipErrorOutOfMemory (out of memory)

after 47.6 s. It is the FIRST cell of the FIRST lane of the run, on a card
with 192 GB of VRAM, and the other eight fixtures of the same lane -- and all
36 cells of the other four lanes -- are STABLE on the same box in the same
process. THE CAUSE IS NOT KNOWN AND IS NOT GUESSED AT HERE. It was seen once,
it was not reproduced serially, and the pod was terminated at the end of the
lease, so nothing further can be asked of that box. What is known is that it
is a refusal and not a divergence: the cell carries no hash to disagree with
anything.

It does not hold the lane back from the count. `verification_matrix.gpu_coverage`
credits a lane to a class on any STABLE cell, and `gbdt-ordered` has eight of
them here.

## The wrapper, because it is the part that is not in the tree

`tools/gemm_remote_leg.sh` runs an extra body as `sh /root/gemm_leg_extra.sh`
with NO environment passthrough, so `MOJOLEARN_GAP_*` set on the driving Mac
reaches the pod in no form at all. Both columns were produced by shipping this
as `MOJOLEARN_GEMM_LEG_EXTRA`, which sets them and execs the body out of the
pinned archive:

    #!/bin/sh
    MOJOLEARN_GAP_LANES=gbdt-ordered,gbdt-ordered-bayesian-noise,gbdt-border-types,gbdt-bfa-quantile,gbdt-catboost-defaults
    MOJOLEARN_GAP_SLUG=gbdt-class-gaps
    MOJOLEARN_GAP_COMMIT_DIR=bench/results/identity_break/2026-09-19_gbdt-class-gaps/
    MOJOLEARN_GAP_BUDGET=2600
    MOJOLEARN_COMPILE_JOBS=16
    export MOJOLEARN_GAP_LANES MOJOLEARN_GAP_SLUG MOJOLEARN_GAP_COMMIT_DIR \
           MOJOLEARN_GAP_BUDGET MOJOLEARN_COMPILE_JOBS
    exec sh /root/mojolearn/tools/gap_column_leg.sh

An earlier attempt named the body itself as `MOJOLEARN_GEMM_LEG_EXTRA`, which
is what that file's own usage header said to do. It rented two pods
(`7whutyzv73i9ul`, `3hz64m4pbs523f`), ran the body with an empty lane list,
exited 8 by name in three minutes and cost about $0.16. The header now spells
the wrapper.

## Cost, and what the binding cache did

| leg | build phase | identity phase | lease used | rate | bill |
|---|---|---|---|---|---|
| NVIDIA | 1731 s, 55 families, ALL FROM SOURCE | 311 s | 32 min | $0.69/h | ~$0.37 |
| AMD | 1082 s, 55 families, ALL FROM SOURCE | 263 s | 26 min | $2.39/h | ~$1.04 |

`bincache_outcomes=55 miss+built-uploaded` on both. That is expected and is
the point: `bincache/v1/` held 372 objects before today and **not one GPU
architecture had ever appeared in it**, so these two legs are the first
entries under `sm_89/` and `gfx942/` and were cold by construction. Both
promoted cleanly (`BINCACHE PROMOTED 55 refused_or_failed=0`).

They will not be hit by the next leg either, and that is a defect in the cache
and not in these runs. See `tools/bincache.py`'s `NON_BUILD_PREFIXES` and the
commit that added `MOJOLEARN_GAP_` to it: the key these 110 objects were
written under carries `MOJOLEARN_GAP_LANES`, `MOJOLEARN_GAP_SLUG`,
`MOJOLEARN_GAP_COMMIT_DIR` and `MOJOLEARN_GAP_BUDGET` in its `build_env`,
none of which can reach `mojo build`, so only a leg running this exact lane
list into this exact directory could ever have matched them.

Evidence, with the full gate files, logs and the cache key fields:
`~/mojolearn-evidence/gbdt-class-gaps-sep19/{nvidia,amd}/`.
