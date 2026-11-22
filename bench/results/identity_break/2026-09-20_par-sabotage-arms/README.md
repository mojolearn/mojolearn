# The 43 unwatched `par-*` sabotage arms, on CPU: 3 closed, 40 unreachable

Branch `lane/gpu-confirm-never-launched`, 2026-09-20. Three RunPod CPU pods,
`runpod/base:1.3.1-ubuntu2204`, x86-64, 16 vCPU, **$0.062 total**.

## What this was for

`tools/verification_matrix.py` stood at **217 `seen(build)`, 38 `none`, 7
`declared`**. The 45 not-`seen(build)` were 43 `par-*` lanes plus the two
decode-session lanes. A sabotage nobody has watched fail is indistinguishable
from no sabotage.

`seen(build)` is decided **empirically**, by `sabotage_moves`: a sabotage
column paired against a clean column of the same device class whose HASHES
DIFFER. A lane in no family's `training_lanes` can still earn it. That is how
16 other `par-*` lanes earned theirs on 2026-09-17, from a plain
`-D MOJOLEARN_HOST_SABOTAGE=1` CPU host build. This is the same recipe.

## How each pair was taken

Per shard, the host families were built TWICE from the same tree in the same
run: production into `python/mojolearn/host/`, the negative control into
`python/mojolearn/host-sabotage/` with

    -D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_BORDER_TYPES_SABOTAGE=1
    -D MOJOLEARN_ORDERED_SABOTAGE=1 -D MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE=1
    -D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1 -D MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1
    -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1 -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1

(`host_surface.sabotage_build_defines`'s union over all 32 families), then

    python3 tools/identity_break.py --lanes <shard> --fixtures base,ties --repeats 2 --json cpu-x86.json
    env MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        python3 tools/identity_break.py --lanes <shard> --fixtures base,ties --repeats 2 --json cpu-x86.sabotage.json

**`--repeats 2` on BOTH sides.** `verification_matrix.stable_digest` refuses a
part with fewer than two repeats, so a one-repeat sabotage column reads STABLE,
falls through to a `MOVED/DIVERGENT` fallback a STABLE verdict does not
satisfy, and the move is **silently discarded**. That flag once cost 135
lanes' worth of real negative controls.

`arm_bytes_differ.txt` records `identical-byte arms: 0` in every shard: all 32
bindings DIFFER between the two sets, so the defines did something.
`missing_bindings.txt` is empty in every shard: a binding absent from the
sabotage set would load nothing and read REFUSED, which a diff would miscount
as a catch.

## What moved: 3 of 43

| lane | shard | parts moved | before → after (train) |
|---|---|---|---|
| `par-forest-et-clf` | a | batch, infer, model, train | `c586b27a3b049614` → `6567d13a392cede0` |
| `par-forest-reg` | a | batch, infer, model, train | `f032017cfd1c73dd` → `51c47ab53a694b17` |
| `par-forecast-holtwinters` | c | batch, train | `5e8ffa04d8c2fb3d` → `7bf9c2cccc9035df` |

`par-forest-et-clf`'s batch part and `par-forecast-holtwinters`'s batch part
read `BATCH_MOVED` with the position and both bit patterns, e.g.

    BATCH_MOVED:predict_proba:row 0 of 20000 alone:output 0 element 0:
      whole 0x3fdfc9cb40000000 vs alone 0x3fdffc6160000000

## Why the other 40 cannot be closed this way, in the harness's own words

    NotImplementedError: no CPU implementation of the cooperative multi-GPU
    driver kmeans_fit yet: its shards are device row tiles, chunks or ranges
    inside the GPU binding, which no host binding restates

Measured over every committed CPU column in the tree: **19 `par-*` lanes have
ever produced a STABLE cell on one, and 35 have only ever refused.** The 19 are
the `cooperative=False` drivers, whose shards are cut in Python by
`_parallel_pool.DevicePool` with each worker masked to a single device. The 35
are the `cooperative=True` drivers, whose shards live inside the GPU binding
and which set `MOJOLEARN_*_DEVICE_COUNT` for one worker that sees all devices.

A sabotage column whose cells REFUSE rather than move is **not counted**, and
should not be: a refusal is a build that did not run, not arithmetic that
changed. So for those lanes the negative control is only stateable on real
devices, which is what the two-device GPU leg in
`bench/results/identity_break/2026-09-20_gpu-confirm-never-launched/` is for.

## Four lanes that ran clean and refused under the arm

`par-ivf`, `par-queries-nn`, `par-rbf-sampler` (shard b) and
`par-forecast-arima` (shard c) produce STABLE cells in the clean column and
REFUSE in the sabotage column. They are correctly **not** credited.

This record does not say why. Two explanations fit the evidence here — the arm
perturbs something those lanes reach in a way that breaks the binding rather
than the arithmetic, or the sabotage build lacks a binding they reach — and
nothing in this run distinguishes them. Naming a cause would be a claim this
column cannot support.

## The two decode-session lanes

`mamba1-decode-session` and `transformer-decode-session` are in shard c and
refuse on CPU by construction: `mamba1_session_create` and
`transformer_decode_session_create` exist only in `bindings/_mojolearn_mamba.mojo`
and `bindings/_mojolearn_transformer.mojo`, and each constructor refuses BY
NAME on the host route. Their GPU column is owed elsewhere; no CPU run can
supply it.

## Shards

| shard | lanes | pod | spend |
|---|---|---|---|
| `a-trees-gbdt` | 13 | `zaho1l0oqoy2hh` | $0.0205 |
| `b-classical-graph` | 24 | `g60mkatgi6epm4` | $0.0220 |
| `c-neural-decode` | 6 par + 2 decode | `epn4g2y79weyad` | $0.0199 |

Every pod DELETEd and verified gone (HTTP 204, then GET 404, then absent from
the listing); see each `teardown.txt`.

**An earlier attempt at these three shards cost $0.34 and recorded nothing**,
because `import mojolearn` cannot complete without
`python/mojolearn/.libs/libMojolearnMath.so`, which `.gitignore` excludes and
no `bindings/` script builds. `tools/runpod_cpu_leg.sh` now builds it. The
`bincache/` directories and per-family `build_*.log`s from these runs are not
committed: they are regenerable build artifacts, not testimony.
