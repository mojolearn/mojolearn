# `par-queries-nn`'s batch part on two AMD devices: what it is not, and what it is

Written 2026-09-20, lane/par-verify-and-queries-nn. No GPU was rented for this;
every claim below is from code already in the tree and from captures already on
disk.

## The symptom, exactly

On 2026-09-19, pod `162bilnfsj8nmn`, 2x MI300X (gfx942), commit `331cdfaa3`,
`par-queries-nn`'s **batch** part refused on all nine fixtures x two repeats =
18 serial attempts with `MOJOLEARN_PAR_DEVICES=0,1`, and on none with
`MOJOLEARN_PAR_DEVICES=0`. Its train, infer and model parts were IDENTICAL one
against two. The capture is
`bench/results/identity_break/2026-09-19_par-lane-amd-class/README.md`; the raw
columns are `amd-amd-instinct-mi300x-gfx942.par-{one,two}.json` in
`~/mojolearn-evidence/e1g/2026-09-19_amd-2gpu-par-class/remote/two-device-par/`.

The cell carries, and carries nothing more:

    batch: RuntimeError: GPU worker failed:
    Traceback (most recent call last):
      File ".../_parallel_worker.py", line 386, in main
        response = (True, execute(request))
      File ".../_parallel_worker.py", li

That is the 300-character cap keeping the OUTERMOST frames, fixed later the
same day (`identity_break.ERROR_TEXT_LIMIT`, `_clip_error`,
`write_error_sidecar`). **The cause inside the worker is not known from that
run and is not guessed at here.**

## THE PEER-COPY HYPOTHESIS IS FALSIFIED, by reading the path

The standing suspicion was `amd-mi300x-sriov-peer-copy-stale-read`: a device-1
kernel reading a cross-device copy's destination before it is written, because
the bytes did not go through `transfer_bytes` host staging.

**`par-queries-nn`'s batch path performs no cross-device transfer of any kind.**
The whole path is four files and none of them moves a byte between devices:

  * `tools/identity_break.py::_batch_pq` opens ONE `ParallelQueries` per part
    and asks it for the whole batch, sixteen rows alone, and the 1/7/56 split.
  * `python/mojolearn/parallel_neighbors.py::ParallelQueries.query` cuts the
    QUERY ROWS into `rows_per_shard` ranges in Python, sends one
    `('neighbor_query', state, ...)` request per range, and joins the shard
    outputs in Python with `_join` (a host `memcopy` into a host array).
  * `python/mojolearn/_parallel_pool.py::DevicePool._start` gives each worker
    its own process with `HIP_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES`) set
    to ONE index. A worker cannot address a second device even in principle.
  * results come back to the parent as pickles over a pipe.

There is no device-to-device copy to stage, so there is no unstaged one. A
driver that does it right is a better specification than prose, and the
comparison to make here is not to another `par-*` driver but to
`parallel_ivf`/`parallel_forecasting`, which use the same `DevicePool` the same
way. The `par-*` drivers as a family move nothing between devices; they shard
in Python and merge in Python.

## AND THE SAME LANE PASSED ON TWO AMD DEVICES THE DAY BEFORE

This is the fact that changes the shape of the bug. In the tree already:

    bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-queries-nn-two.json

`vendor: amd-gfx942`, `package.par_devices: "0,1"`, commit `c3b5783cf`
(2026-09-18 21:04), same `batch_protocol` (`alone 16, split [1,8,n]`), and the
batch part reads **STABLE on all nine fixtures x two repeats**. Its nine batch
hashes are, cell for cell, the nine the failing lease's ONE-device column
produced:

    base da944e45126bd8e0   ties f6f06727f6f8caec   hashed 1c5e222f1cd206a3
    wide 62dabd8ae8cf5b4f   denormal/denormal_ftz 9af443b575315e61
    dupes 4c05d8d24e5cfa27  odd 7ebe1444150c0ee5   negative 9f59fac4020afea4

So "two AMD devices cannot run this lane's batch part" is false. What is true is
that it stopped working between `c3b5783cf` and `331cdfaa3`, or between the two
environments those captures ran in, which differ in more than the commit:

| | PASS `amd-par-queries-nn-two.json` | FAIL `...par-two.json` |
|---|---|---|
| commit | `c3b5783cf` 2026-09-18 21:04 | `331cdfaa3` 2026-09-19 12:38 |
| Python | 3.10.12 | **3.14.7** |
| NumPy | 2.2.6 | 2.5.2 |
| install | wheel in site-packages | source checkout |

The source diff over that window touches this path in exactly two places, and
neither explains it: `_parallel_pool.py` only adds three IVF names to
`CPU_OPERATIONS`, and `neighbors.py` only swaps `import math` for
`from . import _portable_math as math`, where the module's only use of `math`
is `isfinite` on a radius, which `NearestNeighbors` never reaches.

## The Python sharding is device-count invariant, measured

Reproduced 2026-09-20 on an M4 with a CPU-only install (host bindings, no
`identical/` set, `_backend.vendor() == 'cpu'`, Python 3.14.7, NumPy 2.5.3),
where `DevicePool` gives each logical device its own worker PROCESS and
`neighbor_query` is an admitted `CPU_OPERATIONS` route:

    MOJOLEARN_PAR_DEVICES=0    -> nine cells, batch STABLE
    MOJOLEARN_PAR_DEVICES=0,1  -> nine cells, batch STABLE, SAME nine hashes

and those nine are the nine above. The driver's split, its shard order and its
merge do not depend on how many workers there are. Whatever the failure is, it
is below Python.

## The one structural thing that singles this lane out

In the thirteen-lane set that lease ran, `par-queries-nn` is the ONLY lane
whose BATCH declaration opens a `DevicePool` at all. Every other batch
declaration in that set calls the plain fitted estimator. So it is the only
batch part that can depend on device count, and the only one that holds ONE
pool open across 26 sequential `map` calls rather than making one call and
closing. The failing batch part took 32.17s and 32.16s on the two repeats
against 1.36s on one device, and 47.5s on the `hashed` fixture, which scales
with the fixture rather than with a fixed timeout. Its train (1.40s) and infer
(2.72s) parts were the same wall time on one device and on two.

That is where to look first: a long-lived two-worker pool on HIP making many
small queries, not the arithmetic of a shard.

## What is owed, and the exact command that pays it

Nothing here needs a fix committed, because the cause is not known and a fix
for a guessed cause is worse than none. What is owed is ONE rerun on a two-GPU
AMD box at current main, where the cap no longer throws the cause away:

    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0,1 \
      python3 tools/identity_break.py --lanes par-queries-nn --repeats 2 \
        --vendor amd-gfx942 --json /tmp/par-two.json
    # then, beside it
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0 \
      python3 tools/identity_break.py --lanes par-queries-nn --repeats 2 \
        --vendor amd-gfx942 --json /tmp/par-one.json
    python3 tools/identity_break.py --diff /tmp/par-one.json /tmp/par-two.json

The error sidecar written beside `/tmp/par-two.json` carries the untruncated
text, which is the frame inside `_parallel_worker.execute` that nine
reproductions did not preserve.

`python -m mojolearn verify --par --lanes par-queries-nn` (this lane's
companion change) does the same thing in one process on any two-GPU CUDA or HIP
box and gates on it.

## A second defect, in the leg script, that is why nobody saw the cause

`tools/two_device_par_class_amd_leg.sh` re-runs disagreeing lanes SOLO before
reporting, which is the right idea. Its selector is

    grep -E 'DIVERGENT|MOVED' ... # then --lanes "$_bad"

so a `ONE-COLUMN`/`REFUSED` cell is not a disagreement it re-runs, and the
lease ended without ever asking the question again. A refusal on exactly one of
two columns is the strongest single-lane finding that leg can produce and it is
the one shape the solo re-run does not cover.
