# The k-means saved-model part, on the CPU host route

`lane/kmeans-save` gave `KMeans` a `save` and a `load` on 2026-09-16. Scanned
over the committed identity_break column files on main BEFORE this record, the
number carrying a real `model` hash for any k-means lane was ZERO: every one
read `n/a:no-save`, because the last k-means recording
(`2026-09-15_kmeans-transform`) was taken before the format existed. The code
shipped and no column had ever hashed its bytes.

This is that column, on the CPU host route, taken on the M4 at one core.

    PYTHONPATH=<worktree>/python MOJOLEARN_NUMERIC_MODE=identical \
      python tools/identity_break.py \
      --lanes kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,\
kmeans-classic-pp,kmeans-cosine --fixtures base --repeats 2

`cpu-apple-m4.json`, commit `a8b7e6b3b`, vendor `cpu-apple-m4`,
`host.column = cpu`, run from this lane's own worktree. Host bindings: the
thirty-two-family CPU set preserved at
`~/mojolearn-evidence/ship-cpu-host-families/hostbuild/host-v2/`.

    cells=7 stable=7 moved=0 refused=0
    infer: stable=6 n/a=1        model: stable=6 n/a=1        batch: stable=6 n/a=1
    n/a: kmeans-cosine batch n/a:fit-refused, kmeans-cosine infer n/a:function,
         kmeans-cosine model n/a:no-save

## It reproduces the shipped references, so the column is not a new arithmetic

`kmeans/base` reads `infer 3d50c4c50bf078b2` and `batch ce34789a012168b5`.
Those are the exact `ref` values a table rebuilt from the records on main
carries for that cell, which the Apple (Metal) and CPU columns of
`2026-09-15_kmeans-transform` already agree on. The new information here is the
`model` part, which had no reference on any column.

## Both sabotage arms were SEEN TO FIRE

**Host build.** `bindings/build_core_host.sh` with
`MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1"` into a separate
directory, swapped in by byte copy and swapped back the same way.
`cpu-apple-m4.host-sabotage.json`: **24 of 24 comparable cells MOVED**, every
`model` cell among them.

One cell is UNCHANGED and it is the right one: `kmeans-cosine` train, whose
cell is the hash of the REFUSAL SENTENCE by design (`metric='cosine'` is
refused by name in `cluster/impl/kmeans_params.mojo::validate`). A sabotage
build cannot move a refusal, so an arm that moved it would be the surprise.

**The first attempt at this arm produced REFUSED on every cell**, which is not
a control. The guard refuses a sabotage build outside the gate by name:

    ImportError: mojolearn: .../_mojolearn_core_host.so is a SABOTAGE build and
    computes wrong answers on purpose; it is refused outside the gate
    (MOJOLEARN_HOST_ALLOW_SABOTAGE=1)

This is the same shape `lane/kmeans-save` hit on its NVIDIA leg from a
different cause, and the same lesson: a REFUSED cell is indistinguishable from
a cell that was never asked. `cpu-apple-m4.host-sabotage.log` is the rerun with
the flag set.

**Batch.** `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, exit 1,
`batch: batch_moved=6 n/a=1`, each naming the row and both bit patterns, e.g.
`BATCH_MOVED:predict:row 0 of 20000 alone:output 0 element 0: whole 0x00000004
vs alone 0x00000005`.

## What is still owed

* The other eight fixtures, and the Apple, NVIDIA and AMD columns of the
  `model` part. `lane/kmeans-save` wrote `tools/kmeans_save_nvidia_leg.sh` for
  the NVIDIA one and fixed the `env FOO=1 -u BAR` ordering bug that made its
  first attempt read REFUSED (`1647059a0`); the leg has not been rerun on a box.
* `kmeans` is still not a declared inference lane. `host_surface`'s
  `SAVED_MODEL_INFERENCE_OWED` says why, and the condition is a
  `tools/classical_host_gate.py record` run on a GPU box, which a CPU-only
  install refuses by design. `bench/results/classical_host/` has no `kmeans`
  directory.
