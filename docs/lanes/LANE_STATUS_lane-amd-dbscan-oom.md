# lane/amd-dbscan-oom

`lane/classical-host-recordings` took the AMD column for the saved-model
predict lanes on a RunPod MI300X and read `cells=54 stable=31 moved=0
refused=23`, with every refusal a `hipErrorOutOfMemory` raised inside
`dbscan_fit_core` while fitting 6000 rows of four columns "on a 192 GB card".
It claimed nothing about the cause and said so: one card, one process, no
instrumentation, no replication. This lane was opened to find the leak.

**There is no leak of the size that reading needs, and the card was not
192 GB.** It was 360.4 MiB.

## The three readings that settle it

Two boxes, both AMD gfx942, both terminated and verified gone (HTTP 404).

| | DigitalOcean MI325X, droplet 601157435 | RunPod MI300X, pod o3paueazuvo87t |
|---|---|---|
| cards in `/sys/class/drm` | 1 | **8**, and the one the container owns is `card33` (PCI `0000:85:00.0`), not `card0` |
| device memory free before `import mojolearn` | 260689.6 MiB of 261824.0 | **360.4 MiB of 196592.0** |
| held by a process outside this container | none | **178.6 GiB**, kfd pid 4109547 |
| `identity_break`, the three DBSCAN lanes | `cells=27 stable=27 moved=0 refused=0` | two sub-cells `REFUSED`, both `hipErrorOutOfMemory` |
| `identity_break`, the original six predict lanes | not run | `cells=54 stable=54 moved=0`, `dbscan/base` model and `dbscan-brute-l1/negative` batch `REFUSED` |
| `classical_host_gate.py record` | not run | **died**, `hipErrorOutOfMemory` in `labeled_reference_predict` |
| one `wide` fit, 36,000,000 edges | 0.44 s | **31.21 s** |
| device memory across 54 fits in one process | 285.8 MiB to a 1.5 GiB plateau | unreadable, the card was already full |

The full transcript is
`~/mojolearn-evidence/amd-dbscan-oom/runpod-mi300x-card-was-full.txt`.

### Why the RunPod figure cannot be ours

Two confirmations from inside our own process, in
`json/rocmsmi-cold.json` of the leg:

1. `used_before_any_gpu_work = 196231.6 MiB of 196592.0 MiB`. The probe reads
   that BEFORE `import mojolearn`, so none of it can be this library's.
2. Across our first fit the card's used memory **fell** by 12031.2 MiB. A fit
   of ours cannot release 11.7 GiB it never held. Another tenant did.

And `rocm-smi --showpids` named a kfd pid of 4109547 holding 178.6 GiB while
this container's own pids were five figures.

## So what was the original column measuring

A rented card that was already full, and nothing about the code that raised.
Every feature of that reading follows from it and from nothing else:

* **"ordered, not shaped"** was never ordered. The JSON, not the prose, says
  `dbscan` fitted `base`, `ties` and `hashed`, refused `wide`, `denormal` and
  `denormal_ftz`, then **fitted `dupes`**, then refused `odd` and `negative`.
  A fit that succeeds between three that failed is not an exhausted card, and
  the eighteen `agglomerative` and `spectral` cells that ran clean after it are
  not either. It is a neighbour's allocation moving under us.
* **`odd` refused and `base` succeeded** although they are the same shape and
  within 3% of the same edge count (667,520 against 687,484, measured). No
  property of the data separates them. Arrival time does.
* **DBSCAN and not the other lanes**, because DBSCAN asks for the largest
  single buffers in that lane set. On a card with 360 MiB free it is the first
  thing to fail and the only thing that has to.
* **`refused=0` on an 80 GB A100** was never about 80 GB being enough. That
  A100 was not shared.

## What the instrumentation found instead

`tools/dbscan_memory_probe.py` on the dedicated MI325X, 24 fits of one
unchanging shape in one process:

    fits 2-4   +179 MiB each     the HIP context and the code objects
    fits 5-25  +2 MiB every second fit, exactly, with no plateau

That is **1.000 MiB per fit**, and it is real. It is also the same 1.000 MiB
on every path:

| arm | MiB per fit, warm |
|---|---|
| `prediction_data=True` (`dbscan_fit_core`) | +1.000 |
| `prediction_data=False` (`dbscan_fit`) | +1.000 |
| `algorithm='brute'` (no ball cover) | +1.000 |
| `sample_weight` (the weighted `ja1` scratch) | +1.000 |
| `base` (687,484 edges) against `negative` (5,034,848 edges) | +1.000 either way |

**A residual that does not move when the code path changes and does not move
when the edge count changes by a factor of seven is not a buffer `dbscan/`
allocates.** Every buffer in `dbscan/estimator.mojo`,
`dbscan/impl/dbscan.mojo` and `dbscan/impl/runner.mojo` is sized by `n_rows`,
by `batch * n_rows` or by the edge count, and all of them would have moved.
What is per-call and identical across all four paths is the
`var ctx = DeviceContext()` that `bindings/_mojolearn_estimators.mojo::
_dbscan_fit_run` constructs fresh inside every call, and MAX's own bookkeeping
behind it.

At 1 MiB per fit a 256 GB card is about 250,000 fits from caring, and a
release record is a few thousand. **It is not what refused on the MI300X**,
which needed 190 GB, not 1 MiB. It is written down here because it had not
flattened by fit 25 and nobody has looked at it, not because this lane thinks
it matters yet.

## What was considered and deliberately not changed

`dbscan/impl/dbscan.mojo:248` derives the batch budget from **80% of TOTAL**
device memory when `max_mbytes_per_batch` is left at its default, following
`dbscan.cuh:157-158` and their note that free memory cannot be relied on. On a
card where 99.8% of the memory belongs to a stranger that is a fantasy, and it
picks the largest possible batch exactly when the least memory is available.

Changing it to read FREE memory was considered and **rejected**. It would make
the batch count, and therefore the reduction order, depend on what another
tenant happened to be holding at that instant. This library refuses results
that depend on transient machine state, and a budget that reads free memory is
exactly that. The `budget` arm of the probe exists to ask the question on a
box rather than in an argument, and on the MI325X every budget from the
default down to 8 MB ran clean.

## What is owed, and to whom

* **The AMD column for the DBSCAN predict lanes is still owed.** It is owed on
  a card that is ours. Nothing is wrong with the code it would exercise: on the
  MI325X the same three lanes read `cells=27 stable=27 moved=0 refused=0`, and
  the six-lane run on the MI300X read `cells=54 stable=54 moved=0` with only
  two sub-cells lost to the full card.
* **`bench/results/identity_break/2026-09-16_amd-mi300x/README.md` and the
  comment at `python/mojolearn/host_surface.py:236` are corrected by this lane**,
  because both say "on a 192 GB card" and one reads the refusals as ordered.
* **A recommendation this lane did not implement.** `tools/gemm_remote_leg.sh`
  and `tools/do_extra_leg.sh` log `rocm-smi` and `nvidia-smi` at box acceptance
  but not free memory, so a column can be taken on a card with 360 MiB free and
  nothing in the leg record will say so. Adding `--showmeminfo vram` to the
  acceptance log is one line in each and would have saved this lane and the
  last one. It is left unwritten because both runners are shared infrastructure
  and there is no box left to run the change on, and unrun code stays off main.

## The probe, and the gate it had to have

`tools/dbscan_memory_probe.py` reads the device from outside this library
before and after every fit and keeps fitting past every raise, so the answer is
a trend rather than a stopping point. It refuses to report before it has shown
it can move: the self check reads the device before any GPU work, runs one fit,
reads again, and prints PROBE CANNOT MOVE and exits 3 under the floor.

**That gate fired on a paid box and was right to.** On the RunPod pod the sysfs
reader was watching `/sys/class/drm/card0` while the container owned `card33`,
so it read a stranger's counter, saw it never move, and refused every arm
rather than writing a flat line into a file. The flat line would have been
indistinguishable from a clean allocator. The reader now asks `rocm-smi`, which
enumerates the devices the container actually owns, and the floor is on the
absolute movement because on a shared card the figure can fall across our fit.

It was rehearsed on the Mac against a stub before either box: with nothing
allocated it printed PROBE CANNOT MOVE and exited 3 without running an arm,
with 64 MiB held per fit it read `+64.0 MiB per fit`, and against a stub whose
constructor raises `TypeError` it stopped on fit one with THE PROBE IS BROKEN,
NOT THE FIT rather than filling a file with 54 rows that would have read like a
reproduction.

## Spend

Two boxes, both destroyed and verified gone by HTTP 404.

| box | what it answered | minutes |
|---|---|---|
| DigitalOcean `gpu-mi325x1-256gb`, droplet 601157435 | the control: 54 fits, `refused=0`, a 1.5 GiB plateau | 6 |
| RunPod `AMD Instinct MI300X OAM`, pod o3paueazuvo87t | the reproduction, and the card that was already full | 12 |

Ten RunPod NVIDIA specs answered "There are no instances currently available"
and billed nothing; nothing was pinned, which is why there was an AMD box at
all.
