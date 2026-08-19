# What runs on the host, and the rule that keeps it safe

CatBoost's GPU trainer is not all-GPU, and the split is deliberate. Copying
where they put the boundary is part of porting the design, not a concession.

## The rule

**Host work must be O(leaves) or O(candidates). It must never be O(rows).**

CatBoost never touches a row on the host during training. Everything that
scales with the dataset is on the device; everything that scales with the
TREE is on the host, where a branchy decision costs nothing and a kernel
launch would cost more than the work.

## What they run on the host, verified in their source

| step | where | why the host |
|---|---|---|
| final argmax over block-best records | `greedy_search_helper.cpp:513-532` | the device reduces to at most 64 records; a plain C++ loop over 64 structs beats a second kernel and a second synchronization |
| `BuildNecessaryHistograms`: which leaves to build, which to derive, which sibling is smaller | `split_properties_helper.cpp:1288-1334` | O(leaves), and it is branchy hashing by leaf path, which is miserable on a GPU and free here |
| `SelectLeavesToVisit`, `FindMaxDepth` | `greedy_search_helper.cpp:398-418` | O(leaves) |
| the per-leaf sort loop | `split_points.cu:658-689` | NOT a good example. This is the one their own TODO calls wrong, and our port replaced it with a device-side segmented partition |
| leaf sizes, read without a copy | `split_points.cu:372`, `:379` | the split kernel writes each partition to device memory AND pinned host memory in the same store, so the host learns sizes with no readback in the critical path |

## Where this port stands against that

**Already host, correctly:** `split_properties_helper.mojo`
(`build_necessary_histograms`, the smaller-sibling rule and the terminal-pair
skip) and `structure_searcher_template.mojo` (the level schedule). Both are
O(leaves) and both are verified.

**Currently on device where CatBoost is on host:** the final argmax. Our
`compute_optimal_splits_kernel` does a block-wide tree reduction and writes
one record per block, which matches theirs; what we have not written is the
host loop over those records, because the driver launches a single block and
the question has not arisen. It arises the moment the candidate count exceeds
one block's worth, and the fix is a host loop over at most 64 structs, not a
second kernel.

**Not yet done, and it is the one with a measured payoff on their side:**
pinned host partitions. `update_partitions_after_split_kernel` already takes
and writes `host_offset` / `host_size`, so the device half exists. The driver
allocates them as ordinary device buffers, so today the host still has to
copy to learn a leaf's size. Tracked in UNWIRED.md.

## The second rule, added 2026-08-19

**"The host DECIDES" and "the host WAITS" are separate properties, and only
the second one costs anything.** The rule above constrains the first. This
one constrains the second, and it exists because a proposal to remove host
waits arrived and half of it was in scope and half was not.

### What CatBoost actually does, read at `13ce0e1`

The pin for this port is `54a8143a` and the lines below were read at
`13ce0e1`, so treat them as current-upstream evidence rather than
pin-exact; `tools/check_upstream.sh` is the drift check.

| host read | file | how often |
|---|---|---|
| `bestProps.Read(propsCpu)`, then a host argmin over the block records, `UpdateBestSplit`, `PrintBestScore`, `FeaturesManager.IsCtr` | `greedy_search_helper.cpp:517-545` | **once per level** |
| `RebuildLeavesSizes`: `currentParts.Read(partsCpu)` | `split_properties_helper.cpp:803` | once per level |
| `FastUpdateLeavesSizes`: `currentParts.Read(partsCpu)` | `split_properties_helper.cpp:822` | **once per NEW LEAF**, so 32 of them in the last level of a depth-6 tree |
| `ReadReduce(currentPartStats)` | `greedy_search_helper.cpp:632` | once per tree, at terminate |
| `DefaultStream().Synchronize()` | `split_properties_helper.cpp:961` | per level |

**So CatBoost's host learns the chosen split before it enqueues the split
kernel, and it blocks to do it.** That is not incidental. The value is
printed, it gates a CTR bookkeeping call, and the argmin across score blocks
and devices is done in C++.

### The rule

**Cut our host waits down to THEIR count. Not below it, not yet.**

Removing a wait we have and they do not is restoring fidelity: it is our
artifact, caused by Mojo having no counterpart to a CUDA stream plus pinned
host memory, and it inflates a number that is supposed to measure THEIR
design. That is in scope and it is not an optimization.

Removing a wait they also have is an improvement on CatBoost. It may well be
a good idea. It is out of scope until this port produces its number, because
the question this tree exists to answer is "is CatBoost's design fast on
Metal", and a port that runs a better control plane than CatBoost's cannot
answer it.

### The specific proposal, recorded for after the measurement

An oblivious tree has a **data-independent shape**: depth levels, `2^d`
leaves, and a launch sequence that does not depend on anything the data says.
So the whole per-tree schedule could be enqueued up front, with the argmax
feeding the next level as a DEVICE BUFFER instead of a host value, and the
host would never wait. Lossguide cannot do this and symmetric can, which is
plausibly part of why CatBoost chose oblivious trees.

Two things to keep straight about it.

**It departs from CatBoost**, per the table above: their host does read the
split. This is the first optimization to take once the port concludes, not a
fidelity fix to fold in now.

**Its arithmetic is smaller than it looks.** `RESUME.md` priced the 54 round
trips: 54 bare drains cost 0.76 ms, and 54 copy-plus-drain cost 13.0 ms. The
barrier is nearly free and the HOST TRANSFER attached to it is the cost. So
going from 54 drains to about 1 buys roughly **13 ms of the fixed 34**, which
takes a 50 ms tree to about 37 and leaves us still above CatBoost's 30.1.
The other ~21 ms is launch COUNT, about 96 per tree, which enqueuing early
does not reduce. Both halves are needed and neither alone closes the gap.

**Already in flight:** a peer session is cutting the drains in
`ported/methods/greedy_subsets_searcher/greedy_search_helper.mojo` right now,
uncommitted in this shared checkout, with the removals annotated against
`split_points.cu`. Do not duplicate that work. This document is the rule it
should be checked against.
