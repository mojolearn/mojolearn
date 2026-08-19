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
