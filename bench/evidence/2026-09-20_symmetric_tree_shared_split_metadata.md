# Symmetric-tree shared split metadata

`compute_bins_and_add_kernel` previously reread five split descriptors from
global memory for every row and level of every oblivious tree. The revised
kernel stages each tree's at-most-32 level records once per thread block and
keeps the leaf-index construction and tree-order cursor additions unchanged.
This benefits both the repeated learn-set apply during boosting and resident
oblivious inference.

## Million-row end-to-end fit

Apple M4, FAST mode, public `bench/lanes_price_main.mojo` GBDT lane:
1,000,000 rows, 100 float features, SymmetricTree Logloss, depth 6, five
trees. The clock includes raw-float border construction, quantization, all
boosting work, and model construction.

- Baseline measured rounds: 0.480853 and 0.489440 s (median 0.485147 s).
- Shared-metadata rounds: 0.380675, 0.379520, 0.379155, 0.376772, and
  0.375618 s (median 0.379155 s).
- End-to-end improvement: 21.8%.

Every warmup and measured round produced the complete split/bin/type,
leaf-value, and loss-ledger hash `35b01921a435ca40`. The candidate's first
qualification run also produced 0.381060 and 0.396353 s with that hash.

The optimization does not change arithmetic, predicates, tree order, or
cursor addition order. Depth is already bounded below the 32-record shared
capacity by the GBDT admission checks.
