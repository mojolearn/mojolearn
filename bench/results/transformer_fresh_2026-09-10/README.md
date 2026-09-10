# Transformer discarded-cache prefill — September 10, 2026

Opt-in candidate913a2cca with reproduction driver d26a32e3. Stateless forward's documented contract discards its cache. The candidate keeps the original zero device cache, full block, capacity/window metadata and ordered mutable-weight refusals, while omitting two host-zero cache allocations plus two cache uploads/downloads. Original large GQA shapes save64MiB of transfers per request. Explicit-state forward/decode, linear/ring cache contents and backward are unchanged.

Apple correctness passed:82 complete cross-build array hashes,188,928 fresh-vs-explicit-zero-cache output cells, actual fresh dispatch without host cache allocation, ordered mutable-weight and8193-position capacity refusals,116 public surface checks and the HD128 suite. The recorded sequential tiny/midsize Apple prices regress and are not evidence to enable Apple by default. H100 results and reversed-order confirmation are pending; no opponent ratio is claimed because original transformer Torch admission remains unqualified.

Exact Apple script/logs/JSON maps are under apple/. Root owns builds, shared GPU scheduling and opponent records. The candidate is compile-guarded until measured and admitted.
