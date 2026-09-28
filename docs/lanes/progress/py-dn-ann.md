# py-dn-ann (sub-lane of py-decomp-nbrs): the ANN index stays resident; the distributed IVF merge is native

Branch `lane/py-dn-ann`, cut from `lane/py-decomp-nbrs`. Scope: audit ann rows 1, 2 and 5 (ranks 7 and 15 of
~/mojolearn-evidence/python_work_audit.md).

## What changed

1. **IVF-Flat index resident (retires DEVIATION 1804).** `ivf_flat_search_traced` is now
   `IvfFlatDevice(ctx, index)` (the centroids, the list data and its row norms on the device, the list norms
   downloaded once, the host CSR layout, the list sizes, the device CSR pair the Apple scan reads) followed by
   `ivf_flat_search_prepared`, the old body minus that preparation. `ivf/resident.mojo` keeps an admitted index
   and its `IvfFlatDevice` per handle (`ivf_flat_index_prepare` / `_search` / `_release`, contract in
   `bindings/ivf_index_arrays.mojo`); the two host bindings keep an admitted `IvfHostIndex` per handle under the
   same three names (`bindings/ivf_host_search.mojo`). `IVFIndex.search` and the `DistributedIVFIndex` shard
   search (`partial_storage`) go through a handle made at the first search, keyed on the five index arrays'
   identity, address and shape, released on `fit`, `extend`, collection, and never pickled. A binary without the
   doors takes the old per-call entry.
2. **x_ann indexes resident.** `x_ann/resident.mojo`: IVF-PQ, IVF-SQ, IVF-RaBitQ and CAGRA uploaded once per
   handle (`x_ann_index_prepare` / `_search` / `_release`, GPU binding), the CAGRA graph admitted once (same
   refusal text), a device all-ones filter made once, so a search uploads only its queries (and a caller's
   filter). The searches are `ivf_{pq,sq,rabitq}_search_on` and `cagra_search_on`, the bodies the one-shot
   entries now call after their own upload. The CPU host binding already read the caller's arrays in place; its
   only per-search O(n) work in Python, the all-ones filter array, is now made once per row count.
3. **DistributedIVFIndex merge native.** `ivf_merge_shards` (bindings/ivf_index_arrays.mojo, registered in the
   GPU and both host IVF bindings) is the Python merge statement for statement: per query each shard's first
   min(k, count) candidates mapped to original ids, ordered by (float32 distance, original id), the first k
   kept; the same three errors; a NaN distance is now refused by name (the Python tuple sort had no defined
   order for it). The shard id maps are int32 arrays built once in `from_index`.

## DEVIATION changes

- 1804 (IVF-Flat index host resident, uploaded every search): RETIRED. The note in
  `ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo` now records the closure; the `_ivf_impl.py` header says so.
- The Parallel IVF "driver's own Python" merge text in `_parallel_pool.py` is updated; there was no DEVIATION
  number.

## Proof and timing

PENDING (the shared NVIDIA pod went down at 19:19Z with the base job nvc1-0031 queued; see the table below once
it is back).

| machine | column | what | base s | head s | base sha | head sha | job |
|---|---|---|---|---|---|---|---|

## Unproven

Everything above until the table is filled.
