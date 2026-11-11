# Mojolearn GP mixed scalar index proof scope

This is a source proof for the two insertion-only temporary indices removed by `b25d2e9357`. It does not establish equivalence for general Python dictionaries, mutable/custom keys, deletion, arbitrary interpreters, or a future CPython implementation. There is no runtime or speed evidence from this lane.

## Preconditions from the actual GP path

`gp_labels_native::_gp_sorted_classes` first admits every label. `_gp_number_kind` permits exact builtin bool/int/float and exactly connected NumPy scalar classes whose dtype kind is bool, signed/unsigned integer or real float. Text takes its existing separate path. Subclasses, custom protocols, array-valued keys, complex values and other scalar kinds do not enter this index; NaNs are refused before indexing. The surviving scalar values have immutable values and stable hashes during the operation. Signed zero and infinities are retained. NumPy mixed promotion need not give a transitive mathematical equivalence relation, which is why the implementation retains the actual scalar C equality operation and its operand order rather than normalizing keys.

The former first index was a new dict from a flat exact list, with None values. The second was a new dict from zip of the sorted public classes and range of their indices. Neither deletes keys or publishes its dict. A duplicate updates the value and retains the first key object. There are no supported user callbacks that mutate these private indices. Sorting can reorder the public class list but does not mutate scalar values.

## Invariant and induction

Fix one table capacity and stable hash h. All keys of hash h use the same deterministic probe sequence. Consider the first occurrence of each physical slot on that sequence; repeated visits to a slot do not create a new position. A new key occupies the first available slot. With no deletion, every slot before an earlier key's first encounter was occupied when it was inserted and remains occupied. A later key of hash h therefore cannot occupy an earlier first-encounter position. Intervening different-hash insertions only consume slots; they cannot create a vacancy or move the earlier key. This proves insertion order for first encounters of same-hash keys while capacity is fixed.

A resize copies the dense entry array without reordering, then rebuilds indices by traversing that array from its first entry to its last. The same insertion argument applies in the new capacity. Induction over all insertions/resizes proves that a lookup's first encounters of same-hash keys follow their insertion order. Value overwrites do not move entries.

Dictionary lookup checks key identity first, and only compares nonidentical keys when their stored hash equals the query hash, calling equality with the stored key as the first operand. The native chain does exactly those meaningful tests in that first-encounter order. It omits different-hash slots and repeated false comparisons against an immutable scalar. It therefore returns the same first matching key or the same first scalar comparison error, without requiring equality to be symmetric or transitive. The second index is constructed in sorted-class order, and every original input is looked up again in it; the code does not incorrectly remap the first index's equivalence classes.

## Supported source anchors

Raw CPython source and hashes are retained under `~/mojolearn-evidence/python-mojo-migration-20261110/gp-completion-20261111/mixed/cpython/`. The audited release tags are 3.10.0 through 3.14.0. Entry append, first-empty probing, generic comparison and insertion-ordered rebuild retain this invariant in each inspected source:

| CPython | Probe / insertion | Ordered rebuild | Dense-order preservation |
| --- | --- | --- | --- |
| 3.10.0 | find_empty_slot 1042; insertdict 1069 | build_indices 1195 | dictresize 1279–1310 |
| 3.11.0 | find_empty_slot 1165; insertdict 1226 | build_indices_generic 1360 | dictresize 1483–1493 |
| 3.12.0 | find_empty_slot 1176; insertdict 1236 | build_indices_generic 1379 | dictresize 1501–1511 |
| 3.13.0 | find_empty_slot 1657; insert_combined_dict 1679 | build_indices_generic 1906 | dictresize 2040–2050 |
| 3.14.0 | find_empty_slot 1701; insert_combined_dict 1723 | build_indices_generic 1937 | dictresize 2081–2091 |

Primary sources: [CPython 3.10](https://github.com/python/cpython/blob/v3.10.0/Objects/dictobject.c), [3.11](https://github.com/python/cpython/blob/v3.11.0/Objects/dictobject.c), [3.12](https://github.com/python/cpython/blob/v3.12.0/Objects/dictobject.c), [3.13](https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c), [3.14](https://github.com/python/cpython/blob/v3.14.0/Objects/dictobject.c). No private CPython layout or probe implementation is copied into product code; product uses public C APIs and Mojo containers. Patch-release source variations were not exhaustively audited.

## Complexity and allocation limits

Both forms have expected linear total insertion/query cost under ordinary hash distribution and quadratic worst-case work for adversarial distinct same-hash keys. The native form uses a Mojo hash-to-bucket index plus contiguous key-pin, link and code lists. It preserves the meaningful equality-call order, but its native metadata allocation pattern and physical probing differ; it does not claim equal or improved wall time. The old first dict and zip/range/code dict containers are removed. Public classes, scalar identities, hash/equality operations, list sorting, numerical kernels and device copies are retained. No measured numerical winner is replaced. End-to-end measurement remains owed before any speed claim.

The first urllib source download failed certificate validation; that failure is recorded. System curl subsequently downloaded all five sources successfully using its configured certificate trust. No product code was executed.
