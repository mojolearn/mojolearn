# Mojolearn native sparse connectivity preparation

Source-only migration; compilation, installed boundary execution and timing
are not yet run. The hierarchy numerical kernels, routes and defaults are
unchanged. The existing COO ABI remains rows, then columns, then values, all
float32, with mode 2 and every stored edge retained.

`providers/python/hierarchy_connectivity_pyapi.mojo` admits exact supported
SciPy storage and owns its Python pins. It captures numeric ndarray addresses,
dimensions, strides, dtype widths/kinds and byte order into
`core/hierarchy_connectivity_plan.mojo.ConnectivityStorage`. The native plan
owns its final COO `List[Float32]`. The existing `HierarchyConnectivity`
signature and six-argument constructor remain compatible; its native plan
now keeps sparse output alive through the caller's descriptor call. Dense
input retains the original snapshot owners in separate boundary fields.

All current consumers are in `bindings/cfb_agglo.mojo`: construction at the
existing admission stage, reading address/count/edges/mode, and retaining
`aux` until the descriptor call finishes. No caller reads the former private
Python output Array. The later Agglomerative boundary migration preserves
this shared API and call order.

| Input | Native conversion and order |
| --- | --- |
| CSR/CSC | Validate compressed pointers in parallel; convert disjoint stored-entry spans in original major/entry order. Empty majors, duplicate entries, zeros and ignored trailing storage remain unchanged. |
| BSR | Same compressed plan, preserving block row/column order. Output spans split even a single large row or block across tasks. |
| COO | Parallel per-entry word conversion with the original coordinate count/bounds refusals; entries remain in stored order. |
| DIA | Parallel diagonal counts and native prefix metadata, followed by disjoint output spans; out-of-square positions stay excluded and repeated diagonals remain separate. |
| DOK/LIL | Ordered CPython storage admission with the existing native scalar normalization/refusals; typed admitted records are packed in parallel. |

Compressed conversion finds the first malformed pointer, then reads only the
proven complete-row prefix. Invalid indices in that prefix win over the later
pointer error, matching the original interleaved refusal order. Each output
task performs one upper-bound search and advances a linear major cursor; it
does not search for a row for every edge. An index is loaded once per stored
entry per task, including a possible split block at a task boundary. DIA uses
the same bounded task-summary prefix pattern as existing native label plans.
Independent work uses the existing host core/grain policy. No new hard-coded
thread limit, row-targeted route, or serial numerical fallback is introduced.

DOK dictionary iteration and LIL object-array/list unpack cannot release the
GIL or reorder input admission without changing the first refused key/value.
Those loops remain explicitly boundary work. They reuse `_SparseKinds`,
`_sparse_coordinate_k` and `_sparse_scalar_word_k`; native CPython dictionary,
tuple and list access replaces private pair-list allocation and item calls.
Scalar normalization required for the existing per-item conversion refusal
stays in admission and is not claimed as parallel computation. No SciPy
conversion method or user scalar hook is invoked.

Removed: three incremental Float32 scratch lists and the private Python Array
used for sparse output, the serial final packing loop, numeric sparse
conversion loops from the Python boundary, the dense pin tuple, and the DOK
`list(dictionary.items())` envelope. Numeric formats now write their one
native output buffer directly. DOK/LIL retain one packed native record list
plus the final output, the same total float payload volume as the old three
lists plus final buffer. DIA adds integer prefix metadata proportional to
stored diagonals; compressed formats use only task-summary scratch. Source
structure is not a measured performance claim.

The 21 native fixtures exercise actual conversion plans with literal stored
edge oracles, duplicate/zero retention, strided byte-swapped storage, BSR
layout, clipped/repeated/extreme diagonals, pointer/index refusal order,
empty spans, detached/moved output ownership, and large skewed/prefix cases.
Thirteen retained Python boundary cases cover all seven public sparse formats
without conversion callbacks, unsupported scalar hooks, and ordered
key/value refusals. Existing tests remain. No numerical backend is duplicated
and no source guard, experiment switch or Python fallback is added.
