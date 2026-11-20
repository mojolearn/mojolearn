# Native NearestCentroid code scratch and shared label packing

Baseline: `f29d59c52e` in `lane/mojo-finite-tests-20261110`. This follow-up
closes the remaining private prediction-code Array identified by the seven-family
inventory. It does not change that inventory's frozen `b7f177d1db` comparison.
No lane compilation, test execution, timing, identity or box work occurred.

`NcLabelCodes` owns the int32 prediction buffer in Mojo. Its constructor uses
the same `NativeArrayLayout` dimension/byte-count checks as the former private
Array, and its empty address remains zero. `_nc_predict_values` passes this
address to the same fused, distance-label and score-label requests, then keeps
the owner alive through final packing. No Array, memoryview or buffer export is
constructed for those codes. Fit publication, class normalization, integer/bool
promotion, real-label conversion and class ordering remain unchanged.

`labels_decode_i32_codes` accepts the borrowed native List at PACK. It shares one
decode implementation with the existing public object-input decoder. Two small
native source types specialize address access at compile time: the legacy source
pins its Python input and obtains its address at the existing stages; the native
source carries its already owned address. There is no runtime source switch or
Python wrapper around the native codes. Legacy class admission, code admission,
size/dtype reads and address-access order remain in place.

The decoder's private int64 widening buffer and numeric class table are now
Mojo Lists too. Int32 widening still calls the same parallel cast once; numeric
labels still call the same typed gather once. The existing native list fill
handles strings, booleans and integers outside int64, preserving negative-index
and out-of-range behavior. Typed table storage keeps the original conversions
and int64 range test. Scratch remains alive through released-GIL gather work
and list packing. The same logical data-buffer sizes, zero initialization, numerical
calls and data-copy count remain; Python wrapper objects and their metadata are
removed. No new serial numerical loop, kernel, route, default or fallback was
added. No performance or identity result is claimed.

The new native-code decoder has one product consumer:
`neighbors_local_native::_class_array`. The shared legacy decoder still serves
`labels_native_entry`, `logistic_native`, `neighbors_knn_native`,
`neighbors_semi_native`, `parallel_neighbor_worker`, `prep_classify_native`,
`x_linear_native`, and internal score paths in `classical_labels`. Exact call
sites are retained in `semantics-and-consumers.json`. No live interface was
removed and no new Python type registration is needed.

Removed: three private Array roles (NC int32 codes, decoder int64 widening,
decoder numeric table), NC's object-input decoder import/use, and their repeated
private Array address lookups. Public result Arrays/lists, fitted attributes,
class-label input owners, and boundary integer/real class normalization lists
remain. This is not a claim that the whole public family is Python-free.

Source comparison confirms the existing NC numerical plan body and fit body
are byte-identical; class coercion differs only in the typed code signature and
decoder name. Underlying `x_neighbors` operations, label cast/gather kernels and
list-fill implementation are unchanged. No intended bit change or A/B define.

Coverage authored, not executed:

- The existing native plan suite grows from 20 to 22 cases, adding negative and
  overflow shape refusals before allocation.
- The existing 11 actual descriptor cases use `NcLabelCodes`, keeping their
  closed-form numerical oracles and adding the empty-code address check.
  CPU and ordered GPU wrappers reuse those cases. The original concurrent GPU
  wrapper remains parked for the separately recorded MAX/Metal failure.
- Four new Python boundary definitions expand to 30 cases: empty/nonempty int32/int64 decode,
  exact integer/real/bool/Unicode/large-integer labels, storage lifetime, negative
  indices and range refusal, rebound NC class order/dtype, and short-class
  refusal. Existing Python label/NC tests are retained.

Static source audit exited 0 over 89 product roots. Changed product closure:
18 platform records across nine physical host groups (`byte_trainer`,
`classical_decomp`, `classical_family`, `classical`, `classical_prep`,
`classical_trees`, `orchestration`, `parallel`, `xseq`). The native NC plan closure
has 29 sources and does not reach either boundary importer. Exact stems, flags,
roots and source evidence are in `source-audit.json`; this is rebuild scope,
not a compile result.

Complete baseline/repaired sources, patches, source audit, consumer inventory
and push evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/nc-native-label-codes/`.
RUN OWED: combined affected compilation, native 22/11 cases and the 30 boundary
cases. No lane job was queued.
