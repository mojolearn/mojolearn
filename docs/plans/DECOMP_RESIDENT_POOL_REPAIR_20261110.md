# Resident decomposition pool concurrency repair

Follow-up to finding PCA-R2 in
`docs/plans/PCA_INDEPENDENT_REVIEW_20261110.md` (review commit `c4a858b6af`).
The original frozen source and defects remain recorded there and in the
evidence directory. This repair changes pool ownership plumbing only.

`x_decomp/resident.mojo` now guards allocation/reuse metadata, free-list
changes, counters, live-count inspection and pointer validation with the
existing native lock components. Its changed layout uses
`MojolearnXDecompPoolV2`, layout 2. Critical sections protect metadata; they do
not span whole fits, kernel launches, device allocation, device synchronization
or the final destruction of a retired buffer. A single trim owner drains
over-cap idle batches, including concurrent returns during the drain.

Power-of-two classes, same-class LIFO reuse, released-ID reuse, the byte/count
caps and the existing two trim synchronizations remain. No numerical kernel,
launch shape, fold order, transfer route or experiment default changes.

`pool_buffer(id, n)` validates under the lock and returns an owned DeviceBuffer
reference. Direct buffer-list borrows in `s_linalg_fast`, `lanczos_dev`,
`dict_fast`, `mds_iso_dev` and `kit_device` use this accessor or its owned
sub-buffer view. Raw `_ptr(id, n)` addresses remain valid while their caller
owns the live ID: metadata reallocation does not move device storage and trim
never evicts live entries. A copied buffer pins allocation lifetime; it does
not authorize using an ID after `pool_free` or prevent subsequent reuse of its
contents. No API promises concurrent free/use of the same ID.

Removed: unguarded shared metadata mutations and reads; direct imports/borrows
of `X_DECOMP_POOL` outside its owner; duplicate pointer lookups where the owned
buffer now provides the same address. The complete filename inventory now
finds that pool symbol only in `resident.mojo`; no unknown direct importer
remains in the inspected source tree. `fa_fast` uses the live-ID pointer API
and needed no edit. This inventory is a source fact, not a concurrency verdict.

Static review compared 41 named kernel bodies in the six edited source files;
all were unchanged. `git diff --check` passed. The 89-root import analysis
identified these identical device groups:

- Darwin: `_mojolearn_native`, `_mojolearn_native_m1`,
  `_mojolearn_native_cluster_m1`.
- Linux: `_mojolearn_native`, `_mojolearn_native_cluster`.

No host group was reached by this device-pool edit. Detailed roots, flags,
consumer inventory and comparisons are in
`~/mojolearn-evidence/python-mojo-migration-20261110/resident-pool-repair/source-audit.json`.
No compilation, runtime test, benchmark, identity run or box job occurred in
this lane. Native concurrency fixtures are owned by the independent test lane;
their execution is owed to the orchestrator. No speed or bits claim is made,
and no A/B define was added.

Integration dependencies: the separate `xd_ctx` and pinned-stage lease repairs
belong to the PCA/TSVD owner. Root owns `core/device_pool` repair `c07626c77d`;
preserve its quoted `MojolearnXDecompCls2ScanV2` and
`MojolearnXDecompOrthWSV2` caller names when integrating these edits. Those are
different pools from this resident handle owner. The fast-only `GRP_STAGE`
scratch in `resident.mojo` is unchanged and has its own unguarded shared state;
it is outside this identical/PCA repair and the stopped dedicated fast work.
This batch does not certify every decomposition operation as concurrency-safe.
