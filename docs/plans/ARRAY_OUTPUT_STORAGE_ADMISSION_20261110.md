# Shared Array output storage admission

Producer baseline: lane adoption `bf5279c0a8` of root `b7f177d1db`. Root's
frozen build was not changed. No lane compilation, product/test execution,
timing, identity or box work occurred.

The independent NMF review found that `classical_wrap_layout` allocated an
Array with `PyType_GenericAlloc` and then called its mutable attribute setter.
Replacing `Array.__setattr__` could therefore execute Python while packing
outputs. The two common Array-address branches and NMF's local address helper
also used ordinary `_addr` lookup, allowing replacement descriptor/getter
hooks. These are boundary defects; the numerical migration did not create them.

`native_array_storage_admission` now owns the existing from-buffer class/storage
checks. Before allocation it admits exact native class metadata, passive
subclass inheritance, native allocation/access/lifetime protocols and the full
fourteen-member Array layout. Every registry key is an exact string, every
descriptor is an exact native member descriptor owned by the connected base,
and its name and live class identity must match. Removing a key or replacing a
descriptor refuses before descriptor invocation. Class and dictionary mutation
stamps jointly cache that admission; the previous class-only cache could miss
registry changes. Where watchers are unavailable, the full admission remains.

The shared output packer keeps its store-size check first, then admits storage
before allocating the owner. Its fourteen existing native C-API field writes
retain their order, values and ownership. The GIL stays held through admission,
allocation and publication, so a Python thread cannot replace a checked setter
between those stages. Common and NMF Array address reads use
`PyObject_GenericGetAttr` after the same admission and require exact integer
address metadata before conversion. Array owners remain pinned throughout;
no C-API table expansion or toolchain workaround was needed.

Both classical and Sequential import-time contexts reuse the complete descriptor
inventory already captured by `_ARRAY_CONTEXT`. Their `array_fields` getters,
order and public registration remain unchanged. The existing ndarray-view
consumer still reads its same seven fields. From-buffer allocation, retained
setters, exporter pins, shape/stride handling, readonly behavior and passive
subclass support are unchanged.

Removed: duplicate from-buffer admission implementation, its class-only cache
kind, two redundant seven-slot registry comprehensions/dictionaries, admission
key-list materializations, and the AST assertion requiring those comprehensions.
Existing native view/lifetime tests remain. No numerical source, kernel,
dispatch/default, data allocation, data copy or algorithm was changed. No bit
change is intended; no identity or performance result is claimed. No A/B define
or fallback was introduced.

Four new Python boundary definitions expand to eight independent cases:
replaced output setter/getter/address descriptor; changed descriptor registry,
missing field and colliding foreign key; passive subclass storage sharing;
and normal fitted-output metadata plus memoryview ownership after model deletion.
These exercise public operations, not source patterns. They are authored and
unrun; root's existing parallel runner owns execution. Numerical NMF coverage
remains in its native fixtures.

Static source analysis completed with exit 0. It inspected 89 product roots;
the helper closure contains 13 files and does not reach any of its three
importers. Common admission affects 32 platform records across 16 physical
groups. Exact group members, flags and changed paths are in `source-audit.json`.
This is compile scope, not a compilation result. No new native registration is
required. The common Array and Python boundary remain Python-aware; this repair
does not claim every public Array operation is callback-free.

Complete baseline/repaired sources, patch, static source audit and push output:
`~/mojolearn-evidence/python-mojo-migration-20261110/array-output-admission/`.
RUN OWED: bundled affected product compilation and the eight boundary cases;
no lane job was queued.
