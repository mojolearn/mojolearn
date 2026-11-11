# Independent NMF migration review

Reviewed product snapshot: root `2963338519`, including boundary `f22168ca30`,
native API/provider `d0acf6f7a9` and qualification repair `1aae5030b1`.
Comparison baseline: `0eb10b0aeb`. This was a source review; no lane compilation,
test execution, timing, identity or box work occurred.

The five descriptor operations and their fixed field layouts agree between the
driver and dispatcher. Fit carries seven addresses, ten integer words and six
Float64 words; inverse carries three addresses and three integer words. Provider
pointer, dimension, options and output checks keep the previous exported
provider's order. Host/device registrations retain their original Exec/RDev/
resident/DecompDev specializations and call the same `nmf_run` and
`nmf_inverse_run`. All 217 tracked paths under `x_decomp` and `decomposition`
are unchanged against the comparison baseline.

Regularization retains the Float64 operand grouping. The initialization routes,
custom W/H shape check before seed conversion, ignored noncustom W/H arguments,
mode/availability check before regularization, max_iter/shuffle/tol conversion
order, finite refusal before publication, component/iteration/feature/error
publication order and fixed-H transform behavior match in the reviewed paths.
Private fit-only W and transform-only H storage is fully overwritten by the
same successful kernel path; failures publish neither its bytes nor new fitted
outputs. No extra copy or replacement numerical algorithm was identified.

Compiled provider pins and copied drivers survive GIL release; the provider
cache is accessed under the GIL and no borrowed cache element escapes the
release window. Native fit result spans, finite status and W/H scratch owners
survive the synchronous call. Public X/custom W/H/fitted H owners remain pinned
at their existing stages. This does not establish GPU runtime concurrency;
root's separate MAX/Metal concurrent-context failure remains parked.

The native closures contain zero Python imports: API 26 source files, scratch
state three files, provider 40 files. Public admission, option conversion,
ownership and publication still use Python boundary objects. Primitive error
messages remain at their original stages; invalid ndarray options use the
already-repaired fixed representation rather than NumPy's Python formatter.
Callback/protocol scalar objects are refused before their conversion slots.

Four inherited transitive callback findings were identified:

1. `decomp_public_begin` converted fitting depth before the existing exact-int
   guard. Repair `f3079b3612` factors and reuses that guard, preserving the
   surrounding callback/overflow refusal order. Eight NMF/plain-subclass trap
   cases are authored, unrun.
2. Raw model dictionaries were probed before key admission. A stored key with a
   colliding hash could run Python equality. Repair `e7fddf8669` reuses native
   key iteration and dictionary mutation stamps before lifecycle probes,
   cleanup and completion, and after NMF's GIL release before publication.
   Four key-collision trap cases are authored, unrun.
3. Both public X validation and the subsequent sparse probe read
   `cls.__module__` before source admission, permitting a Python metaclass
   callback. Repair `35a2cb149a` reads the native type getset after existing
   watched class-key admission. It preserves ordinary native class metadata
   and does not reject arbitrary metaclasses just to make the probe safe.
   Four nested/non-nested trap cases are authored, unrun. Its 15-file helper
   closure does not reach either importing module, so no import cycle was added.
4. Common `classical_wrap_layout` output construction uses `py_setattr` after
   `PyType_GenericAlloc`, and direct Array address reads use `_addr` lookup.
   A replaced Array setter/address descriptor can therefore enter Python.
   This predates the NMF migration and is not repaired by the three commits
   above. Root assigned the common output-owner repair to this lane for the
   following snapshot; the active product freeze remains immutable.

The three completed callback repairs add three definitions / 16 cases. Static
closure analysis inspected 89 product roots and identifies 32 platform records
across 16 physical groups because the real defects are in common admission
code. Exact groups, flags, roots and touched files are recorded in
`source-audit.json`; this is rebuild scope, not build evidence.

No broader callback-free or performance claim follows from this review. NMF's
public seams and the new trap cases still require installed runtime execution.
The raw legacy NMF exports remain registered for compatibility/testing, while
the live NMF operation no longer dispatches through them; removal requires the
remaining raw-interface consumer audit.

Full baseline/frozen/repaired sources, complete patches, source-audit script,
logs, JSON and push evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/nmf-independent-review/`.
