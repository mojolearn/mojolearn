# NMF native requests, results and private output ownership

Base: frozen integration `0eb10b0aeb`; adoption merge `e5bf08a626` matches
that tree and is not a selective product commit. Dependencies for this batch:

- `9d39093e14`: additive NMF native API/provider over the existing kernels.
- `7a4a338545`: compiler repair qualifying generic struct parameters with
  `Self`; also repairs the TSVD provider. No numerical body changes.
- `69095a3335`: existing PCA/TSVD factory host-export inventory correction.
- Shared scalar admission `0c24c4ca90` and safe diagnostic representation
  `26dd5ed3b4` from the finite lane. This lane adopted those as `dcaf70a5f0`
  and `9eba936d3f`; integrate their originals once, not these duplicate picks.
- The NMF boundary follow-up adds provider exports, native scratch, public
  admission/packing, import connection and its matching export inventory.

NMF's native driver already implements initialization, NNDSVD variants,
random generation, coordinate descent, multiplicative updates, stopping,
negative/finite tests, reconstruction error and inverse projection. This
migration calls that same `nmf_run` / `nmf_inverse_run` through a typed fixed
native descriptor. The GPU provider still specializes the same
`DevExec/DevRDev/True/DevDecomp`; the CPU provider still specializes
`HostExec/NoRDev/False/HostDecomp`. No algorithm, tuned kernel, seed policy,
initialization branch, numeric default or transfer count is replaced.

`NMFFitRequest` carries addresses, dimensions, initialization/solver enums,
existing `NmfArgs`, and finite-test state. `NMFFitResult` carries iteration
count, binary64 reconstruction error and finite status. `NMFInverseRequest`
carries the native matrix descriptors. The five descriptor operations are
mode, vendor, capabilities, fit and inverse. Metadata is cached once per
connected provider; execution copies its driver and pins its compiled image
before releasing the GIL. The provider preserves the previous pointer,
dimension, solver-option and output-pointer refusal order.

`NMFOptions` and `NMFRegularization` are typed Mojo state. Regularization's
four binary64 products keep their former order/grouping. Shapes remain native
rank-two lists throughout output allocation, including a zero second dimension.
An admitted X retains native dimensions/address; custom W/H and fitted H use
the same existing admission, with their owner pins held through execution.

Public W/H outputs still use the established Mojolearn Array allocator and
packing. Private fit-only W and transform-only H now use `NMFScratch` Mojo
owners. Their shape/size validation reuses `NativeArrayLayout`; storage is
uninitialized because the existing native driver overwrites every output word
before successful return. Refusals discard it without publication. Every old
output copy still runs; only unnecessary Python Array construction and the
unused private zero fill are removed. There is no extra serial data loop.

Public behavior retains these stages:

- `numeric_mode_` publishes before X admission; fit outputs publish in their
  existing order only after success. A failed initial fit publishes no learned matrix,
  iteration count or reconstruction error.
- Beta and solver validation precede initialization. Custom initialization
  derives count from H, ignores requested `n_components`, validates W/H shapes
  before the seed, and keeps the same input copies and ownership.
- Fit provider refusal precedes regularization conversion/allocation. Maximum
  iteration, shuffle and tolerance conversions keep their prior order.
- Transform uses the fitted H with updates disabled and does not publish its
  scratch H or change fitted iteration/error state. Rebound and writable public
  components are admitted again and remain observable.
- Inverse provider refusal precedes input/component admission; inner dimension
  refusal precedes output allocation and the same native matrix product.

Exact scalar admission precedes comparison/conversion hooks. Common string
options compare native strings; admitted native NumPy scalar comparisons use
direct CPython C slots, not `operator.eq`. Beta values compare against 0/1/2
before binary64 conversion, so a wider scalar adjacent to an accepted value
cannot round into admission. Existing exact native numeric and zero-dimensional
Unicode/bytes conversion forms are retained. Seeds keep the builtin-int-only
counter-based modulo-2^32 policy. Invalid ndarray-valued options use the shared
safe diagnostic text and never call NumPy print formatters.

Removed: temporary beta-name dictionaries, option membership tuples,
regularization Python lists, fit/inverse argument lists, compiled-callable
lookup/dispatch for NMF, driver result tuples, the private `(W,H,it,error)`
tuple, Python shape readback between compiled calls, and private output array
packing. Legacy exported fit/inverse functions remain in the supported host
surface for direct callers. Other decomposition families' existing function
maps remain; NMF's two entries were removed from those maps.

Boundary staging remains explicit: public lifecycle, field dictionaries,
mode/scalar/native-array admission, Python kwargs enumeration, storage/image
pins and output/public-state packing. This is not a claim that the whole family
or package has no Python imports. NMF's three native modules plus their concrete
host/device executor roots have a 187-file transitive native closure with zero
Python/binding imports. Eight numerical/executor files are byte-identical to
`0eb10b0aeb`.

Compile scope: GPU `x_decomp` (`_mojolearn_x_decomp`), host `x_decomp`
(`_mojolearn_x_decomp_host`), and host `runtime_decomp`
(`_mojolearn_runtime_decomp_host`). The Python factory inventories now name the
new exported entries; this is source coherence, not tested load evidence.

Authored public boundary coverage: 13 definitions / 59 expanded cases, covering
all initialization names and both solvers, legacy-call/factory poisoning,
scalar callbacks, custom and ignored W/H, zero-dimensional text conversions,
publication/refusal order, fixed-H state preservation and writable component
rebinding. Native contract/correctness/test-migration fixtures belong to the
native-test lane. AST parsing and source/diff checks only; no compilation, tests,
benchmarks, identity or boxes were run in this lane. The orchestrator owns the
combined build and all runtime evidence. No A/B define or intended numerical
bit change; no speed claim.

Evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/nmf-native-owner/`.
