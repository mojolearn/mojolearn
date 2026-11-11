# TruncatedSVD source review and scalar repair

Reviewed root freeze `ab6d28e96dec6b99c53543494ee5a6c7e3df860c`
(`c9d74974aa` plus `ab6d28e96d`) against `e53128fe7f`.
The original reviewed sources and complete boundary diff are retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/tsvd-independent-review/`.
This lane ran no compilation, runtime tests, timing, identity or box jobs.

Concrete finding: the new TSVD `_scalar` admitted native NumPy string scalar
classes but rejected exact zero-dimensional Unicode/bytes ndarrays. The old
boundary passed these values to builtin integer/float conversion; the new
numeric-array-only admission narrowed that behavior. Constructors retain raw
parameter values and lifecycle admission does not exclude these scalar forms.

The isolated repair moves that admission to
`bindings/decomp_scalar_admission.mojo::decomp_scalar_admit`. Exact native U/S
zero-dimensional storage is admitted by ndarray identity, rank/size and exact
native dtype class before conversion. Object dtypes and ndarray subclasses
remain refused before element or conversion hooks. Each caller retains its
existing conversion and error order; no parser or algorithm was replaced.
The TSVD-local duplicate helper is removed. The NMF lane can reuse this narrow
boundary helper instead of introducing another admission implementation.

Five Python boundary definitions / 26 cases cover fit scalars, rebound query
dimensions, builtin parse-error text and partial publication, ndarray subclass
hook refusal, and invalid-algorithm native repr. These cases inspect the Python
API seam; numerical oracles remain the native test lane's responsibility.
The Python source parses and `git diff --check` passes; no case was executed.

PCA was checked separately: its fit scalar snapshot parser is byte-identical
to `e53128fe7f` and already handles native U/S text. Its query dimensions use
an integer-index contract. They are different contracts, so the repair does
not weaken PCA query admission or replace its snapshot parser.

No further concrete migration defect was found in the reviewed descriptor
field/operation layouts, host `_bind` removal, route/capability selection,
publication order or resident ownership. Native requests retain exact/fused,
randomized, TSQR and resident-fold implementations. Host estimator selection
preserves the former late mode refusal; randomized host fits retain their
previous decomposition-provider route. Query input/output/component owners and
resident lease owners remain pinned through GIL-free calls; native close takes
the lease once, with the provider image still retained. The existing resident
store provides locked metadata and owned device-buffer references. Non-weak
models retain the existing retired-lease per-call behavior.

Seven of eight sampled numerical/route files are byte-identical between the
two root freezes. The eighth, `decomposition/estimator.mojo`, differs only in
root's `c07626c77d` pool-name change from `MojolearnEstimatorsPcaFastXV1` to
`MojolearnEstimatorsPcaFastXV2`; the complete diff is preserved separately.
The earlier randomized-context/pool/staging repairs are separate integration
dependencies, not numerical changes made by this scalar repair.

The repair affects the host `runtime_decomp` owner. Import-time provider
connection, public lifecycle/publication, scalar and buffer admission and
storage/image pins still use the boundary's Python support. This is not a
package-wide Python-free, runtime-correctness or performance claim. Root owns
the next bundled affected compilation and installed boundary execution.
