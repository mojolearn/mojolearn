# Kernel-method provider migration — source candidate

This batch routes KernelRidge, Nystroem and RBFSampler computation through the
same `NeuralDescriptorProvider`/`NativeDescriptorCall` used by neural execution.
The historical helper name does not create a second classical provider.
Algorithm-specific tables expose the existing typed kernel-method owners;
no kernels, arithmetic order, precision or pointer-route switches change.

`MOJOLEARN_CLASSICAL_NATIVE_DISPATCH=1` selects the candidate on an identical GPU
binding. It refuses an unavailable table and non-identical builds. Default
behavior remains unchanged while root compiles and runs the focused public probe.
There is no new Apple fast work or cross-vendor validation campaign in this lane.

Python public fit/predict/transform frontdoors now make one native call. Their
binding images connect at import; the common estimator binding admission reads
the current mode-set registry at each operation, so replaced modules are checked
without invoking `_bind`, `_backend.binding`, or `_ModeSet.__getattr__` in product
execution. The existing shared provider checks factory/module identity before
reuse. `score` and parallel wrappers remain separately listed work.

The native table includes ordinary/pointer fits and predictions, Nystroem's
joint fit/transform where the old build exposes it, and exact gamma-scale limbs.
KRR resident handles use the generic native resource owner. A refit, array or
provider replacement closes the old owner through its original table. A holder
retains its source arrays, provider and compiled image; final destruction invokes
the generic native destructor, not a per-estimator Python callback. Legacy
holders and legacy binding routes remain supported.

Candidate compound `fit_transform` reuses its input and provider for the second
native operation. This does **not** complete metadata migration in shared family
input validation: `family_view`/`family_cast_view` still repeat some metadata
reads, and a following shared snapshot repair is tracked in the library
inventory. Sparse, pandas, Torch and foreign-buffer admission also need their
whole-operation audits; this batch does not claim those implementations are
Python-free merely because a provider table exists.

`public_runtime_probe.py` is authored for root's rebuilt Mac candidate. It compares
model/output bytes against the retained packed entry route, exercises weighted
and precomputed KRR, repeated resident predict/refit/array replacement, Nystroem,
RBF including gamma scale, poisoned old loader/computation entries, replaced
factories and native resource finalization. It records audits and exact coverage.
The lane has not run that probe, compiled code, measured performance or qualified
another device. Source tests and logs live under
`~/mojolearn-evidence/library-provider-completion-20261027/`.

The subsequent metadata tranche introduces `ArrayMetadata` in the common provider
boundary, retaining the input owner, native `CastIn`, shape/count and dtype/order.
The shared family f32 admission uses the existing cast implementation and those
snapshots; kernel X and fitted f32 consumers reuse them for validation, gamma,
resident preparation and copying. The authored public probe includes strided,
Fortran, f64, integer and bool input conversion. It has NOT been run by this lane.
Targets, sample weights, direct index/offset arrays, finite-only/f64 admission,
foreign input preparation and other family consumers still need further work;
this is not a claim of exactly one metadata read across every public call.
