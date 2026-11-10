# Shared native label driver — 2026-11-10

The numerical label path used by Logistic and other classical estimators now
passes typed Mojo requests and results. The old v1 descriptor unpacked native
Lists and rebuilt Python arguments before calling the compiled Python wrappers;
those envelopes and calls are removed. The public factory, v1 operation IDs,
metadata slots, three-word reduction result and scalar refusal text remain.

`core/labels_native_api.mojo` contains the request values, `LabelsNativeDriver`,
native span dispatch and reduction wire packing. The driver uses the previously
introduced fixed-span helper. Reduction returns the existing typed `HpStat`;
its writable three-word output retains its existing native heap owner.

`core/labels_prepare.mojo` contains the existing ordered preparation: capped
encoding (cap 4096), widening, uncapped unique fallback, then finite/negative/
all-zero weight checks. Typed host and device implementations invoke the same
existing kernels. `core/label_device_steps.mojo` retains the current hotpath
mode/experiment switches, covered dimensions/dtypes, device reductions/cast/
gather and existing host refusal replay. No kernel or numerical algorithm changed.

`bindings/labels_descriptor_boundary.mojo` owns provider admission and Python
image pins. Its versioned provider record stores a native optional driver and
the admitted image owner. The GIL is released around the complete numerical
operation; the pin survives success and refusal. Host calls invoke typed host
kernels directly. No numerical Python objects or callbacks occur inside the
native dispatch. Existing legacy exported wrappers remain UNPACK/RUN/PACK over
the same typed implementations because actual callers still use those exports.

Caller signatures for `classical_accuracy`, `labels_score_from_codes`, encoding,
argmax and gather remain. `labels_reduce_stat` now returns `HpStat`; its direct
consumers in `classical_labels`, `prep_labels_native` and `neighbors_knn_native`
read native integer/float fields. The old private reduction adapter in
`classical_labels` returns Float64. Label range checks retain their short-circuit
order, and fitted classes remain admitted from their live writable public owners.

Removed: `bindings/labels_native_abi.mojo`, its Python reduction encode/decode,
the label-only `_labels_device_call`/`kit_call` path, `_lw` reboxing and
`labels_native_operation`, copied metadata Lists in the descriptor, duplicate
prepare-target control flow, obsolete hotpath Python reduction peeking and its
dispatch body, and imports made unused by those moves. `kit_call` itself and
other consumers remain; the dynamic descriptor helper is unchanged.

Remaining work is explicit: class-label Python snapshots, string/object
publication, score set/union/program containers and other classical boundary
algorithms are not removed by this numerical-driver batch. Their actual API and
lifetime tests remain. No speedup, runtime success or identity result is claimed.
No intended arithmetic/bit changes; no A/B define or box job.

Root owns compilation and runtime. Native host fixtures, v1 protocol/refusal
checks, GPU/host public label/weighted-fit/score seams and affected development
group compilation are required. Build closure and fixture inventory will be
recorded with the next test batch before the combined freeze.

Complete before-source snapshots, hashes and implementation diff are retained
in `~/mojolearn-evidence/python-mojo-migration-20261110/native-label-driver/`.
No lane compiler, test process or box was used.
