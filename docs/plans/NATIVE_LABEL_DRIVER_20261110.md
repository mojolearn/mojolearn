# Shared native label driver — 2026-11-10

The numerical label path used by Logistic and other classical estimators now
passes typed Mojo requests and results. The old v1 descriptor unpacked native
Lists and rebuilt Python arguments before calling the compiled Python wrappers;
those envelopes and calls are removed. The v1 operation IDs, metadata slots,
three-word reduction result and scalar refusal text remain. New callers require
`labels_native_api_v2` / `mojolearn.labels.v2`: an old v1 image may still construct
Python values and cannot safely be called with the GIL released. The existing
v1 factory/name remains a compatibility export over the same native entry for
its GIL-held consumers; there is no duplicate numerical implementation.

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

Root owns compilation and runtime. `checks/native_tests/labels_driver_suite.mojo`
contains 100 parallel native fixtures: all six numeric label dtypes, independent
class/inverse-code expectations, label/weight refusal order, reduction signed
zero/NaN/triple words, gather refusal before writes, first-max argmax, known ratio
and exact-sum results, both widened and already-wide uncapped fallback, raw ABI
refusals, and 32 independent concurrent encode/reduce units. The two uncapped
fixtures initialize and verify their own disjoint slices with the existing host
pool; source/destination owners remain live through every join.

Native command: `bash tools/compile_slot.sh pixi run check-native-label-driver`.
The suite exposes importable `run_checks` for the existing combined native CPU
runner. No native fixture has been compiled or executed by this lane.

`test_labels_provider_version.py` adds two thin installed-provider seams in fresh
processes: the v2 factory replaced by the extension's real compiled v1 factory
must fail capsule-version admission, and a missing v2 factory must fail entry
admission. Both also inspect the retained v1 and new v2 capsule identities. They
perform no numerical kernel call on success. CPU-only installs have no GPU base
provider and explicitly skip these two seams; that is pending provider coverage,
not a pass. Existing public label/weighted-fit/score/lifetime tests remain intact.

Static closure analysis uses the existing `compile_closures`/binary-cache source
walker, with ten independent workers: 89 distinct roots scanned, 13 affected,
zero source-walk errors. Both Darwin and Linux need identical
`_mojolearn_dev_core` and these host groups:

- `_mojolearn_dev_core_host`
- `_mojolearn_dev_runtime_byte_trainer_host`
- `_mojolearn_dev_runtime_classical_host`
- `_mojolearn_dev_runtime_decomp_host`
- `_mojolearn_dev_runtime_family_host`
- `_mojolearn_dev_runtime_linear_host`
- `_mojolearn_dev_runtime_manifold_host`
- `_mojolearn_dev_runtime_model_selection_host`
- `_mojolearn_dev_runtime_parallel_host`
- `_mojolearn_dev_runtime_prep_host`
- `_mojolearn_dev_runtime_trees_host`
- `_mojolearn_dev_runtime_xseq_host`

The source maps are in `closure.json`; root resolves the combined freeze through
the normal controller because other lanes may add edges. Source reachability is
not compilation or runtime evidence. GPU/host public label, weighted-fit, score,
provider and ownership seams remain owed against the rebuilt installed code.

Complete before-source snapshots, hashes and implementation diff are retained
in `~/mojolearn-evidence/python-mojo-migration-20261110/native-label-driver/`.
No lane compiler, test process or box was used.
