# Logistic fixed request spans — 2026-11-10

The six QN driver methods now construct their fixed ABI request envelopes in
local `InlineArray` storage. They call the new small
`core/native_descriptor_fixed_call.mojo` wrapper, which delegates to unchanged
`native_descriptor_call_spans`. The dynamic helper remains available and
unchanged for variable-length consumers. Driver signatures and ABI slots do
not change; Logistic config/state work can continue against the same methods.

| Driver method | Address slots | Integer slots | Float64 slots |
| --- | ---: | ---: | ---: |
| Dense fit | 6 | 9 | 5 |
| Dense query | 4 | 5 | 0 |
| Dense link | 2 | 2 | 0 |
| Dense logarithm | 2 | 1 | 0 |
| Fold fit | 5 | 12 | 5 |
| Fold query | 3 | 8 | 0 |

This removes the native heap-backed request Lists from these methods. The
result/error owner remains the established one-element `List[NativeCallOutput]`:
the prior supported-toolchain reproduction requires stable heap storage for its
writable field addresses. Result reset, error clearing, callback accumulation,
status handling and refusal propagation are the same existing implementation.
There is no global mutable scratch, lock, provider cache, new loader or Python
call. Empty metadata spans use address zero and count zero. `finally` keeps
all borrowed request arrays and output owners live through synchronous success
or refusal. Nested and concurrent calls each have independent storage.

Supported storage evidence: the installed SDK metadata reports Mojo 1.0.0;
`bindings/native_scalar_decode.mojo:80` already uses `InlineArray` and raw
addresses for call-local C ABI storage. The official [1.0 release notes](https://github.com/modular/modular/blob/main/Mojo/docs/site/releases/v1.0.0.md)
describe `InlineArray` as the compatibility alias for `Array` and fixed array
literal construction. Current [official Array source](https://github.com/modular/modular/blob/main/Mojo/stdlib/std/collections/array.mojo)
allows length zero. Root compilation must still qualify this implementation on
the pinned SDK; source documentation is not compile evidence.

## Checks queued, not run

The existing importable native contract suite now contains 118 fixtures:

- The original 35 pointer and malformed metadata-span cases remain.
- Eight dense and five resident-fold request fixtures check independent ABI
  slot sentinels, shape counts and iteration output through fake native entries.
- Six helper fixtures cover returned values, a successful unwritten zero result,
  complete multi-fragment UTF-8 refusal text despite a written result, empty
  spans, nested success/refusal, recovery after refusal and compatibility with
  the unchanged dynamic helper.
- Sixty-four independent concurrent units use distinct request sentinels and
  nested calls. The existing `host_parallelize` pool schedules all 118 units;
  every fixture owns its metadata and output storage. No global test state.

Root command: `bash tools/compile_slot.sh pixi run check-native-logistic-contracts`.
The suite is also importable as `checks.native_tests.logistic_contracts_suite.run_checks`
for the existing combined CPU executable. Python boundary tests remain intact.

Static closure scan used the existing `compile_closures.closure_of` and
development group plan, with ten independent source-walk workers. It scanned
89 distinct binding roots, found six affected roots and reported zero errors.
The complete affected development groups for this lane's source are:

| Set | Group | Platforms |
| --- | --- | --- |
| identical | `_mojolearn_dev_core` | Darwin, Linux |
| identical | `_mojolearn_dev_estimators_m1` | Darwin |
| identical | `_mojolearn_dev_estimators` | Linux |
| host | `_mojolearn_dev_estimators_host` | Darwin, Linux |
| host | `_mojolearn_dev_runtime_family_host` | Darwin, Linux |
| host | `_mojolearn_dev_runtime_linear_host` | Darwin, Linux |
| host | `_mojolearn_dev_runtime_trees_host` | Darwin, Linux |

`closure.json` records every affected root's full transitive source/hash map.
Root should resolve its combined freeze with the normal controller, since other
lanes may add import edges or change the group manifest. This source scan did
not compile or execute any product code.

No numerical kernel, formula, reduction, allocation of tensor data, synchronization
or device route changes. No intended bit changes. No A/B define or box queued.
Native compile/run, product compilation and installed boundary execution remain
owed; no performance improvement is claimed from source allocation removal.

Complete source diff, whitespace-check result, SDK source snapshot and static
compile-closure evidence are retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/logistic-fixed-spans/`.
The dynamic descriptor helper was deliberately not edited, so its unrelated
consumers do not acquire a new source dependency.
