# Native score-label keys and containers

Frozen lane work, 2026-11-10. Depends on native label driver0268d8b446/521abd4e6c
and native COUNT plan9fe781d99d. No compile, test, timing or identity run by this lane.

`bindings/classical_labels.mojo` now keeps `ScoreLabels` as native key state with
boundary-owned input/output pins. Its integer/bool fast admission still uses the
existing `prepare_targets` backend; the distinct integer words stay native until
public class PACK. The fallback keeps the original shape/type refusal sequence and
normalizes only already-admitted exact Python/NumPy integers, bools and strings.

`core/labels_score_keys.mojo` contains the Python-free request/state operations:
parallel open-addressed native hash insertion and lookup, deterministic first/last
representatives, parallel class compaction and merge-path sorting, joined views,
and int32 inverse-code writes. Integer keys are exact Int64 or canonical signed
hexadecimal magnitudes outside Int64; text keys use borrowed UTF-8 or owned UCS4.
The host's existing work/worker policy controls parallelism, with comparison levels
included in sorting work. The bounded prefix loop traverses worker summaries.

`bindings/labels_score_boundary.mojo` owns scalar UNPACK, immutable storage pins and
public class PACK. Three CPython APIs are local to this boundary's shared slot;
`bindings/cpython_table.mojo` is unchanged. Exact type admission precedes C coercion,
so these conversions cannot call a user's scalar hooks. Big integers use base16,
which has no Python decimal-digit limit. Only UnicodeEncodeError takes the surrogate
fallback; unrelated pending exceptions propagate. Native planning releases the GIL.

Removed: score-path Python class intermediate lists, Python set construction/union,
Python sorted calls, per-class dictionaries and per-row C dictionary probes; obsolete
`_label_set`, exact-string and integer-list helper imports. Numeric gather and COUNT
remain their existing kernels. Returned public arrays/class lists remain Python API
objects and their owners remain pinned through every synchronous native join.

Semantics retained: bool/integer equality, full uint64 and arbitrary-size Python ints,
Unicode code-point order including non-BMP and lone surrogates, unknown class -1,
last-assignment mapping for duplicate rebound fitted classes, error ordering, and
float/NaN/mixed-label refusals. Ordinary label encoding's supported floats are not
changed; this batch only touches classification score/union admission.

No tuned numerical kernel was edited. `unchanged-kernels.json` records the preserved
source bytes; this is source evidence, not a speed or bitwise-identity result. The
metadata representation and scheduling changed; end-to-end performance remains
unmeasured. A/B define: none (typed plumbing migration, no tuning default).

Coverage queued:28 independent native key/index fixtures (`check-native-label-score-keys`)
cover empty/small/large integer domains through262271rows, duplicate representatives,
UTF8/UCS4 equality/hash/order, combining/non-BMP/surrogate text, arbitrary-size signed
integers, unions, inverse codes and output guards. Six Python-boundary definitions /
eight cases cover NumPy scalar normalization/public exact types, huge integers beyond
the decimal digit limit, rebound duplicate classes, unchanged float refusals and
callback rejection. These Python seams cannot be replaced by native-only fixtures.
No tests were executed. Independent additional native stress fixtures are owned by
the native-tests lane and are separate commits.

Static source reachability:89 distinct roots,9 affected physical groups on each of
Darwin and Linux (18 platform/group entries),0 walker errors. The new native core's
five-file source closure has no Python imports. Build these through the integrator's
combined freeze (other lanes can broaden it):

- `host:_mojolearn_native_byte_trainer_host`
- `host:_mojolearn_native_classical_decomp_host`
- `host:_mojolearn_native_classical_family_host`
- `host:_mojolearn_native_classical_host`
- `host:_mojolearn_native_classical_prep_host`
- `host:_mojolearn_native_classical_trees_host`
- `host:_mojolearn_native_orchestration_host`
- `host:_mojolearn_native_parallel_host`
- `host:_mojolearn_native_xseq_host`

Raw evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/label-score-keys/`
(`closure.json`, `closure-scan.log`, `unchanged-kernels.json`). Integration compilation,
native execution, installed public boundary tests, performance and release identity
remain distinct pending evidence. No new source-compliance guard was introduced.
