# Native debugging interruptions and authorized fallbacks

Andrew authorized narrow Python fallbacks on 2026-10-25 where a working native
path cannot be completed. Each actual fallback must document its operation,
Python work, reason and evidence inline, preserve behavior, and be explicitly
identified in release qualification. This is not blanket qualification of the
remaining source inventory. The lookup repair needs no Python fallback. The
meta-estimator paths retain the explicit metadata exception below.

## Retained Python metadata exception

`array-length-metadata` permits the existing `mojolearn._array.Array.__len__`
callback for fit/fit_transform/predict/predict_proba/transform/score on the eight
connected Voting, Stacking, MultiOutput, OneVsRest and Calibrated meta-estimators.
It reads dimensional metadata, including the existing zero-dimensional refusal;
it performs no model computation. This is retained implementation work under
Andrew's release exception after the screening interruption, not proof that
Mojo cannot implement a length query.

The callback definition documents this INLINE in `_array.py`; the exact caller
scope and connected callback code are in `_runtime_fallbacks.py`. Modified
callback bodies and other Python callees are not admitted by that exception.
`audit_compatible_call` retains all Python frames and reports the named fallback.
The strict `passed` field and `assert_native()` remain false/failing when the
callback executes. Compatibility qualification uses `assert_compatible()` and
is recorded separately in retained runtime-test JUnit properties.

## Reproduced Mojo issue

The independent standard-library-only lookup reproducer crashes on macOS arm64
with Mojo 1.0.0 (ed45d567) under CPython 3.13.15 and 3.14.6. The existing instance
field case fails; class, descriptor-refusal and missing-field controls pass.
The exact compiler/stdlib component is unconfirmed. The source, full commands,
result matrix and investigation interruptions were submitted to Modular:
https://github.com/modular/modular/issues/7399

`bindings/native_instance_item.mojo` documents the source failure and repair
inline. Its documented native dictionary API repair passes the original public
failures and affected identity/lifetime tests without a Python callback.
The source bytes of that compiled helper are retained unchanged; the upstream
URL is maintained in this report and the linked `MODULAR_REPORT.md` so updating
report status does not cause an unnecessary native rebuild.

## Codex screening interruptions

At least four delegated turns were halted by an automated message saying the
content was flagged for possible cybersecurity risk and directing the user to
Daybreak access. The retained team statuses and handoffs identify these scopes:

| Lane | Known interrupted turns | Work affected |
| --- | ---: | --- |
| native_execution | 2 | Native parameter/instance lookup reduction |
| max_adapter | 1 | Native lookup ownership/API review |
| meta_runtime | 1 | Public meta-estimator lifecycle and class-label repair |

The precise classifier trigger is unavailable. These are Codex/OpenAI screening
events, separate from library diagnoses. An aborted tool dispatch with no
screening message is not counted as a screening event.

The meta-estimator failure was traced to our retained Python lifecycle wrappers
and incorrect handling of `ClassLabels` as a buffer. It is not established as a
Mojo defect. The root agent completed the ordinary native API implementation and
source tests after the delegated interruption; compilation and public runtime
results must establish whether that implementation works.

## Evidence and release status

Full original failure evidence remains under
`~/mojolearn-evidence/native-params-crash-20261025/` and repaired-runtime evidence
under `~/mojolearn-evidence/native-instance-repair-20261025/`.
The latter's `retained-lookup-repair-1/receipt.json` records a downloaded/hash-
verified R2 archive including the repaired binary and full test results.
Meta repair source results are under `~/mojolearn-evidence/meta-runtime-20261025/`.

The original 167-case suite first completed with 147 passed and 20 failed;
that failure evidence remains retained. At source `70447150b`, the complete
original selection passes **167/167**. The separate affected composition suite,
including ten finite-check connection cases, passes **68/68**. Both runs use the
same verified 79-group activation, entirely reused from valid cached builds.
Reports are `existing-compatible-freeze/run/summary.json` and
`compatibility-targeted/run/summary.json` under the repaired-runtime evidence
directory above. The earlier 37 extra attribute cases and separate Torch
parameter case retain their passing evidence.

The eight meta-estimator fit/predict/score probes retain 24 operation reports:
12 are strictly native and 12 use the named metadata callback. All 24 satisfy
the scoped compatibility check. The original callback frames, counts and strict
native failures remain in JUnit properties and `compatibility-details.json`;
the passing test totals do not mean every operation is Python-free.

This is source-development runtime qualification, not installed-wheel or full
release qualification. Adapter, cross-vendor identity, source inventory and
other release requirements remain separate and must retain their own evidence.
