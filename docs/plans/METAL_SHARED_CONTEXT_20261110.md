# Concurrent Metal context submission: upstream reproducer

The 2026-11-10 native ownership tests exposed a runtime abort when independent
host workers submit through one shared `DeviceContext`. The standalone
`checks/device_context_concurrent_repro.mojo` reproduces it using only the
standard library and MAX: no Mojolearn imports, Python objects, resident pools,
shared-slot code or host staging helper. Ten host workers own distinct input,
output and device buffers; all buffers remain live through synchronization.
The kernel is a parallel elementwise multiply. This is a correctness
reproducer, not a timing or identity result.

Observed environment: Apple arm64, macOS 26.5.2, Mojo 1.0.0 release,
MAX 26.5.0. The standalone source compiled successfully and exited with signal
6. The first exception was `NSInvalidArgumentException` from
`NSCountedSet addObject:`, reporting an attempt to insert nil, through Metal
command-buffer residency/submission frames. Initial test-bundle failures and
both unsuccessful reproducer compilation attempts are preserved separately.

Ask for Modular (draft, not sent): is concurrent host submission through the
same `DeviceContext` supported? If so, please fix the Metal submission failure
reproduced here. If it is unsupported, please document the ownership contract
and provide a supported concurrent submission API that retains persistent
contexts and parallel GPU execution. The
[DeviceContext API](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext/)
is the relevant public surface. This report does not establish NVIDIA/AMD
behavior or characterize every MAX version.

The affected concurrency work is parked for a supported native solution under
the owner's "Wait for Mojo and Modular" rule. No Python fallback, compiler
patch, reduced worker count or whole-fit serialization was added. The context
publication and pool-metadata ownership repairs remain useful independent
fixes; they do not cure the lower-level concurrent submission failure.
Product compilation and the rest of the Python-to-Mojo migration continue.

Native diagnostic results remain separate: 133 CPU and 26 GPU fixtures passed
as isolated suites. Twenty NearestCentroid plan fixtures failed due to missing
test-buffer lifetime pins; their repair is committed and awaits execution.
Eleven GPU NearestCentroid fixtures and five GPU concurrency phases failed or
aborted. There is no aggregate native-suite pass. The original bundled CPU
execution also crashed; isolated CPU suites narrowed its source to the
NearestCentroid fixture group, but the repaired aggregate run is still owed.

Evidence is retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/`:
`metal-shared-context-upstream-evidence.json`, the exact standalone source and
binary, full build/run logs, `metal-concurrency-environment.json`,
`native-wave8-diagnostic-results.json` and
`native-wave8-partial-accepted-results.json`. The tracked reproducer is
byte-identical to the source that produced the abort. Re-run only after a
relevant runtime/source change or a specific additional diagnostic question.

Removed in this report: nothing identified, because this records a runtime
limitation and adds its minimal reproducer; product algorithms are unchanged.
No A/B define, release, box allocation or upstream message was initiated.
