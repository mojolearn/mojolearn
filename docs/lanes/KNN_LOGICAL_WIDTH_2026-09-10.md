# kNN logical reduction width prerequisite

Source starts at2419895f; initial candidate d827ea24; corrected source5b552e26. Only the IDENTICAL kNN selector
calls the extracted shuffle primitive. Its native-width dispatch and integer
comparison order are unchanged. No floating reduction, numerical floor,
FAST/DETERMINISTIC path, or tree source is changed.

`neighbors/checks/lane_minimum.mojo` defines UInt64 minimum for compile-time
logical widths that are positive powers of two and divide the block size.
The shuffle implementation is valid only when the logical width divides a
known native width. The generic implementation uses shared memory otherwise,
including when physical subgroup width is compiler-selected. It requires all
block threads to participate and includes a final barrier before scratch reuse.
Integer minimum is associative, commutative and idempotent; the schedule may
change without changing composite-key ordering, ties, NaN payload ordering,
or selected distance bits. This argument does not permit reordering Float32
sums, and the helper deliberately offers no floating-sum overload.

The qualification driver checks every output lane for widths1,2,4,8,16,32,
64,128 in both admitted-native and forced-shared schedules. Two consecutive
calls use differing high-word patterns, low-word ties and high-bit values,
with a serial full-group UInt64 oracle. On Apple32, width64/128 exercises
shared-memory grouping across native SIMD groups; it does not validate an
AMD64 device or establish fourth-device identity.

Repository column audit at2419895f:

| Declared column | Lane declaration | Detection / existing kNN route |
|---|---:|---|
| Apple | fixed32 | default fallback; native shuffle |
| NVIDIA | fixed32 | detected; native shuffle |
| AMD/CDNA | fixed64 | all detected AMD maps here; native shuffle |
| AMD/RDNA | fixed32 | explicit column override only; native shuffle |
| Qualcomm | variable floor8 | explicit simulation override; shared selector tree |
| Intel | variable floor8 | explicit simulation override; shared selector tree |
| spec baseline | variable floor1 | explicit simulation override; shared selector tree |

These are source declarations, not evidence that a compiler/backend/device
is available. In particular, an AMD device detection flag does not by itself
establish a64-lane wavefront; RDNA admission still requires selecting its
separate column and executing the relevant device gates. Intel/Qualcomm floor
values cannot substitute for actual subgroup-width knowledge.

No performance claim is attached to the extraction. Existing opponent prices
are reused unchanged. A new device must pass native-width collectives, full
selector ordering (including ties and NaNs), exact distance seams, public
output hashes and its own numerical-floor gates before an identity claim.
Root owns Apple build/test execution; non-Apple hardware execution is RUN OWED.

Explicit override flags now include MOJOLEARN_COLUMN_QUALCOMM,
MOJOLEARN_COLUMN_INTEL and MOJOLEARN_COLUMN_SPEC_BASELINE. These change only
the declared policy; backend detection and column_is_buildable are unchanged.
The qualification driver prints simulated status and declared buildability.
Every simulated column forces the shared reduction path, even when its
claimed native width is fixed. Spec-baseline uses its declared128-thread
maximum (4096 checked cells), while other arms use256 (8192 cells).

Installed package inspection: the Apple Pixi environment ships compiled
std.mojoc and max.mojoc, with CUDA/ROCm wrapper packages (_cublas,_cudnn,
_cufft,_rocblas,_miopen); no readable backend source was present at
lib/mojo. This inventory cannot establish emit/link/runtime support for a
fourth physical backend. Explicit-policy execution on Metal is only policy
and portable-kernel evidence, never Intel/Qualcomm/RDNA device admission.
