# Exact kNN logical widths, RDNA compilation and distance reuse

Final source c03bbcfc, based on2419895f. kNN native subgroup minimum now
has an explicit compile-time width. NVIDIA IDENTICAL distance tiles reuse
index loads across eight query rows instead of four, preserving every cell's
ascending feature/FMA chain. Apple and other columns retain four rows.
The ROWS4 diagnostic flag restores NVIDIA's prior distance tile.
No floating reduction order, identity floor, tree source, FAST or DETERMINISTIC
path was changed.

## Exact logical reduction

`neighbors/checks/lane_minimum.mojo` reduces UInt64 composite keys in logical
groups whose widths are positive powers of two dividing the block size.
Shuffle is admitted only when a known physical subgroup width is divisible
by the logical width. Otherwise shared memory and block barriers implement
the same exact minimum. All block threads must participate. A final barrier
protects scratch reuse between successive calls.
Integer minimum is associative, commutative and idempotent. This permits
changing its tree while retaining signed-zero/NaN key ordering and ties;
it does not justify reordering floating-point sums.

The native gate checks widths1,2,4,8,16,32,64,128, all lanes, two successive
calls, high-word ties and high-bit values against an independent serial
UInt64 oracle. Apple and NVIDIA each pass16cases/8192cells. On32-lane
hardware,64/128-wide logical groups use shared memory across physical groups.
Existing36 long-selector cases pass on both devices. Alternating H100 prices
show the extraction alone is neutral: k10 request39.309916 ->39.314031ms;
k15 44.817819 ->44.801711ms, with reverse order agreeing.

## Declared policy versus actual backend

| Column | Lane declaration | Final selection and evidence |
|---|---:|---|
| Apple | fixed32 | detected/fallback; native gate executed |
| NVIDIA | fixed32 | detected; native gate executed |
| AMD/CDNA | fixed64 | generic AMD detector after RDNA; device run owed |
| AMD/RDNA | fixed32 | architecture-specific detector; native-target compile passed |
| Qualcomm | variable floor8 | explicit policy override only; Apple shared simulation passed |
| Intel | variable floor8 | explicit policy override only; Apple shared simulation passed |
| spec baseline | variable floor1 | explicit policy override only; Apple shared simulation passed |

TARGET_COLUMN and DETECTED_COLUMN check has_amd_rdna_gpu_accelerator before
generic AMD. The installed Apple compiler imports that detector successfully.
Actual RDNA cross-compilation passes for gfx1100 and gfx1201 with an assertion
requiring TARGET_COLUMN=RDNA and not simulated. The production long-selector
caller also compiles for gfx1100. These binaries were never executed: physical
RDNA numerical identity, collectives and timing are RUN OWED.

The [official requirements](https://mojolang.org/docs/requirements/) list
RDNA gfx1100/gfx1201 as known-compatible targets; the
[detector reference](https://mojolang.org/docs/std/sys/info/has_amd_rdna_gpu_accelerator/)
documents the architecture-specific predicate. Installed compiled packages
alone cannot establish emit/link/runtime support for Intel or Qualcomm.
Their explicit defines change declared policy, not the detected backend or
column_is_buildable. Simulated columns always use shared reduction, avoiding
accidental64-lane shuffles on a32-lane host. Spec-baseline's declared128-thread
limit is retained: its simulation checks4096cells; the other simulations8192.
These Apple policy simulations do not admit another physical device.

## Measured NVIDIA distance reuse

H10080GBHBM3, driver580.126.09, Mojo1.0.0(ed45d567), IDENTICAL dyadic-v1;
two warmups, seven rounds, exclusive GPU timing. At400k index rows/4000
queries/32features, k10 request39.31 ->35.46ms (~9.8% less), device38.23
->34.38; k15 request44.80 ->40.93 (~8.6% less), device43.66 ->39.79.
Phase instrumentation shows distance25.13 ->21.23ms while selection stays
13.45 ->13.42 and merge about0.8ms. Timers add synchronization, so phase
numbers are diagnostic rather than uninstrumented prices.

All16 public-grid fingerprints match the archived corrected baseline;
six paired d8/d32/d128 andq32/q1000 fixtures also match. The exact396584x3
FMA oracle and24 full distance-layout fixtures pass for the eight-row tile.
Small-feature performance is a measured tradeoff: d8/q32 is flat, d8/q1000
is about0.9% slower in both measurement orders. d32/q1000 is about8% faster,
d128/q1000 about15%, and d128/q32 about5%. No universal speedup is claimed.

The existing opponent table receives updated16 IDENTICAL prices through
`bench/results/knn/2026-09-10-logical-width/opponent-table.patch`. Its archived
cuML tuple is reused unchanged; opponent admission checks indices, not bitwise
cuML distances. No opponent or fourth-device benchmark was invented.
Final flag-absent default source c03bbcfc passes the396584x3 oracle and24
layout fixtures. Seven-round request/device medians are35.473274/34.374416ms
for k10 and40.942684/39.805605ms for k15; both fingerprints match the
prior corrected baseline. All final logs were fetched before GPU release.

Evidence: `bench/results/knn/2026-09-10-logical-width/`, including commands,
compressed build/gate/pricing logs, rejected initial compile error, metadata
and grid summary. No GPU binaries or generated17MB oracle fixture are committed.
