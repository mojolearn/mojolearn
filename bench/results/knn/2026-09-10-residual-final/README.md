# Exact kNN residual optimization, 2026-09-10

Final source: 8266f1c0. NVIDIA uses rounded FP32 FMA followed by FTZ
multiplication by exactly one. Apple proves all products in a complete
sixteen-cell feature chain lie on the 2^-149 lattice before bypassing its
integer zero-result repair. Unsafe chains retain the repair. Summation order
and selector are unchanged. Trees and FAST/DETERMINISTIC are untouched.

NVIDIA H100 80GB HBM3, driver580.126.09, Mojo1.0.0(ed45d567), root-owned
pod gdkumy79olopt6. No concurrent GPU timing in this lane's slots.
Pristine5011239a versus final, seven rounds after two warmups,400k x32,
4000 queries: k10 request49.452488 ->39.310253ms (-20.51%), device48.374526
->38.228002; k15 request54.973115 ->44.817340 (-18.48%), device53.826727
->43.671144. Both index and distance fingerprints match. All16 public
shape prices pass; grid-summary.json lists the repeated final grid.
The existing opponent table receives these16 rows through opponent-table.patch.
Its archived cuML reference is reused, with identicalGPU/driver metadata;
its admission checks indices, not bitwise cuML distances.

Final NVIDIA strict actual/simulated/production-tile oracle396584 x3,
24 distance-layout fixtures and36 long-selector fixtures pass.
The final-status log preserves one erroneous nonexistent harness path;
the correct bench/knn_index_layout_main.mojo rerun passes as final-layout.log.
The first launch failed before execution because the Pixi environment was
not activated; all reported results use Pixi activation. No compiler-version
change is established relative to Sep9: both logs say1.0.0(ed45d567).
The idle clock snapshot is metadata, not proof of timed clock frequency.

Apple alternating five-round runs used the same baseline/preflight binaries:
baseline device488.827/473.205ms; preflight439.199/445.295ms; cached chunk4
441.047/445.368ms. Across six paired smaller-shape coverage runs (d8/q32,
d8/q1000,d32/q32;400k index,k15), preflight device ratios were
0.979,0.944,0.957,0.931,0.941,0.818. All outputs pass. Host request results
are noisy, including one17% regression despite a7% device improvement;
concurrent unrelated CPU builds were observed. These observations support
modest device improvement, not a robust universal request-speed claim.
Root final default396584 x3 oracle, eight boundary cases,24 layout fixtures,
and rebuilt Python exact binding smoke all pass (apple-final-default/).
Root source06e6c93d byte-equals8266f1c0 for all four kNN changed files.
Root deleted the shared pod; GET404 verified at04:55EDT.

Rejected: two-word warp redux and composite-key value decode add <1% on
NVIDIA, so both were removed; cached four-feature Apple admission performs
similarly to the simpler complete-chain preflight and was removed.
Diagnostic flags retain SOFTWARE_FLUSH on NVIDIA and NO_PREFLIGHT on Apple.
Bare fma.rn.ftz remains forbidden: it fails the smallest-normal rounding
boundary. Apple's prior exact repair remains active for unsafe inputs;
this does not repair unrelated Apple FMA consumers.
