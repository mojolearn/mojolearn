# Performance default decisions and workload size

User direction, September 10: performance defaults must be decided on large
intended workloads. Small correctness fixtures can reject incorrect code;
they cannot qualify a performance default. Numerical admission remains mandatory
at every size. A correctness repair must not be disabled because it costs time.

This is a scoped audit of the recent GEMM, kNN, Mamba and Transformer decisions,
not a claim that every historical kernel-matrix row has been audited. Decision
history is retained in source, raw logs and handoffs; where evidence is noisy or
missing, the decision below says so instead of reconstructing an unsupported win.

| Decision | Large evidence that drove it | Status and limits |
|---|---|---|
| NVIDIA GEMM 128x128/KS16 staging | Llama t512 actual m,n,k: 512/4096/4096, 512/14336/4096, 512/4096/14336; two seven-round H100 runs | Enabled on NVIDIA; 9.7–11.3 useful TFLOP/s. Other columns keep their previous dispatch. |
| Forced GEMM 64x64 candidate | Same three large Llama shapes; candidate ran 0.747–0.841x baseline speed | Rejected. Historical untuned/dispatch labels were wrong: this was forced candidate versus current dispatch. |
| GEMM transpose swizzle plan 19 | Same three large Llama shapes on M4, four alternating samples per arm | Forced experiment only. Small timing differences do not justify a default; small correctness gates did not enable it. |
| NVIDIA kNN 512-query batches | 400k index / 4k queries / d32 / k10 and k15; five rounds per arm in both orders, final seven-round default | Enabled through the measured 400k index range. Request savings 5.8%/5.2%; complete outputs matched. Small/ragged cases were controls. |
| NVIDIA kNN eight query rows per distance thread | 400k rows / 4k queries / d32: paired k10 request 39.31→35.46 ms, k15 44.80→40.93 ms; 8-feature coverage flat to 0.9% slower | Enabled on NVIDIA, four rows elsewhere. See opponent table historical eight-row entry for exact shapes and device provenance. |
| Apple kNN complete-chain exponent preflight | Alternating five-round 400k-index / 1k-query / d32 / k15 runs (not the full 4k-query target): baseline device 488.827/473.205 ms, preflight 439.199/445.295 ms | Enabled with unsafe chains still repaired. Evidence supports modest device improvement, not a universal request-speed win: host contention and request noise were recorded. Smaller coverage supplemented this evidence. |
| Apple zero-FMA repair / GEMM rounding seam | Adversarial smallest-normal boundary and independent integer/device oracles | Correctness requirements, not optional speed gates. Keep enabled even if large timing regresses; recover cost with equivalent arithmetic. |
| Mamba scratch-initialization removal | Original large B8/L4096/D512 and B8/L1024/D2048 measurements; same baseline binary subsequently entered slow regimes | Not enabled. No stable positive large-run evidence; tiny output equality cannot promote it. |
| Shared GEMM staging effect on Mamba | Wide B8/L1024/D2048 initially improved, then same-binary repeats became unstable | No stable new Mamba price or ratio. GEMM dispatch decision has its own large GEMM evidence. |
| Transformer shared GEMM staging | B8/L4096/D512 and B8/L1024/D2048 in both orders, full output hashes unchanged | Own-arm savings measured; original Torch numerical admission still fails. No qualified opponent ratio. |
| Byte runtime shapes / Transformer stage tools | Only small native numerical or CPU diagnostic checks in the latest pass | Correctness/tooling progress only; no large-model speed/default claim. |

Evidence entry points:

- `bench/OPPONENT_REFERENCE.md`, sections Sep10 bounded kNN query batches and
  Sep10 end-to-end effect of GEMM staging.
- `bench/results/staging_performance_2026-09-10/`: gemm-stage, knn-batch,
  knn-final, mamba-stage, mamba-repeat and final-stage raw evidence.
- `bench/results/knn/2026-09-10-residual-final/README.md` and retained compressed
  Apple alternating/coverage logs. The previous 39% Apple repair cost predates
  accepted preflight and is not the current remaining overhead.
- `docs/lanes/HANDOFF_speed_gemm_2026-09-10.md` and
  `bench/results/gemm_swizzle_2026-09-10/`.

## Required evidence for subsequent performance switches

1. Name the target device, actual dimensions, input fixture, arithmetic mode,
   current baseline and candidate source/binary hashes before measurement.
2. Run the large target: GEMM the three Llama shapes above; kNN 400k/4k/d32
   at k10 and k15; Mamba/Transformer both original B8 large configurations.
   Include low-feature/ragged/small cases as controls, not promotion evidence.
3. Require numerical admission and complete-output comparisons appropriate to
   the change. A small numerical pass never substitutes for the large output check.
4. Warm both arms, reverse execution order, retain individual samples and report
   drift. Phase-instrumented timings diagnose costs; default adoption requires
   ordinary end-to-end timings with instrumentation disabled. Unstable arm
   rankings leave the candidate experimental.
5. Record the switch or rejection and its evidence here or in the lane handoff.
   Reuse matched cached opponent timings; append any newly measured opponent
   to `bench/OPPONENT_REFERENCE.md`. No new opponent is needed for own-arm tuning.

Trees remain outside this lane. No performance default was changed by this audit.

## Full Apple target measured after the audit

The continuation ran 400k/ 4000/d32 at k10 and k15 with ordinary phase-disabled
timing, five rounds after two warmups, both arm orders. Current preflight
request times are 3.6–7.4% lower at k10 and 4.1–6.3% lower at k15 than the
safe NO_PREFLIGHT control. All eight paired outputs and cross-order hashes
match, including controls. Decision: retain current preflight on this evidence;
no new switch. Small-case timing does not drive this decision. Details and
limitations: `bench/results/knn_large_gate_audit_2026-09-10/README.md`.

Mamba original large-shape regime diagnosis, Transformer original large-shape
numerical admission, and physical H100 GEMM occupancy remain open. No small
check in this pass qualified any of those.


## Scoped Apple metadata promotion

The subsequent continuation promotes per-request vector exponent minima only
on Apple IDENTICAL, transposed register distance path, Euclidean return_sqrt,
exactly 400k index / 4000 queries / 32 features, k10 or 15. This is intentionally
narrow: other shapes keep the previous per-tile preflight. The forced candidate
saved 23.0–25.7% in a stable large paired window. A freshly compiled actual
default saved 15.7–25.0% against NO_METADATA in both orders; its later device
window drifted, so those absolute timings are not a universal price. The
ranking remained favorable. Five rounds after two warmups, full outputs across
all arms/orders/windows equal, 396584-case four-arm integer oracle, mutation
check and 24 distance layouts pass. Two below-scope controls confirm fallback;
their timings do not drive promotion. Evidence and exact flags/source hashes:
`bench/results/knn_metadata_2026-09-10/README.md`.

The NVIDIA GEMM scalar shared-read candidate was rejected on the actual three
Llama t512 shapes: flat QKV, ~0.6% up improvement, slight down regression.
Its static occupancy remained one block/SM. No GEMM switch.
