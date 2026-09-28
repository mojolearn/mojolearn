# Apple round 3 consolidation snapshot — 2026-09-28

Integration starts from main `243f74dc79b99b9457fe89dda557aeb843c62012` and aggregate `deb7a56f5cd80911e604bbfeb0386dfa2d673aa1`. Existing worktrees and running job snapshots were not modified. This records source integration, not a new correctness or performance qualification.

## Frozen family tips

| Family | Included tip | Evidence and default policy |
|---|---|---|
| ann | `c5ac7e737efbd0bddab77dfdd9286a74f35c21fc` | Host passes and resident preparation enabled after same-job M3 A/B with equal digests in both tiers. Other new kernels remain opt-in/unqualified. |
| cluster | `fc96682e7f10a5f6636e36821983bca3061b4298` | Ward has M4 Pro equal-label evidence, with taxi merge-tree differences. Latest laptop g1 records Ward, OPTICS, MeanShift, BGMM and AP gains; AP_SPLIT has no gain. Experimental compile switches remain opt-in; final combined correctness is owed. |
| decomp | `55dcd1f5e92a77b45499e396c76128f43dd69708` | Measured FAST Metal eigensolver and small Lanczos cases are inherited. IDENTICAL retains established default. Newly returned round-robin solvers are runtime opt-in and unbuilt; unconditional imports still require the consolidated build. Larger-size Lanczos quality and final dispatch validation remain owed. |
| linear | `7a46a4fb26c0113ffd42a6493d60642bce352f9d` | Blocks have preliminary timing gains but no accepted paired quality/convergence result; all new optimization switches stay opt-in. Latest fold wrappers/plain-loop option is included. |
| metrics | `652233910debb95d768c1059ed1ccd9f7d8cf88d` | Only baseline measured; nine changes are opt-in under MOJOLEARN_MSEL3, with original default epilogues preserved. |
| neighbors | `bba62c43d7630c4efceab9225387829b51591cf8` | Row-vector Jacobi enabled after equal-digest evidence in both modes. Slower SVM host-block and quality-regressing Nystroem host solver stay opt-in, as do unqualified newer kernels. Latest resident-K module isolation and split job commands included. |
| prep | `e6085d625a719814e4a26517676733393b9d3f7f` | Already in aggregate: measured host improvements and FAST radix, with 60 cases per tier matching prior digests. Abandoned unbuilt additions at c56cc2923 remain in history, not reintroduced. |
| trees | `adddb13b1cf947f5c7a8adecaf14affe59db7abb` | Lossguide batch has timing gains but compiled leaf-budget quality incomplete. Forest sessions, wider RF batches, zero-after-read, device gather and inherited partitions remain opt-in/unqualified. |

## Explicit exclusions and resolution

- `lane/apple3-merged-check` at `505f5e1cec9bdb54ac25006c7e45da77f45428bf` adds the verification-only light driver (`d144aafa6`), explicitly marked never merged. It remains available on that branch; it does not replace the fail-closed consolidated checker.
- `lane/decomp-apple3-roundrobin` is not blindly merged over the newer family tip: the selected family tip reintroduces its solvers with additional convergence guards.
- `lane/metrics-apple3-merge` contains the earlier measurement-only contribution already in the aggregate; the full selected metrics tip includes its final opt-in policy.
- The metrics benchmark add/add conflict was resolved to the final family's `MOJOLEARN_MSEL3` reporting, consistent with its actual opt-in switch. No numerical conflict was resolved by dropping one side.
- Historical timing figures are specific to their machines, fixtures, modes and switches. An opt-in switch is not a claim that its implementation builds or passes.

## One coordinated qualification campaign

Freeze the final combined source and explicit switch manifest before any campaign. Build baseline/candidate once per required binding and mode; preserve source closures, binary hashes and portable-math hashes. Run lightweight identity and batch invariance on the combined applicable inventory, with cross-vendor comparison where shared native or Python code changed. Use existing validated historical records only for unchanged source closures and exact recorded artifacts.

Time representative workloads with baseline and candidate arms interleaved on the same hardware and inputs, outside profiling/stage timers. Measure warm and cold behavior separately where relevant. FAST arithmetic changes require paired quality and convergence checks; IDENTICAL must retain its bitwise contract. Mutually exclusive algorithm variants are separate selected arms, not all switched on simultaneously. Keep wins that pass, leave regressions/unqualified options disabled, and investigate failures with targeted cases rather than repeating per-family full sweeps.

No M2 workload action, M3 restart, Apple allocation, cloud mutation, numerical run or release is authorized by this manifest.
