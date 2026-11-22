# Every public estimator on three GPUs, second run (2026-09-14, 46 lanes, 414 cells, all three columns complete)

The rerun the 2026-09-13 record owed: the same `tools/identity_break.py` lanes (46, every public
estimator), nine hostile fixtures, each cell fitted twice, three columns (train, infer, model), after
the `ExperimentalTwoLevelFeatureFreq` fix (DEVIATION 2710, shipped in 0.8.5) and with the leg body
building the CPU host bindings so the byte-LM host lanes have cells on every box.

| column | box | commit archived | cells |
|---|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal, two-core cap; the 17 IDENTICAL bindings the 0.8.5 macOS release run built at 8d16ce2f | labeled 7c534e713 (the checkout's HEAD; no GPU source differs from 8d16ce2f) | 414: stable 414, moved 0, refused 0 |
| nvidia-h100-sm_90a | RunPod H100 SXM, sm_90a (`bench/results/e1g/2026-09-14_005702-nvidia-h100-identity-46-lanes-b`) | ca1db545 | 414: stable 414, refused 0 |
| amd-mi300x-gfx942 | Hot Aisle MI300X, gfx942, rocm/dev-ubuntu-22.04 container (`bench/results/e1g/2026-09-14_005702-amd-mi300x-hotaisle-identity-46-lanes-b`) | ca1db545 | 414: stable 414, refused 0 |

`diff.apple-nvidia-amd.txt`: `summary: IDENTICAL=414` and `summary (infer/model): IDENTICAL=459,
N/A=369` (N/A is a lane with no infer or model column). No DIVERGENT, MOVED or ONE-COLUMN cell.
`gbdt-feature-freq` is IDENTICAL x3 on all nine fixtures, where the 2026-09-13 run had it divergent
on eight between every pair of vendors. `byte-lm`, `byte-lm-host-infer` and `byte-lm-host-train`
have all 27 cells on all three boxes (18 + 9 were ONE-COLUMN before).

The commits differ between the Apple column and the GPU columns because main moved twice while the
legs ran (the CPU training phase 1 merge, then the leg body fix). Between 8d16ce2f, 7c534e713 and
ca1db545 the only `.mojo` files outside `bindings/_mojolearn_*_host.mojo` that change are the four
host oracles (`gemm/checks/gemm_oracle.mojo`, `holtwinters/checks/hw_oracle.mojo`,
`kde/checks/kde_oracle.mojo`, `svm/checks/smo_oracle.mojo`), which gain sabotage-define blocks that
are inert without `-D MOJOLEARN_HOST_SABOTAGE=1`; the hashes above are the evidence that the
columns agree, the commit list is the provenance.

The first attempt at these two legs (`...-identity-46-lanes`, no `-b`, 00:39Z) built every GPU
binding and then wrote no column: `identity_break.py` requires a commit witness since the fourth
column landed and the runners ship a git archive with no `.git`, and every `build_*_host.sh` refused
the exported GPU architecture and column. `tools/identity_three_columns_leg.sh` at ca1db545 takes
`MOJOLEARN_COMMIT` or a `commit.txt` a wrapper body writes, and builds the host set as the CPU column.
