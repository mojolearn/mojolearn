# L40S: PR #58 confirm, #59, #60, #64 arms, #62 (Oct 1, replacement nvc1 pod)

The `ab_job` jobs on the L40S are main against each branch, built from source over 0.8.33 and timed twice, interleaved, plus a phase-timer build. Four jobs ran at once on the 4-GPU pod, one per GPU.
Raw files are in R2 at `measurements/2026-10-01/{ab-p5960,ab-pr58b,ab-arms57,pr62}-nvidia.tar.gz`; summary-raw.txt holds the condensed lines.
Phase timer values here are sums over the run, not per layer, so compare them within a table only.

## Same bits, faster

| arm | lm-train-step ms (main → arm) | phase |
|---|---|---|
| #58 head c119d731b (TQ16 default) | 41.7 → 39.0 | dq 1.34 → 0.58 |
| #60 (4-wide dkdv/zdot) | 40.9 → 39.6 | dkdv 0.68 → 0.58, zdot 0.70 → 0.61 |
| #59 (4-wide fwd/dq) | 40.9 → 40.9 | dq 1.33 → 1.26 |

## #64 arms on lane/neural-pass57
- **DKDV_BJ16:** slower on NVIDIA (lm-train-step 41.2 → 42.5; dkdv 0.69 → 0.78), same bits.
- **FWD_TQ16 (`-D MOJOLEARN_ATTN_FWD_TQ16=1`, from #54): WRONG ON NVIDIA.** The lm-forward digest is c1ff2015add6814a instead of 4a8e781b0739a038, and the losses start at 27.97 / 29.10 instead of 9.02 / 8.42. This is a correctness failure, not a bit difference. The arm is define-only, and the shipped default (32) is unaffected.

## #62 (NVIDIA tile step-down from k 4096): MERGED

| run | lm-forward ms | lm-train-step ms | gemm 4096³ race ms |
|---|---|---|---|
| default run 1 | 50.1 | 42.3 | 49.5 |
| default run 2 | 51.9 | 42.1 | n/a |
| MIN_BLOCKS=0 (#55 as merged) | 50.4 | 41.3 | 52.7 |
| MIN_K=0 | 50.8 | 49.8 | 50.1 |

The LM GEMMs (k 384..2048) take the same plan as MIN_BLOCKS=0 by construction, so the lm-train-step spread is noise from four jobs sharing the pod. The gemm digest is 535b4c27bd9313d1 for all three. check_device_default_dispatch is OK (5 shapes bit-identical to FLAT and the old plan), and the 8 GEMM gates are green. The AMD default is unchanged from #55 (TILE_MIN_K 0).
