# The two low-bit profiles on an AMD MI325X (2026-09-17)

DigitalOcean gpu-mi325x1-256gb (gfx942), one guarded leg through
`tools/do_extra_leg.sh amd --minutes 60 --skip-gates` with
`MOJOLEARN_GEMM_LEG_EXTRA=tools/lowbit_mma_leg.sh` and
`MOJOLEARN_GPU_ARCHS=gfx942`, at the source commit in `commit.txt`. Droplet
destroyed and verified gone. Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md`.

| phase | exit | expected | held |
|---|---|---|---|
| lowbit (`pixi run check-gemm-lowbit`) | 0 | pass | yes |
| lowbit-force-flat (`-D MOJOLEARN_INT8_FORCE_FLAT=1`) | 0 | pass | yes |
| lowbit-sabotage (`-D MOJOLEARN_LOWBIT_SABOTAGE=1`) | 1 | fail | yes |
| lowbit-host-sabotage (`-D MOJOLEARN_HOST_SABOTAGE=1`) | 1 | fail | yes |

The honest build read `column: amd  int8 dispatch: mma (lib_int8_matrix_unit_for)`,
10 gates, 0 failed, `check_int8_mma_matches_flat` included: the MFMA
`v_mfma_i32_16x16x32_i8` kernel equal to the flat kernel on 15 shapes and both
equal to the host oracle on the 8 ragged ones. With the Apple M4 record
(`bench/results/identity_break/2026-09-17_lowbit-m4/`, flat int8) and the H100
record (`bench/results/lowbit/2026-09-17_h100-lowbit-mma/`, IMMA), the nine
GEMM-gate shapes of both profiles now have three vendor columns, each against
the same host oracle. The cells are the gates' fixtures, not a lane record;
the fourteen harness lanes still have the M4 columns only.
