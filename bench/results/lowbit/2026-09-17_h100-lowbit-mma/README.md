# The two low-bit profiles on an NVIDIA H100 (2026-09-17)

RunPod H100 80GB HBM3 (driver 580.126.09), Mojo 1.0.0, one guarded leg through
`tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60` with
`MOJOLEARN_GEMM_LEG_EXTRA=tools/lowbit_mma_leg.sh`, at the source commit in
`commit.txt` (the integration branch carrying lane/int8-mma d02447962 over
lane/identical-lowbit-inference). Pod terminated and verified gone with 56
minutes of the lease unused. Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md`.

| phase | exit | expected | held |
|---|---|---|---|
| lowbit (`pixi run check-gemm-lowbit`) | 0 | pass | yes |
| lowbit-force-flat (`-D MOJOLEARN_INT8_FORCE_FLAT=1`) | 0 | pass | yes |
| lowbit-sabotage (`-D MOJOLEARN_LOWBIT_SABOTAGE=1`) | 1 | fail | yes |
| lowbit-host-sabotage (`-D MOJOLEARN_HOST_SABOTAGE=1`) | 1 | fail | yes |

What the honest build read on this column: `int8 dispatch: mma
(lib_int8_matrix_unit_for)`, 10 gates, 0 failed: the bf16 narrowing, integer
seams and quantizer bounds; device bf16 conversions equal to the host; bf16
device equal to the oracle on nine shapes; fused plan equal to widen plan; bf16
batch invariance; int8 device (through the IMMA kernel) equal to the host
oracle; int8 batch invariance; and `check_int8_mma_matches_flat`, the matrix
unit against the flat kernel on 15 shapes (8 ragged: 3x5x17, 17x33x31, 33x17x33,
1x47x100, 50x70x1000, 13x21x4097, 2x16x32, 100x3x64) with both equal to the
oracle on the ragged 8. The forced-flat run reads the same 10 gates with
`int8 dispatch: flat`. The device value arm fails the two oracle gates, the
plan gate and the mma gate; the host value arm fails the two oracle gates and
the mma gate, and leaves the plan gate passing, as the contract predicts.

The leg's own fp32.v1 card diff (Apple M4 against this box, 60 stages) read
IDENTICAL as well; it is in the evidence directory, not here.

Columns now: Apple M4 (Metal, flat int8) and NVIDIA H100 (IMMA). AMD owed.
