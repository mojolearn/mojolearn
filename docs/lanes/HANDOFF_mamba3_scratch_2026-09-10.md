# Mamba3 definite-write scratch candidate

Base main d557b851; candidate bcc511d7. Opt-in only, IDENTICAL. No builds or runs were performed by the implementing agent. Root owns local compilation, GPU scheduling and performance admission. Trees and other numeric modes are outside this lane.

`Mamba3DeviceStages` previously zero-filled and synchronized all 33 buffers. The binding can now request uninitialized storage for 27 buffers whose complete logical contents are written before consumption. This removes 1,884.234375 MiB of fills at B8/L4096/D512 and 1,845.46875 MiB at B8/L1024/D2048, plus their fill/synchronize calls. The previous complete phase log puts all stage allocation near 1.1 ms; the expected gain is bounded and must be measured, not inferred from the memory volume alone.

The constructor's default remains zero initialization. The binding opts in only with `MOJOLEARN_MAMBA3_UNINITIALIZED_SCRATCH=1`. Its device state, all weights, inputs, outputs and reports retain their contracts. Six working arrays whose writes are split between buffered and new-token producers keep their zero initialization: adt_work, sig_work, dt_work, rotq_work, rotk_work and v_work. No state zero initialization is removed.

| Selected stages | Complete producer before consumption |
| --- | --- |
| norm_sumsq, norm_out | Existing RMSNorm writes every new-token row/cell |
| in_proj, out_proj | GEMM with accumulate=False writes every output |
| a_out, dt_out | m3_a_dt_kernel writes every M×H cell |
| gamma_work, betap_work, scale_work | m3_scale_kernel writes every B×T×H cell, including the structural final shift |
| bcnorm_b, bcnorm_c | m3_bcnorm_kernel writes both full M×N outputs |
| theta_out, theta_last | Angle increment/recurrence writes all new-token angles and each final report |
| qkdot, kscale_work | Their existing kernels write every logical output cell |
| dacs | Every Q slot written, including the copied padded tail |
| seg_l | Every column explicitly writes its structural zero triangle and remaining decay entries |
| qk_s | Decay scratch writes its complete used prefix; QK later overwrites the entire Q×Q matrix |
| pass_states | Increment writes every cell before state-scan reads/overwrites each chunk entry |
| yintra, ystate | Existing scalar/tiled plans cover every new-token output |
| skip_out, gate_out, residual_out | Their elementwise kernels write every logical output |
| h_last, k_last, v_last | State scan and report kernels write complete report arrays |

`MOJOLEARN_MAMBA3_POISON_SCRATCH=1`, together with the opt-in, fills all selected buffers with quiet-NaN bits 0x7fc01234. The normal native gate then compares every recorded stage to its independent oracle and repeated/batch/continuation gates. A stale scratch read should become an observable failure. All six native-check construction sites accept the same opt-in, and logs report both switches. The normal public constructor remains untouched unless the caller explicitly requests scratch.

`tools/mamba3_scratch_leg.sh` runs baseline, uninitialized and poison builds. It compares native default/B2-L65-D64 traces, runs decode-cross/continuation/refusal, checks 39,087,232 fresh/report cells and mutable refusals, runs the combined public surface, and compares all three original public output files by complete SHA256 across arms. It records all timing samples; poison is a correctness diagnostic, not the performance candidate. Existing matched fixture helpers and an explicit Python runtime are reused; no opponent is measured.

Root's first Apple gate should compile/run the native check with IDENTICAL, UNINITIALIZED_SCRATCH and POISON_SCRATCH. If it passes, run the long shape and decode-cross/continuation before committing time to public GPU pricing. No production default should change until bitwise admission and a repeatable benefit are recorded.
