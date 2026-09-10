# Mamba3 definite-write scratch candidate

Base main d557b851; candidate bcc511d7. Opt-in only, IDENTICAL. No builds or runs were performed by the implementing agent. Root owns local compilation, GPU scheduling and performance admission. Trees and other numeric modes are outside this lane.

`Mamba3DeviceStages` previously zero-filled and synchronized all 33 buffers. The binding can now request uninitialized storage for 27 buffers whose complete logical contents are written before consumption. This removes 1,884.234375 MiB of fills at B8/L4096/D512 and 1,845.46875 MiB at B8/L1024/D2048, plus their fill/synchronize calls. The previous complete phase log puts all stage allocation near 1.1 ms; the expected gain is bounded and must be measured, not inferred from the memory volume alone.

The constructor's default remains zero initialization. The binding opts in only with `MOJOLEARN_MAMBA3_UNINITIALIZED_SCRATCH=1`. Its device state, all weights, inputs, outputs and reports retain their contracts. Six working arrays whose writes are split between buffered and new-token producers keep their zero initialization: adt_work, sig_work, dt_work, rotq_work, rotk_work and v_work. No state zero initialization is removed.

| Selected stages | Complete producer before consumption |
| --- | --- |
| norm_sumsq, norm_out | Existing RMSNorm writes every new-token row/cell |
| in_proj, out_proj | GEMM implements C=op(A)*op(B), with no old-C term, and writes every output |
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

## Definite-write audit (candidate bcc511d7)

References below name functions, with line numbers at this candidate revision. `block` means `mamba/impl/modules/mamba3.mojo`; `siso` means `mamba/impl/ops/mamba3_siso.mojo`; `m1` means `mamba/impl/modeling/modeling_mamba.mojo`. Let M=B*L, T=q0+L, C=ceil(T/Q), N=128 and P=64. Only logical cells are consumed or traced; allocation's max(n,1) sentinel is not a logical cell.

| Field(s) | Definite write and extent |
| --- | --- |
| norm_sumsq, norm_out | m1 `mamba_rms_norm_kernel`:626 owns each token, initializes its local accumulator, stores M sums and all M*D normalized cells. |
| in_proj, out_proj | block `mamba3_block_forward`:1106 calls `identical_gemm[False]` for both projections. The False parameter disables vendor dispatch; overwrite follows the GEMM C=op(A)*op(B) contract, not that flag. All positive output dimensions are written. |
| a_out, dt_out | block `m3_a_dt_kernel`:504 stores both outputs for every cell below M*H. |
| bcnorm_b, bcnorm_c | block `m3_bcnorm_kernel`:558 owns each token and stores both complete N-element rows. |
| gamma_work, betap_work, scale_work | siso `m3_scale_kernel`:327 stores all B*T*H cells; its final shifted-beta zero is explicitly stored. |
| theta_out | siso `m3_angle_increment_kernel`:384 first writes every M*H*R increment if enabled; `m3_angle_kernel`:411 subsequently reads and overwrites exactly those cells. The legacy path computes increments directly and never reads old theta_out. |
| theta_last | siso `m3_angle_kernel`:411 stores each B*H*R owner's final run after all L tokens. |
| qkdot | siso `m3_qkdot_kernel`:589 stores all M*H cells. |
| kscale_work | siso `m3_kscale_kernel`:644 stores all B*T*H*N cells. |
| dacs | siso `m3_dacs_kernel`:674 owns each B*H*C row and stores all Q entries. Padded entries explicitly receive the last real prefix sum. |
| seg_l | siso `m3_seg_l_kernel`:736 owns every column in B*C*H*Q*Q; rows i<=j explicitly receive +0, remaining rows receive their decay, including padded rows. |
| qk_s | siso `m3_state_decay_kernel`:870 writes the B*H*C*(Q+1) prefix used as decay scratch. Increment/scan reads only that prefix. Later `m3_qk_s_kernel`:1211 or `m3_qk_s_tiled_kernel`:1161 overwrites the full B*C*H*Q*Q allocation before Y_intra. Scalar output starts at zero and is always stored. Tiled structural-zero and padded-row branches explicitly store zeros before returning. |
| pass_states | siso `m3_state_increment_kernel`:889, tiled:935 or shared_v:987 writes all B*C*H*P*N increments before `m3_state_scan_kernel`:1037 reads and overwrites every chunk entry. Legacy `m3_statepass_kernel`:1071 directly stores every entering-state entry before downstream use. |
| h_last | Both state-scan:1037 and legacy statepass:1071 store the complete B*H*P*N report after the chunk loop. |
| yintra | siso scalar:1330 stores every M*H*P cell. Tiled:1271 has two channel tiles per head and eight token rows per tile; stores exactly q0<=t<T. Whole-buffer-prefix tiles return uniformly, but correspond to no output rows. |
| ystate | siso scalar:1453 and tiled:1395 cover the same complete new-token extent as Y_intra. |
| skip_out, gate_out | siso `m3_skip_gate_kernel`:1521 stores both results for every M*H*P cell. |
| residual_out | m1 `residual_add_kernel`:1113 stores every M*D cell. |
| k_last, v_last | siso `m3_reports_kernel`:1636 partitions B*H*(N+P) owners between the two reports and copies the final real row T-1. |

`mamba3_block_forward` rejects B<=0 or L<=0 before any stage consumer, then rejects stage/call/state shape disagreement. Thus an L=0 caller cannot read uninitialized reports, and admitted calls always have T>0 and C>=1. For L=1 decode and arbitrary admitted q0, only new-token outputs are allocated at M; their producer offsets subtract q0. Q is 32 or 64, divisible by the tiled token width 8, so a tile never crosses a chunk boundary. Partial final tiles mask only non-output lanes. Dacs, seg_l and qk_s explicitly write their padded tails; increment folds retain structural padded zeros. Buffered-prefix/new-token assembly still uses zero-initialized adt_work, sig_work, dt_work, rotq_work, rotk_work and v_work. Recurrent theta/h, pending state, and unused buffered capacity retain their original initialization and update rules.

This proof is for admitted, unsabotaged calls. Deliberately armed upstream-recurrence sabotage can skip producers; the public binding already refuses sabotage builds. The proof does not turn skipped-producer negative-control configurations into supported scratch consumers.

## Allocation lifetime

`_m3_stage_buffer` returns `dev^`, transferring the owning DeviceBuffer into its stage field; no borrowed local pointer or host storage escapes. The public binding constructs `dstages` at line879, then `_m3_upload_addr` at line884 synchronizes the same DeviceContext before entering the block. `_m3_upload_addr`:257 synchronizes both its direct-copy and legacy upload paths. Each output/report download completes before the explicit `dstages` teardown at line915. Native block calls also perform synchronous refusal readbacks before numerical stage consumers. Thus this change does not depend on a borrowed allocation surviving a helper return, nor on an unsynchronized public first use. Enqueue ordering on the same context remains the existing API pattern; no claim about undocumented cross-stream ordering is needed. A metadata refusal before kernel launch holds no enqueued stage-consuming pointer. Six retained zero-filled fields introduce additional synchronization, but this incidental fact is not the lifetime justification.

Root reports Apple baseline/poison native default and long traces byte-equal, plus decode-cross, continuation and refusal passes. H100 admission/performance remains pending at the time of this audit.

## Transformer follow-on review (no edits)

`LlamaDeviceStages.__init__` in `transformer/impl/llama/modeling_llama.mojo`:1193 also zero-fills projection, norm, MLP and attention scratch. A bounded follow-on could opt out only for fully written linear stages: both norm sums/outputs, q/k/v projections, q/k rotary outputs, context/output projection, residuals and MLP intermediates. These are produced by `llama_rms_norm_kernel`:1315, GEMM with no accumulation, `apply_rotary_pos_emb_kernel`:1601, attention context/scatter or fused forward, `silu_kernel`:2404 and `mlp_gated_kernel`:2431. Attention context needs route-specific proof before inclusion.

Do NOT mechanically apply the Mamba list to transformer cache storage: `kv_append_kernel`:1723 writes only the active packed B*nkv*S*HD prefix, while stage cache buffers have S_cap capacity. The non-window path at line3298 copies whole stage buffers into the persistent cache; an unwritten capacity tail could therefore become observable. Preserve k_cache/v_cache zeros unless a separate capacity-tail proof or exact clearing rule is implemented. Preserve LlamaKVCache's own zeros, lean attention placeholders, and unused packed-head scratch until their backward/materialization consumers are audited. The RoPE table kernel:1473 fully writes both tables, but removing those two fills has much smaller volume than linear stages. Measure current allocation phase before selecting this follow-on; no transformer source changes are proposed here.

## Initial H100 performance rejection

Root reports first medians baseline/uninitialized: narrow 56.767391/236.131446 ms, wide 104.340496/103.713224 ms, tiny 1.1889/1.0680 ms. These are preliminary single-order measurements; output hashes matched so far and poison arm remained pending. Do not enable this candidate by default on this evidence, including a tiny-shape carveout. Final combined GEMM validation should use zero-initialized Mamba3 stages.

Read-only regression inspection finds identical allocation sizes/order and numerical launches, and a synchronized input upload before numerical stage use. No missing allocation-readiness barrier has been established. Deferred allocation, physical placement/first-touch behavior, and external runtime conditions are hypotheses only. Poison restores the fills and synchronization while retaining the new helper/constructor code, making its timing a useful discriminator. If poison restores narrow speed, a bounded next diagnostic is uninitialized allocation with a synchronize per allocation; this separates the removed synchronization from removed memory writes. Existing PHASE_TIMERS synchronizes stage construction, so compare against an uninstrumented arm and do not mistake that extra barrier for a neutral observer. No additional code or device work was performed for this diagnosis.

## Final H100 decision: rejected

Root completed the three arms on the same H100, with all GPU jobs serialized by predecessor completion files. There was no overlapping kNN GPU work. Each arm used the baseline GEMM implementation to isolate scratch allocation; no opponent was measured.

| Public shape | Baseline ms | Uninitialized ms | Poison ms |
| --- | ---: | ---: | ---: |
| tiny | 1.188908 | 1.068041 | 1.220964 |
| narrow B8/L4096/D512 | 56.767391 | 236.131446 | 56.415820 |
| wide B8/L1024/D2048 | 104.340496 | 103.713224 | 103.963660 |

All native default and long trace comparisons, decode-cross, continuation, refusals, large fresh/report checks and public surface checks passed. All six complete-output SHA256 comparisons (three shapes each for uninitialized and poison against baseline) passed. Poison fills therefore found no observed stale scratch read in the tested cases. The static definite-write audit remains useful, but arithmetic admission alone does not justify production adoption.

The initial uninitialized narrow measurement was 4.16x the initial baseline, and poison measured approximately the initial baseline. Subsequent reuse of the identical original baseline library also produced the slow narrow regime (see update below). Consequently, the original three-arm sequence does not isolate a causal effect of scratch initialization. No allocator, synchronization, first-touch or clock cause is established. The tiny improvement does not justify a separate default dispatch from this single three-arm run; wide offers no material repeatable win here.

**Do not adopt the scratch optimization: stable positive performance evidence is absent.** Root archived its candidate patch and reproduction driver, and restored the three production/check source files to d557b851 behavior in `9c10d749`. Final combined Mamba3 validation uses the accepted GEMM change only, retaining zero-initialized stages. The raw remote evidence was produced under `/root/evidence/mamba-stage`, including `price-baseline.log`, `price-uninitialized.log`, `price-poison.log`, `timings.json` and `full-output-identity.json`; root owns its durable local archive and final combined evidence. This handoff records root-reported completed results; the implementing agent performed no build, GPU execution or independent retiming.

## Subsequent same-library timing invalidates scratch attribution

Root's final GEMM-only comparison reused the identical `baseline.so` from the scratch experiment, with zero-initialized stages. Narrow baseline medians became 229.788 and 225.317 ms in two comparison orders, versus the original 56.767391 ms. The new GEMM default measured 222.377 and 222.876 ms. Thus both original and modified libraries can occupy the slow narrow regime; the initial 4.16x contrast cannot be attributed to removing scratch fills. Earlier statements describing it as an optimization-induced regression are superseded by this observation.

Wide remained comparatively stable: baseline 102.725/103.468 ms versus GEMM default 95.521/93.319 ms, with unchanged complete outputs. These rounded root-reported values provide context only; the final GEMM evidence and opponent ledger are owned by root. The scratch candidate remains unadopted because no stable positive evidence supports it, not because a scratch-specific slowdown has been established. Root is collecting an isolated identical-binary repeat with hardware context after all jobs; no source change or causal conclusion follows from that pending diagnosis.

## Final isolated identical-binary repeat: no qualified Mamba3 speed claim

After every other GPU job completed, root repeated the original baseline, final GEMM default, and identical original baseline again. Baseline library SHA256 checks matched, and all complete output SHA256 values remained unchanged.

| Isolated arm | Tiny ms | Narrow ms | Wide ms |
| --- | ---: | ---: | ---: |
| baseline0 | 1.267016 | 248.337356 | 102.489213 |
| default0 | 1.246534 | 251.352344 | 97.158484 |
| identical baseline1 | 1.271963 | 250.333956 | 283.757282 |

The wide baseline now also changes timing regime within an identical-binary repeat. The earlier two-order wide 7–10% improvement is an observation, not a stable current price. **No new Mamba3 speedup claim or ratio against a cached opponent is qualified for any shape by this run.** The scratch candidate stays unadopted, the accepted GEMM arithmetic change has separate GEMM evidence, and Mamba3 end-to-end latency remains unresolved. Root archived before/after GPU query and process snapshots alongside these results. No hardware, allocator, scheduling or compiler cause is inferred from the available evidence.
