# Candidate A/B measurements

One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| A01 | identical | PENDING_MEASUREMENT | 0 |
| A02 | identical | PENDING_MEASUREMENT | 0 |
| A03 | identical | PENDING_MEASUREMENT | 0 |
| A04 | identical | PENDING_MEASUREMENT | 0 |
| A05 | identical | PENDING_MEASUREMENT | 0 |
| A06 | identical | PENDING_MEASUREMENT | 0 |
| A07 | identical | PENDING_MEASUREMENT | 0 |
| A08 | identical | PENDING_MEASUREMENT | 0 |
| F01 | fast | FAILED | 0 |
| F02 | fast | PARTIAL_MEASUREMENTS_RETAINED | 11 |
| F03 | fast | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| F04 | fast | FAILED | 0 |
| F05 | fast | FAILED | 0 |
| F06 | fast | FAILED | 0 |
| F07 | fast | PARTIAL_MEASUREMENTS_RETAINED | 6 |
| F08 | fast | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| F09 | fast | FAILED | 0 |
| F10 | fast | PARTIAL_MEASUREMENTS_RETAINED | 163 |
| F11 | fast | PARTIAL_MEASUREMENTS_RETAINED | 24 |
| F12 | fast | PARTIAL_MEASUREMENTS_RETAINED | 42 |
| F13 | fast | FAILED | 0 |
| F14 | fast | PARTIAL_MEASUREMENTS_RETAINED | 15 |
| F15 | fast | PARTIAL_MEASUREMENTS_RETAINED | 36 |
| F16 | fast | PENDING_MEASUREMENT | 0 |
| F17 | fast | PENDING_MEASUREMENT | 0 |
| F18 | fast | PENDING_MEASUREMENT | 0 |
| F19 | fast | PENDING_MEASUREMENT | 0 |
| F20 | fast | PENDING_MEASUREMENT | 0 |
| I01 | identical | PENDING_MEASUREMENT | 0 |
| I02 | identical | PENDING_MEASUREMENT | 0 |
| I03 | identical | PENDING_MEASUREMENT | 0 |
| I04 | identical | PENDING_MEASUREMENT | 0 |
| I05 | identical | PENDING_MEASUREMENT | 0 |
| I06 | identical | PENDING_MEASUREMENT | 0 |
| I07 | identical | PENDING_MEASUREMENT | 0 |
| I08 | identical | PENDING_MEASUREMENT | 0 |
| I09 | identical | PENDING_MEASUREMENT | 0 |
| I10 | identical | PENDING_MEASUREMENT | 0 |
| I11 | identical | PENDING_MEASUREMENT | 0 |
| I12 | identical | PENDING_MEASUREMENT | 0 |
| I13 | identical | PENDING_MEASUREMENT | 0 |
| I14 | identical | PENDING_MEASUREMENT | 0 |
| I15 | identical | PENDING_MEASUREMENT | 0 |
| I16 | identical | PENDING_MEASUREMENT | 0 |
| I17 | identical | PENDING_MEASUREMENT | 0 |
| I18 | identical | PENDING_MEASUREMENT | 0 |
| I19 | identical | PENDING_MEASUREMENT | 0 |
| I20 | identical | PENDING_MEASUREMENT | 0 |
| I21 | identical | PENDING_MEASUREMENT | 0 |
| I22 | identical | PENDING_MEASUREMENT | 0 |
| I23 | identical | PENDING_MEASUREMENT | 0 |
| I24 | identical | PENDING_MEASUREMENT | 0 |
| N01 | identical | PENDING_MEASUREMENT | 0 |
| N02 | identical | PENDING_MEASUREMENT | 0 |
| N04 | identical | PENDING_MEASUREMENT | 0 |
| N05 | identical | PENDING_MEASUREMENT | 0 |
| N06 | identical | PENDING_MEASUREMENT | 0 |
| N07 | identical | PENDING_MEASUREMENT | 0 |
| N08 | identical | PENDING_MEASUREMENT | 0 |

## Captured evidence

| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |
|---|---|---|---|---|---:|---|
| F01 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F01/default/result.json |
| F02 | apple/cholesky | cholesky | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/cholesky/result.json |
| F02 | apple/default | default/257/solve_ms | public_caller_component | MEASURED | 1.4119 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/default | default/257/factor_ms | public_caller_component | MEASURED | 1.6020 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/default | default/319/solve_ms | public_caller_component | MEASURED | 1.2316 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/default | default/319/factor_ms | public_caller_component | MEASURED | 1.0685 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/default | default/513/solve_ms | public_caller_component | MEASURED | 0.9948 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/default | default/513/factor_ms | public_caller_component | MEASURED | 1.0441 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/default/result.json |
| F02 | apple/mcd | mcd | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/mcd/result.json |
| F02 | apple/pca | pca | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/pca/result.json |
| F02 | apple/sdk | sdk/1-509-37-129-0/call_ms | public_caller_component | MEASURED | 0.9922 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/sdk/result.json |
| F02 | apple/sdk | sdk/1-137-137-521-1/call_ms | public_caller_component | MEASURED | 0.9800 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/sdk/result.json |
| F02 | apple/sdk | sdk/2-521-31-133-0/call_ms | public_caller_component | MEASURED | 1.3143 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/sdk/result.json |
| F02 | apple/sdk | sdk/2-509-41-131-0/call_ms | public_caller_component | MEASURED | 0.6982 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/sdk/result.json |
| F02 | apple/sdk | sdk/2-17-1-67-0/call_ms | public_caller_component | MEASURED | 1.5301 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F02/sdk/result.json |
| F03 | apple/default | default/cold-and-repeated-session/train_step_ms_trajectory_total | public_caller_component | MEASURED | 0.5125 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F03/default/result.json |
| F03 | apple/default | default/alternate-vocabulary-shape/train_step_ms_trajectory_total | public_caller_component | MEASURED | 0.5565 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F03/default/result.json |
| F04 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F04/default/result.json |
| F05 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F05/default/result.json |
| F06 | apple/active_compact | active_compact | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F06/active_compact/result.json |
| F06 | apple/active_cov | active_cov | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F06/active_cov/result.json |
| F06 | apple/cov_reuse | cov_reuse | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F06/cov_reuse/result.json |
| F06 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F06/default/result.json |
| F07 | apple/default | default/L17-KV4/forward_ms | public_caller_component | MEASURED | 0.1752 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/default/result.json |
| F07 | apple/default | default/L129-KV1/forward_ms | public_caller_component | MEASURED | 0.8973 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/default/result.json |
| F07 | apple/default | default/L513-KV2/forward_ms | public_caller_component | MEASURED | 0.7777 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/default/result.json |
| F07 | apple/gqa | gqa/L17-KV4/forward_ms | public_caller_component | MEASURED | 2.7525 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/gqa/result.json |
| F07 | apple/gqa | gqa/L129-KV1/forward_ms | public_caller_component | MEASURED | 0.7077 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/gqa/result.json |
| F07 | apple/gqa | gqa/L513-KV2/forward_ms | public_caller_component | MEASURED | 0.9083 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/gqa/result.json |
| F08 | apple/default | default/train-checkpoint-refusal/train_step_ms_trajectory_total | public_caller_component | MEASURED | 0.8766 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F08/default/result.json |
| F08 | apple/fused | fused/train-checkpoint-refusal/train_step_ms_trajectory_total | public_caller_component | MEASURED | 0.9494 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F08/fused/result.json |
| F08 | apple/views | views/train-checkpoint-refusal/train_step_ms_trajectory_total | public_caller_component | MEASURED | 0.9765 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F08/views/result.json |
| F09 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F09/default/result.json |
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6187 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/prefix_ms | public_caller_component | MEASURED | 0.7255 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/forward_ms | public_caller_component | MEASURED | 0.8065 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6761 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.7614 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.6362 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6037 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/prefix_ms | public_caller_component | MEASURED | 0.6032 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/forward_ms | public_caller_component | MEASURED | 0.6877 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6317 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.6204 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.5985 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6078 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.6273 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.6378 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l1_d32/forward_ms | public_caller_component | MEASURED | 0.5931 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6332 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/prefix_ms | public_caller_component | MEASURED | 0.6413 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/forward_ms | public_caller_component | MEASURED | 0.6745 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.7187 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/prefix_ms | public_caller_component | MEASURED | 0.7363 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/forward_ms | public_caller_component | MEASURED | 0.8485 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6677 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/prefix_ms | public_caller_component | MEASURED | 0.6872 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/forward_ms | public_caller_component | MEASURED | 0.6787 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6066 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/prefix_ms | public_caller_component | MEASURED | 0.5519 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/forward_ms | public_caller_component | MEASURED | 0.5911 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5969 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.6557 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/forward_ms | public_caller_component | MEASURED | 0.6309 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5495 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.5848 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.5691 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5679 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.4768 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.4967 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_a_near_zero_b3_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0188 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_a_near_zero_b3_l8_d8/prefix_ms | public_caller_component | MEASURED | 0.9989 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_a_near_zero_b3_l8_d8/forward_ms | public_caller_component | MEASURED | 1.0136 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_gate_saturation_b1_l8_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0149 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_gate_saturation_b1_l8_d16/prefix_ms | public_caller_component | MEASURED | 1.0451 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_gate_saturation_b1_l8_d16/forward_ms | public_caller_component | MEASURED | 0.9603 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_signed_zeros_b2_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0191 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_signed_zeros_b2_l8_d8/prefix_ms | public_caller_component | MEASURED | 0.8980 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_signed_zeros_b2_l8_d8/forward_ms | public_caller_component | MEASURED | 0.9983 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_softplus_guard_b2_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9774 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_softplus_guard_b2_l8_d8/prefix_ms | public_caller_component | MEASURED | 0.9997 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/adv_softplus_guard_b2_l8_d8/forward_ms | public_caller_component | MEASURED | 0.9963 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l1_d16/forward_ms | public_caller_component | MEASURED | 1.0660 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l1_d8/forward_ms | public_caller_component | MEASURED | 0.9911 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0135 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d16/prefix_ms | public_caller_component | MEASURED | 0.9528 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d16/forward_ms | public_caller_component | MEASURED | 0.9762 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9806 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d8/prefix_ms | public_caller_component | MEASURED | 0.9830 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b1_l64_d8/forward_ms | public_caller_component | MEASURED | 1.0063 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b2_l4_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0223 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b2_l4_d8/prefix_ms | public_caller_component | MEASURED | 1.0054 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b2_l4_d8/forward_ms | public_caller_component | MEASURED | 1.0265 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b3_l4_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0205 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b3_l4_d16/prefix_ms | public_caller_component | MEASURED | 0.8921 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/base_b3_l4_d16/forward_ms | public_caller_component | MEASURED | 0.9973 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_b2_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9751 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_b2_l257_d8/prefix_ms | public_caller_component | MEASURED | 0.9700 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_b2_l257_d8/forward_ms | public_caller_component | MEASURED | 0.9524 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row0_b1_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0006 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row0_b1_l257_d8/prefix_ms | public_caller_component | MEASURED | 0.9578 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row0_b1_l257_d8/forward_ms | public_caller_component | MEASURED | 1.0394 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row1_b1_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9907 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row1_b1_l257_d8/prefix_ms | public_caller_component | MEASURED | 1.0196 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/chunk-scan | chunk-scan/comp_row1_b1_l257_d8/forward_ms | public_caller_component | MEASURED | 0.9087 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0240 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/prefix_ms | public_caller_component | MEASURED | 1.0669 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/forward_ms | public_caller_component | MEASURED | 2.2935 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9509 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.2725 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.9880 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0154 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/prefix_ms | public_caller_component | MEASURED | 1.0277 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/forward_ms | public_caller_component | MEASURED | 1.0044 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0246 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.0255 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 1.0081 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0221 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.0028 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 1.0073 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l1_d32/forward_ms | public_caller_component | MEASURED | 1.0162 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9833 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/prefix_ms | public_caller_component | MEASURED | 0.9368 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/forward_ms | public_caller_component | MEASURED | 0.9747 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9340 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/prefix_ms | public_caller_component | MEASURED | 0.9274 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/forward_ms | public_caller_component | MEASURED | 0.9629 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0009 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/prefix_ms | public_caller_component | MEASURED | 1.0532 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/forward_ms | public_caller_component | MEASURED | 1.0120 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9878 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/prefix_ms | public_caller_component | MEASURED | 1.0104 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/forward_ms | public_caller_component | MEASURED | 1.0151 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0209 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/prefix_ms | public_caller_component | MEASURED | 1.1263 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/forward_ms | public_caller_component | MEASURED | 1.1754 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9441 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 1.1788 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.9769 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9668 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.8310 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.8446 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_a_near_zero_b3_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9143 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_a_near_zero_b3_l8_d8/prefix_ms | public_caller_component | MEASURED | 1.0799 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_a_near_zero_b3_l8_d8/forward_ms | public_caller_component | MEASURED | 1.4946 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_gate_saturation_b1_l8_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9240 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_gate_saturation_b1_l8_d16/prefix_ms | public_caller_component | MEASURED | 0.9699 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_gate_saturation_b1_l8_d16/forward_ms | public_caller_component | MEASURED | 0.9606 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_signed_zeros_b2_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9651 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_signed_zeros_b2_l8_d8/prefix_ms | public_caller_component | MEASURED | 0.9988 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_signed_zeros_b2_l8_d8/forward_ms | public_caller_component | MEASURED | 0.9486 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_softplus_guard_b2_l8_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9667 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_softplus_guard_b2_l8_d8/prefix_ms | public_caller_component | MEASURED | 0.9605 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/adv_softplus_guard_b2_l8_d8/forward_ms | public_caller_component | MEASURED | 0.8608 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l1_d16/forward_ms | public_caller_component | MEASURED | 1.0599 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l1_d8/forward_ms | public_caller_component | MEASURED | 0.8600 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9607 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d16/prefix_ms | public_caller_component | MEASURED | 0.9303 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d16/forward_ms | public_caller_component | MEASURED | 0.9418 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9541 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d8/prefix_ms | public_caller_component | MEASURED | 0.9500 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b1_l64_d8/forward_ms | public_caller_component | MEASURED | 0.8834 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b2_l4_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9647 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b2_l4_d8/prefix_ms | public_caller_component | MEASURED | 0.9635 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b2_l4_d8/forward_ms | public_caller_component | MEASURED | 0.9552 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b3_l4_d16/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9462 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b3_l4_d16/prefix_ms | public_caller_component | MEASURED | 0.9552 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/base_b3_l4_d16/forward_ms | public_caller_component | MEASURED | 0.9721 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_b2_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9572 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_b2_l257_d8/prefix_ms | public_caller_component | MEASURED | 1.0048 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_b2_l257_d8/forward_ms | public_caller_component | MEASURED | 0.8577 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row0_b1_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9707 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row0_b1_l257_d8/prefix_ms | public_caller_component | MEASURED | 0.9095 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row0_b1_l257_d8/forward_ms | public_caller_component | MEASURED | 0.8432 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row1_b1_l257_d8/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9572 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row1_b1_l257_d8/prefix_ms | public_caller_component | MEASURED | 0.8297 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba1-input | mamba1-input/comp_row1_b1_l257_d8/forward_ms | public_caller_component | MEASURED | 0.9414 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_a_floor_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.7083 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_a_floor_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.6775 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_a_floor_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.7993 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_softplus_band_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.7955 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_softplus_band_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.0143 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_adv_softplus_band_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.9373 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b1_l1_d32/forward_ms | public_caller_component | MEASURED | 0.8438 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b1_l64_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.8349 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b1_l64_d32/prefix_ms | public_caller_component | MEASURED | 0.9068 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b1_l64_d32/forward_ms | public_caller_component | MEASURED | 0.8347 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b2_l4_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.8040 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b2_l4_d32/prefix_ms | public_caller_component | MEASURED | 0.7934 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b2_l4_d32/forward_ms | public_caller_component | MEASURED | 0.8994 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b3_l4_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.8258 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b3_l4_d64/prefix_ms | public_caller_component | MEASURED | 0.8338 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_base_b3_l4_d64/forward_ms | public_caller_component | MEASURED | 0.8545 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_state_b2_l257_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.8406 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_state_b2_l257_d64/prefix_ms | public_caller_component | MEASURED | 0.8791 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F10 | apple/mamba3-elementwise | mamba3-elementwise/m3_state_b2_l257_d64/forward_ms | public_caller_component | MEASURED | 0.9060 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise/result.json |
| F11 | apple/compensated-pca | compensated-pca/collinear/inverse_ms | public_caller_component | MEASURED | 1.1052 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/collinear/embedding_ms | public_caller_component | MEASURED | 1.1234 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/collinear/fit_ms | public_caller_component | MEASURED | 2.1360 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/collinear/transform_ms | public_caller_component | MEASURED | 1.1057 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/low-rank/inverse_ms | public_caller_component | MEASURED | 0.9826 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/low-rank/embedding_ms | public_caller_component | MEASURED | 1.0034 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/low-rank/fit_ms | public_caller_component | MEASURED | 0.9876 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/low-rank/transform_ms | public_caller_component | MEASURED | 0.9639 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/clustered/inverse_ms | public_caller_component | MEASURED | 0.9411 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/clustered/embedding_ms | public_caller_component | MEASURED | 0.9418 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/clustered/fit_ms | public_caller_component | MEASURED | 0.8987 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/compensated-pca | compensated-pca/clustered/transform_ms | public_caller_component | MEASURED | 1.0456 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca/result.json |
| F11 | apple/default | default/509-17-100.0/fit_ms | public_caller_component | MEASURED | 0.9952 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/default | default/range-509-17-100.0/range_ms | public_caller_component | MEASURED | 0.9869 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/default | default/521-19-10000.0/fit_ms | public_caller_component | MEASURED | 0.9704 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/default | default/range-521-19-10000.0/range_ms | public_caller_component | MEASURED | 0.9878 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/default | default/997-31-1000000.0/fit_ms | public_caller_component | MEASURED | 0.9882 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/default | default/range-997-31-1000000.0/range_ms | public_caller_component | MEASURED | 1.0015 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/default/result.json |
| F11 | apple/grid | grid/509-17-100.0/fit_ms | public_caller_component | MEASURED | 0.9861 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F11 | apple/grid | grid/range-509-17-100.0/range_ms | public_caller_component | MEASURED | 0.9882 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F11 | apple/grid | grid/521-19-10000.0/fit_ms | public_caller_component | MEASURED | 0.9873 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F11 | apple/grid | grid/range-521-19-10000.0/range_ms | public_caller_component | MEASURED | 0.9642 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F11 | apple/grid | grid/997-31-1000000.0/fit_ms | public_caller_component | MEASURED | 0.9882 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F11 | apple/grid | grid/range-997-31-1000000.0/range_ms | public_caller_component | MEASURED | 0.9867 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/grid/result.json |
| F12 | apple/arena | arena/1009-11-depth4/fit_and_predict_ms | public_caller_component | MEASURED | 0.9407 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/arena | arena/1009-11-depth4/predict_ms | public_caller_component | MEASURED | 0.9969 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/arena | arena/1031-17-depth6/fit_and_predict_ms | public_caller_component | MEASURED | 1.0082 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/arena | arena/1031-17-depth6/predict_ms | public_caller_component | MEASURED | 0.9336 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/arena | arena/2017-23-depth5/fit_and_predict_ms | public_caller_component | MEASURED | 1.0003 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/arena | arena/2017-23-depth5/predict_ms | public_caller_component | MEASURED | 0.9978 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/arena/result.json |
| F12 | apple/categorical | categorical/categorical-511-11/repeated_predict_ms | public_caller_component | MEASURED | 1.0349 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/categorical/result.json |
| F12 | apple/categorical | categorical/categorical-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 1.0105 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/categorical/result.json |
| F12 | apple/categorical | categorical/categorical-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 0.9445 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/categorical/result.json |
| F12 | apple/categorical | categorical/categorical-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 1.0110 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/categorical/result.json |
| F12 | apple/default | default/1009-11-depth4/fit_and_predict_ms | public_caller_component | MEASURED | 0.9547 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/default | default/1009-11-depth4/predict_ms | public_caller_component | MEASURED | 1.0011 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/default | default/1031-17-depth6/fit_and_predict_ms | public_caller_component | MEASURED | 0.9860 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/default | default/1031-17-depth6/predict_ms | public_caller_component | MEASURED | 0.9964 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/default | default/2017-23-depth5/fit_and_predict_ms | public_caller_component | MEASURED | 0.9923 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/default | default/2017-23-depth5/predict_ms | public_caller_component | MEASURED | 1.0050 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/default/result.json |
| F12 | apple/depthwise | depthwise/depthwise-511-11/repeated_predict_ms | public_caller_component | MEASURED | 1.1062 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/depthwise/result.json |
| F12 | apple/depthwise | depthwise/depthwise-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.8870 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/depthwise/result.json |
| F12 | apple/depthwise | depthwise/depthwise-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 1.0276 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/depthwise/result.json |
| F12 | apple/depthwise | depthwise/depthwise-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.8646 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/depthwise/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/1009-11-depth4/fit_and_predict_ms | public_caller_component | MEASURED | 0.9627 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/1009-11-depth4/predict_ms | public_caller_component | MEASURED | 1.0045 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/1031-17-depth6/fit_and_predict_ms | public_caller_component | MEASURED | 1.0063 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/1031-17-depth6/predict_ms | public_caller_component | MEASURED | 1.0146 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/2017-23-depth5/fit_and_predict_ms | public_caller_component | MEASURED | 0.9904 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/leaf-inputs | leaf-inputs/2017-23-depth5/predict_ms | public_caller_component | MEASURED | 0.9858 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/leaf-inputs/result.json |
| F12 | apple/lossguide | lossguide/lossguide-511-11/repeated_predict_ms | public_caller_component | MEASURED | 1.0691 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/lossguide/result.json |
| F12 | apple/lossguide | lossguide/lossguide-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.4914 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/lossguide/result.json |
| F12 | apple/lossguide | lossguide/lossguide-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 1.0883 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/lossguide/result.json |
| F12 | apple/lossguide | lossguide/lossguide-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.4561 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/lossguide/result.json |
| F12 | apple/ordered-docids | ordered-docids/ordered-docids-511-11/repeated_predict_ms | public_caller_component | MEASURED | 0.9904 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-docids/result.json |
| F12 | apple/ordered-docids | ordered-docids/ordered-docids-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.9203 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-docids/result.json |
| F12 | apple/ordered-docids | ordered-docids/ordered-docids-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 1.0334 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-docids/result.json |
| F12 | apple/ordered-docids | ordered-docids/ordered-docids-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.8957 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-docids/result.json |
| F12 | apple/ordered-storage | ordered-storage/ordered-storage-511-11/repeated_predict_ms | public_caller_component | MEASURED | 0.9203 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-storage/result.json |
| F12 | apple/ordered-storage | ordered-storage/ordered-storage-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.7802 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-storage/result.json |
| F12 | apple/ordered-storage | ordered-storage/ordered-storage-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 0.9955 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-storage/result.json |
| F12 | apple/ordered-storage | ordered-storage/ordered-storage-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.8846 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ordered-storage/result.json |
| F12 | apple/ranking | ranking/ranking-511-11/repeated_predict_ms | public_caller_component | MEASURED | 1.0304 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ranking/result.json |
| F12 | apple/ranking | ranking/ranking-511-11/fit_and_first_predict_ms | public_caller_component | MEASURED | 0.9972 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ranking/result.json |
| F12 | apple/ranking | ranking/ranking-1023-17/repeated_predict_ms | public_caller_component | MEASURED | 1.0015 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ranking/result.json |
| F12 | apple/ranking | ranking/ranking-1023-17/fit_and_first_predict_ms | public_caller_component | MEASURED | 1.0009 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F12/ranking/result.json |
| F13 | apple/default | default | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F13/default/result.json |
| F13 | apple/packed-layout | packed-layout | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F13/packed-layout/result.json |
| F13 | apple/shap | shap | public_caller_component | FAILED | — | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F13/shap/result.json |
| F14 | apple/ann | ann/1031-7-probes5-skewFalse-filterFalse/query_ms | public_caller_component | MEASURED | 1.1574 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes5-skewFalse-filterFalse/fit_ms | public_caller_component | MEASURED | 0.9487 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes5-skewFalse-filterTrue/query_ms | public_caller_component | MEASURED | 0.8496 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes5-skewFalse-filterTrue/fit_ms | public_caller_component | MEASURED | 0.9487 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/4093-33-probes8-skewTrue-filterFalse/query_ms | public_caller_component | MEASURED | 0.7893 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/4093-33-probes8-skewTrue-filterFalse/fit_ms | public_caller_component | MEASURED | 1.0311 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/4093-33-probes8-skewTrue-filterTrue/query_ms | public_caller_component | MEASURED | 0.8012 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/4093-33-probes8-skewTrue-filterTrue/fit_ms | public_caller_component | MEASURED | 1.0311 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes17-skewFalse-filterFalse/query_ms | public_caller_component | MEASURED | 1.2984 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes17-skewFalse-filterFalse/fit_ms | public_caller_component | MEASURED | 0.9799 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes17-skewFalse-filterTrue/query_ms | public_caller_component | MEASURED | 1.4357 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/ann | ann/1031-7-probes17-skewFalse-filterTrue/fit_ms | public_caller_component | MEASURED | 0.9799 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/ann/result.json |
| F14 | apple/default | default/1031-7-k5/query_ms | public_caller_component | MEASURED | 0.4242 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/default/result.json |
| F14 | apple/default | default/2017-15-k11/query_ms | public_caller_component | MEASURED | 0.9289 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/default/result.json |
| F14 | apple/default | default/4093-23-k13/query_ms | public_caller_component | MEASURED | 1.2074 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F14/default/result.json |
| F15 | apple/center-accumulation | center-accumulation/2017-11-5/cold_fit_ms | public_caller_component | MEASURED | 1.0013 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/center-accumulation | center-accumulation/2017-11-5/repeated_fit_ms | public_caller_component | MEASURED | 1.0222 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/center-accumulation | center-accumulation/2053-13-7/cold_fit_ms | public_caller_component | MEASURED | 1.0041 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/center-accumulation | center-accumulation/2053-13-7/repeated_fit_ms | public_caller_component | MEASURED | 1.0116 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/center-accumulation | center-accumulation/4093-19-11/cold_fit_ms | public_caller_component | MEASURED | 0.9863 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/center-accumulation | center-accumulation/4093-19-11/repeated_fit_ms | public_caller_component | MEASURED | 0.9952 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/center-accumulation/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/509-7-3/cold_fit_ms | public_caller_component | MEASURED | 1.7658 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/509-7-3/repeated_fit_ms | public_caller_component | MEASURED | 1.3171 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/997-13-4/cold_fit_ms | public_caller_component | MEASURED | 1.0036 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/997-13-4/repeated_fit_ms | public_caller_component | MEASURED | 0.9371 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/1031-67-5/cold_fit_ms | public_caller_component | MEASURED | 0.9789 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/dbscan-outputs | dbscan-outputs/1031-67-5/repeated_fit_ms | public_caller_component | MEASURED | 0.9910 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/dbscan-outputs/result.json |
| F15 | apple/default | default/2017-11-5/cold_fit_ms | public_caller_component | MEASURED | 0.9470 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/default | default/2017-11-5/repeated_fit_ms | public_caller_component | MEASURED | 1.1250 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/default | default/2053-13-7/cold_fit_ms | public_caller_component | MEASURED | 1.0060 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/default | default/2053-13-7/repeated_fit_ms | public_caller_component | MEASURED | 0.9673 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/default | default/4093-19-11/cold_fit_ms | public_caller_component | MEASURED | 0.9927 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/default | default/4093-19-11/repeated_fit_ms | public_caller_component | MEASURED | 0.9981 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/default/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/509-7-3/cold_fit_ms | public_caller_component | MEASURED | 1.0389 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/509-7-3/repeated_fit_ms | public_caller_component | MEASURED | 0.9972 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/997-13-4/cold_fit_ms | public_caller_component | MEASURED | 0.9777 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/997-13-4/repeated_fit_ms | public_caller_component | MEASURED | 0.9964 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/1031-67-5/cold_fit_ms | public_caller_component | MEASURED | 0.9843 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-core | hdbscan-core/1031-67-5/repeated_fit_ms | public_caller_component | MEASURED | 0.9529 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-core/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/509-7-3/cold_fit_ms | public_caller_component | MEASURED | 0.9506 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/509-7-3/repeated_fit_ms | public_caller_component | MEASURED | 0.8997 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/997-13-4/cold_fit_ms | public_caller_component | MEASURED | 0.9225 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/997-13-4/repeated_fit_ms | public_caller_component | MEASURED | 0.9363 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/1031-67-5/cold_fit_ms | public_caller_component | MEASURED | 0.8915 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-downloads | hdbscan-downloads/1031-67-5/repeated_fit_ms | public_caller_component | MEASURED | 0.8688 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-downloads/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/509-7-3/cold_fit_ms | public_caller_component | MEASURED | 0.8085 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/509-7-3/repeated_fit_ms | public_caller_component | MEASURED | 0.6957 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/997-13-4/cold_fit_ms | public_caller_component | MEASURED | 0.6857 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/997-13-4/repeated_fit_ms | public_caller_component | MEASURED | 0.6656 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/1031-67-5/cold_fit_ms | public_caller_component | MEASURED | 0.6613 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |
| F15 | apple/hdbscan-linkage | hdbscan-linkage/1031-67-5/repeated_fit_ms | public_caller_component | MEASURED | 0.6698 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/hdbscan-linkage/result.json |

## Campaign notes

- Candidate A/B first; missing GPU opponents after candidate queues and applicable repairs.
- Rentals delete after 1800 seconds genuinely idle, after captured evidence; M3 is retained.
- Existing identity and compilation accepted by owner; no separate validation passes.
- Reused native component executables warm up in a separate process; scored first calls may include JIT. These are not steady-state or full-workload promotion evidence.
- Winners and losers are recorded beside source toggles as sufficient measurements arrive; partial component screens leave defaults unchanged.

## Recorded source decisions

| Candidate / arm | Decision | Source commit | Evidence |
|---|---|---|---|
| F03 / resident | CONFIRMED EXISTING DEFAULT: B/A trajectory 0.5125–0.5565; already enabled | b953bc9a2 | M3 measured heldout and resume results; source _byte_lm_impl.py _is_resident |
| F08 / BWD_NOSYNC | PROMOTED: Apple FAST only; 12-step train B/A0.8766 with matching loss/resume; explicit OFF escape; one scored sample | b953bc9a2 | training/byte_lm_afn.mojo:53; M3 F08/default retained result |
| F08 / fused, views | RETAIN OFF: small or mixed gain; views cold call regressed | b953bc9a2 | M3 F08 independent-arm results; baseline manifests explicitly disable NOSYNC |
| F07, F10, F11 | RETAIN OFF for evaluated experimental arms: mixed/regressing measured cases; see source for each scoped outcome | b953bc9a2 | Inline toggle annotations retain case timing and quality counts |
