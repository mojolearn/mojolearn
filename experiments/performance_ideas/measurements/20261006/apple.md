# Candidate A/B measurements

One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Defaults remain unchanged.

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
| F10 | fast | PARTIAL_MEASUREMENTS_RETAINED | 74 |
| F11 | fast | PENDING_MEASUREMENT | 0 |
| F12 | fast | PENDING_MEASUREMENT | 0 |
| F13 | fast | PENDING_MEASUREMENT | 0 |
| F14 | fast | PENDING_MEASUREMENT | 0 |
| F15 | fast | PENDING_MEASUREMENT | 0 |
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
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/forward_ms | public_caller_component | MEASURED | 0.8065 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6187 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_a_near_zero_b3_l64_d32/prefix_ms | public_caller_component | MEASURED | 0.7255 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.6362 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6761 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_dt_limit_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.7614 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/forward_ms | public_caller_component | MEASURED | 0.6877 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6037 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_gate_saturation_b1_l8_d64/prefix_ms | public_caller_component | MEASURED | 0.6032 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.5985 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6317 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_signed_zeros_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.6204 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.6378 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6078 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_adv_softplus_band_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 0.6273 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l1_d32/forward_ms | public_caller_component | MEASURED | 0.5931 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/forward_ms | public_caller_component | MEASURED | 0.6745 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6332 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l256_d32/prefix_ms | public_caller_component | MEASURED | 0.6413 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/forward_ms | public_caller_component | MEASURED | 0.8485 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.7187 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b1_l257_d64/prefix_ms | public_caller_component | MEASURED | 0.7363 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/forward_ms | public_caller_component | MEASURED | 0.6787 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6677 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b2_l4_d32/prefix_ms | public_caller_component | MEASURED | 0.6872 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/forward_ms | public_caller_component | MEASURED | 0.5911 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.6066 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_base_b3_l4_d64/prefix_ms | public_caller_component | MEASURED | 0.5519 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/forward_ms | public_caller_component | MEASURED | 0.6309 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5969 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_b2_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.6557 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.5691 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5495 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row0_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.5848 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.4967 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.5679 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/arena | arena/m2_comp_row1_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.4768 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/forward_ms | public_caller_component | MEASURED | 2.2935 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0240 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_a_near_zero_b3_l64_d32/prefix_ms | public_caller_component | MEASURED | 1.0669 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 0.9880 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9509 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_dt_limit_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.2725 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/forward_ms | public_caller_component | MEASURED | 1.0044 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0154 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_gate_saturation_b1_l8_d64/prefix_ms | public_caller_component | MEASURED | 1.0277 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 1.0081 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0246 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_signed_zeros_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.0255 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/forward_ms | public_caller_component | MEASURED | 1.0073 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0221 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_adv_softplus_band_b2_l8_d32/prefix_ms | public_caller_component | MEASURED | 1.0028 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l1_d32/forward_ms | public_caller_component | MEASURED | 1.0162 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/forward_ms | public_caller_component | MEASURED | 0.9747 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9833 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l256_d32/prefix_ms | public_caller_component | MEASURED | 0.9368 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/forward_ms | public_caller_component | MEASURED | 0.9629 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9340 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b1_l257_d64/prefix_ms | public_caller_component | MEASURED | 0.9274 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/forward_ms | public_caller_component | MEASURED | 1.0120 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0009 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b2_l4_d32/prefix_ms | public_caller_component | MEASURED | 1.0532 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/forward_ms | public_caller_component | MEASURED | 1.0151 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9878 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_base_b3_l4_d64/prefix_ms | public_caller_component | MEASURED | 1.0104 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/forward_ms | public_caller_component | MEASURED | 1.1754 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 1.0209 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_b2_l257_d32/prefix_ms | public_caller_component | MEASURED | 1.1263 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.9769 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9441 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row0_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 1.1788 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/forward_ms | public_caller_component | MEASURED | 0.8446 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/decode_ms_trajectory_total | public_caller_component | MEASURED | 0.9668 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |
| F10 | apple/default | default/m2_comp_row1_b1_l257_d32/prefix_ms | public_caller_component | MEASURED | 0.8310 | /Users/andrewhendel/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default/result.json |

## Campaign notes

- Candidate A/B first; missing GPU opponents after candidate queues and applicable repairs.
- Rentals delete after 1800 seconds genuinely idle, after captured evidence; M3 is retained.
- Existing identity and compilation accepted by owner; no separate validation passes.
- Reused native component executables warm up in a separate process; scored first calls may include JIT. These are not steady-state or full-workload promotion evidence.
- Winners and losers are recorded beside source toggles as sufficient measurements arrive; partial component screens leave defaults unchanged.
