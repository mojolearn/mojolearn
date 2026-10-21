## Zero codes of the backward operands: fp32.v1 baseline, step 3999, mean over seeds [0, 1, 2, 3, 4]

### G^T, the weight-gradient GEMM's left operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.5883 (0.5883) |
| block3.w_down | 0.0000 | 0.0346 (0.0346) |
| block3.w_up | 0.0000 | 0.0671 (0.0671) |
| block3.w_gate | 0.0000 | 0.1022 (0.1022) |
| block3.w_o | 0.0000 | 0.0333 (0.0333) |
| block3.attn_pv | 0.0000 | 0.0257 (0.0257) |
| block3.attn_qk | 0.4962 | 0.4410 (0.7183) |
| block3.w_v | 0.0000 | 0.0057 (0.0057) |
| block3.w_k | 0.0000 | 0.0055 (0.0055) |
| block3.w_q | 0.0078 | 0.0643 (0.0716) |
| block2.w_down | 0.0000 | 0.0025 (0.0025) |
| block2.w_up | 0.0000 | 0.0116 (0.0116) |
| block2.w_gate | 0.0000 | 0.0397 (0.0397) |
| block2.w_o | 0.0000 | 0.0025 (0.0025) |
| block2.attn_pv | 0.0000 | 0.0015 (0.0015) |
| block2.attn_qk | 0.4962 | 0.4873 (0.7417) |
| block2.w_v | 0.0000 | 0.0042 (0.0042) |
| block2.w_k | 0.0000 | 0.0040 (0.0040) |
| block2.w_q | 0.0078 | 0.0197 (0.0273) |
| block1.w_down | 0.0000 | 0.0019 (0.0019) |
| block1.w_up | 0.0000 | 0.0083 (0.0083) |
| block1.w_gate | 0.0000 | 0.0276 (0.0276) |
| block1.w_o | 0.0000 | 0.0018 (0.0018) |
| block1.attn_pv | 0.0000 | 0.0010 (0.0010) |
| block1.attn_qk | 0.4962 | 0.3716 (0.6834) |
| block1.w_v | 0.0000 | 0.0048 (0.0048) |
| block1.w_k | 0.0000 | 0.0042 (0.0042) |
| block1.w_q | 0.0078 | 0.0125 (0.0202) |
| block0.w_down | 0.0000 | 0.0016 (0.0016) |
| block0.w_up | 0.0000 | 0.0082 (0.0082) |
| block0.w_gate | 0.0000 | 0.0150 (0.0150) |
| block0.w_o | 0.0000 | 0.0016 (0.0016) |
| block0.attn_pv | 0.0000 | 0.0009 (0.0009) |
| block0.attn_qk | 0.4962 | 0.0842 (0.5386) |
| block0.w_v | 0.0000 | 0.0023 (0.0023) |
| block0.w_k | 0.0000 | 0.0031 (0.0031) |
| block0.w_q | 0.0078 | 0.0067 (0.0144) |

### A^T, the weight-gradient GEMM's right operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0034 (0.0034) |
| block3.w_up | 0.0000 | 0.0001 (0.0001) |
| block3.w_gate | 0.0000 | 0.0001 (0.0001) |
| block3.w_o | 0.0000 | 0.0002 (0.0002) |
| block3.attn_pv | 0.4961 | 0.3924 (0.6939) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0001 (0.0001) |
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0034 (0.0034) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0002 (0.0002) |
| block2.attn_pv | 0.4961 | 0.4936 (0.7448) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0038 (0.0038) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0002 (0.0002) |
| block1.attn_pv | 0.4961 | 0.3685 (0.6818) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0078 (0.0078) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.4961 | 0.0795 (0.5361) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0001 (0.0001) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |

### G, the input-gradient GEMM's left operand (one row per token)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.7739 (0.7739) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0014 (0.0014) |
| block3.w_gate | 0.0000 | 0.0096 (0.0096) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.4962 | 0.3593 (0.6772) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0003 (0.0003) |
| block3.w_q | 0.0078 | 0.0010 (0.0088) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0013 (0.0013) |
| block2.w_gate | 0.0000 | 0.0089 (0.0089) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.4962 | 0.4391 (0.7174) |
| block2.w_v | 0.0000 | 0.0002 (0.0002) |
| block2.w_k | 0.0000 | 0.0002 (0.0002) |
| block2.w_q | 0.0078 | 0.0019 (0.0097) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0013 (0.0013) |
| block1.w_gate | 0.0000 | 0.0073 (0.0073) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.4962 | 0.3273 (0.6611) |
| block1.w_v | 0.0000 | 0.0002 (0.0002) |
| block1.w_k | 0.0000 | 0.0003 (0.0003) |
| block1.w_q | 0.0078 | 0.0017 (0.0095) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0018 (0.0018) |
| block0.w_gate | 0.0000 | 0.0040 (0.0040) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.4962 | 0.0530 (0.5228) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0002 (0.0002) |
| block0.w_q | 0.0078 | 0.0005 (0.0083) |

### B^T, the input-gradient GEMM's right operand

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0001 (0.0001) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0001 (0.0001) |
| block3.w_gate | 0.0000 | 0.0001 (0.0001) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0001 (0.0001) |
| block3.w_k | 0.0000 | 0.0000 (0.0000) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0000 (0.0000) |
| block2.w_k | 0.0000 | 0.0000 (0.0000) |
| block2.w_q | 0.0000 | 0.0000 (0.0000) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0000 (0.0000) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0001 (0.0001) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |


## Zero codes of the backward operands: F1-pv32, forward only, step 3999, mean over seeds [0, 1, 2, 3, 4]

### G^T, the weight-gradient GEMM's left operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.5635 (0.5635) |
| block3.w_down | 0.0000 | 0.0317 (0.0317) |
| block3.w_up | 0.0000 | 0.0641 (0.0641) |
| block3.w_gate | 0.0000 | 0.0989 (0.0989) |
| block3.w_o | 0.0000 | 0.0295 (0.0295) |
| block3.attn_pv | 0.0000 | 0.0216 (0.0216) |
| block3.attn_qk | 0.4962 | 0.4420 (0.7189) |
| block3.w_v | 0.0000 | 0.0050 (0.0050) |
| block3.w_k | 0.0000 | 0.0046 (0.0046) |
| block3.w_q | 0.0078 | 0.0586 (0.0659) |
| block2.w_down | 0.0000 | 0.0021 (0.0021) |
| block2.w_up | 0.0000 | 0.0110 (0.0110) |
| block2.w_gate | 0.0000 | 0.0385 (0.0385) |
| block2.w_o | 0.0000 | 0.0021 (0.0021) |
| block2.attn_pv | 0.0000 | 0.0011 (0.0011) |
| block2.attn_qk | 0.4962 | 0.4846 (0.7403) |
| block2.w_v | 0.0000 | 0.0037 (0.0037) |
| block2.w_k | 0.0000 | 0.0037 (0.0037) |
| block2.w_q | 0.0078 | 0.0176 (0.0253) |
| block1.w_down | 0.0000 | 0.0015 (0.0015) |
| block1.w_up | 0.0000 | 0.0079 (0.0079) |
| block1.w_gate | 0.0000 | 0.0261 (0.0261) |
| block1.w_o | 0.0000 | 0.0015 (0.0015) |
| block1.attn_pv | 0.0000 | 0.0008 (0.0008) |
| block1.attn_qk | 0.4962 | 0.3642 (0.6796) |
| block1.w_v | 0.0000 | 0.0042 (0.0042) |
| block1.w_k | 0.0000 | 0.0041 (0.0041) |
| block1.w_q | 0.0078 | 0.0122 (0.0199) |
| block0.w_down | 0.0000 | 0.0014 (0.0014) |
| block0.w_up | 0.0000 | 0.0076 (0.0076) |
| block0.w_gate | 0.0000 | 0.0143 (0.0143) |
| block0.w_o | 0.0000 | 0.0013 (0.0013) |
| block0.attn_pv | 0.0000 | 0.0007 (0.0007) |
| block0.attn_qk | 0.4962 | 0.0911 (0.5421) |
| block0.w_v | 0.0000 | 0.0019 (0.0019) |
| block0.w_k | 0.0000 | 0.0029 (0.0029) |
| block0.w_q | 0.0078 | 0.0064 (0.0141) |

### A^T, the weight-gradient GEMM's right operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0035 (0.0035) |
| block3.w_up | 0.0000 | 0.0002 (0.0002) |
| block3.w_gate | 0.0000 | 0.0002 (0.0002) |
| block3.w_o | 0.0000 | 0.0002 (0.0002) |
| block3.attn_pv | 0.4961 | 0.3969 (0.6961) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0002 (0.0002) |
| block3.w_q | 0.0000 | 0.0002 (0.0002) |
| block2.w_down | 0.0000 | 0.0036 (0.0036) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0002 (0.0002) |
| block2.attn_pv | 0.4961 | 0.4914 (0.7437) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0037 (0.0037) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0002 (0.0002) |
| block1.attn_pv | 0.4961 | 0.3617 (0.6783) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0077 (0.0077) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.4961 | 0.0887 (0.5408) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0000 (0.0000) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0000 (0.0000) |

### G, the input-gradient GEMM's left operand (one row per token)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.7691 (0.7691) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0014 (0.0014) |
| block3.w_gate | 0.0000 | 0.0096 (0.0096) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.4962 | 0.3649 (0.6800) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0003 (0.0003) |
| block3.w_q | 0.0078 | 0.0011 (0.0089) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0013 (0.0013) |
| block2.w_gate | 0.0000 | 0.0087 (0.0087) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.4962 | 0.4392 (0.7174) |
| block2.w_v | 0.0000 | 0.0002 (0.0002) |
| block2.w_k | 0.0000 | 0.0002 (0.0002) |
| block2.w_q | 0.0078 | 0.0017 (0.0095) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0013 (0.0013) |
| block1.w_gate | 0.0000 | 0.0071 (0.0071) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.4962 | 0.3201 (0.6575) |
| block1.w_v | 0.0000 | 0.0002 (0.0002) |
| block1.w_k | 0.0000 | 0.0003 (0.0003) |
| block1.w_q | 0.0078 | 0.0016 (0.0094) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0017 (0.0017) |
| block0.w_gate | 0.0000 | 0.0039 (0.0039) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.4962 | 0.0576 (0.5252) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0002 (0.0002) |
| block0.w_q | 0.0078 | 0.0005 (0.0083) |

### B^T, the input-gradient GEMM's right operand

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0000 (0.0000) |
| block3.w_gate | 0.0000 | 0.0001 (0.0001) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0000 (0.0000) |
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0000 (0.0000) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0000 (0.0000) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.0000 | 0.0002 (0.0002) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |


## Zero codes of the backward operands: F1-pv32, forward and backward, step 3999, mean over seeds [0, 1, 2, 3, 4]

### G^T, the weight-gradient GEMM's left operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.5816 (0.5816) |
| block3.w_down | 0.0000 | 0.0342 (0.0342) |
| block3.w_up | 0.0000 | 0.0661 (0.0661) |
| block3.w_gate | 0.0000 | 0.1018 (0.1018) |
| block3.w_o | 0.0000 | 0.0325 (0.0325) |
| block3.attn_pv | 0.0000 | 0.0249 (0.0249) |
| block3.attn_qk | 0.4962 | 0.4489 (0.7223) |
| block3.w_v | 0.0000 | 0.0057 (0.0057) |
| block3.w_k | 0.0000 | 0.0052 (0.0052) |
| block3.w_q | 0.0078 | 0.0623 (0.0696) |
| block2.w_down | 0.0000 | 0.0024 (0.0024) |
| block2.w_up | 0.0000 | 0.0117 (0.0117) |
| block2.w_gate | 0.0000 | 0.0403 (0.0403) |
| block2.w_o | 0.0000 | 0.0024 (0.0024) |
| block2.attn_pv | 0.0000 | 0.0015 (0.0015) |
| block2.attn_qk | 0.4962 | 0.4698 (0.7329) |
| block2.w_v | 0.0000 | 0.0041 (0.0041) |
| block2.w_k | 0.0000 | 0.0043 (0.0043) |
| block2.w_q | 0.0078 | 0.0183 (0.0259) |
| block1.w_down | 0.0000 | 0.0019 (0.0019) |
| block1.w_up | 0.0000 | 0.0084 (0.0084) |
| block1.w_gate | 0.0000 | 0.0272 (0.0272) |
| block1.w_o | 0.0000 | 0.0018 (0.0018) |
| block1.attn_pv | 0.0000 | 0.0011 (0.0011) |
| block1.attn_qk | 0.4962 | 0.3745 (0.6849) |
| block1.w_v | 0.0000 | 0.0047 (0.0047) |
| block1.w_k | 0.0000 | 0.0041 (0.0041) |
| block1.w_q | 0.0078 | 0.0123 (0.0200) |
| block0.w_down | 0.0000 | 0.0016 (0.0016) |
| block0.w_up | 0.0000 | 0.0081 (0.0081) |
| block0.w_gate | 0.0000 | 0.0146 (0.0146) |
| block0.w_o | 0.0000 | 0.0016 (0.0016) |
| block0.attn_pv | 0.0000 | 0.0011 (0.0011) |
| block0.attn_qk | 0.4962 | 0.0923 (0.5426) |
| block0.w_v | 0.0000 | 0.0022 (0.0022) |
| block0.w_k | 0.0000 | 0.0032 (0.0032) |
| block0.w_q | 0.0078 | 0.0066 (0.0144) |

### A^T, the weight-gradient GEMM's right operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0033 (0.0033) |
| block3.w_up | 0.0000 | 0.0001 (0.0001) |
| block3.w_gate | 0.0000 | 0.0001 (0.0001) |
| block3.w_o | 0.0000 | 0.0002 (0.0002) |
| block3.attn_pv | 0.4961 | 0.4019 (0.6986) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0002 (0.0002) |
| block3.w_q | 0.0000 | 0.0002 (0.0002) |
| block2.w_down | 0.0000 | 0.0035 (0.0035) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0002 (0.0002) |
| block2.attn_pv | 0.4961 | 0.4726 (0.7342) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0038 (0.0038) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0002 (0.0002) |
| block1.attn_pv | 0.4961 | 0.3711 (0.6831) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0079 (0.0079) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.4961 | 0.0888 (0.5409) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0000 (0.0000) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0000 (0.0000) |

### G, the input-gradient GEMM's left operand (one row per token)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.7698 (0.7698) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0014 (0.0014) |
| block3.w_gate | 0.0000 | 0.0097 (0.0097) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.4962 | 0.3702 (0.6827) |
| block3.w_v | 0.0000 | 0.0003 (0.0003) |
| block3.w_k | 0.0000 | 0.0003 (0.0003) |
| block3.w_q | 0.0078 | 0.0011 (0.0089) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0013 (0.0013) |
| block2.w_gate | 0.0000 | 0.0088 (0.0088) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.4962 | 0.4219 (0.7087) |
| block2.w_v | 0.0000 | 0.0002 (0.0002) |
| block2.w_k | 0.0000 | 0.0003 (0.0003) |
| block2.w_q | 0.0078 | 0.0018 (0.0096) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0013 (0.0013) |
| block1.w_gate | 0.0000 | 0.0073 (0.0073) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.4962 | 0.3291 (0.6620) |
| block1.w_v | 0.0000 | 0.0002 (0.0002) |
| block1.w_k | 0.0000 | 0.0002 (0.0002) |
| block1.w_q | 0.0078 | 0.0017 (0.0095) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0018 (0.0018) |
| block0.w_gate | 0.0000 | 0.0039 (0.0039) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.4962 | 0.0583 (0.5255) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0002 (0.0002) |
| block0.w_q | 0.0078 | 0.0005 (0.0083) |

### B^T, the input-gradient GEMM's right operand

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0001 (0.0001) |
| block3.w_gate | 0.0000 | 0.0001 (0.0001) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0000 (0.0000) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0000 (0.0000) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0000 (0.0000) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0000 (0.0000) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0000 (0.0000) |

