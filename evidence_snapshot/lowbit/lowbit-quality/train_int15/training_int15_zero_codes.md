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


## Zero codes of the backward operands: int15-both+attn, forward only, step 3999, mean over seeds [0, 1, 2, 3, 4]

### G^T, the weight-gradient GEMM's left operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.5806 (0.5806) |
| block3.w_down | 0.0000 | 0.0319 (0.0319) |
| block3.w_up | 0.0000 | 0.0629 (0.0629) |
| block3.w_gate | 0.0000 | 0.0977 (0.0977) |
| block3.w_o | 0.0000 | 0.0302 (0.0302) |
| block3.attn_pv | 0.0000 | 0.0228 (0.0228) |
| block3.attn_qk | 0.4962 | 0.4388 (0.7172) |
| block3.w_v | 0.0000 | 0.0054 (0.0054) |
| block3.w_k | 0.0000 | 0.0051 (0.0051) |
| block3.w_q | 0.0078 | 0.0577 (0.0651) |
| block2.w_down | 0.0000 | 0.0022 (0.0022) |
| block2.w_up | 0.0000 | 0.0113 (0.0113) |
| block2.w_gate | 0.0000 | 0.0396 (0.0396) |
| block2.w_o | 0.0000 | 0.0023 (0.0023) |
| block2.attn_pv | 0.0000 | 0.0012 (0.0012) |
| block2.attn_qk | 0.4962 | 0.4777 (0.7368) |
| block2.w_v | 0.0000 | 0.0040 (0.0040) |
| block2.w_k | 0.0000 | 0.0039 (0.0039) |
| block2.w_q | 0.0078 | 0.0177 (0.0254) |
| block1.w_down | 0.0000 | 0.0017 (0.0017) |
| block1.w_up | 0.0000 | 0.0081 (0.0081) |
| block1.w_gate | 0.0000 | 0.0269 (0.0269) |
| block1.w_o | 0.0000 | 0.0017 (0.0017) |
| block1.attn_pv | 0.0000 | 0.0009 (0.0009) |
| block1.attn_qk | 0.4962 | 0.3773 (0.6863) |
| block1.w_v | 0.0000 | 0.0046 (0.0046) |
| block1.w_k | 0.0000 | 0.0041 (0.0041) |
| block1.w_q | 0.0078 | 0.0132 (0.0209) |
| block0.w_down | 0.0000 | 0.0015 (0.0015) |
| block0.w_up | 0.0000 | 0.0078 (0.0078) |
| block0.w_gate | 0.0000 | 0.0146 (0.0146) |
| block0.w_o | 0.0000 | 0.0014 (0.0014) |
| block0.attn_pv | 0.0000 | 0.0008 (0.0008) |
| block0.attn_qk | 0.4962 | 0.0870 (0.5400) |
| block0.w_v | 0.0000 | 0.0021 (0.0021) |
| block0.w_k | 0.0000 | 0.0026 (0.0026) |
| block0.w_q | 0.0078 | 0.0059 (0.0137) |

### A^T, the weight-gradient GEMM's right operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0034 (0.0034) |
| block3.w_up | 0.0000 | 0.0002 (0.0002) |
| block3.w_gate | 0.0000 | 0.0002 (0.0002) |
| block3.w_o | 0.0000 | 0.0002 (0.0002) |
| block3.attn_pv | 0.4961 | 0.3948 (0.6950) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0001 (0.0001) |
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0035 (0.0035) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0002 (0.0002) |
| block2.attn_pv | 0.4961 | 0.4860 (0.7410) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0038 (0.0038) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0002 (0.0002) |
| block1.attn_pv | 0.4961 | 0.3747 (0.6849) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0076 (0.0076) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.4961 | 0.0826 (0.5377) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0001 (0.0001) |
| block0.w_k | 0.0000 | 0.0001 (0.0001) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |

### G, the input-gradient GEMM's left operand (one row per token)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.7725 (0.7725) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0014 (0.0014) |
| block3.w_gate | 0.0000 | 0.0095 (0.0095) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.4962 | 0.3632 (0.6792) |
| block3.w_v | 0.0000 | 0.0002 (0.0002) |
| block3.w_k | 0.0000 | 0.0003 (0.0003) |
| block3.w_q | 0.0078 | 0.0011 (0.0089) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0013 (0.0013) |
| block2.w_gate | 0.0000 | 0.0088 (0.0088) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.4962 | 0.4331 (0.7143) |
| block2.w_v | 0.0000 | 0.0002 (0.0002) |
| block2.w_k | 0.0000 | 0.0003 (0.0003) |
| block2.w_q | 0.0078 | 0.0018 (0.0096) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0013 (0.0013) |
| block1.w_gate | 0.0000 | 0.0071 (0.0071) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.4962 | 0.3321 (0.6635) |
| block1.w_v | 0.0000 | 0.0002 (0.0002) |
| block1.w_k | 0.0000 | 0.0003 (0.0003) |
| block1.w_q | 0.0078 | 0.0019 (0.0097) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0018 (0.0018) |
| block0.w_gate | 0.0000 | 0.0039 (0.0039) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.4962 | 0.0547 (0.5237) |
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
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0000 (0.0000) |
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
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0003 (0.0003) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0000 (0.0000) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |


## Zero codes of the backward operands: int15-both+attn, forward and backward, step 3999, mean over seeds [0, 1, 2, 3, 4]

### G^T, the weight-gradient GEMM's left operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.5665 (0.5665) |
| block3.w_down | 0.0000 | 0.0312 (0.0312) |
| block3.w_up | 0.0000 | 0.0626 (0.0626) |
| block3.w_gate | 0.0000 | 0.0976 (0.0976) |
| block3.w_o | 0.0000 | 0.0293 (0.0293) |
| block3.attn_pv | 0.0000 | 0.0215 (0.0215) |
| block3.attn_qk | 0.4962 | 0.4400 (0.7179) |
| block3.w_v | 0.0001 | 0.0048 (0.0049) |
| block3.w_k | 0.0000 | 0.0043 (0.0043) |
| block3.w_q | 0.0078 | 0.0571 (0.0644) |
| block2.w_down | 0.0000 | 0.0019 (0.0019) |
| block2.w_up | 0.0000 | 0.0104 (0.0104) |
| block2.w_gate | 0.0000 | 0.0383 (0.0383) |
| block2.w_o | 0.0000 | 0.0020 (0.0020) |
| block2.attn_pv | 0.0000 | 0.0012 (0.0012) |
| block2.attn_qk | 0.4962 | 0.4818 (0.7389) |
| block2.w_v | 0.0001 | 0.0034 (0.0035) |
| block2.w_k | 0.0000 | 0.0036 (0.0036) |
| block2.w_q | 0.0078 | 0.0166 (0.0243) |
| block1.w_down | 0.0000 | 0.0015 (0.0015) |
| block1.w_up | 0.0000 | 0.0076 (0.0076) |
| block1.w_gate | 0.0000 | 0.0253 (0.0253) |
| block1.w_o | 0.0000 | 0.0015 (0.0015) |
| block1.attn_pv | 0.0000 | 0.0008 (0.0008) |
| block1.attn_qk | 0.4962 | 0.3688 (0.6820) |
| block1.w_v | 0.0001 | 0.0041 (0.0042) |
| block1.w_k | 0.0000 | 0.0040 (0.0040) |
| block1.w_q | 0.0078 | 0.0117 (0.0194) |
| block0.w_down | 0.0000 | 0.0014 (0.0014) |
| block0.w_up | 0.0000 | 0.0075 (0.0075) |
| block0.w_gate | 0.0000 | 0.0141 (0.0141) |
| block0.w_o | 0.0000 | 0.0013 (0.0013) |
| block0.attn_pv | 0.0000 | 0.0008 (0.0008) |
| block0.attn_qk | 0.4962 | 0.0849 (0.5389) |
| block0.w_v | 0.0001 | 0.0018 (0.0019) |
| block0.w_k | 0.0000 | 0.0026 (0.0026) |
| block0.w_q | 0.0078 | 0.0059 (0.0136) |

### A^T, the weight-gradient GEMM's right operand (rows over the tokens)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.0002 (0.0002) |
| block3.w_down | 0.0000 | 0.0033 (0.0033) |
| block3.w_up | 0.0000 | 0.0002 (0.0002) |
| block3.w_gate | 0.0000 | 0.0002 (0.0002) |
| block3.w_o | 0.0000 | 0.0002 (0.0002) |
| block3.attn_pv | 0.4961 | 0.3965 (0.6959) |
| block3.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block3.w_v | 0.0000 | 0.0001 (0.0001) |
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0034 (0.0034) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0002 (0.0002) |
| block2.attn_pv | 0.4961 | 0.4881 (0.7421) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0001 (0.0001) |
| block2.w_k | 0.0000 | 0.0001 (0.0001) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0037 (0.0037) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0002 (0.0002) |
| block1.attn_pv | 0.4961 | 0.3666 (0.6808) |
| block1.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block1.w_v | 0.0000 | 0.0001 (0.0001) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0076 (0.0076) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0002 (0.0002) |
| block0.attn_pv | 0.4961 | 0.0814 (0.5371) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0002 (0.0002) |
| block0.w_k | 0.0000 | 0.0002 (0.0002) |
| block0.w_q | 0.0000 | 0.0002 (0.0002) |

### G, the input-gradient GEMM's left operand (one row per token)

Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).

| product | exactly zero as float32 | int15 |
|---|---|---|
| lm_head | 0.0000 | 0.7694 (0.7694) |
| block3.w_down | 0.0000 | 0.0001 (0.0001) |
| block3.w_up | 0.0000 | 0.0014 (0.0014) |
| block3.w_gate | 0.0000 | 0.0097 (0.0097) |
| block3.w_o | 0.0000 | 0.0001 (0.0001) |
| block3.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block3.attn_qk | 0.4962 | 0.3638 (0.6794) |
| block3.w_v | 0.0001 | 0.0002 (0.0003) |
| block3.w_k | 0.0000 | 0.0003 (0.0003) |
| block3.w_q | 0.0078 | 0.0010 (0.0088) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0013 (0.0013) |
| block2.w_gate | 0.0000 | 0.0088 (0.0088) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.4962 | 0.4387 (0.7172) |
| block2.w_v | 0.0001 | 0.0002 (0.0003) |
| block2.w_k | 0.0000 | 0.0003 (0.0003) |
| block2.w_q | 0.0078 | 0.0018 (0.0096) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0014 (0.0014) |
| block1.w_gate | 0.0000 | 0.0072 (0.0072) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.4962 | 0.3243 (0.6595) |
| block1.w_v | 0.0001 | 0.0002 (0.0003) |
| block1.w_k | 0.0000 | 0.0003 (0.0003) |
| block1.w_q | 0.0078 | 0.0017 (0.0095) |
| block0.w_down | 0.0000 | 0.0001 (0.0001) |
| block0.w_up | 0.0000 | 0.0018 (0.0018) |
| block0.w_gate | 0.0000 | 0.0040 (0.0040) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block0.attn_qk | 0.4962 | 0.0538 (0.5233) |
| block0.w_v | 0.0001 | 0.0001 (0.0002) |
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
| block3.w_k | 0.0000 | 0.0001 (0.0001) |
| block3.w_q | 0.0000 | 0.0001 (0.0001) |
| block2.w_down | 0.0000 | 0.0001 (0.0001) |
| block2.w_up | 0.0000 | 0.0001 (0.0001) |
| block2.w_gate | 0.0000 | 0.0001 (0.0001) |
| block2.w_o | 0.0000 | 0.0001 (0.0001) |
| block2.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block2.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block2.w_v | 0.0000 | 0.0000 (0.0000) |
| block2.w_k | 0.0000 | 0.0000 (0.0000) |
| block2.w_q | 0.0000 | 0.0001 (0.0001) |
| block1.w_down | 0.0000 | 0.0001 (0.0001) |
| block1.w_up | 0.0000 | 0.0001 (0.0001) |
| block1.w_gate | 0.0000 | 0.0001 (0.0001) |
| block1.w_o | 0.0000 | 0.0001 (0.0001) |
| block1.attn_pv | 0.0000 | 0.0001 (0.0001) |
| block1.attn_qk | 0.0000 | 0.0000 (0.0000) |
| block1.w_v | 0.0000 | 0.0000 (0.0000) |
| block1.w_k | 0.0000 | 0.0001 (0.0001) |
| block1.w_q | 0.0000 | 0.0001 (0.0001) |
| block0.w_down | 0.0000 | 0.0002 (0.0002) |
| block0.w_up | 0.0000 | 0.0001 (0.0001) |
| block0.w_gate | 0.0000 | 0.0001 (0.0001) |
| block0.w_o | 0.0000 | 0.0001 (0.0001) |
| block0.attn_pv | 0.0000 | 0.0000 (0.0000) |
| block0.attn_qk | 0.0000 | 0.0001 (0.0001) |
| block0.w_v | 0.0000 | 0.0000 (0.0000) |
| block0.w_k | 0.0000 | 0.0000 (0.0000) |
| block0.w_q | 0.0000 | 0.0001 (0.0001) |

