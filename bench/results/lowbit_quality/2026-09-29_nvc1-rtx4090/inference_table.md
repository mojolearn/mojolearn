text enwik8: corpus/enwik8/input.txt bytes [99000022, 99374116), sha256 a2bf2dea5108207b0653ffb0aa46cc550ba9cd6e67e82ff2d1123fa890c0880d; 0 invalid UTF-8 bytes in them; 102400 ids in 200 windows of 512, ids sha256 d5be2324ffc522bc306eaa9baa63e626c9fe8df690eec7c552b4117f1ad8a56f; 102200 scored positions; baseline fp32.v1 perplexity 7.937106; numerical floor +9.93e-09
text pile_github: corpus/pile_github/input.txt bytes [96000030, 96240053), sha256 91cc52d4b375dd0b7a8b0a256f1b5b5261967eba025a44ce3b6072633e3919ef; 0 invalid UTF-8 bytes in them; 102400 ids in 200 windows of 512, ids sha256 2d337ff1deeb3846125e558b8521f9f47f92e51abf03396d083255a9827a2f86; 102200 scored positions; baseline fp32.v1 perplexity 3.361991; numerical floor +2.12e-08

THE INTERVAL: within each of the 200 windows of a text the per-position difference of nll (arm minus baseline) is averaged; the interval is the mean of those window means plus and minus 1.96 of their standard error, 95 percent under a normal approximation, mapped through exp(x) - 1; the table prints its upper end. It bounds the sampling error on THESE texts and says nothing about other text or tasks.
TOP-1 AGREEMENT: the share of scored positions, each with the true context supplied, where the arm's top token equals the baseline's. It is not a rate of changed tokens in generated text: free-running generation diverges from the first changed token on.

| profile | weights | activations | attention products | enwik8: change (upper end), top-1 agreement | pile_github: change (upper end), top-1 agreement | verdict |
|---|---|---|---|---|---|---|
| bf16f32.v1 | bf16 | fp32 | no | +0.0000% (to +0.0000%), top-1 1.0000 | +0.0000% (to +0.0000%), top-1 1.0000 | PASS, bit-equal to the baseline |
| bf16-both | bf16 | bf16 | no | +0.0093% (to +0.0168%), top-1 0.9962 | +0.0000% (to +0.0061%), top-1 0.9976 | PASS |
| int8i32.v1 | int8 | int8 | no | +32.1772% (to +34.6155%), top-1 0.7659 | +28.8948% (to +31.8559%), top-1 0.8501 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int15-both | int15 | int15 | no | -0.0027% (to +0.0017%), top-1 0.9979 | +0.0046% (to +0.0089%), top-1 0.9986 | PASS |
| int15w-int8a | int15 | int8 | no | +27.2758% (to +29.2969%), top-1 0.7844 | +22.5919% (to +24.8538%), top-1 0.8648 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| bf16-both+attn | bf16 | bf16 | yes | +0.0085% (to +0.0179%), top-1 0.9953 | +0.0127% (to +0.0224%), top-1 0.9967 | PASS |
| int8i32.v1+attn | int8 | int8 | yes | +31.6108% (to +33.9116%), top-1 0.7653 | +27.8914% (to +30.6804%), top-1 0.8489 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int15-both+attn | int15 | int15 | yes | -0.0015% (to +0.0033%), top-1 0.9980 | +0.0055% (to +0.0099%), top-1 0.9987 | PASS |
| int15w-int8a+attn | int15 | int8 | yes | +27.5892% (to +29.6136%), top-1 0.7825 | +23.3498% (to +25.6442%), top-1 0.8626 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-fp32a | int8 | fp32 | no | +1.5935% (to +1.7110%), top-1 0.9327 | +1.0552% (to +1.1562%), top-1 0.9630 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| fp32w-int8a | fp32 | int8 | no | +27.1015% (to +29.0833%), top-1 0.7847 | +22.5666% (to +24.7941%), top-1 0.8658 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int15w-fp32a | int15 | fp32 | no | -0.0007% (to -0.0002%), top-1 0.9998 | not measured | OWED (one text only) |
| fp32w-int15a | fp32 | int15 | no | -0.0001% (to +0.0048%), top-1 0.9981 | not measured | OWED (one text only) |
| int8w-int15a | int8 | int15 | no | +1.5932% (to +1.7108%), top-1 0.9327 | +1.0575% (to +1.1588%), top-1 0.9629 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int12a | int8 | int12 | no | +1.6642% (to +1.7849%), top-1 0.9317 | +1.1312% (to +1.2361%), top-1 0.9625 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int10a | int8 | int10 | no | +3.8966% (to +4.1019%), top-1 0.9199 | +3.2618% (to +3.4575%), top-1 0.9566 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int8s1a | int8 | int8s1 | no | +213.8971% (to +221.4520%), top-1 0.5325 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int8s2a | int8 | int8s2 | no | +559630.9725% (to +620651.4059%), top-1 0.0266 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8m-both | int8m | int8m | no | +6.1208% (to +6.4305%), top-1 0.8840 | +4.8054% (to +5.0657%), top-1 0.9317 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-fp32head | int8 | int8 | no; lm_head=fp32/fp32 | +25.2703% (to +27.4545%), top-1 0.8084 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-fp32mlp | int8 | int8 | no; gate_proj=fp32/fp32, up_proj=fp32/fp32, down_proj=fp32/fp32 | +6.4975% (to +6.8266%), top-1 0.8639 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-fp32down | int8 | int8 | no; down_proj=fp32/fp32 | +30.3730% (to +32.7833%), top-1 0.7750 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-fp32attnproj | int8 | int8 | no; q_proj=fp32/fp32, k_proj=fp32/fp32, v_proj=fp32/fp32, o_proj=fp32/fp32 | +34.9608% (to +37.7741%), top-1 0.7667 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-int15down | int8 | int8 | no; down_proj=int15/int15 | +30.8507% (to +33.3011%), top-1 0.7754 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1-int15head-down | int8 | int8 | no; down_proj=int15/int15, lm_head=int15/int15 | +23.5794% (to +25.7448%), top-1 0.8222 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| fp32+attn-int8 | fp32 | fp32 | no; attn_qk=int8/int8, attn_pv=int8/int8 | +0.5947% (to +0.6657%), top-1 0.9636 | not measured | OWED (one text only); dropped 2026-09-29, Andrew (not offered by the flag) |
| fp32+attn-int15 | fp32 | fp32 | no; attn_qk=int15/int15, attn_pv=int15/int15 | +0.0011% (to +0.0020%), top-1 0.9995 | not measured | OWED (one text only) |
| int8i32.v1+qk | int8 | int8 | no; attn_qk=int8/int8 | +32.6264% (to +35.1070%), top-1 0.7646 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1+pv | int8 | int8 | no; attn_pv=int8/int8 | +31.6053% (to +33.9066%), top-1 0.7672 | not measured | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int8a | int8 | int8 | no | +32.1772% (to +34.6155%), top-1 0.7659 | +28.8948% (to +31.8559%), top-1 0.8501 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int10w-int15a | int10 | int15 | no | +0.0711% (to +0.1022%), top-1 0.9799 | +0.0680% (to +0.0899%), top-1 0.9902 | PASS |
| int10w-int12a | int10 | int12 | no | +0.2547% (to +0.3023%), top-1 0.9755 | +0.2187% (to +0.2646%), top-1 0.9871 | PASS |
| int10w-int10a | int10 | int10 | no | +2.3524% (to +2.5206%), top-1 0.9519 | +2.2926% (to +2.4579%), top-1 0.9722 | MISS |
| int10w-int8a | int10 | int8 | no | +27.3662% (to +29.4088%), top-1 0.7833 | +22.9826% (to +25.2779%), top-1 0.8630 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int12w-int15a | int12 | int15 | no | +0.0082% (to +0.0159%), top-1 0.9953 | -0.0064% (to +0.0004%), top-1 0.9973 | PASS |
| int12w-int12a | int12 | int12 | no | +0.1803% (to +0.2163%), top-1 0.9867 | +0.1242% (to +0.1612%), top-1 0.9920 | PASS |
| int12w-int10a | int12 | int10 | no | +2.2949% (to +2.4591%), top-1 0.9559 | +2.3255% (to +2.4727%), top-1 0.9749 | MISS |
| int12w-int8a | int12 | int8 | no | +26.8771% (to +28.8954%), top-1 0.7856 | +22.2292% (to +24.4248%), top-1 0.8654 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int15w-int15a | int15 | int15 | no | -0.0027% (to +0.0017%), top-1 0.9979 | +0.0046% (to +0.0089%), top-1 0.9986 | PASS |
| int15w-int12a | int15 | int12 | no | +0.1776% (to +0.2123%), top-1 0.9875 | +0.1133% (to +0.1493%), top-1 0.9921 | PASS |
| int15w-int10a | int15 | int10 | no | +2.3114% (to +2.4727%), top-1 0.9563 | +2.2947% (to +2.4428%), top-1 0.9743 | MISS |
| int8mw-int15a | int8m | int15 | no | +0.6184% (to +0.6910%), top-1 0.9572 | +0.4856% (to +0.5505%), top-1 0.9758 | PASS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8s1w-int15a | int8s1 | int15 | no | +79.6025% (to +82.9192%), top-1 0.6680 | +77.9630% (to +84.9203%), top-1 0.7521 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int15a-fp32head | int8 | int15 | no; lm_head=fp32/fp32 | +0.7069% (to +0.7867%), top-1 0.9628 | +0.5312% (to +0.6032%), top-1 0.9781 | PASS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int15a-int15head | int8 | int15 | no; lm_head=int15/int15 | +0.7075% (to +0.7874%), top-1 0.9626 | +0.5329% (to +0.6049%), top-1 0.9781 | PASS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int15a-int15mlp | int8 | int15 | no; gate_proj=int15/int15, up_proj=int15/int15, down_proj=int15/int15 | +1.1123% (to +1.2078%), top-1 0.9380 | +0.6430% (to +0.7290%), top-1 0.9680 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8w-int15a-int15attnproj | int8 | int15 | no; q_proj=int15/int15, k_proj=int15/int15, v_proj=int15/int15, o_proj=int15/int15 | +1.4556% (to +1.5661%), top-1 0.9358 | +0.9637% (to +1.0525%), top-1 0.9650 | MISS; dropped 2026-09-29, Andrew (not offered by the flag) |

WIDTH TABLE: rows weight width, columns activation width; each cell: int8 pieces of the weight x pieces of the activation = products, then per text the change (upper end), then the verdict

| weight \ activation | 8 bits (1 piece) | 10 bits (2 pieces) | 12 bits (2 pieces) | 15 bits (2 pieces) |
|---|---|---|---|---|
| 8 bits (1 piece) | 1x1 = 1 product; enwik8 +32.177% (+34.615%); pile_github +28.895% (+31.856%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 1x2 = 2 products; enwik8 +3.897% (+4.102%); pile_github +3.262% (+3.457%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 1x2 = 2 products; enwik8 +1.664% (+1.785%); pile_github +1.131% (+1.236%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 1x2 = 2 products; enwik8 +1.593% (+1.711%); pile_github +1.058% (+1.159%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) |
| 10 bits (2 pieces) | 2x1 = 2 products; enwik8 +27.366% (+29.409%); pile_github +22.983% (+25.278%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 2x2 = 4 products; enwik8 +2.352% (+2.521%); pile_github +2.293% (+2.458%); MISS | 2x2 = 4 products; enwik8 +0.255% (+0.302%); pile_github +0.219% (+0.265%); PASS | 2x2 = 4 products; enwik8 +0.071% (+0.102%); pile_github +0.068% (+0.090%); PASS |
| 12 bits (2 pieces) | 2x1 = 2 products; enwik8 +26.877% (+28.895%); pile_github +22.229% (+24.425%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 2x2 = 4 products; enwik8 +2.295% (+2.459%); pile_github +2.325% (+2.473%); MISS | 2x2 = 4 products; enwik8 +0.180% (+0.216%); pile_github +0.124% (+0.161%); PASS | 2x2 = 4 products; enwik8 +0.008% (+0.016%); pile_github -0.006% (+0.000%); PASS |
| 15 bits (2 pieces) | 2x1 = 2 products; enwik8 +27.276% (+29.297%); pile_github +22.592% (+24.854%); MISS; dropped 2026-09-29, Andrew (not offered by the flag) | 2x2 = 4 products; enwik8 +2.311% (+2.473%); pile_github +2.295% (+2.443%); MISS | 2x2 = 4 products; enwik8 +0.178% (+0.212%); pile_github +0.113% (+0.149%); PASS | 2x2 = 4 products; enwik8 -0.003% (+0.002%); pile_github +0.005% (+0.009%); PASS |

cheapest passing cells (4 products): int10w-int12a, int12w-int12a, int10w-int15a, int12w-int15a, int15w-int12a, int15w-int15a
