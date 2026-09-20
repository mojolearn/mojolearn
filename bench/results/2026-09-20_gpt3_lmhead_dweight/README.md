# GPT-3-small chunked LM-head dWeight audit

Apple M4, FP32 IDENTICAL, production `chunked_lm_head_v2_train` call including
forward loss, dHidden, dWeight, allocation and copies. The candidate replaced
each retained logit chunk in-place with exact dLogits, reused those values for
dHidden, and computed dWeight with exact OP_TN GEMM. OP_TN's `k=rows` fold is
ascending-row and matched the scalar dWeight loop bit for bit.

The candidate was rejected and production source was reverted. At
`rows=256,vocab=8193,width=768`, baseline median was 3.072 s and candidate
median was 3.023 s (1.6%). At `128,8193,768`, four rotated pairs overlapped:

```
baseline 2.652416 2.676303   candidate 2.607289 2.597811
candidate 2.621019 2.618225 candidate 2.637185 2.631354
baseline 2.642218 2.618505  baseline 2.887575 2.623057
candidate 2.647409 2.632937 baseline 2.633869 2.632393
```

All baseline/candidate runs had the same full loss/maxima/denominator/dHidden/
dWeight SHA-256: `b0d71d02873a45dddc51d2109b325b9693bae82ab4cab1725d7f367e5e8e05d2`.
The smaller cross-boundary run also matched exactly. The extra dLogit
write/read is safe, but forward logit recomputation and the dHidden fold leave
no stable whole-call improvement on Apple.

The device gate now uses vocab 1025, explicitly crossing the 1024-token chunk
boundary and comparing every dHidden and dWeight cell with the scalar oracle.
