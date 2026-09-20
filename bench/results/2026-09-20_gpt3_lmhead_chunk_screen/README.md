# Chunked GPT LM-head launch-count screen

Apple M4, FP32 IDENTICAL, complete `chunked_lm_head_v2_train` calls. Four
otherwise-identical bindings used chunk sizes 512, 1024 (production), 2048,
and 4096. Chunk boundaries do not change token order. Every arm produced the
same full loss/maxima/denominator/dHidden/dWeight hash.

At rows=256, vocab=8193, width=768, rotated samples for all arms overlapped
between 2.925 and 3.003 seconds. SHA-256 was
`703e726993142f9fa8c645622b4d205b37b1d4a246e8ebb9cdaa89f9f71632e1`.
Chunk buffers were respectively 0.5, 1, 2 and 4 MiB.

At rows=512, vocab=16385, width=768, SHA-256 was
`77d656981f3fcab9fae280d85234b1f9f5efe5d80cc0a174801eb139f1c4b957`.
Chunk buffers were 1, 2, 4 and 8 MiB. An initial cold pair favored 4096
(6.34 s versus 6.74/8.43 s), but stricter rotation did not reproduce it:

```
1024  6.277399 6.325229
4096  6.373970 6.405994
4096  6.594241 6.720021
1024  6.642402 6.895950
```

Disposition: reject a fixed larger chunk. Launch-count savings are below
whole-call noise on this device, while retained memory rises linearly (at
rows=2048, 8 MiB for production 1024 versus 32 MiB for 4096). No production
routing change was retained.
