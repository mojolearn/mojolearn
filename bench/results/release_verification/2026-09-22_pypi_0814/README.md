# mojolearn 0.8.14

Source commit d181d97922496b860959c7b68d8e1ab85f453804. Holt-Winters estimated initialization
by default, GradientBoosting host route for small and one-border pools, non-finite input
refusals across the tree, cluster and embedding families, classical scikit-learn interop fixes,
native libraries rebuilt from source.

## Release verification (CPU and Apple GPU, side by side, 646 s)

Full sweep, base, denormal and odd fixtures, every cell fitted once.

| column | lanes | cells | complete |
|---|---|---|---|
| CPU | 215 | 645 | yes |
| Apple M4 Metal | 209 | 627 | yes |

CPU against Metal, every lane both columns ran: infer/model IDENTICAL=978, DIVERGENT=0;
batchgrad IDENTICAL=48, batchscale IDENTICAL=81, ragged IDENTICAL=54, stepfull IDENTICAL=54.
The ONE-COLUMN rows are host-only lanes Metal does not run and the GPU-saved-file model parts
the CPU column does not write.

- cpu/column.json sha256 710fbc776613da09aa7f94c8cb93a6e9ad450bb2cdb1cf424146168d663578e7
- metal/column.json sha256 cd1faf9be40d397ae63879b89e0392abbae4754b0beb2a3e806e680a26c2df7a

## Wheels (light route)

| platform | sha256 | smoke | tag |
|---|---|---|---|
| manylinux_2_35_x86_64 | f60a125b2fd89e616212c38b2d1c739ed6b3c614e0b5a8f3c08aca697977dc86 | NVIDIA H100, 13 jobs PASSED | alpha-api-0.8.14-linux-20260922 |
| macosx_11_0_arm64 | d6b124042907084463e1d3987b7ba3f9e6ea102087cbf510f2f8e343bf9c0eda | Apple M4, 13 jobs PASSED | alpha-api-0.8.14-macos-20260922 |

The Linux wheel packs the sm_89, sm_90a and gfx942 builds of the source commit; the 32 host
bindings are byte-identical across them and auditwheel repair passed. The macOS wheel carries
all 32 host bindings.
`pip install mojolearn==0.8.14` on the M4: version 0.8.14, no other package required.
