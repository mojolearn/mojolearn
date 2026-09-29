| arm | boxes that ran it | verdict |
|---|---|---|
| fp32.v1 | h100, h100r6, h100r4, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.flat | h100, h100r6, h100r4, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.mma | h100, h100r6, h100r4, mi325x | AGREE 12 of 12 |
| convert.int8.quantize.a.par | h100, h100r6, h100r4, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.pack.b.par | h100, h100r6, h100r4, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w32x32.b64x64.k64.l16 | h100, h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w32x32.b128x128.k64.l16 | h100, h100r6, h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w16x32.b128x128.k64.l16 | h100, h100r4 | AGREE 4 of 4 |
| int8i32.v1.mma.staged.w32x32.b256x64.k64.l16 | h100 | ONE BOX, nothing to compare |
| int8i32.v1.mma.staged.w32x32.b512x32.k64.l16 | h100 | ONE BOX, nothing to compare |
| int8i32.v1.mma.staged.w32x16.b512x16.k64.l16 | h100 | ONE BOX, nothing to compare |
| inference.int8i32.v1.tuned | h100, h100r6, h100r4, mi325x | AGREE 12 of 12 |
| inference.4x.int8i32.v1.tuned | h100, h100r6, h100r4, mi325x | AGREE 12 of 12 |
| training.4x.int8i32.v1.tuned | h100, h100r6, h100r4, mi325x | AGREE 12 of 12 |
| pieces.int8.flat | h100, h100r6, mi325x | AGREE 12 of 12 |
| pieces.int8.mma.staged.w16x16.b32x32.k64.l16 | h100, h100r6, mi325x | AGREE 12 of 12 |
| pieces.int8.mma.staged.w16x32.b64x128.k64.l16 | h100, h100r6, mi325x | AGREE 12 of 12 |
| pieces.int8.mma.staged.w32x32.b64x128.k64.l16 | h100, h100r6, mi325x | AGREE 12 of 12 |
| pieces.int8.mma.staged.w16x16.b64x64.k64.l16 | h100, h100r6, mi325x | AGREE 12 of 12 |
| pieces.int8.mma.staged.w16x32.b128x64.k32.l16 | h100 | ONE BOX, nothing to compare |
| pieces.int8.mma.staged.w16x32.b256x32.k32.l16 | h100 | ONE BOX, nothing to compare |
| pieces.int8.mma.staged.w16x16.b256x16.k32.l16 | h100 | ONE BOX, nothing to compare |
| inference.pieces.int8.tuned | h100, h100r6, mi325x | AGREE 12 of 12 |
| training.pieces.int8.tuned | h100, h100r6, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w16x16.b32x32.k64.l16 | h100r6, h100r4, mi325x | AGREE 12 of 12 |
| training.int8i32.v1.tuned | h100r6, h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w16x16.b32x32.k32.l4 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w32x32.b64x64.k32.l4 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x128.k32.l4 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x256.k32.l4 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x128.k32.l16 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x128.k64.l16 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x128.k128.l16 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w64x64.b128x256.k64.l16 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.staged.w16x64.b16x256.k64.l16 | h100r4, mi325x | AGREE 12 of 12 |
| int8i32.v1.mma.direct.reference-loads | h100r4 | ONE BOX, nothing to compare |
| int8i32.v1.mma.direct.aligned-loads | h100r4 | ONE BOX, nothing to compare |
| int8i32.v1.mma.direct.aligned-loads.scalar-acc | h100r4 | ONE BOX, nothing to compare |
| probe.int8.mma.loads-hoisted | h100r4 | ONE BOX, nothing to compare |
| probe.int8.mma.one-half | h100r4 | ONE BOX, nothing to compare |
| probe.int8.mma.raw-store | h100r4 | ONE BOX, nothing to compare |
| int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.quantize.a | m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.pack.b | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1 | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1 | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.parq | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.parq | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.applechunk.parq | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.applechunk.parq | m3ultra, m2pro | AGREE 12 of 12 |

compared 472, disagree 0
