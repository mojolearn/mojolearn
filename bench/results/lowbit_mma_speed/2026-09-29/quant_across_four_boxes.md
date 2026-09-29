| arm | boxes that ran it | verdict |
|---|---|---|
| fp32.v1 | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.flat | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.mma | h100, mi325x | AGREE 12 of 12 |
| convert.int8.quantize.a | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.pack.b | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1 | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1 | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.quantize.a.par | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| convert.int8.pack.b.par | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.parq | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.parq | h100, mi325x, m3ultra, m2pro | AGREE 12 of 12 |
| int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.applechunk | m3ultra, m2pro | AGREE 12 of 12 |
| inference.int8i32.v1.applechunk.parq | m3ultra, m2pro | AGREE 12 of 12 |
| training.int8i32.v1.applechunk.parq | m3ultra, m2pro | AGREE 12 of 12 |

compared 192, disagree 0
