# RBFSampler direct public output qualification

IDENTICAL mode, Apple Metal, public native ABI. Shape: 1,000,000 query rows,
4 input features and 4 random features. The input is a deterministic uint32
LCG mapped to `[0,1)`; weights and offsets are fixed nonuniform Float32
arrays. Round zero is warmup.

Merged baseline warmed milliseconds: 60.389, 47.283, 45.510, 47.976,
49.428; median 47.976 ms. Direct caller-owned output warmed milliseconds:
32.377, 32.277, 34.248, 34.681, 33.015; median 33.015 ms, 31.2% faster.
Every output on both arms hashed `24cb4fc5e277443b` over all 4,000,000
Float32 cells.

Baseline process maximum RSS grew from 152,158,208 bytes after warmup to
184,713,216 after six calls. Candidate grew from 129,744,896 to 130,187,264
bytes. The candidate removes the device-to-host-buffer-to-List-to-Python
materialization while retaining the original GEMM, epilogue, trace points,
queue order and final device synchronization.

`bindings/build_kernel_methods.sh` completed in IDENTICAL mode and its binary
checks passed after the change.
