# Nystroem direct public output qualification

IDENTICAL mode, Apple Metal, public native ABI. Shape: 1,000,000 query rows,
4 input features and 4 components, linear kernel, nonuniform deterministic
Float32 input/components and identity normalization. Round zero is warmup.

Merged baseline warmed milliseconds: 60.124, 57.686, 56.571, 59.890,
60.544; median 59.890 ms. Direct caller-owned output warmed milliseconds:
52.259, 55.294, 59.385, 57.706, 58.672; median 57.706 ms, a 3.6% timing
improvement. Every output on both arms hashed `8fba527c9edd816d` over all
4,000,000 Float32 cells.

The qualification is the major repeated-call RSS win: baseline maximum RSS
grew from 153,485,312 bytes after warmup to 234,143,744 after six calls,
approximately one 16 MB output allocation per call. Candidate maximum RSS
remained exactly 130,662,400 bytes for every call. The candidate removes the
device-to-host-buffer-to-List-to-Python materialization while retaining the
original kernel matrix, GEMM transpose, trace points, queue order, error
contract and final synchronization.

`bindings/build_kernel_methods.sh` completed in IDENTICAL mode and its binary
checks passed after the change.
