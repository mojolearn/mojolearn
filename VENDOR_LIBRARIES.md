# Vendor primitives: what they call, what Mojo ships, what we still hand-write

**The rule, from Andrew, 2026-08-19: where the incumbent calls a vendor
primitive, call OURS. Hand-write only where no equivalent exists, and CHECK
that rather than assume it.**

Every one of these libraries sits on cuBLAS, cuSOLVER, CUB and Thrust. Those
are where a large share of their performance lives, and reimplementing one by
hand is almost always worse than calling the tuned equivalent. This file
exists because I assumed twice that no equivalent existed and was wrong both
times, at a measured cost.

Everything below is **probed against this toolchain**, not read from
documentation. `AVAILABLE` means the import compiled. `NOT FOUND` means it
did not, and the exact path tried is recorded so nobody assumes the search
was never done.

---

## THE CORRECTION THIS FILE OPENS WITH

**Mojo HAS warp primitives.** This repository has claimed since its first
commit that it does not, in `PORTING.md 2`, `PORTING.md 8`, in the reasoning
that made RAFT's `select_warpsort.cuh` "untranslatable", and in the memory
notes. That claim was wrong. The namespace is `std.gpu.primitives.warp`, and
the earlier searches looked under `std.gpu`, `max.gpu`, `std.gpu.block` and
`max.gpu.block`, all of which miss the `primitives` level.

    AVAILABLE  std.gpu.primitives.warp   shuffle_down, shuffle_idx,
                                         shuffle_xor, lane_id, prefix_sum,
                                         reduce, sum, max, broadcast
    AVAILABLE  max.gpu.sync              syncwarp
    AVAILABLE  max.gpu.primitives.block  sum, max, min, broadcast, prefix_sum

What follows from it has to be re-derived rather than assumed:

- `cub::WarpScan` and `cub::ShuffleIndex` in `histogram_utils.cu`, which
  `PORTING.md 8` replaced with a block scan and called a fidelity loss, may
  now be portable directly.
- RAFT's `select_warpsort.cuh`, ruled out at 14 warp intrinsics, may be
  translatable. **RAFT's own dispatch prefers it for every k a k-NN user asks
  for** (`select_k-inl.cuh:38` sends `2 < k <= 256` there), so this is not a
  small door.
- Metal still has **no float `atomicAdd`**, which is a HARDWARE limit and is
  unaffected by any of this. `mojo_only/fixed_point.mojo` and its overflow
  proof stand.

None of that is done. It is written here so it is not forgotten, and it is
the largest open item in the repository.

---

## Device-wide calls: real substitution candidates

| their file | vendor call | Mojo equivalent | ours today |
|---|---|---|---|
| `split_points.cu` | `cub::DeviceRadixSort::SortPairs` | `nn.argsort.argsort` AVAILABLE | hand-written stable partition |
| `split_points.cu` | `cub::DeviceScan::ExclusiveSum` | `nn.cumsum.cumsum` AVAILABLE (inclusive; no `cumsum_exclusive`) | hand-written chunk offsets |
| `cuda_util/reduce.cu` | `cub::DeviceSegmentedReduce` | **NOT FOUND** | hand-written `partitions_reduce` |
| `split_points.cu` | `cub::DeviceSegmentedRadixSort` | **NOT FOUND** | segmented stable partition |
| cuVS/cuML | `cublasGemmEx` | `linalg.matmul.matmul` AVAILABLE | **WIRED**, N-T shape only |
| RAFT `lstsq.cuh` | `raft::linalg::gemv` | `linalg.gemv.gemv` AVAILABLE | ported contraction with `n = 1` |
| RAFT `pca.cuh` | cuSOLVER `syevj` | **NOT FOUND** as a dense eigensolver; `linalg.qr_factorization` AVAILABLE | `jacobi_eigh_device.mojo` |
| CatBoost multiclass | cuSOLVER dense Newton solve | same gap | not ported |
| RAFT distance | `raft::stats::cov` (`OP_T, OP_N`) | **BLOCKED**: `transpose_a not yet supported` | `covariance_kernel` |

## Block and warp scope: ordinary kernel code, not swap candidates

These are things a kernel does inside itself. They are not device-wide calls
and substituting them is writing normal kernel code, not plugging in a
library. Listed so the distinction is explicit.

| vendor call | Mojo equivalent | ours today |
|---|---|---|
| `cub::BlockReduce` | `max.gpu.primitives.block.sum` / `max`/`min` AVAILABLE | hand-written tree reduction |
| `cub::BlockScan` | `max.gpu.primitives.block.prefix_sum` AVAILABLE | hand-written Hillis-Steele |
| `cub::WarpScan` | `std.gpu.primitives.warp.prefix_sum` AVAILABLE | hand-written serial scan |
| `cub::ShuffleIndex` | `std.gpu.primitives.warp.shuffle_idx` AVAILABLE | **was called blocked. It is not.** |
| `cub::BlockRadixSort` | **NOT FOUND** | n/a |
| `ThreadLoad` / `ThreadStore` cache hints | **NOT FOUND** | plain loads |
| `LoadDirectWarpStriped` | **NOT FOUND** | plain strided loads |

## Also confirmed available, unused so far

`nn.gather_scatter.gather`, `nn.topk.top_k`, `nn.softmax.softmax`,
`nn.concat.concat`, `linalg.transpose.transpose`,
`linalg.qr_factorization.qr_factorization`.

`nn.topk.top_k` is worth a second look: `neighbors/` hand-ports RAFT's radix
select for exactly this, and the ported version is currently a 1.93x win over
scikit-learn. Whether MAX's is faster is a measurement nobody has taken.

## Probed and NOT FOUND, recorded so the search is not repeated blindly

    algorithm.reduce, algorithm.reduction.sum, nn.reduction.reduce
        -> no device-wide reduce with a custom operator
    nn.cumsum.cumsum_exclusive
        -> cumsum is inclusive only
    nn.sort.sort
        -> no free-standing sort; argsort exists
    std.gpu.primitives.block.*
        -> block primitives are under MAX, warp primitives under STD.
           They are NOT in the same package, which is what made the first
           search miss both.

## Vendor calls that EXIST, COMPILE, and DO NOT WORK HERE

Three so far, and the pattern is the same each time: the symbol resolves, the
call typechecks, and it fails at a level the signature does not advertise.
Recorded in full because "AVAILABLE" in the table above means *the import
compiled* and nothing more.

| call | how it fails | what we do instead |
|---|---|---|
| `linalg.matmul` with `transpose_a=True` | `constraint failed: transpose_a not yet supported` at compile time | `covariance_kernel`, the hand-ported contraction |
| `linalg.matmul` with `n = 1` | returns zeros for some outputs, no error | ported contraction; RAFT calls `gemv` here anyway |
| `linalg.transpose` on device buffers | **SIGNALS at runtime** inside `linalg::transpose::_copy_with_strides[...] rank=2, dtype=f32`, a HOST strided-copy path handed DEVICE pointers | nothing; the transpose route to the Gram shape is blocked |

That third one killed the planned unblock for PCA and OLS. The identity was
right (`Xt . Xt^T == X^T X` turns the unsupported T-N shape into the
supported N-T one), it compiled, and it died on execution.
`linalg.transpose` takes an `Optional[DeviceContext]`, and **accepting one is
not the same as dispatching on it.**

The remaining routes to the Gram shape, neither yet measured against the
other: write a twenty-line device transpose kernel and then use MAX's matmul,
or apply the register-tiling port directly to `covariance_kernel`.

## Limits of the ones we do use, both compiler-verified

`linalg.matmul`:

    constraint failed: transpose_a not yet supported

so the N-T shape (every distance) works and the T-N shape (every covariance:
`raft::stats::cov`, `lstsqEig` step 1, `tsvd_fit`) does not. **That single
limit is why PCA and OLS did not move across six benchmark rounds.** The
answer is `linalg.transpose` followed by an N-T matmul, and it is not yet
wired.

`linalg.matmul` with `n = 1` returns zeros for some outputs. RAFT does not
call gemm there either; it calls `gemv`.

## How to add a row

Probe the import, record AVAILABLE or NOT FOUND with the exact path tried,
and say what we do instead. A row asserting an equivalent does not exist is
only worth having if it names what was searched.
