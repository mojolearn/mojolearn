# Mamba-3 S17 operands arm gated on the threadgroup limit (PR #28, lane/neural-pass24), 2026-10-01

Bug in 0.8.32: `mamba3_s17_operands_shared_kernel` allocated 35,328 bytes of threadgroup memory with no
`lib_smem_page_fits_for` gate, so every Mamba-3 backward on Metal failed at pipeline creation ("Threadgroup memory size
(35328) exceeds the maximum threadgroup memory allowed (32768)"; bench/results/apple-verify-0832-20261001).

On the M3 Ultra, base and mamba IDENTICAL bindings built for Metal from the branch:
| check | result |
|---|---|
| tools/strides_digest.py 2 512 384 (y + 10 gradients, two calls) | runs; JSON md5 26abf7d3 = NVIDIA L4/L40S and AMD MI325X |
| tools/strides_digest.py 8 512 768 | runs; JSON md5 6dcc4669 = NVIDIA and AMD |
NVIDIA and AMD keep the shared arm (their pages fit), so their kernels are unchanged.
