# lane/apple-fast-shap: tree-shap, kernel-shap, permutation-shap

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code unchanged.
Binding: `x_trees` (bindings/build_x_trees.sh); bench driver: `algos` (tools/bench_board_algos.py).

| switch | site | what it changes under FAST on Apple |
|---|---|---|
| `-D MOJOLEARN_SHAP_TREE_TAB` | `xtrees/shap_tab.mojo` (units), `xtrees/shap_device.mojo` `_tab_values` from `tree_shap_values` | the per-leaf TreeSHAP polynomial (extend + unwound sums) tabulated once per leaf and one-fraction pattern (<= 64 patterns x 6 elements per leaf); per (row, tree) one decision word over the internal nodes, n mask tests per leaf and n tabulated terms added into the same (row, tree, slot) cells in the same order with the same statement, so the SHAP values are the same bits. Forest deeper than 6, > 64 internal nodes per tree, > 6 merged path elements or a table > 512 MB: falls back to main's kernels (one host wait per call reads the flag). |
| `-D MOJOLEARN_SHAP_KERNEL_DEV` | `xtrees/shap_fast.mojo`, `xtrees/api.mojo` (entry points registered only under the define), `python/mojolearn/_expansion_trees.py` `KernelExplainer._masks_dev` / `_coalitions_dev` / `shap_values` (branch on `x_trees_kernel_solve_dev` existing) | per explained row: the coalition draws on the device from the same counter-RNG stream as `x_trees_uniform` (software float64 for the cdf product and the Fisher-Yates index, so the same draws), the first-earlier-duplicate per draw, the mask matrix gathered on the device; `mask_expand` on the device into a kept buffer; the normal equations' integer count matrices per weight group on the device (`gram_kernel`), A and b formed in float64 on the host and `kernel_solve`'s elimination unchanged. Values differ from main by float64 association only; > 16 distinct weights falls back to `x_trees_kernel_solve`. |
| `-D MOJOLEARN_SHAP_PERM_CACHE` | `xtrees/shap_fast.mojo` `perm_synthetic_cached`, `xtrees/api.mojo` `perm_synthetic_binding` | `perm_synthetic`'s four device buffers (388 MB per row at Istella) kept for the process and reused by every row of the same size; the same words. |

Why: tree-shap (M3 338 / 816 ms vs LightGBM 73 / 148 ms) recomputed every leaf's path polynomial per row
(120 divisions and ~300 indexed register-array accesses per leaf, 64 leaves, 1e6 (row, tree) units);
the row only enters through <= 6 one-fraction bits per leaf. kernel-shap (14.4 s vs shap 7.7 s) spent its
144 ms per row on host work: Python Fisher-Yates per draw and `u.tolist()` of 1.4M doubles, the serial
`mask_expand` (45M words), and `kernel_solve`'s m q^2 float64 loop. permutation-shap (17.2 s vs 12.3 s)
allocates and first-touches 388 MB of device buffers per row.

Risky compile sites: `shap_tab.mojo` UInt64 shifts by `UInt64(rk)` and `InlineArray[UInt64, 8]`;
`shap_device.mojo::_tab_values` passes `_Forest` by read and moves `dx`/`cover`/`fo` in the early-return
branch of `tree_shap_values`; `shap_fast.mojo` `Optional(DeviceBuffer)` slots in a `_Global` struct,
`cdf.bitcast[UInt64]()` for the float64 upload, `ctx.enqueue_copy(dst_buf=slot[].perm_x.value(), ...)`,
the host buffer `cnt` read with `unsafe_ptr()[unsafe_offset=...]`, `draw`/`sf64_*` called from a kernel
(as `ops_device.bag_mark_kernel` does).

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
(tree-shap: additivity error; kernel/permutation: relative error vs the exact linear-model values) holds.
