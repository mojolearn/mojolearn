# dart: the DART boosting round on the device

Gap: dart / dart-reg istella FAST 69.4 s vs LightGBM 35.4 s (taxi is already
2.5x ahead), `docs/apple-fast/m3-gaps-0834.tsv`.

## Switch

`-D MOJOLEARN_DART_DEVICE=1` on the `x_trees` binding (`bindings/build_x_trees.sh`,
`python/mojolearn/_mojolearn_x_trees.so`). Compiled only under
`is_defined["MOJOLEARN_DART_DEVICE"]() and GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`
(`xtrees/dart_device.mojo::DART_DEVICE`); that guard also gates the registration of the four
`x_trees_dart_*` entries, and the Python layer takes the device loop only when `x_trees_dart_open`
exists on the binding, so every other build runs main's loop unchanged.

## What it changes

Main's `_DARTBase._boost_loop` keeps the running score on the host: per round a host walk of every
row through the new tree (`apply_trees`), host gradients, host leaf Newton sums, and
`tree_score_add` once per dropped tree to take it off and once more to put it back rescaled
(tens of host passes over the million rows per round at drop_rate 0.1 over 200 trees).

Under the switch one session holds X, y, the score, the per-tree leaf index rows (uint16) and the
per-tree leaf values on the device for the whole fit (`x_trees_dart_open`). Per round:

- `x_trees_dart_step`: the drop draws on the device (`dart_drop_kernel`, the counter RNG of
  `xtrees/ops.mojo`; the decision is the exact integer compare of the draw's top 53 bits against
  host-prepared `ceil(rate_i * 2^53)`, so the drop set is main's bit for bit), then ONE row launch
  (`dart_row_kernel`) that gathers the dropped trees' leaf values through their leaf rows, takes
  them off the score and writes the gradient, hessian and fit target. The target and the flags
  come back once (the forest fit takes host labels): the round's one sync.
- the tree fit in the forest data session, as on main.
- `x_trees_dart_add` (per class tree, no wait): the leaf row of the new tree (`dart_apply_kernel`),
  chunked leaf g/h sums folded in a fixed order (`dart_leaf_sum_kernel`, no atomics), the Newton
  leaf values (`dart_newton_kernel`), and one row launch (`dart_add_kernel`) that adds the scaled
  new tree and the dropped trees' rescale (`factor x` the gathered sum) back onto the score. The
  leaf values go back to the host asynchronously (landed by the next sync) for the model.

No per-tree launches over the kept or the dropped set; no host loop over rows. Predict (`_raw`) is
main's.

## FAST-tier deviations from main's spelling

Score, gradients and leaf values are float32 on the device (Metal has no float64); the leaf sums fold
per 8192-row chunk then across chunks instead of one row-order chain; the dropped trees come off and
go back as one gathered sum per row instead of one add per tree. Drop set, shrink factors and tree
shapes are main's.

## Files

- `xtrees/dart_device.mojo` (new): kernels, the session registry, `dart_open/step/add/close`.
- `bindings/_mojolearn_x_trees.mojo`: the four entries, registered under `comptime if DART_DEVICE`.
- `python/mojolearn/_expansion_trees.py`: `_DARTBase._dart_device`, `_dart_thr`, `_boost_loop_device`;
  the branch in `_boost` (host loop otherwise, also when bagging or column sampling closes the session).
- `bindings/build_x_trees.sh`: passes `MOJOLEARN_EXTRA_DEFINES` through (what `tools/aft_ab.sh` sets).

## Risky compile sites (no toolchain here)

- `comptime if DART_DEVICE:` inside the binding's `try:` block around `m.def_function` calls.
- `DeviceBuffer.create_sub_buffer[...]` on a buffer reached through `reg[].sessions[idx].<buf>`.
- `DeviceBuffer[DType.uint16]` / `DType.int64` buffers and `UInt64` shifts and multiplies in
  `dart_drop_kernel` (SplitMix64 on Metal: 64-bit integer arithmetic, no atomics).
- `from std.math import exp` used on `Float32` inside a kernel.
- `List[DartSession].pop(idx)` of a struct holding device buffers (the `_mojolearn_rf` registry idiom).
