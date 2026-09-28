# py-shared: progress

Lane `py-shared` (Python-work fix lanes, ~/mojolearn-evidence/py_work_brief.md;
audit ~/mojolearn-evidence/python_work_audit.md, cross-cutting patterns 1, 2, 4, 5).
Worktree `~/mojolearn-wt/py-shared`, branch `lane/py-shared`, base
lane/apple2-merged 0a11b50c7.

## SHARED API READY

Merge `origin/lane/py-shared` to use these. Status per item: Python-only items
are proven here on one core (pure Python, no binding); the Mojo items are
written, pushed and COMPILE on Linux (x_metrics and x_prep GPU bindings built
on nvc1, 2026-09-28); their NVIDIA + x86 CPU bit proof is queue job
nvc1-0015 (below), queued behind the other lanes' jobs. Until that job passes, treat
the Mojo entries as UNPROVEN and keep your fallbacks (every caller below
already falls back when an entry is missing).

### 1. `_portable_math.fsum`, `isfinite`, `isinf`, `isnan` (READY, same bits)
Nothing to change: every module that does `from . import _portable_math as math`
gets it. `fsum` takes CPython's compiled `math.fsum` when the result is finite
(the same correctly rounded sum; about 20 ns instead of about 300 to 500 ns per
term) and the exact big-integer sum otherwise; `MOJOLEARN_HOTPATH=python` keeps
the exact sum. The predicates take IEEE comparisons for a Python float (about 40
ns instead of about 190 ns). `_expansion_metrics._fsum` is now `pmath.fsum`.
Proof: `python/mojolearn/tests/test_portable_math_fast.py` (4000+ sums with
overflow, subnormals, NaN, inf, strings, big ints; 20000 random bit patterns).
For an O(n) scan over a BUFFER prefer `_buffer.all_finite(arr)` (native).

### 2. Label encoding: `_labels.encode_labels(y)` (READY)
Returns `(classes, int32 codes Array)` under the ORDER RULE; native in the base
binding AND in `_mojolearn_core_host` (the CPU-only binding has exported
`encode_labels_*` since the python-hotpath lane; the audit's "exports no
encoder" was stale). Pass the buffer (`Array`, ndarray) or a flat list; do NOT
call `sorted_classes(flatten_labels(y))` or `sorted_classes(y.tolist())`.
Decode with `_labels.decode_labels(classes, codes)` (native gather). Routed in
this lane: LinearSVC.fit, SVC `_as_labels` and `_c_rows`, silhouette,
model_selection `_encode_sorted`, KNeighborsClassifier fit/load, x_prep's
`encode_labels`/`decode_labels`. NUMPY_FREE_CONTRACT no longer permits O(rows)
Python label loops.

### 3. Arena ranges runner (Mojo written, proof pending)
For any arena-style binding (a float32 arena, stages of units): upload ONLY
the input words, zero the rest on the device, download ONLY the outputs.
- Mojo: `core/arena_io.mojo`: `check_in_ranges`, `check_out_ranges`,
  `upload_ranges(ctx, df, host_f, arena_len, ins_addr, nins, store)`,
  `download_ranges(ctx, df, host_f, outs_addr, nouts)`. `ins` are Int32
  triples `[lo, hi, src]` (src -1 host, else a `DeviceStore` id); `outs` are
  x_metrics' quads `[lo, hi, CNT, mult]`.
- Python: `mojolearn._arena_io`: `input_ranges(spans)`, `complement(ranges,
  size)`, `output_ranges(pairs)`, `pack_ins`, `pack_outs`, `ranges_enabled()`
  (`MOJOLEARN_ARENA_RANGES=0` is the whole-arena A/B arm).
- Wired: `x_metrics_run_ranges` (x_metrics/device.mojo
  `run_program_device_ranges`) and `x_prep_run_ranges` (x_prep/device.mojo).
  Python `_expansion_metrics._execute` and `_expansion_prep._Prog.run` use them
  when the binding has them. x_prep now refuses (AssertionError, every
  backend) a read of an INPUT slot, since inputs never come back.
- To adopt in another binding: build a triple list of your inputs and a quad
  list of what Python reads, call the two functions around your launches.

### 4. Resident inputs: `DeviceStore` + `DeviceCache` / `resident()` (Mojo written, proof pending)
- Mojo: `core/device_store.mojo` `DeviceStore`: `put(ctx, addr, n_words) -> id`,
  `write`, `read`, `copy_into(ctx, id, dst_buf, at, n)`, `ptr(id, need)` (for
  your own launches), `free(ctx, id)` (waits, drops the buffer now; nothing
  pooled). One store per binding, next to its own context:
  `comptime MY_STORE = _Global[StorageType=DeviceStore, name="Mojo<Me>Store<Tier>", init_fn=DeviceStore.__init__]`,
  and three entries `<prefix>_dev_put(addr, n_words)`, `<prefix>_dev_free(id)`,
  `<prefix>_dev_live()` (copy x_metrics' `dev_*_binding`, bindings/_mojolearn_x_metrics.mojo).
- Python: `mojolearn._arena_io.DeviceCache(binding, prefix)`: `id_of(arr)`
  (uploads once, keyed by buffer address + word count, holds the owner),
  `get`, `release(arr)`, `close()`, context manager, `live()`.
  `with _arena_io.resident(): ...` makes every x_metrics / x_prep program in
  the block keep inputs of at least `RESIDENT_MIN_WORDS` (65536) words on the
  device and frees them all at the end of the block. The caller promises not
  to write into a buffer it already handed over inside the block.
- For optimizer state, LM weights, an index: keep the id in your Python object,
  launch on `store.ptr(id, n)`, free in `close()` / on refit.

## Changes
| commit | what |
|---|---|
| d30392468 | `_portable_math` fsum / isfinite / isinf / isnan fast paths; test_portable_math_fast.py |
| 05bf6ad0d | label callers through `encode_labels`; NUMPY_FREE_CONTRACT and DEVIATION 2377 text |
| 5a9379251 | core/arena_io.mojo, core/device_store.mojo, x_metrics / x_prep ranges + store entries, `_arena_io.py`, test_arena_ranges.py |
| b946c9aac | tools/py_shared/ab_job.sh (lane checks base vs head, tests, timing), bench/py_shared_micro.py |

## DEVIATION changes
- NUMPY_FREE_CONTRACT.md Rules: the O(rows) Python label-loop permission is
  retired (native encoder and argmax); O(classes) Python stays.
- DEVIATION 2377 (silhouette's O(rows) label pass): retired in `_metrics_impl`.
  The `_native_union_codes` fallback text stays for the no-binding case.
- DEVIATION 6106: unchanged (O(classes) scalars); its `_fsum` wrapper is now
  the shared `_portable_math.fsum`.

## Proof plan (light, one job per step)
`tools/py_shared/make_base_patch.sh` then `tools/nvidia_central.sh sync py-shared
<worktree>`, stage taxi/higgs speed npz on the pod, and
`tools/nvidia_central.sh submit py-shared /root/mojolearn-py-shared/tools/py_shared/ab_job.sh <out>`:
base tree (this diff reversed) and head tree, `tools/algos_lane_check.sh` per
lane in `tools/py_shared/lanes.txt` (72 lanes: every x-prep and x-metrics lane,
metrics, svc, linear-svc, knn-clf, cross-val, permutation-test and three tree
lanes that call fsum; GPU arm = RTX 4090, CPU arm = the pod's x86 CPU), then
every hash base == head per column; then the tests; then timing
(`bench/py_shared_micro.py`, and the x_prep / x_metrics 1M boards with
`MOJOLEARN_ARENA_RANGES` 0 / 1 / 1 / 0).

## Results
- Pure Python, one core (Mac, M-series): fsum fast path 21 ns vs 307 ns per
  term (14x), isfinite 42 vs 186 ns; 0 differences over 3013 sums and every
  predicate case. The x86 numbers come from the job's `PYSHARED` lines.
- Linux compile (nvc1, A40 pod, no GPU): `_mojolearn_x_metrics.so` and
  `_mojolearn_x_prep.so` identical tier built.
- Queue job nvc1-0015 (submitted 2026-09-28 ~20:10Z, `tools/py_shared/ab_job.sh`,
  output /root/ev-py-shared/ab-<stamp>): pending.

## Unproven
- Every Mojo change (items 3 and 4): compiled, not yet run on a GPU.
- x_prep's new input-read refusal: the head test run will show any program
  that reads an input slot.
