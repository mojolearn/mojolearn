# LANE BRIEF, the boundary tax

Written 2026-09-10 against main 36d48b07. Self-contained. Numbers here are
Apple M4 unless a column is named; every other vendor is OWED.

Implementation status and measured attribution now live in
[BOUNDARY_TAX_PROGRESS_2026-09-10.md](BOUNDARY_TAX_PROGRESS_2026-09-10.md).
The historical motivation below is retained; it is not a current speed claim.
WP6/WP7 belong to a separate lane.

## The claim this lane tests

Every estimator pays boundary costs per call that scale with input and model
size: moving X, y and the model across the three boundaries Python -> Mojo
host -> pinned host memory -> device, and back. That cost is paid by scalar
loops, staging copies, and per-element Python object construction. None of it
is in the paper's speed numbers as a mechanism, but all of it is in the
wall-clock those numbers were taken from.

Measured facts that motivate the lane, in the order they were found:

1. **The ExtraTrees fit spends about 15 percent of wall-clock outside its own
   phases.** `bench/results/et_profile/APPLE_M4_2026-09-01.log`, higgs 1M rows
   x 28, 100 trees depth 16, 1.83M nodes: the in-Mojo `fit_once` is 11,145 ms
   unclocked, the same fit through the binding is 13,158 and 12,998 ms. The
   1.85 to 2.0 s in between is the boundary. Nothing has itemized it yet.
2. **The trees fit copies X across the host three times before the device sees
   it.** `bindings/_mojolearn_trees.mojo::_copy_f32` (scalar List.append from
   the Python buffer, GIL held), then `upload_dataset` in
   `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo` (scalar
   `List[i]` store into a pinned host buffer), then the DMA.
3. **Isolated pass timing at 2M x 20 float32**
   (`bench/results/boundary_tax_2026-09-10/run1_m4.txt`, min of 3 warm reps):

   | pass | today | replacement |
   |---|---|---|
   | `_copy_f32` List staging | 9.3 ms | 0, read the Python buffer directly |
   | scalar List -> pinned store | 68.6 ms | 3.6 ms, SIMD-8 pointer copy |
   | same scalar loop into plain malloc | 21.8 ms | (diagnostic: pinned memory on Metal punishes scalar stores 3:1) |
   | DMA to device | 4.5 to 5.2 ms | unchanged |
   | fused f64 -> f32 SIMD cast straight into the pinned buffer | not done today | 5.9 ms |

   So roughly 75 ms per fit at that shape is recoverable for EVERY caller,
   float32 included, from one builder function and one binding helper.
4. **The model comes back as Python objects, one per field per node.**
   `_forest_out` (trees, and two byte-identical copies in `_mojolearn_rf.mojo`)
   appends `PythonObject(Int(colid))`, `PythonObject(Float64(quesval))`,
   `PythonObject(Int(left_child))` per node plus one per leaf output, and the
   Python side then walks the lists again (`Array.from_list([int(v) for v in
   ...])` since 36d48b07). Pure-Python mirror of that per-node work measures
   133 ns per node (5M nodes: 440 ms to build the lists, 223 ms to pack). The
   Mojo side constructing PythonObjects through the C API will not be faster.
   At 1.83M nodes that is on the order of half a second to a second, UNMEASURED
   in situ; WP0 measures it before anyone attributes anything to it.
5. **Stage 2 of the NumPy-free landing is on main** (36d48b07), so the
   converters from DEVIATION 2470 to 2472 are now on the ExtraTrees fit path.
   The Python side of the boundary is settled; this lane is the Mojo side.

This lane preserves split-search algorithms and numeric reduction schedules.
It changes host staging, upload scheduling and, in WP4, reuses a device row-fill
kernel. Therefore launch/synchronization counts can change. Benefits must be
measured at large-data scale and on repeated fits/predictions; the historical
profile does not put an upper bound on current-source savings. See the
implementation progress document for isolated measurements.

## The four moves

Every work package below is one of these, applied to one site. They are the
same moves DEVIATION 2470 to 2472 made on the Python side.

- **P, pointer-through.** Take the Python buffer address, build a
  `MutPointer(unsafe_from_address=...)`, hand it to the estimator under
  `GILReleased`. No `List`. The caller holds the object alive across the call
  (the `_buffer.addr_ro` contract), so this is safe with the GIL released;
  `kmeans_fit_binding` and every binding in `_mojolearn.mojo`,
  `_mojolearn_estimators.mojo`, `_mojolearn_training.mojo`, `_mojolearn_arima`,
  `_mojolearn_tsa`, `_mojolearn_solver`, `_mojolearn_linalg` already do this.
  The dividing line is only whether the estimator signature takes a pointer or
  a `List`.
- **V, vectorize the pinned store.** Where a pinned host buffer must be filled,
  fill it with `unsafe_load[width=8]` / `unsafe_store[width=8]` from a raw
  pointer, or `memcpy`, never `for i: buf.unsafe_store(i, list[i])`. Same bytes
  by construction. The measured ratio on Metal is 19:1 (68.6 vs 3.6 ms).
- **W, write into the caller's buffer.** Outputs go into an `out_*_addr` the
  Python side preallocated with `_buffer.empty()`, never into a `Python.list()`
  one element at a time. Sizes the caller cannot know in advance (a fitted
  forest) use a two-call protocol: fit returns a handle and the counts, an
  export call fills the buffers, a release call frees the handle. The forest
  inference side already has exactly this shape (`forest_prepare_gpu` /
  `forest_predict_resident_*` / `forest_release_gpu`).
- **R, stay resident.** Do not re-upload what the device already holds, do not
  download what the host already holds. Once per fit, once per model.

## Work packages, ranked

Ranking is by expected win per unit of work, on evidence. Each package names
its files, its move, its DEVIATION number, its gate, and what is owed. A lane
takes ONE package and touches only the files it names.

### WP0. Itemize the 1.9 s. DEVIATION 2480. Do this first.

Nothing below may claim a share of the boundary until this has run. Add
host-side timers (the `MOJOLEARN_STAGE_TIMES` shape, `ensemble/instruments.mojo`)
around, in `_mojolearn_trees.mojo`: `_copy_f32` for X and y; the call into
`fit_extra_trees_*_device` split at `upload_dataset` (the builder already
reports its phases); `_forest_out`. And in Python `extratrees._fit_arrays`:
the `Array.from_list` packing. Run the Sep 1 shape (higgs 1M x 28, 100 trees,
depth 16) twice interleaved with an untimed fit, on the M4, `nice -n 19`.
Report each slice as ms and as a share of binding total minus `fit_once`.

Files: `bindings/_mojolearn_trees.mojo`, `python/mojolearn/extratrees.py`
(timers only, behind the env define, default off). Output:
`bench/results/boundary_tax_2026-09-10/wp0_itemized_m4.txt`.

Gate: none needed, this adds no behavior. RUN OWED on H100 and MI325X.

### WP1. Trees and RF fit input path. DEVIATION 2481. Measured 75 ms at 2M x 20.

Move P + V. Delete `_copy_f32` for X and y in `_mojolearn_trees.mojo`; pass
the two Python addresses through to the fit. In
`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo::upload_dataset`
take `MutPointer[Float32]` instead of `List[Float32]` for X (and the labels
pointer), fill `h_data` with the SIMD-8 copy from
`bench/results/boundary_tax_2026-09-10/hostpass.mojo` pass B2. The RF twin is
`bindings/_mojolearn_rf.mojo:341-351` and `:443-453`, the same scalar store
into `hx`/`hy` straight from the Python pointer; vectorize it the same way.
The regressor's extra magnitude pass over y (`:459-462`) stays, it does work.

Follow-on 1b, same DEVIATION: fuse the float64 cast. Pass the source dtype
code through the binding; when it is float64 C-order, run
`_tiled_transpose_to_f32` (already in `_mojolearn.mojo`) straight into the
pinned buffer and skip the Python-side `as_f32_colmajor` copy entirely. Pass D
in the evidence file is the flat version at 5.9 ms. Only pays for float64
callers; do 1a first and land it alone.

Gate: fingerprint cells in the existing identity_break harness
(`et-clf`, `et-reg`, rf) stay `stable`, all three tiers; a host copy is
bit-exact by construction so fast is gated the same way here. Timing: WP0's
itemized run before and after, inside one thermal window, 1M rows minimum
(the tree timing floor). RUN OWED per vendor.

### WP2. Forest return without Python objects. DEVIATION 2482. Estimated 0.5 to 1 s at 1.8M nodes, WP0 measures.

Move W + R. Replace `_forest_out` (trees) and `_forest_out` /
`_forest_out_i32` (rf, two copies "because the two metadata types do not
unify") with the two-call protocol:

    handle, n_trees, n_nodes, num_outputs, meta... = et_classifier_fit(...)
    et_forest_export(handle, offsets_addr, colid_addr, quesval_addr,
                     left_child_addr, leaves_addr)     # SIMD/memcpy per tree
    et_forest_release(handle)

Python allocates the five arrays with `_buffer.empty()` at the reported
sizes and the estimator holds `Array`s as it does now. This also deletes the
float32 -> Float64 -> float32 widening detour the lists take today.

Follow-on 2b, same DEVIATION, bigger win for fit-then-predict: hand the
handle straight to `ResidentForest` (`core/forest_inference_model.mojo`) so
the first predict after a fit does not re-upload the tables it just
downloaded. That needs the fitted `Forest` converted to the resident layout
on the device, not on the host; scope it after 2a lands.

Files: `bindings/_mojolearn_trees.mojo`, `bindings/_mojolearn_rf.mojo`,
`python/mojolearn/extratrees.py`, `python/mojolearn/randomforest.py`,
`python/mojolearn/_forest_protocol.py`. Four estimators, one protocol; do
trees first, rf second, in separate commits.

Gate: the five arrays byte-identical to the list path on the same fit (dump
both, `cmp`). Model file hashes from `save()` unchanged. Timing per WP0.

Implementation status (current candidate): shared typed ownership registry in
`bindings/forest_export_binding.mojo`, ET/RF `*_fit_export` entrypoints and one
Python allocate/export/finally-release protocol are wired. The registry owns
ET's FitResult or RF's native tree list; it does not flatten or re-upload a
second model. `forest_export_legacy(handle)` reads the same fit for diagnostics.
`MOJOLEARN_FOREST_EXPORT=into` is the measured default; `legacy` selects the comparison arm
and `verify` additionally requires five-array byte equality against that
same-handle diagnostic before releasing it. Invalid selection names are refused.
The default was promoted after native gates and valid large-forest export timing.

Eleven focused Python export checks passed, including archive-byte equality,
allocation/export/diagnostic failure cleanup, negative-control corruption and
retaining the exported Array owners without repacking. The native host-only
registry/count/copy check compiled and passed locally. Both bindings built in
all three tiers; all fifteen Metal same-fit checks passed (four estimators plus
weighted RF classification in each tier), including exact saved NPZ bytes.
Evidence: `bench/results/boundary_tax_2026-09-10/wp2/`.

The interleaved FAST Metal export-only comparison used one fitted HIGGS forest:
1M rows, 28 features, 100 depth-16 trees, 1,823,474 nodes. After warmup, five
pairs gave minimum **2394.039 ms** for the List export plus Array packing and
**3.663 ms** for caller-buffer export plus Array allocation. Every output's
five-array SHA256 agreed. Baseline endpoint drift was 1.424%, within the 20%
limit. This measures export only, not a whole-fit speedup. Caller-buffer export is now the default on this evidence; whole-fit timing
and CUDA/HIP qualification remain RUN OWED.

Reproduction after serialized builds:

    pixi run mojo build -I . -I bindings checks/forest_export_protocol.mojo -o /tmp/wp2-export-host
    /tmp/wp2-export-host
    PYTHONPATH=python python checks/forest_export_public.py --mode fast --vendor metal

Run the public gate again with `--mode deterministic` and `--mode identical`
using each rebuilt tier; CUDA/HIP use the matching `--vendor` and rebuilt
bindings. Small fixtures certify bytes and lifecycle only. The large-data
export timing is `tools/bench_forest_export.py --data <HIGGS.f32.npy> --mode fast`.
ET timer keys distinguish `boundary_export_handle` and `boundary_export_into`
from the retained List arm's `boundary_python_objects` and Python packing.

### WP3. Predict X path. DEVIATION 2483.

**Already implemented and selected on current main.** Both RF and ET export
`forest_predict_resident_reuse_gpu` as
`forest_predict_resident_into_gpu_binding[..., True]`. The public
`_forest_protocol._resident_prediction_function` selects that export, which
passes X/output pointers through `_predict_into_buffers` and retains the device
I/O allocation for equal-size calls. No input List or scalar output drain runs
on this default. Switching to the plain `_into_` export would discard that
allocation reuse. Its old “experimental” docstring was stale and is corrected.

The List-based `forest_predict_resident_gpu` and uncached `_into_` exports remain
A/B reference arms. `ResidentForest.predict`'s List drain belongs to that
retained reference, not the selected path. Sequential prediction still rebuilds
the forest per call and is outside WP3; no engine default changes here.

The September 10 `bench/results/forest_io_reuse_2026-09-10/` evidence records
prior CUDA IDENTICAL results. No new performance result is attributed to this
boundary audit. Added Python checks sabotage both comparison arms and verify
borrowed input/output addresses for four estimators in all three numeric modes.
The native resident-layout check now compares the List arm directly with both
pointer arms for RF and ET over ragged forests, threshold-edge inputs and output
widths 1, 2, 3, 5, 8 and 9.

Gate status for this audit: 26 focused Python tests passed. Native all-tier
matrix RUN OWED, serialized with other builds/GPU work:

    nice -n 19 bash tools/check_forest_resident_layouts.sh /tmp/wp3-resident-gates

That script explicitly runs FAST, DETERMINISTIC and IDENTICAL for separate and
packed layouts. These small fixtures certify correctness, not speed. Fresh
large-data timing remains RUN OWED: 2M prediction rows, at least five
interleaved List/into/reuse pairs, recorded entrypoints, minimum timings and
baseline first-to-last drift, void above 20 percent. WP0 attribution still
applies before claiming any share of the boundary bill.

### WP4. Per-tree host loops in the ensemble RF. DEVIATION 2484.

Move R. `ensemble/randomforest.mojo::RowSampler.sample` identity arm
(:2178-2189) writes `row_ids[i] = i` on the host into a pinned buffer, uploads
it, and synchronizes, ONCE PER TREE: `4 * n_rows * n_trees` bytes per fit, more
than the whole X upload at 100 trees. ExtraTrees fills the same permutation
with a device kernel (`row_ids_tiled_sequence_kernel`,
`extratrees/.../builder.mojo:1872`). Use it. The weighted arm (:2107) bisects
on the host per tree; leave it, note it.

`compute_oob_score` (:1385) downloads X BACK from the device and rebuilds it
as a `List` with no capacity reserve (:1394-1400), inside the fit that just
uploaded it from a host pointer the caller still holds. Pass the host pointer
down instead.

Gate: rf fingerprints stable. Timing: rf fit 1M rows, 100 trees, oob on and
off.

### WP5. GBDT input staging. DEVIATION 2485.

Move V + R. `gbdt/train.mojo::_build_cindex_from_floats` (:240-250) runs an
`n_rows` scalar loop plus a `synchronize` per FEATURE, and `train()` runs it
twice when an eval set is given (:1414). Its twin
`_build_cindex_from_columns` (:335-343) already does memcpy into an 8-slot
ring; its docstring records the win it took ("~0.4 s of the 1.69 s cindex
bill at 2000 features"). Give the flat-list path the same body.

Do NOT take `partition_from_bins` (`doc_parallel_leaves_estimator.mojo:185-216`,
`n_rows` down and back per tree per permutation) in this lane. It is a host
counting sort, so the fix is a device kernel, not a copy shape. Record it as
the largest weighted host round trip in GBDT and hand it to a tree lane.

Gate: gbdt fingerprints stable; timing at 1M rows, wide (2000 features) and
narrow.

### WP6. The mechanical memcpy sweep. DEVIATION 2486. One shared helper module.

Move V, then P where the estimator signature allows. Sixteen binding files
each carry their own `_f32_ptr`; two carry it under other names
(`_mojolearn_preprocessing.mojo::ptr/load`) or inline
(`_mojolearn_byte_lm.mojo`). Create `bindings/hostptr.mojo` with
`f32_ptr/f64_ptr/i32_ptr/u32_ptr`, `copy_f32(src_ptr, dst_ptr, n)` (SIMD-8,
the hostpass B2 body) and `read_f32(addr, n) -> List[Float32]` (memcpy, the
body `_mojolearn_transformer.mojo::_read_f32` :181 already has). Then change
each file's helper to call it, one binding file per commit, in this order of
bytes moved:

1. `_mojolearn_gp.mojo::gpr_predict_binding` :548-568, copies the `n_train^2`
   Cholesky factor element by element. Quadratic. Plus an `n_train` `yzero`
   fill for a field the docstring says is never read.
2. `_mojolearn_byte_lm.mojo::_byte_lm_run` :126-132, :199-207, :250-254,
   :290-294: three whole-model element-loop reads, four `.copy()`s, four
   element-loop writes, PER TRAINING STEP.
3. `_mojolearn_mamba.mojo::_read_f32` :216 (mamba1, mamba2, and
   `mamba3_backward`, which uses the slow helper while `mamba3_forward` uses
   the memcpy `_m3_read_f32` :237). Two `b*l*dm` buffers per backward.
4. `_mojolearn_svm.mojo`, all five bindings, `x` and `y` and the support
   matrix; `iforest_run_binding` reruns the fit per call (DEVIATION 874) so
   its two matrix copies are paid per prediction.
5. `_mojolearn_metrics.mojo::_load_f32/_load_i32` :95-109 and their 22 call
   sites; `_mojolearn_preprocessing.mojo::load` :19 (every scaler fit and
   transform stages the whole matrix); `_mojolearn_estimators.mojo::kde_score_samples_binding`
   :557-568 (the one L in a 21-binding D file); `_mojolearn_gbdt.mojo::
   gbdt_binary_prediction_binding` :114 and `gbdt_fit_ordered_rmse_binding`
   :564-575.
6. Estimator-side `_upload` idioms that take a `List` and scalar-store into a
   pinned buffer once per call: `kde`, `kernel_methods`, `mixture`,
   `gaussian_process`, `cholesky`, `resample`, `ivf_flat_build`,
   `isolation_forest` (`_upload_f32` writes the buffer TWICE, poison then
   values), `holtwinters`, `arima`, `tsa`, `umap/estimator`,
   `umap/transform` (restages the whole training matrix per transform batch),
   `spectral_embedding`, `trustworthiness_score`. Same one-line body change.

Gate: bytes in equal bytes out, so the gate is the existing surface test of
each binding plus a `cmp` of one output before and after. No timing owed per
site; one representative (gp predict at n_train 20,000) before and after.

### WP7. Duplicate uploads. DEVIATION 2487.

Move R. Documented cases where the same immutable data crosses twice in one
call:

- `kernel_methods/estimator.mojo:255-256` uploads X twice because
  `km_kernel_matrix` wants two MUTABLE operands from one allocation (comment
  at :247-254); `:588-589` and `:684-685` same shape. Fix the kernel
  signature to take one immutable operand twice, then delete the second
  upload.
- `gaussian_process/estimator.mojo:673, 686` `dx`/`dx2`, same cause, no
  comment.
- `mixture/estimator.mojo:1096/1102, 1009/1018, 1197/1203`
  `precisions_cholesky` uploaded twice per call as `dprec` and `dlinv`.
- `spectral/impl/cluster/detail/spectral.mojo:100-108` stages the embedding
  into a pinned buffer only to run `plan_sum_scale` on the host, then stages
  it again to upload; the first pinned buffer never reaches the device.
- `neighbors/estimator.mojo:759-826` `h_idx` comes back from the search and
  is re-uploaded for `class_probs`; the weighted arm crosses `n_queries*k`
  three times.
- `hdbscan/impl/detail/select.mojo:462-492` two `synchronize` per download,
  the first is dead.

Gate: fingerprints stable per estimator. These are correctness-neutral
deletions; timing is a courtesy, one shape each.

### WP8. Byte-granular compare-and-copy in the ET level loop. DEVIATION 2488.

Move V. `extratrees/.../builder.mojo::_stage_upload_if_changed` (:2762-2770)
compares and copies the seven per-level staging slots BYTE BY BYTE over the
full capacity extent (`cap_nodes * size_of[NodeWorkItem]()` etc.), about two
scalar ops per byte, seven slots, once or twice per level. Payload is small,
op count is not. Compare and copy at SIMD width. `ensemble/.../builder_kernels_impl.mojo:674, 1606, 1668`
have the same byte-wise skip-on-equal on struct-sized args (small, leave).

Gate: ET fingerprints stable; `MOJOLEARN_STAGE_TIMES` "stage + feature
sampler" slice before and after at the Sep 1 shape (554 ms today).

## Not in this lane, recorded so nobody rediscovers them

- `partition_from_bins` (GBDT, `n_rows` round trip per tree per
  permutation): needs a device counting sort. Tree lane.
- `et_predict_binding` and the rf sequential predict rebuild the forest per
  call. Legacy engine.
- `gbdt_predict_binding` re-parses the model TEXT on every predict
  (`String(py=model)` :423). A resident parsed model is a gbdt lane.
- `_build_cindex_from_floats` per-feature `synchronize`: WP5 removes it as a
  side effect of the ring.
- The 30 pinned buffers `make_level_workspace` allocates per group: not a
  copy, an allocation; only matters if WP0 shows setup time.

## Gates that bind every package

1. **Bits first.** Every package moves bytes without arithmetic, so the output
   must be BYTE-IDENTICAL to the path it replaces, in all three tiers, before
   any timing. A host copy is the one place a FAST arm may be asked a bitwise
   question, because the copy has no arithmetic; the kernel behind it is not
   asked. If bytes differ, the wiring is wrong: STOP and report, no tolerance.
2. **Fingerprints.** The identity_break harness cells for the touched
   estimator stay `stable`, e.g. `cells=18 stable=18 moved=0` for et.
3. **Timing protocol.** Interleave old and new arms inside one thermal window,
   at least five pairs after one untimed warm-up round, report the MINIMUM and
   the first-to-last spread of the baseline arm; void the window above 20
   percent spread. Trees at 1M rows or more, never below. One box is one
   column: name it, record the others as OWED.
4. **Attribution.** WP0 runs before any package claims a share of the 1.9 s.
   A mechanism located before the experiment that isolates it is a guess.

## Constraints

- Own git worktree off current `origin/main`. Never touch
  `/Users/andrewhendel/CascadeProjects/mojolearn`; it has uncommitted work in
  `checks/`, `bench/results/`, `docs/lanes/`.
- One package per lane, only the files it names. Packages that share a file
  (WP1 and WP2 both touch `_mojolearn_trees.mojo`; WP1 and WP4 both touch
  `ensemble/randomforest.mojo` neighbors) run in SERIES, not parallel.
  Packages in different directories may run in parallel.
- Explicit `git add` paths, never `-A`. Report commits as `%h parent %p`.
  Rebase onto moving `origin/main` and push each green step.
- Build with `bindings/build*.sh` for the binding you touched; read the arch
  and mode back. Run one light thing at a time on the Mac, `nice -n 19`; real
  timing columns on rented GPUs.
- DEVIATION numbers 2480 to 2488 are assigned above; 2489 to 2499 are reserved
  for this lane's follow-ons.
- Do not change a default until its gate is green and recorded. Do not
  convert an estimator's numerics; this lane moves bytes.
- RUN OWED for anything not executed, with the exact command.

## Background

- `docs/lanes/BRIEF_native_convert_2026-09-10.md`: the Python-side half of
  this work, executed; its timing script is the template.
- `bench/results/native_convert_2026-09-10/`: converter numbers and protocol.
- `bench/results/boundary_tax_2026-09-10/hostpass.mojo`, `run1_m4.txt`: the
  pass timings in this brief, reproducible in one build.
- `bench/results/et_profile/APPLE_M4_2026-09-01.log`: the 1.9 s.
- `python/mojolearn/NUMPY_FREE_CONTRACT.md`: what `addr_ro` promises about
  object lifetime, which is what makes move P safe under `GILReleased`.
