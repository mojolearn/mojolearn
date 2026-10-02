# lane/apple-fast-prep3: label-encoder, label-binarizer, multilabel-binarizer, maxabs-scaler, spline

Written without a Mojo toolchain; the first M3 build of `x_prep` (`bindings/build_x_prep.sh`, FAST) is the compile
check. Every switch is a `-D` define, default OFF, honoured by `x_prep/prep3.mojo` only under FAST on the Apple GPU;
IDENTICAL, the host binding and the other vendors compile main's code unchanged. Binding for every line: `x_prep`.

| switch | kind | files | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_PREP3_LABELS` (`PREP3_LABELS`) | define, device runner | `x_prep/fastlabels.mojo` (new); `x_prep/device.mojo` the four launches inside the FAST-only block | the labels' run scan (`_label_unique`: LabelEncoder, LabelBinarizer and MultiLabelBinarizer fit, LabelEncoder fit_transform) and `chunk_neg` (LabelEncoder.transform). Main's `uniq_count` / `uniq_write` are one thread per chunk of ~sqrt(n) sorted rows walking them, and `uniq_scan` is ONE thread over the chunk counts (x_prep/labels.mojo). Here each chunk is a threadgroup of 256: run-start flags, a shared-memory tree for the count, a shared-memory prefix scan for each run start's slot; the chunk counts are scanned by one threadgroup in rounds. Same CNT, OFF, TOT, U and neg words (integer counts; slots are ranks in ascending row order). |
| `-D MOJOLEARN_PREP3_MAXABS` (`PREP3_MAXABS`) | define, new binding entry + Python route | `x_prep/fastmaxabs.mojo` (new); `bindings/_mojolearn_x_prep.mojo` `x_prep_maxabs_fit_direct` (registered under the define only); `python/mojolearn/_expansion_prep.py` MaxAbsScaler.fit | fit reads X from the caller's buffer in ONE upload: main's program route puts the board's 1M x 220 X into a device store slot (880 MB host to device) and copies it device to device into a second 880 MB arena before `colb_part` / `maxabs_fold`, then brings the arena back. Here one buffer, a (chunk x column-group) kernel reading consecutive words of a row, a fold per column, 2 d words back. Same max_abs_ and scale_ words (a maximum has one answer; `zero_to_one` as `maxabs_fold`). The Python branch is taken only when the entry exists, which no other build exports. |
| `-D MOJOLEARN_PREP3_SPLINE` (`PREP3_SPLINE`) | define, probe entry + Python route | `bindings/_mojolearn_x_prep.mojo` `x_prep_prep3_spline` (registered under the define only); `python/mojolearn/_expansion_prep.py` `_spline_prep3`, SplineTransformer.fit / transform | fit allocates no n*d arena block when the knots are 'uniform' or given (main's `so = pr.alloc(n*d)` is never written and comes back from the device unread: 64 MB per fit at raw16); the count / min / max rows come from the blocked units (`_col_stats(var=False)`: one unit per 2048-row block and column, then a fold per column) instead of `col_stats_fast_kernel` (one threadgroup per column, 16 threadgroups at raw16, every row read twice). transform's NaN / range check the same. Exact rows: the same knots and output. |

Where the 0.8.34 clock went (board measured before main's neural-pass137 / gap-prep2 routes): label-encoder and
label-binarizer were `_numeric_labels` + `flatten_labels` Python passes over every label and two one-thread device
stages over the sorted 1M rows (`unique_cols`, `count_neg`); multilabel-binarizer three Python flattenings of the sets.
Main already takes numeric label buffers through `lab_load` + the device sort + the chunked run scan and the int sets
through `_mlb_flat` / `row_ones`; this lane's LABELS switch is the remaining device-side serial work in that route.
Not changed: LabelBinarizer.transform's n x K output (taxi: 265M int32 words, the host materialisation of 1 GB is the
bulk of its fit_transform) and MultiLabelBinarizer's `_mlb_flat` passes over Python sets (the input format's cost;
sklearn pays the same). multilabel-binarizer has no switch of its own: its fit is `_label_unique`, raced under LABELS.

Risky compile sites: `x_prep/fastlabels.mojo` shared memory as `stack_allocation[TGL, Int32, address_space =
AddressSpace.SHARED]()` with `sh[tid] = sh[tid] + add` (Int32 adds; fastred uses Float32 there) and `while` loops
around `barrier()` with uniform bounds; `x_prep/device.mojo` a `comptime if PREP3_LABELS:` nested inside the existing
`comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:` with `continue` inside; `x_prep/fastmaxabs.mojo`
`comptime if not PREP3_MAXABS: raise` at the top of `maxabs_fit_direct` (the prep lane's minmax shape), kernels taking
`FP` launched with `buf.unsafe_ptr()` as x_prep/device.mojo does, `FP(unsafe_from_address=...)` as `src_ptr` /
`dst_ptr` of `enqueue_copy` (x_prep/device.mojo's out copy idiom); `bindings/_mojolearn_x_prep.mojo` two
`comptime if` registrations inside PyInit's try block (the prep lane's shape).
