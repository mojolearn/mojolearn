# Unfilled storage audit (2026-11-15)

Commit 1bcfa0d69b and lane fix-output-zero-fill (2026-11-14) made these allocators return UNINITIALISED storage:
`native_array_unfilled` (bindings/native_array_storage_door.mojo:150) and its wrappers, `admitted_storage_native`'s
Array branch (`as_array=True`, or no NumPy installed; bindings/admitted_input_unpack.mojo:310), `admitted_unfilled`
(:962), `_pack_room(..., True)` (bindings/boundary_read_door.mojo:540), `generation_array_unfilled`
(bindings/generation_buffers_door.mojo:156), `tsa_array_unfilled` (bindings/tsa_hw_native_door.mojo:83), and the
optimizer door's `new_array` (bindings/optimizer_public_door.mojo:182). A fresh optimizer then read heap garbage as its
moments and `buf_initialized`; lane cpu-native-crashes fixed that with `new_zeros`.

Every caller was checked for this: is every element written before anything reads it, on every path (early
returns, refusals, zero length, partial or ragged fills, accumulate-into writers, downloads shorter than the
allocation, state read by a later call)?

Verdicts:
- **SAFE-UNFILLED**: the writer covers the whole count before any read, or the call raises while the buffer is still
  a local that never escapes.
- **NEEDS-ZERO**: something can read an element nothing wrote. The site now zero-fills.
- **STATE**: storage a later call reads before writing (moments, `buf_initialized`, carried accumulators). It is
  zero-filled, always.

A fitted attribute, or an exported state snapshot that one writer fills in full and nothing reads before that write,
is SAFE-UNFILLED. It does not count as STATE.

Scan files (complete): `~/mojolearn-evidence/briefs-20261115/unfilled-storage-audit/`
(`as_array_sites.txt`, `unfilled_sites.txt`, `wrapper_callers.txt`, `new_array_callers.txt`, `table_*.txt`).

## Totals

| allocator family | sites | SAFE-UNFILLED | NEEDS-ZERO | STATE |
|---|---|---|---|---|
| `as_array=True` (`rg -n 'as_array=True' bindings providers`: 28 lines; iforest :396 is a comment) | 27 | 26 | 0 | 0 (+1: `new_array` :184 is the allocator; its callers are the last row) |
| `native_array_unfilled` direct calls and the `admitted_storage`/`_pack_copy`/`admitted_typed_c` uses | 27 | 27 | 0 | 0 |
| wrapper callers (`admitted_unfilled`, `tsa_array_unfilled`, `generation_array_unfilled`), plus 9 transitive callers | 32 | 32 | 0 | 0 |
| `new_array` callers | 36 | 35 | 1 | 0 |
| optimizer moments and `buf_initialized` (`optimizer_public_door.mojo:493,495,578-580`) | 5 | 0 | 0 | 5 (already `new_zeros`, lane cpu-native-crashes) |
| **total** | **127** | **120** | **1** | **5** (+1 allocator) |

## NEEDS-ZERO and STATE sites

| site | verdict | what read the unwritten element | fix | bits |
|---|---|---|---|---|
| bindings/optimizer_device_native_door.mojo:232 (`device_host`) | NEEDS-ZERO | `download_device` copies `fields["size"]` floats into a `shape_count(shape)` array. `optimizer_device_output` (training/optimizer_device_storage_plan.mojo:46) allows a prefix when the public size has been rebound, so the tail went back to the caller as heap garbage. | When `count < shape_count(dims)` the site uses `new_zeros`; a full download keeps `new_array` | The tail changes from garbage to zeros on the prefix path only. The full path is unchanged. |
| bindings/optimizer_public_door.mojo:493/495 (`_m`, `_v`), :578/579 (eager moments), :580 (`buf_initialized`) | STATE | A fresh optimizer step reads the moments and the flags before writing them | Already `new_zeros` on the base (lane cpu-native-crashes) | The step used to read garbage; it now reads zeros. That lane already reported this. |
| bindings/optimizer_public_door.mojo:184 (`new_array`) | allocator | n/a | This lane gives every caller a verdict (table below) | none |

## SAFE-UNFILLED evidence

### `as_array=True`

| site | writer covering every element |
|---|---|
| logistic_state_boundary_unpack:49,50 | core/logistic_model_state.mojo:70-92: tasks of ceil(total/tasks) over `coefficients + targets` (core/host_predict_threads.mojo:190) |
| logistic_native_door:175 | xtrees/ops.mojo:406 `mul_f32` / ops_device_elem.mojo:838 over n rows |
| logistic_native_door:249 | `labels_cast`, `rows` items |
| logistic_native_door:267 (`w`) | QN fit: glm/qn_resident.mojo (download loop over n_param), host core/estimators_host_impl.mojo:325; a non-finite X writes nothing but `admitted_finite_refuse` / `door_rows_refuse` raise before `_w` is published |
| logistic_native_door:309 | the store at :310 (one word) |
| logistic_native_door:438,440,464,486,524 | QN query: host glm/host/logistic_native_host.mojo:50-85 (decision, binary, multi, sigmoid writes 2 columns core/classical_host_predict.mojo:344-345), device estimators_impl.mojo:162-185; fold `qn_infer_fold`; rows==0 skips, count 0 |
| neighbors_local_native_door:296 | core/lof_plan.mojo:96 `unary` over rows*neighbors |
| neighbors_local_native_door:301,305,403,422 | `lof_lrd` / `lof_score` / `unary` over rows (lof_plan.mojo:110-135) |
| neighbors_local_native_door:353,433 | lof_plan.mojo:187 `write_i64` casts all `len(labels)` |
| neighbors_local_native_door:529 | all three prior branches write C (`labels_class_ratio_f64`, `admitted_fill`, cast and normalize) |
| neighbors_local_native_door:552 | x_neighbors/items.mojo:823 `nc_stats_item` / :800 `nc_std_item`: std[f] on every branch |
| neighbors_local_native_door:553,554 | items.mojo:877,910 `nc_shrink(_d)_item`: res[t] and devs[t] on both branches for every t in C*d |
| neighbors_local_native_door:698,704 | `nc_decision` per (row, class) and `nc_binary_decision`'s matmul_item store (items.mojo:198) |
| neighbors_local_native_door:722 | `nc_proba` fused, or softmax_item writing every class (items.mojo:975-988) |
| class_weight_rows_boundary_pack:55 | xtrees/glue_device.mojo:484 / :394 `class_rows`: every row. An out-of-range code raises before the store escapes (host checks first; device bad flag) |

### `native_array_unfilled` direct

| site | writer |
|---|---|
| x_prep_native_inputs_door:213 | core/array_helpers_impl.mojo:39-41 strided copy of nr*d |
| x_prep_native_inputs_door:227 | core/host_helpers_impl.mojo:100-102 gather of n rows; a bad index raises first |
| prep_iterimp_native_door:459,460 | loop at :465/:466 over k |
| prep_iterimp_native_door:461 | normal_dist_helpers_run.mojo:212,234 `truncnorm_draws`, all k |
| optimizer_schedule_native_door:158 | sequence/lr_table.mojo:475-509; nonzero status raises (optimizer_schedule_plan.mojo:113-115) |
| causal_checkpoint_door:213 | memcpy at :217 / core/borrowed_buffer.mojo:64-73 |
| mcls_common_unpack:390 | core/admitted_ids.mojo:100-110; dropped on refusal |
| tokenizer_public_native_door:1556 | core/borrowed_buffer.mojo:83-85 memcpy of total*4 |
| native_process_wire_door:215 | native_posix_transport_unpack.mojo:199-212 to byte_count; a short read raises |
| generation_buffers_door:243,338 | memoryview `[:]` assignment of all nbytes (a length mismatch raises) |
| generation_buffers_door:359 (`generation_copy`) | memcpy of captured.nbytes == prod(dims)*4 (native_array_snapshot_door.mojo:102-111). Callers: samba_state 257-260,302; causal_lm 3125,3126,3159,3255; samba_cpu_inference_pack 50,91; parallel_neural 608 |
| sequential_inputs_door:239 | core/sequential_input_plan.mojo:244-277 over plan.count == layout.count |
| arena_buffer_copy_pack:42 | `PyBuffer_ToContiguous` of view.len at :46 |
| prep_labels_native_door:224 | memcpy at :225 |
| prep_labels_native_door:683,718 | str_labels_native_run.mojo:205-216, :282-288: every i, hit or -1 |
| prep_labels_native_door:957 | p3_labels_metrics_native_door.mojo:300-312: every i, hit or `missing` |
| prep_labels_native_door:1237 | loop at :1240-1241 |
| spectral_assignment_boundary_pack:28 | memcpy at :30 |
| boundary_read_door:556 (`_pack_copy`) | memcpy at :559-560 of rows*max(cols,1)*width |
| admitted_input_unpack:310 | the allocator; its callers are the `as_array=True` rows above, plus every caller of the NumPy branch, which has been unfilled (`native_scratch_bytes`) since before this commit |
| admitted_input_unpack:1174 | core/admitted_convert.mojo:196-207, 223-238, all v.count(); a refusal drops the result |

### Wrapper callers

| site | writer |
|---|---|
| linalg_ops_decomp_door:74,81,116 | memcpy at :91 / :119 of the full matrix or vector |
| lowbit_weight_pyapi_door:108,131,137,206 | decomposition/linalg_host_impl.mojo:85-114, gemm/linalg_native.mojo:344-538 (full count, whole rows exponent buffer downloaded) |
| linalg_ops_lowbit_door:302,355,472,502,515,528,540,541,564,688,689,690,730 | host gemms assign `cp[i]=c[i]` over m*n (no `+=`), linalg_host_impl.mojo:45-176. Device: a fresh device C, downloaded whole (gemm/host_entry.mojo:163, gemm/linalg_fused_run.mojo:26). Empty operands raise |
| training_buffer_pyapi_door:54 | core/training_input_copy.mojo:19-32 |
| byte_lm_trainer_native_door:176 (`_f32_unfilled`, used :420-422) | training/byte_lm_session_ops.mojo:269-271 `copy_f32` of all n. This is an exported state SNAPSHOT that one writer fills in full; nothing reads it before that. Its zero fill was 86.7% of the lm-forward profile, so it stays unfilled. The flags it carries are zero-filled (`_i32`) |
| byte_lm_trainer_native_door:207 | memcpy at :209 |
| transformer_decode_pyapi_door:762 | autograd/block_step_ops.mojo:1695, transformer/attn_step_session.mojo:456,505 |
| causal_lm_public_door:873,990 | block_step_ops.mojo:1039 (`_block_out`, b*l*dm) |
| causal_lm_public_door:890,1113,1140 (`s_last`) | block_step_ops.mojo:2301-2310: the full h_last report. It is a report published as `h_last_`; decode state is `st.h`, a separate owner |
| samba_inference_native_door:408 | training/samba_program.mojo:539 (m*vocab) |
| neural_prefill_native_door:122 (callers :178,:181,:259,:300,:306) | mamba/impl/mamba_shell_impl.mojo:853-862, 1088-1097; host mamba_host_shell_impl.mojo:272-276 |
| samba_inference_native_door:155-157,354; mamba_public_native_door:847 | block_step_ops.mojo:2317-2321; samba_program.mojo:463; mamba_block_program.mojo:412-413 |
| samba_resident_pyapi_door:118 | training/samba_resident.mojo:1020 |

### `new_array` callers

| site | writer |
|---|---|
| optimizer_native_router_door:209 | core/optimizer_state_impl.mojo:40-42, all count |
| embedding_layer_native_door:258 | `cast_elements` over layout.count |
| embedding_layer_native_door:315 | embedding/embedding_words_run.mojo:787 gather of T*d |
| embedding_layer_native_door:576 | core/admitted_input_nd.mojo:50 |
| embedding_layer_native_door:791 (grad) | accumulate=0 for a fresh grad: autograd/embedding_ops.mojo:237-252 stores `+0.0` to untouched and padding rows, or the atomic path seeds zeros on the device; all V*d downloaded. A reused `previous` is a filled owner |
| embedding_layer_native_door:867 | memcpy of v*d at :868 |
| training_cross_entropy_native_door:140,147,155 | core/mojolearn_training_host_impl.mojo:207-217; autograd/kernels/lm.mojo:78-118 stores every N*V cell (ignored rows `+0.0`) |
| training_cross_entropy_native_door:157 | an address placeholder for want_grad=0: never written, read or returned |
| training_embedding_native_door:205,387 | memcpy at :206 / cast at :391 |
| training_embedding_native_door:424 | samba_ops.mojo:199-207 → embedding_ops.mojo:237-252 (`+0.0` rows), samba_afn.mojo:399-400 (`enqueue_fill(0)` before the scatter-add) |
| training_extra_native_door:176,192,193 | training/residual_dropout_host.mojo:34-37, residual_dropout.mojo:91-93 |
| training_extra_native_door:259,260,261 | mojolearn_training_host_impl.mojo:369-384; chunked_lm_head_v2.mojo:347-402 |
| training_extra_native_door:273,274 | chunked_lm_head_v2.mojo:219-238, 274-287: each owner zero-stores its row before its `+=` fold. Download :404-405 |
| training_extra_native_door:351 | training/neural_session_mlp.mojo (y copy), neural_session_mlp_host.mojo:49-50 |
| training_rms_native_unpack:70,79,80 | samba_ops.mojo:247, 305-306; dw is a GEMM overwrite (autograd/kernels/transformer.mojo:3258) |
| samba_head_native_unpack:70-73 | samba_ops.mojo:486-500 (gemm backward `_into` overwrite; loss always written) |
| training_linear_native_unpack:126,135,136 | samba_ops.mojo:390-393 |
| optimizer_public_door:573 (offsets) | training/optimizer_public_common.mojo:278,286 writes offsets[0] and offsets[j+1] for every j. Published registry metadata, fully written, so it is not STATE |
| optimizer_public_door:925 | memcpy at :926 of v.nbytes |
| optimizer_public_door:1144 | optimizer_public_common.mojo:242-246 scatter; accumulate_multi_gpu.mojo:54-89 |

## Native check

`checks/native_tests/fresh_state_zeros.mojo` (registered in registry.mojo; host and device columns; serial because the
optimizer pool is one per process) checks three things:
- a fresh optimizer resident slot downloads zeros;
- a slot reopened on the handle that a dirty close freed downloads zeros;
- a larger slot reopened over a smaller dirty one downloads zeros in every cell.

The Python-side `array.array` state (`new_zeros`) lives in a boundary file that a native check may not import
(tools/python_free_dirs.sh). The optimizer-owner Python tests cover it.

## Open doubts (not unfilled-storage defects)

- `generation_copy` copies physical bytes into a C-order result. An F-order source Array would come out transposed.
  This is about meaning, not unfilled storage, and the zero-filled version behaved the same.
- Mamba report sizes: the door takes `nheads` from the block fields, and the native copy count uses
  `Mamba2Dims/Mamba3Dims.of(dm).nheads`. If these two ever differed, a report would be short.
- The byte-LM export sizes its arrays from the door's `shape.n_total()` and copies the session's
  `cfg_shape.n_total()`, which assumes they are the same config.
- A device kernel filling its own device buffer completely (ragged quantize blocks) is a device-side question. A host
  zero fill would not change it.
