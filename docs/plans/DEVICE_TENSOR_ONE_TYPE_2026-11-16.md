# One device tensor type (inventory and plan, 2026-11-16)

Lane `device-tensor-inventory`, read-only, from `preinteg/wave2-20261113` at 21aefd2a8e. No code changed. This
answers the open row of docs/plans/VERIFY_HARNESS_DECOUPLE_2026-11-15.md ("`to_device`, `SequenceDeviceTensor`,
`train_to_device`: LEFT, decision needed"): decide one public resident tensor first, then point verify and the
board harness at it.

## 1. Inventory

There are three Python classes, but four native pools on four device contexts. `SequenceDeviceTensor` fronts two
pools (the sequence one and the embedding one); the `b` slot says which.

### 1.1 `SequenceDeviceTensor` (sequence and embedding pools)

| item | where |
|---|---|
| Python class | python/mojolearn/_x_sequence_device.py:36. `__slots__ = ("b", "shape", "h", "_base", "__weakref__")`, `dtype = "float32"`, `_device_tensor = True`. Methods `.numpy()`, `__array__`, `copy()`, `upload(a)`, `reshape(*shape)` (a view that keeps `_base` alive), `size`, `ndim`, `__del__`. |
| entry | `to_device(x, numeric_mode=None)` :93 (private module). Public routes: `Embedding.to_device` python/mojolearn/embedding.py:134, `MoEBlock.to_device` _x_sequence_moe.py:111, `LayerNorm.to_device` _x_sequence_norm.py:190, `Embedding.forward(ids, device=True)`. |
| owner binding (door) | `_mojolearn_tokenizer_host`: bindings/sequence_device_native_pack.mojo:111 `sequence_device_operation` (`seqdev_*`), unpack bindings/sequence_device_native_unpack.mojo:23 `seq_device_set`, :32 `seq_device_make`, :39 `seq_device_input`; slot reads bindings/sequence_optimizer_native_door.mojo:80 `seq_device_get`, :86 `seq_device`; finalizer bindings/optimizer_native_router_door.mojo:317 and bindings/entry/dev_verifier_repairs.mojo:30. Class connected at python/mojolearn/_native_optimizer_bridge.py:156-172 (slot getters and setters kept at import). |
| native pool A (sequence) | sequence/seq_tensor.mojo:31 `_SeqTensorPool` (`List[DeviceBuffer[float32]]`, `List[Int]` sizes), SharedSlot `MojolearnXSequenceTensors{Identical,Fast}V1`; handles 1-based. Context `sequence_ctx()` (sequence/exec_device.mojo:200, slot `MojolearnXSequenceContext*V3`). Exported by `_mojolearn_x_sequence` (bindings/_mojolearn_x_sequence.mojo:850 `seq_tensor_alloc`...). |
| native pool B (embedding) | embedding/embedding_words_run.mojo:828 `_EmbTensors` (same two lists plus a lock word, because the forward and backward read handles with the GIL released), SharedSlot `MojolearnEmbeddingTensors*V1`; handles 1-based; context `neural_ctx["MojolearnNeuralEmbeddingContext*V3"]`. The same entry names (`seq_tensor_*`, :890-953), exported by `_mojolearn_embedding`, so one Python class serves both. |
| upload | `seq_tensor_alloc` (:64) then `seq_tensor_upload` (:104): raw `enqueue_copy` from the caller's pageable host memory, then `synchronize`. The alloc also synchronizes (:88), one wait per allocation. |
| download | `seq_tensor_download` (:122): raw `enqueue_copy` into pageable memory, then `synchronize`. No pinned staging. |
| free | `seq_tensor_free` (:92): `synchronize`, then the slot is replaced by a 1-float buffer. No reuse. |
| CPU column | `to_device` returns the float32 host array itself (`seq_device_entries` is None). |
| consumers | sequence optimizers RMSprop, Adagrad, Adamax, NAdam, Lion, LAMB, Adafactor (bindings/sequence_optimizer_native_door.mojo:283, 440, 461; sequence/opt_resident.mojo:789, 900, 956); `layer_norm_forward` / `layer_norm_backward` and `LayerNorm` (sequence/layernorm_dev.mojo, bindings/sequence_layers_native_door.mojo:31, 247-251); `MoEBlock` (bindings/sequence_moe_native_door.mojo:190, 305-308; sequence/moe_dev.mojo); `Embedding` forward and backward (bindings/embedding_layer_native_door.mojo:307, 721, 772). Each door refuses a tensor whose `b` is not its own module. |

### 1.2 `TrainDeviceArray` (training pool)

| item | where |
|---|---|
| Python class | python/mojolearn/_training_impl.py:1356. No `__slots__`: fields in `__dict__` (`h`, `shape`, `size`, `_base`, `_b`). Methods `reshape(shape)`, `to_host()` (returns a `mojolearn.Array`, not NumPy). No `.numpy()`, `copy`, `upload`. |
| entry | `train_to_device(a, numeric_mode=None)` :1375 (private). Public routes: `SGD`/`Adam`/`AdamW` `.params_to_device()`, `.grads_to_device(g)`, `.param_device(j)`, `.step_device(grads)` (:318-345). `linear_forward`, `linear_backward`, `cross_entropy`, `residual_dropout` return one when given one. |
| owner binding (door) | `_mojolearn_tokenizer_host`: bindings/optimizer_device_native_door.mojo:180 `device_operation` (`device_*`), :66 `device_new`, :101 `_device_allocate`, :143 `device_finalize`; native lease `DeviceHandleLease` :73. Class connected at _native_optimizer_bridge.py:74, 111-117. Calls the pool through native function pointers (`OptimizerNativeDriver`: `allocate_device`, `upload_device`, `download_device`, `free_device`), not `py_call`. |
| native pool C | training/dev_tensors.mojo:60 `_DevPool` (same two lists), SharedSlot `MojolearnTrainingDevTensors*V1`; handles 0-based. Context: the caller's `ctx` argument, in practice `neural_ctx["MojolearnNeuralTrainingContext*V3"]` (bindings/_mojolearn_training.mojo:82, training/optimizer_kernels_gpu.mojo:12). |
| upload / download | `train_dev_put` :125 / `train_dev_get` :133: raw `enqueue_copy` plus `synchronize`; no pinned staging. |
| free | `train_dev_free` :102: `synchronize`, slot replaced by a 1-float buffer. No reuse. Alloc :74 does not wait. |
| CPU column | `IDN_TRAIN_DEV_TENSORS` (:51) is False on the CPU column and on `fast`: `train_to_device` refuses by name. |
| consumers | bindings/optimizer_admission_door.mojo:145; bindings/optimizer_public_door.mojo:146, 297, 690-704; bindings/training_linear_native_unpack.mojo:31, 91; bindings/training_extra_native_door.mojo:100, 162; bindings/training_cross_entropy_native_door.mojo:96; native training/optimizer_public.mojo:363, 646; training/residual_dropout_boundary_run.mojo:22-45; training/dev_tensors.mojo:148-243 (linear, CE forms). |

### 1.3 `_expansion_cnn.DeviceTensor` (CNN pool)

| item | where |
|---|---|
| Python class | python/mojolearn/_expansion_cnn.py:169. `__slots__ = ("b", "shape", "h", "_host", "_base", "__weakref__")`, `dtype = "float32"`. `h` is a raw DEVICE ADDRESS, not a handle; `_host` holds the words on a CPU install. Methods `.numpy()`, `__array__`, `reshape`, `size`, `ndim`, `_new`, `_new_out` (no fill), `_wrap`. |
| entry | `to_device(x, numeric_mode=None)` :221 (private). No public route: the board and users reach it only by the private name. |
| owner binding (door) | `_mojolearn_runtime_xseq_host`: bindings/xcnn_tensor_native_pack.mojo:32-104 (`tensor_*`), bindings/xcnn_native_pack.mojo:39, 70; layout and slot access bindings/xcnn_common_door.mojo:228-370 (`xcnn_is_t`, `xcnn_tensor_layout` with a per-class cached verdict, `xcnn_on_dev`, `xcnn_tensor`, `xcnn_new`). |
| native pool D | x_cnn/device.mojo:945-1040 `res_alloc`, `res_alloc_nofill`, `res_free`, `res_upload`, `res_download`: lists `res` (live) and `pool` (freed, kept for reuse up to `RES_POOL_MAX_FLOATS` = 64 Mi floats, :918) inside `_CnnContext`; context `cnn_ctx()` (`MojolearnXCnnContext*V3`, :195). Exported by `_mojolearn_x_cnn` (and a host twin in `_mojolearn_x_cnn_host` :538-603 on host memory). |
| upload | `res_upload` :1012 raw copy plus wait. |
| download | `res_download` :1032 through `down` :742 = `core/staged_download.mojo::download_f32_into` (two pooled PINNED stages, chunk pipeline). |
| alloc / free | reuse of the smallest freed array within 2x (:956), optional +0.0 fill with NO wait; free (:987) returns to the reuse pool with no wait, by an O(live) address search. |
| CPU column | host words behind the same object (`_host`), so layer code is uniform. |
| consumers | Conv2d, Conv1d, MaxPool/AvgPool 1d/2d, adaptive pools, BasicBlock, BatchNorm1d/2d, Dropout2d, GCNConv, SAGEConv, CNNClassifier: bindings/xcnn_conv_native_door.mojo (33 sites), xcnn_pool_native_door.mojo (27), xcnn_graph_native_door.mojo (22), xcnn_bn_native_door.mojo (7). The `_m` entries take the address directly with a residency bit. Layer scratch `_Dev` (:67) uses the same pool through `x_cnn_res_alloc`. |

### 1.4 What differs for real, and what is duplication

Real differences (each must survive in the one type, or be a choice):

1. Context. Four streams: XSequence, NeuralEmbedding, NeuralTraining, XCnn. A buffer is safe to use on another
   context only when nothing is queued on it. Every entry already synchronizes before it returns
   (core/neural_context.mojo, x_cnn/device.mojo:716), so at a call boundary every context is idle; the two
   exceptions are CNN's fill-without-wait on allocation and free-without-wait into its reuse pool, which rely on
   the single in-order XCnn context.
2. Handle form. CNN entries take raw device addresses with a residency bit per argument; the other three take
   integer handles and do an O(1) index lookup plus a size check. Changing the CNN `_m` entries to handles
   would add a lookup per argument per call; the plan does not do that.
3. CPU column. Sequence returns the host array; CNN returns the same object over host words; training refuses.
4. Read-back type. `.numpy()` (sequence, CNN) versus `to_host()` returning a `mojolearn.Array` (training).
5. Embedding pool lock (GIL-released readers while another thread allocates).
6. Multi-GPU: none of the four is multi-GPU. Each lives on its binding's process context, whose root may follow
   a device plan (core/neural_context.mojo); nothing records which device a buffer is on.

Not different: dtype (float32 only in all four); shape metadata; view by `_base`; ring or arena membership (none
of the four pools belongs to one; only CNN downloads use the pinned stage pool).

Duplication: four copies of the same pool (list of buffers, list of sizes, free-slot search: seq_tensor.mojo:31,
dev_tensors.mojo:60, embedding_words_run.mojo:828, x_cnn/device.mojo `res`/`pool`); three copies of
alloc/upload/download/free entries per binding; three Python classes with the same fields; three doors
(sequence_device_native_pack.mojo 229 lines, the device part of optimizer_device_native_door.mojo, xcnn_tensor_native_pack.mojo
112 lines) doing construct/new/wrap/reshape/size/ndim/host/finalize; three `to_device`s plus three layer `to_device`
methods; two A/B env switches for the same idea (`MOJOLEARN_SEQ_DEVICE_IO_OFF`, `MOJOLEARN_XCNN_DEVICE_IO_OFF`; the
LayerNorm one, `MOJOLEARN_LN_CLASS_DEVICE_IO_OFF`, is recorded CONFIRMED neutral in EXPERIMENTS.md).

Cost found on the way (fixed by the plan): the sequence pool waits on every allocation (seq_tensor.mojo:88) and
none of sequence, embedding or training reuse freed storage or stage downloads through pinned memory, which CNN
measured as a gain (x_cnn/device.mojo:742 comment, lane gap-neural-overhead2). A tensor from one family cannot feed
another (a CNN output into LayerNorm, an Embedding output into a training optimizer) without a host round trip.

## 2. The proposal: one type

- ONE public class `mojolearn.DeviceTensor` (float32), `__slots__ = ("shape", "h", "p", "_host", "_base",
  "__weakref__")`: `h` the pool handle (1-based), `p` its device address (cached at allocation, so CNN entries
  read it as today with no lookup), `_host` the words on the CPU column, `_base` the view owner. Methods: `.numpy()`,
  `to_host()` (a `mojolearn.Array`), `__array__`, `copy()`, `upload(a)`, `reshape(*shape)`, `size`, `ndim`.
- ONE `mojolearn.to_device(x, numeric_mode=None)`. On a GPU install it uploads once into the one pool; on the CPU
  column it returns a `DeviceTensor` over host words (`p == 0`, `_host` set), so `.numpy()` always works and every
  consumer has one input form. (Today's training refusal on CPU becomes the host entry on `_host`: a feature, no cost.)
- ONE native pool, core/device_tensors.mojo, SharedSlot `MojolearnDeviceTensors{Identical,Fast}V1`, process-wide
  across every .so: per slot the buffer, its size, its device id and the creating context's slot id; the embedding
  lock word; CNN's reuse list (cap 64 Mi floats) and fill/no-fill allocation; CNN's staged pinned download for
  every read-back. Every op takes the CALLER's `DeviceContext` (the training signature), so each family keeps its own
  stream and nothing new is enqueued on another family's stream. Reuse of a freed array matches only the same context
  slot id (one integer compare inside the existing search loop), which keeps CNN's no-wait free and no-wait fill
  correct by the same argument as today; cross-family USE of a live tensor is safe because every entry synchronizes
  before it returns. A view or pointer refuses a device-id mismatch (one compare inside the existing O(1) lookup).
- The other two become aliases, then go: `SequenceDeviceTensor`, `TrainDeviceArray` and `_expansion_cnn.DeviceTensor`
  are bound to `mojolearn.DeviceTensor`; `_x_sequence_device.to_device`, `train_to_device` and `_expansion_cnn.to_device`
  to `mojolearn.to_device`; deleted in step 6. The layer methods `Embedding.to_device`, `LayerNorm.to_device`,
  `MoEBlock.to_device` become `to_device = staticmethod(mojolearn.to_device)`-style one-liners (documented public
  API; deleting them is Andrew's call, see doubts).

## 3. Steps (each compiles on its own)

Step 1. Native pool, additive. New core/device_tensors.mojo: `dt_alloc(ctx, n, fill) -> h`, `dt_addr(h)`,
`dt_view(ctx, h, n, what)`, `dt_ptr(h, n, what)`, `dt_free(ctx, h)`, `dt_upload(ctx, h, src, n)`,
`dt_download[stage_pool](ctx, h, dst, n)`, `dt_copy(ctx, hd, hs, n)`. New native test
checks/native_tests/device_tensors.mojo (alloc/fill/reuse/free/view refusal/device-id refusal), not run by the lane.
Nothing else imports it. Build: the native test only.

Step 2. Re-point the four pools onto it, Python unchanged. sequence/seq_tensor.mojo `seq_tensor_*` call `dt_*` with
`sequence_ctx()`; embedding_words_run.mojo `seq_tensor_*_words` and the `_EmbTensors` lock move to `dt_*` with the
embedding context; training/dev_tensors.mojo `train_dev_*` call `dt_*` (handles become 1-based: every producer and
consumer is native or the door, which only passes the integer back); x_cnn/device.mojo `res_*` call `dt_*` and return
`dt_addr(h)`; `res_free(addr)` keeps today's address search for this step only. Delete `_SeqTensorPool`, `_DevPool`,
`_EmbTensors`, the `res`/`pool` fields of `_CnnContext` (bump its slot name to V4 per core/shared_slot.mojo), the
per-allocation wait in `seq_tensor_alloc`. Bits: none move (copies and allocation only; a no-fill array is handed only
to an op that writes every word, as today). Build: `_mojolearn_x_sequence`, `_mojolearn_embedding`,
`_mojolearn_training`, `_mojolearn_x_cnn` (its host twin `_mojolearn_x_cnn_host` keeps its host-memory pool, untouched),
plus every merged group that contains them (bindings/development_groups.json `x_sequence`, `embedding`,
`training`, `x_cnn`), NVIDIA and AMD targets.

Step 3. One Python class and one door, additive. New python/mojolearn/_device_tensor.py (class plus `to_device`,
no NumPy import, as _x_sequence_device.py) exported as `mojolearn.DeviceTensor` and `mojolearn.to_device`
(python/mojolearn/__init__.py lazy table). New door bindings/device_tensor_native_door.mojo merging the three doors'
construct/new/wrap/to/reshape/size/ndim/host/array/copy/upload/finalize, calling the pool through ONE native entry
table resolved once at import (the `OptimizerNativeDriver` pattern, no `py_call` per op), with the CNN layout
verdict cache (`class_record`). Registered in both owner bindings' contexts (`_mojolearn_tokenizer_host`,
`_mojolearn_runtime_xseq_host`); it is a Python class connected twice, not a Mojo `add_type`, so part 6's one-binding
rule is not touched. The three old classes and `to_device`s become aliases of the new ones. Build: the two owner
bindings.

Step 4. Consumers read the one type. Type check `py_type(v) is context["device_tensor_class"]` (one identity
compare, as today in all three). Sequence, embedding and training doors read `h` through the import-time slot getter
(training goes from a `__dict__` read plus key admission to a slot read: cheaper); CNN doors read `p` (today's
address, no lookup). The `b is entries["module"]` ownership checks go: one pool. Sites: section 1 consumer rows.
`res_free` takes the handle (O(1)), removing CNN's O(live) address search. Build: the two owner bindings; step-2
bindings only if an entry signature changes (it does not in this plan).

Step 5. Verify and the harness use the public type (same change, AGENTS.md "Verify follows the product"):
tools/identity_lanes/coverage.py:400, 417-445 (`mojolearn.to_device`, `isinstance(value, mojolearn.DeviceTensor)`),
:1172-1182 (`train_to_device` -> `mojolearn.to_device`; `.to_host()` stays); tools/identity_lanes/native_owners.py:138-191
(the fault arm keeps reaching the door's context on purpose, now `mojolearn._device_tensor`'s);
tools/test_sequence_device_identity_lanes.py:28-64 (monkeypatch target); tools/bench_board_algos.py:5716, 5853-5867
(the `from mojolearn._expansion_cnn import to_device` fallback), 6122, 6145; python/mojolearn/_verify_api_neural.py:152-163
unchanged while the layer methods exist. The VERIFY_HARNESS_DECOUPLE row closes. The tests under
python/mojolearn/tests that import the private names (test_sequence_*_native*.py, test_training_device_native_*.py,
test_optimizer_native_*.py) move to the public names. The verify smoke (tools/ops/wheel_verify_smoke.sh) shows verify
still works.

Step 6. Deletions. The alias names; `sequence_device_native_pack.mojo`, the device half of
`optimizer_device_native_door.mojo` (with `DeviceHandleLease`), `xcnn_tensor_native_pack.mojo`, and the tensor parts of
xcnn_common_door.mojo; the per-binding `seq_tensor_*`, `train_dev_*`, `x_cnn_res_*` Python entries where only the old
doors called them (the layer scratch `_Dev` keeps `x_cnn_res_*` unless it moves to the pool too); the
`sequence_device_class`/`device_class`/`xcnn_tensor_class` context keys; and, after the A/B in section 5, the two
env before-arms `MOJOLEARN_SEQ_DEVICE_IO_OFF` and `MOJOLEARN_XCNN_DEVICE_IO_OFF` (no EXPERIMENTS.md row: owed under
"An unmeasured arm is deleted"; tombstone parts per AGENTS.md). Build: the six bindings of steps 2-3.

## 4. Hot paths: what each consumer does today, and after

| consumer (loop it sits in) | today | after |
|---|---|---|
| sequence optimizer `step` with resident params/grads (per training step) | type compare, slot getter for `b`, `h`, `shape`; `b` vs module compare; native `seq_tensor_ptr` O(1) | type compare, slot getter for `h`, `shape`; native `dt_ptr` O(1) plus one device-id compare. No module compare. |
| LayerNorm / layer_norm_* / MoEBlock forward and backward (per call) | same as above; output `seq_tensor_alloc` + a WAIT per output | output `dt_alloc` from the reuse list, no wait (the stream orders it). Faster or equal. |
| Embedding forward(device=True) / backward (per call) | lock, O(1) lookup, fresh allocation per y and dW | lock, O(1) lookup, reuse list. |
| training linear/CE/residual_dropout/optimizer `step_device` (per step) | type compare, `__dict__` fetch plus key admission, native `train_dev_view` O(1) | type compare, slot getter, `dt_view` O(1). Cheaper. |
| CNN layer forward/backward with resident tensors (per call) | type compare, cached layout verdict, slot getter for `h` (address); outputs from reuse list, fill w/o wait | the same with `p`; outputs from the same reuse list. Free by handle O(1) instead of O(live). |
| `.numpy()` read-back (outside every board clock) | sequence/embedding/training: pageable copy plus wait; CNN: pinned staged | pinned staged for all. |
| `to_device` upload (outside every board clock) | raw copy plus wait (sequence also waits at alloc) | raw copy plus wait. |

Nothing adds a copy, a lookup, a lock, an allocation or a wait on a per-call path; two waits and one O(live) search
are removed. The one new compare (device id) sits inside an existing O(1) lookup.

## 5. A/B rows that would show a regression (NVIDIA L40S only while part 8 holds; same box, same job)

B arm: the integration sha before step 2; A arm: after step 4. Board lanes: `layer_norm_forward` (functional),
LayerNorm, MoEBlock, Embedding; the sequence optimizers RMSprop, Adagrad, Adamax, NAdam, Lion, LAMB, Adafactor; SGD,
Adam, AdamW on the resident path; Conv2d, Conv1d, MaxPool2d, AvgPool2d, MaxPool1d, AvgPool1d, AdaptiveAvgPool2d,
AdaptiveMaxPool2d, BasicBlock, BatchNorm2d, BatchNorm1d, Dropout2d, GCNConv, SAGEConv; the
`training-device-cross-entropy` identity lane shape. Per family one neighbouring non-board shape (e.g. LayerNorm at a
width not on the board, Conv2d at another channel count) and one non-board dataset, as part 3 requires. The env
before-arms (`MOJOLEARN_SEQ_DEVICE_IO_OFF=1`, `MOJOLEARN_XCNN_DEVICE_IO_OFF=1`) are timed in the same job once, so
their rows exist before step 6 deletes them. Quality beside every time; bits are expected unchanged (no arithmetic
moves), confirmed only by the release's one verification.

## 6. Doubts

1. Contexts stay four. Keeping each family's stream is the least invasive; merging the four into one process
   context would make cross-family ordering trivial and spend one Metal queue instead of four, but serializes
   concurrent threads in different families. Not proposed; Andrew's call if wanted.
2. The no-wait reuse argument across families rests on "every entry synchronizes before it returns". The plan keeps
   reuse per context slot id so it does not depend on that for frees, but an audit of every entry that takes a
   resident tensor for a missing final wait should precede step 4.
3. Device id: process contexts can follow a device plan's root. Recording the device per slot and refusing a
   mismatch is new behaviour (a refusal, never a copy).
4. The public layer `to_device` methods (Embedding, LayerNorm, MoEBlock): keep as one-line aliases (zero cost, docs
   unchanged) or delete in step 6 (alpha, no users). The plan keeps them.
5. CPU column: training today refuses `train_to_device`; after, it would run the host entry on `_host`. The CNN host
   twin's own host-memory pool (`_mojolearn_x_cnn_host` :538-603) stays; whether the CPU `DeviceTensor` should use
   it or plain `_host` words needs a look at the host twin's `_m` entries.
6. `_CnnContext` layout change needs a SharedSlot name bump (V3 -> V4) in step 2; every binding spelling that name
   must rebuild together.
7. No code was changed by this lane, so nothing was removed yet; section 1.4 and step 6 list what the change removes.
