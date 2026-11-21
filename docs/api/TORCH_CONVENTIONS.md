# PyTorch conventions in Mojolearn

Plan `docs/plans/FUTURE_WORK_2026-11-21.md` section 3, items 4 and 5: someone who knows PyTorch can use the neural
models and the Torch adapter without reading our docs, where the meaning is the same. Every PyTorch name below is
either provided with the same meaning, or listed with the reason it is not. Nothing here adds Python work inside a
call: each method is one binding call, and device and dtype words are resolved at construction (or at `to()`, which
rebuilds through the constructor as `set_params` does).

Lane `api-torch`, 2026-11-21. "Owed" rows name the native entry they need; nothing owed is patched in Python.

## Device names

The device is `None`, `"gpu"` or `"cpu"` only (Andrew, 2026-11-21): `core/device_route.mojo` admits these three
values at construction, on every class that takes `device=`, and `to(device)` takes the same two words. PyTorch's
words (`"cuda"`, `"cuda:0"`, `"mps"`, `torch.device(...)`) are refused like any unknown value
(`device must be 'gpu', 'cpu' or None`). Round 1 of this lane mapped them onto the GPU route; that mapping was
deleted by Andrew's decision. The Torch adapter still reads a tensor's own device (below): that is the tensor's
placement, not a `device=` argument.

## dtype names

The neural models run float32 only, in both modes; `identical` defines its bits in float32.

| PyTorch | Mojolearn |
|---|---|
| `module.to(torch.float32)`, `.to("float32")`, `.float()` | neural blocks: accepted, nothing changes (they already are float32) |
| `.to(torch.float16 / bfloat16 / float64)`, `.half()`, `.double()`, `.bfloat16()` | refused in words: `identical` runs float32 only. `.half()`/`.double()`/`.bfloat16()` are not defined |
| `load_state_dict` of float16/bfloat16/float64 tensors | `Sequential.load_state_dict`: converted to float32 on the device (existing); blocks: refused by the constructor's float32 rule (existing) |
| `dtype=` constructor argument | not added: no neural class has a second dtype to choose |

## Per class

`yes` = provided with PyTorch's meaning; `differs` = provided, meaning differs (reason given); `no` = not provided
(reason given); `new` = added by this lane.

### Neural blocks: `TransformerBlock`, `Mamba1Block`, `Mamba2Block`, `Mamba3Block` (and their `*Inference` CPU twins)

| PyTorch | Mojolearn | Note |
|---|---|---|
| `state_dict()` | new | `{name: array}`, the block's weights under the constructor's names (Hugging Face names in torch's `[out, in]` layout for `TransformerBlock`); a new dict over the same arrays, as torch returns references |
| `load_state_dict(sd)` | new | rebuilds the block from `sd` through the constructor (the `set_params(weights=sd)` path): every key and shape checked before the block changes. Returns the block, not torch's `(missing_keys, unexpected_keys)`: an incomplete or extra key set is refused, which is torch's `strict=True` |
| `parameters()` | new | iterator of the weight arrays, in `state_dict` order |
| `named_parameters()` | new | iterator of `(name, array)` pairs, in `state_dict` order |
| `to(device=None, dtype=None)` | new | `"gpu"` or `"cpu"`: rebuilds the block on that route (as `set_params(device=...)`); dtype: float32 only. Returns the block, like torch's in-place `Module.to` |
| `cuda()`, `cpu()` | no | `to("gpu")` / `to("cpu")` say the same; kept to one door |
| `train()`, `eval()`, `training` | no | the blocks have no dropout or batch statistics: forward is the same in both modes, so the flag would change nothing; `backward` is explicit |
| `forward(x)` / `__call__` | differs | `forward(x, state=None)` exists; `__call__` is not defined (blocks are not `nn.Module`s and carry an explicit state) |
| `zero_grad()`, `requires_grad_()` | no | no autograd: `backward(x, dy)` returns the gradients |

### `Sequential`

| PyTorch | Mojolearn | Note |
|---|---|---|
| `load_state_dict(sd)` | yes | existing: `nn.Sequential` names (`"<i>.weight"`, `"<i>.bias"`), torch layouts, float16/bfloat16/float64 converted to float32 |
| `state_dict()` | new (round 2) | `{"<i>.weight"/"<i>.bias": array}` in torch's layouts, a copy: one native transfer (`MODEL_EXPORT_PARAMS`, `autograd/weight_import.mojo` `model_export_params`: params_ uploaded once, each linear weight cast through its transposed view, one download); loads back with `load_state_dict` and into `torch.nn.Sequential` |
| `parameters()`, `named_parameters()` | new (round 2) | iterators over the same export (copies: params_ is one flat vector, not per-tensor arrays) |
| `train()`, `eval()` | no | `fit` always trains (dropout on) and `predict` never does: the mode is the method, so a flag would have no effect |
| `to(device=None, dtype=None)` | new (round 2) | `"gpu"`/`"cpu"`: the route is forgotten and resolved on the new device at the next call (params_ is host memory on every route); dtype float32 only; returns the model |

### CNN layers (`Conv2d`, `BatchNorm2d`, `Dropout2d`, `BasicBlock`, ...)

| PyTorch | Mojolearn | Note |
|---|---|---|
| `train(mode=True)`, `eval()` | yes | existing on `BatchNorm1d/2d`, `Dropout2d`, `BasicBlock` (native `bn_train`, `block_train`) |
| `parameters()` | differs (`BasicBlock`) | existing native `block_parameters` returns `(layer, weight field, gradient field)` descriptors for the optimizers and tools/identity_lanes/cnn.py; moving it to torch's iterator is owed (its callers move with it) |
| `state_dict()`, `load_state_dict()` | new (round 2) | `Conv2d`/`Conv1d` (`weight`, `bias`; Conv1d's weight torch's `(out, in // groups, k)`), `BatchNorm1d/2d` (`weight`, `bias`, `running_mean`, `running_var`, `num_batches_tracked`), `BasicBlock` (torchvision names `conv1.`, `bn1.`, `conv2.`, `bn2.`, `downsample.0/1.`): copies out; load is torch's `strict=True`, through the layer's `set_weights` admission (native ops `layer_state_dict` / `layer_load_state_dict`, `bindings/xcnn_conv_native_door.mojo`, `xcnn_bn_native_door.mojo`). A BasicBlock load checks every name first, then each sublayer's shapes in turn: a wrong shape in a later sublayer leaves the earlier ones loaded. GCNConv/SAGEConv: not added |

### Recurrent estimators (`LSTMRegressor`, `GRUClassifier`, `RNN*`)

| PyTorch | Mojolearn | Note |
|---|---|---|
| `state_dict()`, `load_state_dict()` | yes | existing: `nn.LSTM`/`nn.GRU`/`nn.RNN` names, `head.weight`, `head.bias` |
| `train()`, `eval()` | no | scikit-learn estimators: `fit` trains, `predict` infers |

### Trainers: `SmallMLPTrainer`, `LanguageModelTrainer` (`SmallByteLanguageModelTrainer`), `SambaStack`

| PyTorch | Mojolearn | Note |
|---|---|---|
| `state_dict()`, `load_state_dict()` | yes | existing: the full resumable snapshot (weights, optimizer moments, step, schedule), like torch's model + optimizer state together. The byte trainer state is lane byte-state's |
| `parameters()`, `named_parameters()` (`SambaStack`, `SambaInference`) | yes (round 2) | iterators of arrays / `(name, array)` pairs (orchestrator decision, Andrew told; was `{name: array}`: use `dict(model.named_parameters())`) |
| `parameters()` (`SmallMLPTrainer`, byte trainer) | differs | named `weights_` / `parameters_` (fitted-attribute style) |

### `CausalLM`

| PyTorch | Mojolearn | Note |
|---|---|---|
| `parameters()`, `named_parameters()` | yes (round 2) | iterators over the existing `causal_lm_parameters` entry (unchanged; `bindings/torch_parameters_pack.mojo` shapes its result); was `{name: array}`: use `dict(model.named_parameters())` |
| `state_dict()` / `load_state_dict()` | new (round 3) | `state_dict()` is the `{checkpoint name: array}` mapping of the `causal_lm_parameters` entry (the model's own arrays; a fused Phi-3 tensor joined, a copy). `load_state_dict(sd)` is torch's `strict=True` (missing / unexpected keys refused in words) and all or nothing: native `causal_model_load_state_dict` (`bindings/causal_lm_construct_door.mojo`) reads the weight record from the mapping through the checkpoint reader's own plan walk (fused rows split, the weight format packed as a checkpoint load packs it), constructs a complete model of the same type with this model's plan and options, and only then gives this model its fields; it drops the prepared model record, so the model generation moves on and the checkpoint fingerprint is hashed again. Differs from torch: returns None (no `(missing_keys, unexpected_keys)`, since only strict loads exist), no `strict=False`, and arrays taken earlier keep the old values (the weights are replaced, not copied into in place). Refused while a resident generation runs |
| `eval()` | no | inference only: there is no training mode |

### Optimizers: `SGD`, `Adam`, `AdamW`

| PyTorch | Mojolearn | Note |
|---|---|---|
| `state_dict()`, `load_state_dict()`, `zero_grad()`, `step()` | yes / differs | existing; `step(grads)` takes the gradients explicitly (no autograd), `zero_grad(grads)` zeroes the arrays you pass |

### Torch adapter (`mojolearn.adapters.torch`, `torch.ops.mojolearn.train`)

| PyTorch convention | Status | Note |
|---|---|---|
| tensors on `cpu`, `cuda:0` (CUDA or ROCm), `mps:0` | yes | existing C++ check in `adapters/torch/registration.cpp`; other indices refused in words |
| dtype | yes | float32 (values) and int32 (codes) tensors; anything else refused in words |
| `requires_grad` inputs | refused | "whole training owns its tape and optimizer; detach input tensors explicitly" |
| contiguous strided storage | required | refused in words otherwise |
| meta tensors / `torch.compile` shape propagation | yes | `Meta` kernel registered |
| per-operation ops | no | whole-model only (Andrew, 2026-10-24); the removed surfaces in `docs/TOMBSTONES.md` do not come back |

The adapter computes nothing of its own and needed no change for these conventions.
