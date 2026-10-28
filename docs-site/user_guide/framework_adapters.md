# PyTorch and MAX

mojolearn runs a whole PyTorch model as one mojolearn program, and trains mojolearn models from PyTorch
and MAX. The framework holds the tensors and moves them; every operation on the data is mojolearn's. mojolearn
runs whole models only: there is no entry point for running single mojolearn operations inside a
PyTorch or MAX graph.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

## Run an exported PyTorch model

Export the model once, then import its graph and weights into mojolearn:

```python
import torch
import mojolearn

model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval()
example = torch.ones(8, 3)
torch.onnx.export(model, (example,), "model.onnx", input_names=["input"],
                  opset_version=17, dynamo=False)
program = mojolearn.import_model("model.onnx", numeric_mode="identical")
x = torch.randn(8, 3, device="cuda")  # "mps" on a Mac
y = program.run_torch(x)
```

The export is explicit preparation in PyTorch. Import creates a native model with
its own graph and weight snapshot. Each `run_torch` call captures the input tensor
metadata once and executes the whole model. Outputs are inference tensors on the
first input's device; an `argmax` output uses `int32`.

- Input shapes are fixed by the artifact or the `shapes` argument at import.
- To use changed weights, export and import an updated artifact. Existing programs
  retain their original weights.
- Unsupported graph operations produce an `UnsupportedModel` error naming the
  unsupported nodes.
- `numeric_mode="fast"` is Apple only. The default is `identical`.

CUDA and ROCm tensors on GPU zero use device buffers. Other tensors, including
Apple MPS tensors, use host staging and return outputs to the input's device.

## Run with array inputs

The same imported program accepts array inputs and can be saved:

```python
import numpy as np

y = program.run(np.ones((8, 3), dtype=np.float32))
program.save("model.mlprog")
```

For an artifact with an open input dimension, supply its fixed shape at import:
`mojolearn.import_model("model.onnx", shapes={"input": (8, 3)})`.

## Train from PyTorch

`torch.ops.mojolearn.train(inputs, model, options, scalars)` fits a whole mojolearn `Sequential` model
in one call: the forward pass, backward pass, optimizer, batches and epochs all run in mojolearn. It
returns the fitted parameters, both optimizer moments, the training state and the loss curve as new
tensors.

```python
import torch
import mojolearn.adapters.torch

x = torch.randn(8, 1, device="cuda")
y = 2 * x + 1
parameters = torch.zeros(2, device="cuda")  # one Linear(1, 1) with bias

model = [1, 1, 1, 1, 1, 1, 0]                   # one input, one Linear(1, 1) with bias
options = [0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0]  # MSE, dense target, SGD, batch 2, 1 epoch
scalars = [0.125, 0, 0, 1e-8, 0, 0, 0, 1, 1, 0]    # learning rate 0.125
fitted, moment1, moment2, state, curve = torch.ops.mojolearn.train([x, y, parameters], model, options,
                                                                   scalars)
```

Inputs are contiguous `float32` tensors (`int32` for class codes) on the CPU or GPU 0, and must not
require gradients. `options` holds 13 integers: loss (0 MSE, 1 cross entropy, 2 BCE with logits,
3 Huber), target (0 dense `float32`, 1 `int32` class codes, 2 `int32` numeric codes), optimizer (0 SGD,
1 Adam, 2 AdamW), flags (bit 0: Nesterov for SGD), schedule (0 constant, 1 step, 2 exponential),
schedule step, batch size, epochs, shuffle, the low and high 32 bits of the seed, patience, and
initialize. `scalars` holds the learning rate, momentum or beta1, beta2, epsilon, weight decay,
dampening, tolerance, schedule gamma and Huber delta, then one value per layer (the dropout rate or
layer-normalization epsilon, else 0). The returned parameters use the `Sequential` parameter layout.

## Train from MAX

`mojolearn.adapters.max` gives a MAX graph two custom operations that call the same trainer:
`mojolearn_train` takes `[x, y, parameters]`, and `mojolearn_train_resume` takes
`[x, y, parameters, moment1, moment2, state, previous_curve]` to continue a fit from its saved state.
Both return `[parameters, moment1, moment2, state, curve]`. Their `model`, `options` and `scalars`
parameters are the lists above written as comma-separated strings:

```python
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from mojolearn.adapters import max as mojolearn_max

gpu = DeviceRef.GPU(0)
x_t = TensorType(DType.float32, [8, 1], device=gpu)
p_t = TensorType(DType.float32, [2], device=gpu)
outs = [p_t, p_t, p_t, TensorType(DType.int32, [24], device=gpu), TensorType(DType.float32, [1], device=gpu)]
params = mojolearn_max.parameters(model="1,1,1,1,1,1,0", options="0,0,0,0,0,1,2,1,0,1,0,0,0",
                                  scalars="0.125,0,0,1e-8,0,0,0,1,1,0")

with Graph("fit", input_types=[x_t, x_t, p_t], custom_extensions=mojolearn_max.custom_extensions()) as g:
    fitted = ops.custom("mojolearn_train", gpu, [v.tensor for v in g.inputs], out_types=outs,
                        parameters=params)
    g.output(*[t.tensor for t in fitted])

trainer = InferenceSession(devices=[Accelerator(0)]).load(g)
```

The state is 24 `int32` words; keep its bytes with the parameters and moments to resume a fit. The
curve holds one loss per epoch.

## Install

The adapters ship in the mojolearn GPU wheels. Each needs its framework, at the release the adapter is
built for; the `torch` and `max` extras install it:

```sh
pip install "mojolearn[torch]"
pip install "mojolearn[max]"
```

On AMD GPUs, install PyTorch's ROCm build before mojolearn:

```sh
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install mojolearn-amd
```

On AMD, PyTorch and mojolearn share one ROCm runtime in a process. Import mojolearn before PyTorch, and
PyTorch uses the runtime mojolearn loaded:

```python
import mojolearn
import torch
```

To import PyTorch first, start Python with PyTorch's bundled ROCm libraries first on the library path,
so mojolearn uses the runtime PyTorch loaded:

```sh
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

In the other order without that setting, `import mojolearn` raises an `ImportError` that names both
setups.

If the installed PyTorch or MAX is not the release the adapter was built for, importing the adapter
raises an `ImportError` that names the release to install. `import mojolearn` never imports PyTorch or
MAX; an adapter loads only when you import it.
