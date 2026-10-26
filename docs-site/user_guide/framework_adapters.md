# PyTorch and MAX

mojolearn runs a whole PyTorch model as one mojolearn program, and trains mojolearn models from PyTorch
and MAX. The framework holds the tensors and moves them; every operation on the data is mojolearn's. mojolearn
runs whole models only: there is no entry point for running single mojolearn operations inside a
PyTorch or MAX graph.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

## Run a PyTorch model with `torch.compile`

Import `mojolearn.adapters.torch` and compile the model with the `"mojolearn"` backend:

```python
import torch
import mojolearn.adapters.torch  # registers the "mojolearn" backend

model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval()
compiled = torch.compile(model, backend="mojolearn", fullgraph=True, dynamic=False)

x = torch.randn(8, 3, device="cuda")  # "mps" on a Mac
with torch.no_grad():
    y = compiled(x)  # every operation runs in one mojolearn program call
```

The first call captures the model's graph, imports it into mojolearn and copies its weights to the GPU
once. Each later call runs the whole graph as one program and returns new tensors on the input's device.

- **Nothing falls back to PyTorch.** If the model uses an operation the import does not support, the
  compile fails with an `UnsupportedModel` error that names every unsupported node. Keep
  `torch._dynamo.config.suppress_errors` at its default, `False`: set to `True`, PyTorch would run the
  model itself instead of raising.
- **`fullgraph=True`** makes a graph break an error, so no part of the model runs between programs in
  PyTorch.
- **`dynamic=False`** compiles each input shape as its own program. The backend compiles fixed shapes
  only and refuses a graph with symbolic (dynamic) sizes.
- **Weights** are read when the graph is imported. If you change them in place (an optimizer step or
  `load_state_dict`), the next call imports the graph again with the new values.
- **Numeric mode**: pass `options={"numeric_mode": "fast"}` to `torch.compile` for the `fast` mode, which
  is available on Apple GPUs only. The default is `identical`.
- Outputs are tensors without gradients: the compiled model is for inference. An `argmax` output is
  `int32`.

Inputs on an NVIDIA or AMD GPU (CUDA and ROCm tensors on GPU 0) are read on the GPU where they are, and
the outputs are written straight into new tensors on that GPU, with no copy through the host. Inputs
anywhere else, including Apple GPU (`mps`) tensors, are copied once from the host into the program and
the outputs copied back to the input's device. Both ways return the same bits.

## Import a PyTorch model without `torch.compile`

`mojolearn.import_model` does the same import directly and returns the program, which you can run on
NumPy arrays and save:

```python
import mojolearn
import numpy as np

program = mojolearn.import_model(model, example_inputs=torch.zeros(8, 3))
y = program.run(np.ones((8, 3), dtype=np.float32))
program.save("model.mlprog")
```

`example_inputs` gives the shapes and dtypes the program will take; their values are not used. An ONNX
file imports the same way: `mojolearn.import_model("model.onnx", shapes={"input": (8, 3)})`. A model the
import cannot run is refused with the same `UnsupportedModel` error.

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

If the installed PyTorch or MAX is not the release the adapter was built for, importing the adapter
raises an `ImportError` that names the release to install. `import mojolearn` never imports PyTorch or
MAX; an adapter loads only when you import it.
