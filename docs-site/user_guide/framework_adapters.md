# PyTorch and MAX adapters

mojolearn's GPU wheels include two optional adapters that let other frameworks call mojolearn
directly:

- **PyTorch**: `import mojolearn.adapters.torch` registers `torch.ops.mojolearn.operation` (one
  mojolearn operation on a tensor) and `torch.ops.mojolearn.train` (a whole mojolearn Sequential
  model fit) with the PyTorch dispatcher.
- **MAX**: `mojolearn.adapters.max` gives a MAX graph mojolearn's custom operations
  (`mojolearn_matmul`, `mojolearn_relu`, `mojolearn_operation0` to `mojolearn_operation3`,
  `mojolearn_train` and `mojolearn_train_resume`).

Both call the same compiled mojolearn library as the rest of your install. `import mojolearn` never
imports PyTorch or MAX; an adapter loads only when you import it.

## What is identical

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. Each adapter call
is one mojolearn operation: for the same input bits it returns the same output bits on every GPU, so a
chain of adapter calls does too. Tensors computed by PyTorch's or MAX's own operations between those
calls are computed by that framework, and their bits are the framework's. To run a whole model as one
mojolearn program, where every operation is mojolearn's and `python -m mojolearn verify` can check the
result, use `mojolearn.import_model` for inference or `mojolearn.Sequential` (also reachable as
`torch.ops.mojolearn.train` and the MAX `mojolearn_train` op) for training.

## Install

Each adapter needs its framework, at the release the adapter is built for. The `torch` and `max`
extras install it:

```sh
# macOS
pip install "mojolearn[torch]"
pip install "mojolearn[max]"

# Linux (the extras live on mojolearn-core, which every Linux install includes)
pip install mojolearn "mojolearn-core[torch]"
pip install mojolearn "mojolearn-core[max]"
```

On AMD GPUs, install PyTorch's ROCm build before mojolearn:

```sh
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install mojolearn-amd
```

If the installed PyTorch or MAX is not the release the adapter was built for, importing the adapter
raises an `ImportError` that names the release to install.

## Example

```python
import torch
from mojolearn.adapters import torch as mojolearn_torch  # registers torch.ops.mojolearn

x = torch.ones(2, 3, device="cuda")       # "mps" on a Mac
y = torch.ops.mojolearn.operation([x, x], 2, [2, 3])   # operation code 2: add
print(y)
```

Inputs are contiguous `float32` (or `int32` indices) tensors on the CPU or GPU 0; each call returns a
new tensor.
