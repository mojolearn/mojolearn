# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ``torch.compile`` backend ``"mojolearn"``: a PyTorch model as one mojolearn program."""
# Engineering notes (Andrew, 2026-10-24: "PyTorch as a container, never a computer").
# - The backend gets the FX GraphModule TorchDynamo captured and its example inputs. It exports that
#   GraphModule to ONNX in this process and imports it with mojolearn.import_model (importers/_program.py
#   _export_torch, the same path as import_model(module, example_inputs=...)). Every operation of the
#   graph then runs inside one ImportedProgram.run call; torch holds the tensors and moves them.
# - Nothing falls back to torch: an UnsupportedModel from the import propagates (it names every
#   unsupported node), so the compile fails. The backend never returns the GraphModule or any eager
#   callable. TorchDynamo itself runs eagerly only when the user sets
#   torch._dynamo.config.suppress_errors=True (default False); the docs say to keep it False and to
#   pass fullgraph=True so a graph break is an error too.
# - Parameters TorchDynamo lifts to graph inputs (torch.nn.Parameter example inputs) are bound into the
#   exported module, so the import keeps them on the device as weights and a call uploads only the
#   activations. An in-place change of a bound parameter (an optimizer step, load_state_dict) bumps its
#   version counter; the next call re-imports the graph with the new weights.
# - Static shapes only: a symbolic size (dynamic=True, or the automatic dynamic recompile) is refused.
# - Glue only: per call the inputs are copied to host arrays (.detach().cpu().numpy()) and the outputs
#   back to the input's device. FOLLOW-UP: hand device buffers to the program without the host copy
#   (DLPack / data_ptr into the session) once the import's session takes caller device buffers.
# - torch is imported lazily: `import mojolearn` and `import mojolearn.adapters` never import it.
from __future__ import annotations

import weakref

__all__ = ["backend", "register", "BACKEND_NAME"]

BACKEND_NAME = "mojolearn"

# the imported programs of each GraphModule, by input signature (shapes and dtypes)
_PROGRAMS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

# dtypes a program input can take: the import reads them as float32, or as integer ids
_INPUT_DTYPES = ("float32", "float16", "float64", "int32", "int64", "int16", "int8", "uint8", "bool")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("the mojolearn torch.compile backend needs PyTorch: pip install \"mojolearn[torch]\" "
                          "(Linux: pip install mojolearn \"mojolearn-core[torch]\")") from exc
    return torch


def _dtype_name(dtype):
    return str(dtype).rsplit(".", 1)[-1]


def _refuse_dynamic(torch, example_inputs):
    """Refuse symbolic sizes and non-tensor graph inputs with what to pass instead."""
    symint = getattr(torch, "SymInt", ())
    for i, x in enumerate(example_inputs):  # glue: one entry per graph input
        if symint and isinstance(x, symint):
            raise ValueError(f"the mojolearn torch.compile backend compiles static shapes only: graph input {i} is "
                             "a symbolic size. Pass dynamic=False to torch.compile; each input shape then "
                             "compiles to its own program")
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"the mojolearn torch.compile backend takes tensor inputs only: graph input {i} is a "
                             f"{type(x).__name__}")
        if symint and any(isinstance(d, symint) for d in x.shape):  # glue: at most eight dimensions
            raise ValueError(f"the mojolearn torch.compile backend compiles static shapes only: graph input {i} "
                             f"has the symbolic shape {tuple(x.shape)}. Pass dynamic=False to torch.compile; each "
                             "input shape then compiles to its own program")
        if _dtype_name(x.dtype) not in _INPUT_DTYPES:
            raise ValueError(f"the mojolearn torch.compile backend cannot take graph input {i} of dtype "
                             f"{_dtype_name(x.dtype)}; the inputs must be one of {', '.join(_INPUT_DTYPES)}")


def _output_layout(gm):
    """How the GraphModule returns its outputs: (is_sequence, count)."""
    node = next((n for n in gm.graph.nodes if n.op == "output"), None)  # glue: one entry per graph node
    if node is None:
        raise ValueError("the captured graph has no output")
    out = node.args[0]
    if isinstance(out, (tuple, list)):
        return True, len(out)
    return False, 1


def _bound_module(torch, gm, bound):
    """``gm`` with the parameters at positions ``bound`` held as buffers, taking the other inputs."""

    class CompiledGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_module = gm
            self._order = []
            for k, (pos, p) in enumerate(bound.items()):  # glue: one entry per model parameter
                self.register_buffer(f"weight_{k}", p.detach())
                self._order.append((pos, f"weight_{k}"))

        def forward(self, *xs):
            args = list(xs)
            for pos, name in self._order:  # glue: one entry per model parameter
                args.insert(pos, getattr(self, name))
            out = self.graph_module(*args)
            return tuple(out) if isinstance(out, list) else out

    return CompiledGraph()


class _Compiled:
    """What the backend returns: each call runs the imported program once."""

    def __init__(self, gm, example_inputs, numeric_mode):
        torch = _torch()
        self._gm = weakref.ref(gm)
        parameter = torch.nn.Parameter
        # the lifted parameters, by graph position in increasing order (a dict keeps insertion order)
        self._bound = {i: x for i, x in enumerate(example_inputs) if isinstance(x, parameter)}  # glue: one entry per graph input
        self._free = [i for i in range(len(example_inputs)) if i not in self._bound]  # glue: one entry per graph input
        self._signature = tuple((tuple(int(d) for d in x.shape), _dtype_name(x.dtype))  # glue: at most eight dimensions
                                for x in example_inputs)  # glue: one entry per graph input
        self._numeric_mode = numeric_mode
        self._sequence, self._count = _output_layout(gm)
        self._examples = [example_inputs[i] for i in self._free]  # glue: one entry per graph input
        self._program, self._versions = self._import(gm)

    def _weight_versions(self):
        return tuple(getattr(p, "_version", 0) for p in self._bound.values())  # glue: one entry per model parameter

    def _import(self, gm):
        torch = _torch()
        versions = self._weight_versions()
        cached = _PROGRAMS.get(gm, {}).get((self._signature, self._numeric_mode))
        if cached is not None and cached[1] == versions:
            return cached
        from ..importers._program import import_model
        # zeros of each input's shape, dtype and device: the export traces their shapes only
        examples = tuple(torch.zeros(tuple(x.shape), dtype=x.dtype, device=x.device) for x in self._examples)  # glue: one entry per graph input
        module = _bound_module(torch, gm, self._bound)
        # an UnsupportedModel (every unsupported node, by name) propagates: the compile fails, nothing runs eagerly
        program = import_model(module, example_inputs=examples, numeric_mode=self._numeric_mode)
        if len(program.outputs) != self._count:
            raise ValueError(f"the imported program returns {len(program.outputs)} outputs and the captured graph "
                             f"{self._count}")
        entry = (program, versions)
        _PROGRAMS.setdefault(gm, {})[(self._signature, self._numeric_mode)] = entry
        return entry

    def __call__(self, *args):
        torch = _torch()
        if len(args) != len(self._signature):
            raise ValueError(f"the compiled graph takes {len(self._signature)} inputs, got {len(args)}")
        if self._bound and self._weight_versions() != self._versions:
            gm = self._gm()
            if gm is None:
                raise RuntimeError("the captured graph was released; compile the model again")
            self._program, self._versions = self._import(gm)
        free = [args[i] for i in self._free]  # glue: one entry per graph input
        device = free[0].device if free else torch.device("cpu")
        # glue: a host copy of each input (FOLLOW-UP: zero-copy device handoff)
        # measured: PENDING. tools/adapter_transfer_probe.py times this call's transfer (host copy in,
        # output rebuild on the device) against the program run for a given shape; record the nv/amd/Mac
        # numbers here (shape, transfer ms, compute ms, run id) before the zero-copy lane changes it.
        arrays = [x.detach().cpu().contiguous().numpy() for x in free]  # glue: one entry per graph input
        got = self._program.run(*arrays)
        got = got if isinstance(got, tuple) else (got,)
        # the program's output bits, moved to the input's device
        outs = tuple(torch.tensor(y, device=device) for y in got)  # glue: one entry per graph output
        return outs if self._sequence else outs[0]


def backend(gm, example_inputs, *, numeric_mode=None, **options):
    """Compile a PyTorch model into one mojolearn program.

    The ``torch.compile`` backend named ``"mojolearn"``. Every operation of
    the captured graph runs inside one mojolearn program call, with the same
    results on Apple, NVIDIA and AMD GPUs; PyTorch only holds the input and
    output tensors. In `identical` mode, mojolearn returns the same bits on
    Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The graph ``torch.compile`` captured.
    example_inputs : list of torch.Tensor
        Its inputs, of the shapes and dtypes the compiled model will run.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode, passed as ``options={"numeric_mode": ...}`` to
        ``torch.compile``. None uses the library default (``'identical'``);
        ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    callable
        Takes the graph's input tensors and returns its outputs as new
        tensors on the inputs' device.

    Raises
    ------
    UnsupportedModel
        The graph uses an operation, attribute or shape the import does not
        support; the message lists every one, by node name. Nothing runs in
        PyTorch instead.
    ValueError
        An input has a symbolic (dynamic) shape, is not a tensor, or has an
        unsupported dtype; or an unknown option was passed.

    Examples
    --------
    >>> import torch
    >>> import mojolearn.adapters.torch  # doctest: +SKIP
    >>> model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    >>> compiled = torch.compile(model, backend="mojolearn", fullgraph=True, dynamic=False)  # doctest: +SKIP
    >>> y = compiled(torch.ones(8, 3))  # doctest: +SKIP
    """
    mode = options.pop("mode", None)
    if mode not in (None, "default"):
        raise ValueError(f"the mojolearn torch.compile backend has no mode {mode!r}; leave mode unset")
    if options:
        raise ValueError(f"the mojolearn torch.compile backend takes the option numeric_mode only, got "
                         f"{', '.join(options)}")  # glue: the option names
    torch = _torch()
    example_inputs = list(example_inputs)
    _refuse_dynamic(torch, example_inputs)
    return _Compiled(gm, example_inputs, numeric_mode)


def register():
    """Register :func:`backend` with TorchDynamo as ``"mojolearn"`` (once per process)."""
    torch = _torch()
    import torch._dynamo  # noqa: F401
    from torch._dynamo.backends import registry
    known = getattr(registry, "_COMPILER_FNS", {})
    if known.get(BACKEND_NAME) is backend:
        return
    if BACKEND_NAME in known:
        raise RuntimeError(f"another torch.compile backend is registered as {BACKEND_NAME!r}")
    torch._dynamo.register_backend(compiler_fn=backend, name=BACKEND_NAME)


# The entry point `torch_dynamo_backends: mojolearn` (python/pyproject.toml) names this module's
# `backend`, so backend="mojolearn" works without importing the adapter first; TorchDynamo loads it on
# first use and registers it under the name.
