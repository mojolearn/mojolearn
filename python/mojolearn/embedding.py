# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Embedding lookup tables with a reproducible backward pass.

`Embedding` follows `torch.nn.Embedding`'s forward and its dense backward. Rows that
receive gradient from several positions are summed in ascending position order, so the
gradient does not depend on the device or on how the backward is scheduled, and
microbatches presented in position order reproduce the unsplit gradient exactly when
each one carries the previous gradient forward (``backward(ids, dy, grad=...)``).

There is no autograd here: the layer holds the table and computes the two calls; it
never updates `weight` itself.
"""
from . import _backend, _native_archive_bridge
from ._buffer import _materialize, addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1}
_NO_PADDING = -1
#: ORDER MATCHES autograd/kernels/embedding_sort.mojo's PLAN_SCAN = 0, PLAN_SORT = 1.
#: PLAN_AUTO = 2 is embedding/checks/embedding_identical.mojo's.
_PLAN_CODE = {"scan": 0, "sort": 1, "auto": 2}
#: The saved-table format tag (`save`, `load`, `mojolearn.host_model`;
#: lane/inference-embedding-ivf-cholesky, 2026-09-15).
_EMBEDDING_FORMAT = "mojolearn-embedding-1"

# The compiled saved-model owner, connected once at import: `save` is one
# native call (bindings/archive_schema_native.mojo writes the schema
# registered after the class).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()


# Profile mojolearn.identical.embedding.fp32.v1: bindings/_mojolearn_embedding.mojo reaches
# embedding/checks/embedding_identical.mojo's gather and ascending fold. Specification:
# embedding/IDENTICAL_EMBEDDING_CONTRACT.md; gate: `pixi run check-embedding` and
# tools/embedding_sabotage_arm.sh. Refusals: max_norm (torch renormalizes W in the forward, outside
# the profile), scale_grad_by_freq, sparse (DEVIATION 1314), weight=None (initialization is not part
# of the profile); out-of-range ids (contract 8) and NaN/inf in W, dY or a carried gradient (contract
# 9.1) are refused by the Mojo host. padding_idx: contract section 8. plan: contract section 6
# (PLAN_SCAN, PLAN_SORT = the device total-key bitonic sort; 'auto' takes sort from V * T = 2**26
# upward on the IDENTICAL tier; clause (d) holds counts, perm and dW bit-identical across plans).
# backward's grad carry: contract 7.4. Device I/O (lane gap-neural-io): forward(device=True) returns
# a SequenceDeviceTensor; a binding without the resident entries refuses device=True by name.
class Embedding(NumericModeMixin):
    """Embedding table with a reproducible gather and dense gradient, as `torch.nn.Embedding`.

    The table is the layer's own float32 copy, taken at construction (and again whenever
    `weight` is assigned): to change it, assign ``layer.weight = new_table``; writing into
    the array you passed in does not reach the layer. On the GPU the table is uploaded once
    and kept on the device until `weight` is reassigned.

    `padding_idx` follows torch: a negative value counts from the end, the forward gathers
    that row like any other, and the backward stores +0.0 in its row. Ids below 0 or at or
    past `num_embeddings` raise an error (they are never clamped), as do NaN or infinite
    values in the table or gradients, and empty `ids` or `dy`.

    Parameters
    ----------
    num_embeddings : int
        Number of rows V; at least 1.
    embedding_dim : int
        Row width d; at least 1.
    padding_idx : int or None, default None
        Row whose gradient is always zero; within [-V, V).
    max_norm : None
        Only None; anything else raises `ValueError`.
    norm_type : float, default 2.0
        Only 2.0 (it has no effect without `max_norm`).
    scale_grad_by_freq : bool, default False
        Only False; True raises `ValueError`.
    sparse : bool, default False
        Only False (the gradient is the dense (V, d) array); True raises `ValueError`.
    weight : array-like of shape (num_embeddings, embedding_dim)
        The table, required (there is no random initialization); copied to float32.
        Use `Embedding.from_pretrained` to build a layer from a table alone.
    plan : {'auto', 'scan', 'sort'}, default 'auto'
        How the backward groups the positions of each row. Both plans give the same bits;
        'auto' picks by problem size.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode; 'fast' is Apple only. None uses the process default.
        In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

    Attributes
    ----------
    weight : Array of shape (num_embeddings, embedding_dim), float32
        The table.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> table = np.random.default_rng(0).random((1000, 64), dtype=np.float32)
    >>> emb = mojolearn.Embedding.from_pretrained(table)
    >>> ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    >>> y = emb.forward(ids)
    >>> dW = emb.backward(ids, np.ones((2, 3, 64), dtype=np.float32))
    """

    _mojolearn_native_public = True
    # Public fields are passive scalar/table storage. Avoid inheriting the
    # Python fitted-output lookup wrapper on every operation lookup.
    __getattribute__ = object.__getattribute__
    _BINDING = "_mojolearn_embedding"

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, weight=None, plan="auto", *, numeric_mode=None):
        _EMBEDDING_NATIVE("emblayer_init", self, (num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, weight, plan, numeric_mode), None, _EMBEDDING_CONTEXT)

    @classmethod
    def from_pretrained(cls, embeddings, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False, numeric_mode=None, plan="auto"):
        """Build a layer from an existing (V, d) table, as torch's `Embedding.from_pretrained`.

        There is no `freeze` argument, because nothing here trains the table. The other
        arguments are the constructor's.
        """
        return _EMBEDDING_NATIVE("emblayer_pretrained", cls, (embeddings, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, numeric_mode, plan), None, _EMBEDDING_CONTEXT)

    def set_params(self, **params):
        """Set constructor parameters and rebuild the table from the supplied settings."""
        return _EMBEDDING_NATIVE("emblayer_set_params", self, (params,), None, _EMBEDDING_CONTEXT)

    # -- the resident device table (bindings/_mojolearn_embedding.mojo) ------
    # The layer's `__setattr__` (assigning `weight` releases the table) and
    # `__del__` are the native lifetime entry, installed after the class
    # (bindings/embedding_layer_native.mojo).

    def __getstate__(self):
        return _EMBEDDING_NATIVE("emblayer_state", self, (), None, _EMBEDDING_CONTEXT)

    def _extension(self):
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "embedding_numeric_mode", None)
        if fn is not None and int(fn()) != _MODE_CODE.get(want):
            raise RuntimeError(
                f"mojolearn Embedding: numeric_mode={want!r} was requested but {mod.__name__} "
                "reports another compile-time mode; rebuild it with bash bindings/build_embedding.sh"
            )
        return mod

    def _ids(self, ids):
        raw, _ = _materialize(ids, "ids")
        if raw.dtype[1] not in "iu":
            # Before this check a float id was cast to int32 and 1.5 read
            # row 1, silently. torch refuses a float index tensor; an id is
            # a row number, so a non-integer one is refused by name.
            raise TypeError(
                f"mojolearn Embedding: ids must be integers, got dtype "
                f"{raw.dtype!r}; a float id is refused rather than truncated"
            )
        a, _ = as_i32_c(raw, ndim=None, name="ids")
        return a.reshape((int(a.size),)), tuple(a.shape)

    # -- device I/O (lane gap-neural-io) ------------------------------------

    _device_io = True

    def _dev_module(self):
        mod = self._extension()
        if not all(callable(getattr(mod, n, None)) for n in  # glue: the named binding entry points
                   ("seq_tensor_alloc", "embedding_forward_dev", "embedding_backward_dev")):
            raise RuntimeError(f"mojolearn Embedding: {mod.__name__} has no device I/O entries; "
                               "use host arrays (device=False)")
        return mod

    def to_device(self, a):
        """Upload a float32 array once, for use as `dy` in `backward` on the device."""
        return _EMBEDDING_NATIVE("emblayer_to_device", self, (a,), None, _EMBEDDING_CONTEXT)

    def _is_dev(self, x):
        from ._x_sequence_device import is_seq_tensor
        return is_seq_tensor(x)

    def forward(self, ids, device=False):
        """Look up rows: ``weight[ids]``.

        Parameters
        ----------
        ids : array of int
            Row indices, any shape.
        device : bool, default False
            Return the result as a device tensor, without copying it back to the host.
            Raises an error when this install has no device path.

        Returns
        -------
        Array of shape ``ids.shape + (embedding_dim,)``, float32
        """
        return _EMBEDDING_NATIVE("emblayer_forward", self, (ids, device), None, _EMBEDDING_CONTEXT)

    __call__ = forward

    def backward(self, ids, dy, grad=None):
        """Dense (num_embeddings, embedding_dim) gradient of `forward(ids)` for `dy`.

        With ``grad=None`` the gradient starts from zero. A `grad` array is a carried
        accumulator: the sum continues from its values (it is not modified), so microbatches
        given in ascending position order reproduce the unsplit gradient bit for bit. A `dy`
        from `to_device` keeps the computation and the result on the device.

        Parameters
        ----------
        ids : array of int
            The forward's ids.
        dy : array of float32, shape ``ids.shape + (embedding_dim,)``
            Gradient of the forward's output.
        grad : array or None, default None
            Gradient to continue from.

        Returns
        -------
        Array of shape (num_embeddings, embedding_dim), float32
        """
        # The gradient is training: on a CPU-only install it runs only inside
        # the internal reference context;
        # the shipped embedding_infer binding carries no backward at all.
        return _EMBEDDING_NATIVE("emblayer_backward", self, (ids, dy, grad), None, _EMBEDDING_CONTEXT)

    def _backward_dev(self, flat, shape, dy, grad):
        """`backward` on a resident dy: the gradient as a resident tensor."""
        from ._x_sequence_device import SequenceDeviceTensor
        mod = self._dev_module()
        t, d, v = int(flat.size), self.embedding_dim, self.num_embeddings
        if dy.b is not mod or tuple(dy.shape) != shape + (d,):
            raise ValueError(f"mojolearn Embedding: dy has shape {tuple(dy.shape)}, want {shape + (d,)} "
                             "(on this layer's binding)")
        accumulate = 0
        if grad is None:
            dw = SequenceDeviceTensor._new(mod, (v, d))
        else:
            if self._is_dev(grad):
                if grad.b is not mod or tuple(grad.shape) != (v, d):
                    raise ValueError(f"mojolearn Embedding: grad has shape {tuple(grad.shape)}, want ({v}, {d})")
                dw = grad.copy()          # the carry's bits; `grad` itself is not modified
            else:
                prev, _ = as_f32_c(grad, ndim=2, name="grad")
                if tuple(prev.shape) != (v, d):
                    raise ValueError(f"mojolearn Embedding: grad has shape {tuple(prev.shape)}, want ({v}, {d})")
                dw = SequenceDeviceTensor._wrap(mod, prev)
            accumulate = 1
        pad = _NO_PADDING if self.padding_idx is None else int(self.padding_idx)
        mod.embedding_backward_dev(
            # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_backward_dev_binding.
            [addr_ro(flat, name="ids")], [dy.h, dw.h],
            [v, d, t, pad, accumulate, _PLAN_CODE[self.plan]],
        )
        return dw

    # -- saved tables (lane/inference-embedding-ivf-cholesky, 2026-09-15) ---

    def save(self, path):
        """Save the table and settings to `path` as an `.npz` file.

        A loaded table can look ids up with `forward`, including on a CPU-only install, where
        `backward` is not available.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a layer written by `save`. The dtype and shape are checked; nothing is cast."""
        return _EMBEDDING_NATIVE("emblayer_load", cls, (path,), None, _EMBEDDING_CONTEXT)


# The 0.8.55 writer's members, in its order (members are written sorted).
# `save` is connected here; `load` is the layer's own native entry
# (bindings/embedding_layer_native.mojo), which builds the layer with its
# native lifetime.
_native_archive_bridge.register(Embedding, _EMBEDDING_FORMAT, (
    *_native_archive_bridge.head(_EMBEDDING_FORMAT),
    ("str", "plan", "plan"),
    ("array", "weight", "weight"),
    ("i8", "meta", (("int", "num_embeddings"), ("int", "embedding_dim"),
                    ("none", "padding_idx", _NO_PADDING))),
))


# Package-import connection: the native forward/load owner and the native
# `__setattr__` / `__del__` of the layer (bindings/embedding_layer_native.mojo).
from . import _training_impl as _embedding_training  # noqa: E402
from ._native_optimizer_bridge import connect_embedding as _connect_embedding  # noqa: E402
_EMBEDDING_NATIVE, _EMBEDDING_CONTEXT = _connect_embedding(_embedding_training, Embedding, _ARCHIVE_CONTEXT)

__all__ = ["Embedding"]
