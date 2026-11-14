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
from . import _native_archive_bridge
from ._mode import NumericModeMixin

_NO_PADDING = -1
#: The saved-table format tag (`save`, `load`, `mojolearn.host_model`;
#: lane/inference-embedding-ivf-cholesky, 2026-09-15).
_EMBEDDING_FORMAT = "mojolearn-embedding-1"

# The compiled saved-model owner, connected once at import: `save` is one
# native call (bindings/archive_schema_native_door.mojo writes the schema
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
    # (bindings/embedding_layer_native_door.mojo).

    def __getstate__(self):
        return _EMBEDDING_NATIVE("emblayer_state", self, (), None, _EMBEDDING_CONTEXT)

    # -- device I/O (lane gap-neural-io) ------------------------------------
    # The id dtype refusal, the binding mode cross-check, the device-entry
    # probe and the resident backward's shape refusals run in Mojo
    # (bindings/embedding_layer_native_door.mojo); the dead Python twins
    # `_extension`, `_ids`, `_dev_module`, `_is_dev` and `_backward_dev` are
    # gone (lane py-lm-neural, 2026-11-14).

    _device_io = True

    def to_device(self, a):
        """Upload a float32 array once, for use as `dy` in `backward` on the device."""
        return _EMBEDDING_NATIVE("emblayer_to_device", self, (a,), None, _EMBEDDING_CONTEXT)

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
# (bindings/embedding_layer_native_door.mojo), which builds the layer with its
# native lifetime.
_native_archive_bridge.register(Embedding, _EMBEDDING_FORMAT, (
    *_native_archive_bridge.head(_EMBEDDING_FORMAT),
    ("str", "plan", "plan"),
    ("array", "weight", "weight"),
    ("i8", "meta", (("int", "num_embeddings"), ("int", "embedding_dim"),
                    ("none", "padding_idx", _NO_PADDING))),
))


# Package-import connection: the native forward/load owner and the native
# `__setattr__` / `__del__` of the layer (bindings/embedding_layer_native_door.mojo).
from . import _training_impl as _embedding_training  # noqa: E402
from ._native_optimizer_bridge import connect_embedding as _connect_embedding  # noqa: E402
_EMBEDDING_NATIVE, _EMBEDDING_CONTEXT = _connect_embedding(_embedding_training, Embedding, _ARCHIVE_CONTEXT)

__all__ = ["Embedding"]
