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
import itertools

from . import _backend, _serialize
from ._array import Array
from ._buffer import _materialize, addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1}
_NO_PADDING = -1
#: ORDER MATCHES embedding/checks/embedding_sort.mojo's PLAN_SCAN = 0, PLAN_SORT = 1.
#: PLAN_AUTO = 2 is embedding/checks/embedding_identical.mojo's.
_PLAN_CODE = {"scan": 0, "sort": 1, "auto": 2}
#: Table tokens for the binding's resident device copy (one per layer and
#: per assignment of `weight`; never pickled or copied with the layer).
_TABLE_TOKENS = itertools.count(1)
#: The saved-table format tag (`save`, `load`, `mojolearn.host_model`;
#: lane/inference-embedding-ivf-cholesky, 2026-09-15).
_EMBEDDING_FORMAT = "mojolearn-embedding-1"


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

    _BINDING = "_mojolearn_embedding"

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, weight=None, plan="auto"):
        for name, v in (("num_embeddings", num_embeddings), ("embedding_dim", embedding_dim)):  # glue: validates two int arguments
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"mojolearn Embedding: {name} must be an int, got {type(v).__name__}")
            if v < 1:
                raise ValueError(f"mojolearn Embedding: {name} must be at least 1, got {v}")
        if max_norm is not None:
            raise ValueError(
                "mojolearn Embedding: max_norm is REFUSED. torch renormalizes the "
                "gathered rows of the weight in place during the forward, and that "
                "write is outside profile mojolearn.identical.embedding.fp32.v1"
            )
        if norm_type != 2.0:
            raise ValueError("mojolearn Embedding: norm_type is only 2.0 (it has no effect without max_norm, which is refused)")
        if scale_grad_by_freq:
            raise ValueError(
                "mojolearn Embedding: scale_grad_by_freq=True is REFUSED; the profile's "
                "backward is the plain ascending fold (contract 5.1) with no division"
            )
        if sparse:
            raise ValueError("mojolearn Embedding: sparse=True is REFUSED (DEVIATION 1314: the dense (V, d) gradient only)")
        if not isinstance(plan, str) or plan not in _PLAN_CODE:
            raise ValueError(
                f"mojolearn Embedding: plan must be 'auto', 'scan' or 'sort' (contract section 6's "
                f"PLAN_SCAN and PLAN_SORT), got {plan!r}"
            )
        if padding_idx is not None:
            if isinstance(padding_idx, bool) or not isinstance(padding_idx, int):
                raise TypeError(f"mojolearn Embedding: padding_idx must be an int or None, got {type(padding_idx).__name__}")
            if not -num_embeddings <= padding_idx < num_embeddings:
                raise ValueError(
                    f"mojolearn Embedding: padding_idx must be within [-{num_embeddings}, {num_embeddings}), got {padding_idx}"
                )
            if padding_idx < 0:
                padding_idx = num_embeddings + padding_idx
        if weight is None:
            raise ValueError(
                "mojolearn Embedding: weight is required. Initialization is not part of "
                "the profile (a host normal draw is not pinned across platforms); pass "
                "the table, or use Embedding.from_pretrained"
            )
        w, copied = as_f32_c(weight, ndim=2, name="weight")
        if tuple(w.shape) != (num_embeddings, embedding_dim):
            raise ValueError(
                f"mojolearn Embedding: weight has shape {tuple(w.shape)}, want ({num_embeddings}, {embedding_dim})"
            )
        if not copied:
            # the layer's own bytes: the binding keeps a device copy of the
            # table across forwards, so the caller's array must not alias it
            w = frombytes(w.tobytes(), "<f4", (num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.plan = plan
        self.weight = w

    @classmethod
    def from_pretrained(cls, embeddings, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False, numeric_mode=None, plan="auto"):
        """Build a layer from an existing (V, d) table, as torch's `Embedding.from_pretrained`.

        There is no `freeze` argument, because nothing here trains the table. The other
        arguments are the constructor's.
        """
        w, _ = as_f32_c(embeddings, ndim=2, name="embeddings")
        v, d = (int(s) for s in w.shape)  # glue: two shape dimensions only
        return cls(v, d, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, weight=w,
                   numeric_mode=numeric_mode, plan=plan)

    # -- the resident device table (bindings/_mojolearn_embedding.mojo) ------

    def __setattr__(self, name, value):
        if name == "weight":
            self._drop_table()
        super().__setattr__(name, value)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_table_tok", None)
        return state

    def __del__(self):
        try:
            self._drop_table()
        except Exception:
            pass

    def _drop_table(self):
        """Release this layer's resident device table, if it has one."""
        tok = self.__dict__.pop("_table_tok", None)
        if tok is None:
            return
        try:
            release = getattr(self._extension(), "embedding_table_release", None)
            if release is not None:
                release(tok)
        except Exception:
            pass

    def _table_token(self, mod):
        """This layer's table token, or None for a binding without the
        resident table (the host bindings)."""
        if getattr(mod, "embedding_table_release", None) is None:
            return None
        tok = self.__dict__.get("_table_tok")
        if tok is None:
            tok = next(_TABLE_TOKENS)
            self.__dict__["_table_tok"] = tok
        return tok

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
        from ._x_sequence_device import SequenceDeviceTensor
        return SequenceDeviceTensor._wrap(self._dev_module(), a)

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
        flat, shape = self._ids(ids)
        t, d = int(flat.size), self.embedding_dim
        if device:
            from ._x_sequence_device import SequenceDeviceTensor
            mod = self._dev_module()
            y = SequenceDeviceTensor._new(mod, shape + (d,))
            tok = self._table_token(mod)
            mod.embedding_forward_dev(
                # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_forward_dev_binding.
                [addr_ro(self.weight, name="weight"), addr_ro(flat, name="ids")], [y.h],
                [self.num_embeddings, d, t, 0 if tok is None else tok],
            )
            return y
        y = empty((t * d,), "<f4")
        mod = self._extension()
        tok = self._table_token(mod)
        mod.embedding_forward(
            # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_forward_binding.
            # weight, ids, y_out
            [addr_ro(self.weight, name="weight"), addr_ro(flat, name="ids"), addr(y, name="y")],
            # V, d, T[, token]
            [self.num_embeddings, d, t] if tok is None else [self.num_embeddings, d, t, tok],
        )
        return y.reshape(shape + (d,))

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
        from ._cpu_reference import require_training
        require_training(self)
        flat, shape = self._ids(ids)
        t, d, v = int(flat.size), self.embedding_dim, self.num_embeddings
        if self._is_dev(dy):
            return self._backward_dev(flat, shape, dy, grad)
        g, _ = as_f32_c(dy, ndim=None, name="dy")
        if tuple(g.shape) != shape + (d,):
            raise ValueError(f"mojolearn Embedding: dy has shape {tuple(g.shape)}, want {shape + (d,)}")
        g = g.reshape((t * d,))
        accumulate = 0
        if grad is None:
            dw = empty((v * d,), "<f4")
        else:
            prev, _ = as_f32_c(grad, ndim=2, name="grad")
            if tuple(prev.shape) != (v, d):
                raise ValueError(f"mojolearn Embedding: grad has shape {tuple(prev.shape)}, want ({v}, {d})")
            dw = frombytes(prev.tobytes(), "<f4", (v * d,))
            accumulate = 1
        pad = _NO_PADDING if self.padding_idx is None else int(self.padding_idx)
        mod = self._extension()
        plan = _PLAN_CODE[self.plan]
        if self.plan == "auto" and getattr(mod, "embedding_table_release", None) is None:
            plan = _PLAN_CODE["scan"]  # a binding without PLAN_AUTO (the host bindings)
        mod.embedding_backward(
            # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_backward_binding.
            # dy, ids, dw (read first when accumulate)
            [addr_ro(g, name="dy"), addr_ro(flat, name="ids"), addr(dw, name="dw")],
            # V, d, T, padding_idx, accumulate, plan
            [v, d, t, pad, accumulate, plan],
        )
        return dw.reshape((v, d))

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
        from .decomposition import _saved_mode
        arrays = {
            "format": _EMBEDDING_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "plan": str(self.plan),
            "weight": self.weight,
            "meta": Array.from_list(
                [int(self.num_embeddings), int(self.embedding_dim),
                 _NO_PADDING if self.padding_idx is None else int(self.padding_idx)],
                "<i8",
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a layer written by `save`. The dtype and shape are checked; nothing is cast."""
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _EMBEDDING_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        v, d, pad = (int(meta[i]) for i in range(3))  # glue: three metadata ints only
        weight = _serialize.exact(arrays, "weight", "<f4")
        if tuple(weight.shape) != (v, d):
            raise ValueError(f"mojolearn: {path!r} weight has shape {tuple(weight.shape)}, not {(v, d)}")
        obj = cls(v, d, padding_idx=None if pad == _NO_PADDING else pad, weight=weight,
                  plan=_serialize.scalar_str(arrays, "plan"))
        _restore_mode(obj, arrays)
        return obj


__all__ = ["Embedding"]
