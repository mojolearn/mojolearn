# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`Embedding`, profile `mojolearn.identical.embedding.fp32.v1`, on the GPU.

The Python door of `bindings/_mojolearn_embedding.mojo`, which reaches
`embedding/checks/embedding_identical.mojo`'s gather and ascending fold.
`embedding/IDENTICAL_EMBEDDING_CONTRACT.md` is the specification; the gate
is `pixi run check-embedding` and `tools/embedding_sabotage_arm.sh`.

The semantics are `torch.nn.Embedding`'s forward and its dense
`embedding_dense_backward`, with the contract's refusals:

    max_norm, norm_type != 2.0           here, by name (a forward that
                                         renormalizes W is outside the profile)
    scale_grad_by_freq=True              here, by name
    sparse=True                          here, by name (DEVIATION 1314: the
                                         dense (V, d) gradient only)
    weight=None                          here, by name: initialization is not
                                         part of the profile; pass the table
    an id below 0 or at or past V        Mojo host, by name (contract 8,
                                         never clamped)
    a NaN or an infinity in W, dY or a   Mojo host, by name (contract 9.1)
    carried gradient
    an empty ids or dy                   here (the buffer layer refuses a
                                         zero-size input)

`padding_idx` follows torch: a negative value counts from the end, the
forward gathers that row like any other, and the backward drops its
positions at the source and STORES +0.0 in its row (contract section 8).
`plan` selects the backward's execution plan for the run structure (contract
section 6): "scan" (PLAN_SCAN, the default) or "sort" (PLAN_SORT, the device
total-key bitonic sort). The plan is not the specification: both enumerate
each row's contributors in ascending position, and the contract's clause (d)
holds counts, perm and dW bit-identical across the two. No dispatch threshold
is measured, so nothing picks "sort" for you.

`backward(ids, dy, grad=...)` is the microbatch CARRY (contract 7.4): the
fold continues from `grad`'s bits, so microbatches presented in ascending
position order reproduce the unsplit gradient bit for bit.

There is no autograd here. The class holds the table and runs the two
calls; `weight` is never updated by this module.

NO SPEED CLAIM.
"""
from . import _backend, _serialize
from ._array import Array
from ._buffer import _materialize, addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}
_NO_PADDING = -1
#: ORDER MATCHES embedding/checks/embedding_sort.mojo's PLAN_SCAN = 0, PLAN_SORT = 1.
_PLAN_CODE = {"scan": 0, "sort": 1}
#: The saved-table format tag (`save`, `load`, `mojolearn.host_model`;
#: lane/inference-embedding-ivf-cholesky, 2026-09-15).
_EMBEDDING_FORMAT = "mojolearn-embedding-1"


class Embedding(NumericModeMixin):
    """A `(num_embeddings, embedding_dim)` float32 table with the pinned gather and fold.

    Parameters
    ----------
    num_embeddings : int
    embedding_dim : int
    padding_idx : int or None, default None
    max_norm : None
        Only None; anything else is refused by name.
    norm_type : float, default 2.0
        Only 2.0 (it has no effect without max_norm).
    scale_grad_by_freq : bool, default False
        Only False.
    sparse : bool, default False
        Only False.
    weight : array-like (num_embeddings, embedding_dim), required
        The table, copied to float32 C order at construction.
    plan : {"scan", "sort"}, default "scan"
        The backward's run-structure execution plan (contract section 6).
        Both give the same bits; anything else is refused by name.

    Attributes
    ----------
    weight : Array (num_embeddings, embedding_dim) float32
    """

    _BINDING = "_mojolearn_embedding"

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, weight=None, plan="scan"):
        for name, v in (("num_embeddings", num_embeddings), ("embedding_dim", embedding_dim)):
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
                f"mojolearn Embedding: plan must be 'scan' or 'sort' (contract section 6's "
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
        w, _ = as_f32_c(weight, ndim=2, name="weight")
        if tuple(w.shape) != (num_embeddings, embedding_dim):
            raise ValueError(
                f"mojolearn Embedding: weight has shape {tuple(w.shape)}, want ({num_embeddings}, {embedding_dim})"
            )
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
                        scale_grad_by_freq=False, sparse=False, numeric_mode=None, plan="scan"):
        """torch's `Embedding.from_pretrained` without `freeze` (nothing here trains the table)."""
        w, _ = as_f32_c(embeddings, ndim=2, name="embeddings")
        v, d = (int(s) for s in w.shape)
        return cls(v, d, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, weight=w,
                   numeric_mode=numeric_mode, plan=plan)

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

    def forward(self, ids):
        """`weight[ids]`, shape `ids.shape + (embedding_dim,)`, float32."""
        flat, shape = self._ids(ids)
        t, d = int(flat.size), self.embedding_dim
        y = empty((t * d,), "<f4")
        self._extension().embedding_forward(
            # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_forward_binding.
            # weight, ids, y_out
            [addr_ro(self.weight, name="weight"), addr_ro(flat, name="ids"), addr(y, name="y")],
            # V, d, T
            [self.num_embeddings, d, t],
        )
        return y.reshape(shape + (d,))

    __call__ = forward

    def backward(self, ids, dy, grad=None):
        """The dense `(num_embeddings, embedding_dim)` gradient of `forward(ids)` for `dy`.

        `grad=None` is a fresh gradient (+0.0 fill). A `grad` array is the
        carried accumulator (contract 7.4): its bits are copied and the fold
        continues from them; `grad` itself is not modified.
        """
        # The gradient is training: on a CPU-only install it runs only inside
        # the internal reference context;
        # the shipped embedding_infer binding carries no backward at all.
        from ._cpu_reference import require_training
        require_training(self)
        flat, shape = self._ids(ids)
        t, d, v = int(flat.size), self.embedding_dim, self.num_embeddings
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
        self._extension().embedding_backward(
            # ORDER MATCHES bindings/_mojolearn_embedding.mojo::embedding_backward_binding.
            # dy, ids, dw (read first when accumulate)
            [addr_ro(g, name="dy"), addr_ro(flat, name="ids"), addr(dw, name="dw")],
            # V, d, T, padding_idx, accumulate, plan
            [v, d, t, pad, accumulate, _PLAN_CODE[self.plan]],
        )
        return dw.reshape((v, d))

    # -- saved tables (lane/inference-embedding-ivf-cholesky, 2026-09-15) ---

    def save(self, path):
        """Write the table to `path` as an npz: `weight` `<f4` (V, d),
        `meta` `<i8` [V, d, padding_idx or -1], `plan` and the tier. A loaded
        table looks ids up (`forward`); on a CPU-only install that is public
        inference through `_mojolearn_embedding_infer_host`, which carries no
        backward."""
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
        """Load a table written by `save`, dtype and shape checked, nothing
        cast."""
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _EMBEDDING_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 3 are needed")
        v, d, pad = (int(meta[i]) for i in range(3))
        weight = _serialize.exact(arrays, "weight", "<f4")
        if tuple(weight.shape) != (v, d):
            raise ValueError(f"mojolearn: {path!r} weight has shape {tuple(weight.shape)}, not {(v, d)}")
        obj = cls(v, d, padding_idx=None if pad == _NO_PADDING else pad, weight=weight,
                  plan=_serialize.scalar_str(arrays, "plan"))
        _restore_mode(obj, arrays)
        return obj


__all__ = ["Embedding"]
