# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE PREP LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `prep` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("prep-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "prep-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_prep_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _prep_transformer(m, X, Xh, **attrs):
    parts = {k: _h(getattr(m, k)) for k in attrs.get("attrs", ())}
    parts["transform"] = _h(m.transform(X[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xh[:256]),))


@lane("x-prep-robust-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RobustScaler().fit(X)
    return _prep_transformer(m, X, Xh, attrs=("center_", "scale_"))


@lane("x-prep-maxabs-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MaxAbsScaler().fit(X)
    return _prep_transformer(m, X, Xh, attrs=("scale_",))


_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "x-prep-robust-scaler", "x-prep-maxabs-scaler")


def _prep_categorical(X):
    """A few categories per column, ties everywhere, -1 from the denormal rows."""
    return np.clip(np.floor(X), -4, 4).astype(np.float32)


@lane("x-prep-ordinal-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Xq)
    parts = {f"cat{j}": _h(c) for j, c in enumerate(m.categories_)}
    parts["transform"] = _h(m.transform(Xq[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


@lane("x-prep-onehot-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.OneHotEncoder(handle_unknown="ignore", drop="if_binary").fit(Xq)
    parts = {f"cat{j}": _h(c) for j, c in enumerate(m.categories_)}
    parts["transform"] = _h(m.transform(Xq[:256]))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical),
            "x-prep-ordinal-encoder", "x-prep-onehot-encoder")


@lane("x-prep-target-encoder")
def _(ml, X, yc, yr, Xh=None):
    Xq, Xhq = _prep_categorical(X), _prep_categorical(Xh)
    m = ml.TargetEncoder(random_state=3)
    cross = m.fit_transform(Xq, yr)
    mb = ml.TargetEncoder(random_state=3)
    cross_b = mb.fit_transform(Xq, yc)
    parts = dict(cross=_h(cross), cross_binary=_h(cross_b), mean=_h(m.target_mean_),
                 enc=_h(*m.encodings_), transform=_h(m.transform(Xq[:256])))
    return _fit(parts, m, lambda e: (e.transform(Xhq[:256]),))


_batch_decl(_rows_calls("transform", sl=slice(0, 256), prep=_prep_categorical), "x-prep-target-encoder")
