"""Testing oracle retained from ForestProtocol at e087692fd.

This is the previous Python orchestration, deliberately outside native-call
audits. It qualifies parity with that behavior, not runtime compliance.
"""
from mojolearn._buffer import empty, as_f32_c
from mojolearn._arrays import _addr, _addr_ro


def legacy_auxiliary(model, x, *, leaves=False, prefixes=False, prediction=False):
    if not hasattr(model, '_offsets'):
        raise RuntimeError('this estimator is not fitted yet')
    xa, _ = as_f32_c(x, ndim=2, name='X')
    rows, features = xa.shape
    if features != model.n_features_in_:
        raise ValueError(f'X has {features} features, fit saw {model.n_features_in_}')
    trees, outputs = int(model._n_trees), int(model._num_outputs)
    leaf = empty((rows, trees), '<i4') if leaves else None
    prefix = empty((rows, trees, outputs), '<f4') if prefixes else None
    raw = empty((rows, outputs), '<f4') if prediction else None
    native = model._bind('_mojolearn_x_trees')
    entry = getattr(native, 'x_trees_forest_auxiliary', None)
    if not callable(entry):
        raise NotImplementedError('this tree binding does not expose combined auxiliary outputs')
    arrays = tuple(getattr(model, name) for name in ('_offsets', '_colid', '_quesval', '_left_child', '_leaves'))
    wrote = entry([_addr_ro(a) for a in arrays], _addr_ro(xa),
                  0 if leaf is None else _addr(leaf),
                  0 if prefix is None else _addr(prefix),
                  0 if raw is None else _addr(raw),
                  [rows, features, trees, outputs, int(model._BINDING == '_mojolearn_rf')])
    if wrote != rows:
        raise RuntimeError(f'forest auxiliary wrote {wrote} of {rows} rows')
    return raw, leaf, prefix
