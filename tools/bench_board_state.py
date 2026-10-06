"""Untimed benchmark receipts for scored outputs and public model exports.

This is evidence glue, never part of the library runtime or measured operation.
Output identity is deliberately separate from model identity. A missing public
export remains explicit and must not be treated as a verified model match.
"""
import hashlib
import json
from pathlib import Path
import tempfile


# Reviewed against the public save/load and fit methods at numerical freeze
# 4fc63c9d5486261f27fdd55eb31d1900e2cc0e8f. These are fitted-state contracts,
# not a declaration that arbitrary public save() exports are complete models.
PUBLIC_FITTED_STATE_CONTRACTS = {
    'PCA': {
        'id': 'mojolearn.public-fitted-state/pca-1',
        'format': 'mojolearn-pca-1',
        'fields': ('components', 'estimator', 'explained_variance', 'explained_variance_ratio',
                   'format', 'mean', 'meta', 'noise_variance', 'numeric_mode', 'singular_values', 'svd_solver'),
        'source': 'python/mojolearn/decomposition.py:PCA.fit/_fit_randomized/save/load',
        'excluded': {'input_copied_': 'input-copy diagnostic, not fitted numerical state',
                     'constructor_settings': 'separately pinned estimator settings; save is not a refit checkpoint'},
    },
    'KMeans': {
        'id': 'mojolearn.public-fitted-state/kmeans-1',
        'format': 'mojolearn-kmeans-1',
        'fields': ('centers', 'estimator', 'format', 'init', 'labels', 'meta', 'metric', 'numeric_mode', 'reals'),
        'source': 'python/mojolearn/cluster.py:KMeans.fit/save/load',
        'excluded': {'init_centroids': 'original refit input, not learned centers; pinned by workload/settings',
                     'input_copied_': 'input-copy diagnostic, not fitted numerical state'},
    },
}
_PUBLIC_FITTED_TYPES = {
    ('mojolearn.decomposition', 'PCA'): 'PCA',
    ('mojolearn.cluster', 'KMeans'): 'KMeans',
    ('mojolearn._classical_host', 'HostPCA'): 'PCA',
    ('mojolearn._classical_host', 'HostKMeans'): 'KMeans',
}


def public_fitted_state_paths(estimator):
    """The reviewed complete path set for a future frozen worker recipe."""
    return ['$.%s' % name for name in PUBLIC_FITTED_STATE_CONTRACTS[estimator]['fields']]


def _saved_text(arrays, name):
    value = arrays[name]
    if value.dtype.kind != 'U' or value.shape != (1,):
        raise ValueError(name + ': expected one Unicode export value')
    return str(value[0])


def normalize_public_fitted_state(arrays, estimator, saved_estimator):
    """Validate a known saved schema, without loading or running a model.

    Preserve numerical bytes and dtypes. Only the export's Python class name is
    normalized: the explicit public host subclasses inherit these save formats.
    The original full export hash remains in the capture's provenance.
    """
    import numpy as np
    contract = PUBLIC_FITTED_STATE_CONTRACTS[estimator]
    if set(arrays) != set(contract['fields']):
        missing = sorted(set(contract['fields']) - set(arrays))
        extra = sorted(set(arrays) - set(contract['fields']))
        raise ValueError('export field set changed; missing=%r unexpected=%r' % (missing, extra))
    if any(not isinstance(v, np.ndarray) or v.dtype.hasobject for v in arrays.values()):
        raise ValueError('export requires typed non-object arrays')
    if _saved_text(arrays, 'format') != contract['format']:
        raise ValueError('unreviewed saved format version')
    if saved_estimator not in (estimator, 'Host' + estimator) or _saved_text(arrays, 'estimator') != saved_estimator:
        raise ValueError('saved estimator class differs from reviewed owner')
    if _saved_text(arrays, 'numeric_mode') not in ('fast', 'deterministic', 'identical'):
        raise ValueError('invalid saved numeric mode')

    def exact(name, dtype, shape):
        value = arrays[name]
        if value.dtype.str != dtype or value.shape != shape:
            raise ValueError('%s: expected dtype=%s shape=%r, observed dtype=%s shape=%r' %
                             (name, dtype, shape, value.dtype.str, value.shape))

    if estimator == 'PCA':
        exact('meta', '<i8', (4,))
        nc, nf, ns, whiten = map(int, arrays['meta'])
        if nc < 1 or nf < 1 or ns < 2 or whiten not in (0, 1):
            raise ValueError('invalid PCA fitted counts/whitening metadata')
        exact('components', '<f4', (nc, nf))
        exact('mean', '<f4', (nf,))
        for name in ('singular_values', 'explained_variance', 'explained_variance_ratio'):
            exact(name, '<f4', (nc,))
        exact('noise_variance', '<f8', (1,))
        if not _saved_text(arrays, 'svd_solver'):
            raise ValueError('missing PCA solver')
    elif estimator == 'KMeans':
        exact('meta', '<i8', (9,))
        exact('reals', '<f8', (5,))
        k, d, nfit, niter, metric, init, _, _, _ = map(int, arrays['meta'])
        if k < 1 or d < 1 or nfit < 1 or niter < 0:
            raise ValueError('invalid KMeans fitted counts')
        exact('centers', '<f4', (k, d))
        exact('labels', '<i4', (nfit,))
        metrics = {'euclidean': 0, 'l2_expanded': 0, 'l2_sqrt_expanded': 1}
        inits = {'k-means++': 0, 'random': 1, 'array': 2}
        if metrics.get(_saved_text(arrays, 'metric')) != metric or inits.get(_saved_text(arrays, 'init')) != init:
            raise ValueError('KMeans saved names and codes differ')
    state = dict(arrays)
    state['estimator'] = estimator
    return state


def public_fitted_state(model):
    """Read one public save export after clocks stop; never inspect handles.

    Returns (typed state, provenance), or (None, explicit unavailable evidence).
    No existing receipt is upgraded, and incomplete exports remain partial.
    """
    key = (type(model).__module__, type(model).__name__)
    estimator = _PUBLIC_FITTED_TYPES.get(key)
    if estimator is None:
        missing = ['reviewed complete typed fitted-state export']
        if key in (('mojolearn.linear_model', 'LinearRegression'),
                   ('mojolearn._classical_host', 'HostLinearRegression')):
            missing = ['retained training mean _x_mean', 'retained training mean _y_mean']
        return None, dict(status='UNAVAILABLE', missing_state=missing,
                          reason='Public export is not a reviewed complete fitted-state contract',
                          partial_export=model_receipt(model))
    contract = PUBLIC_FITTED_STATE_CONTRACTS[estimator]
    metadata = dict(contract=contract['id'], contract_source=contract['source'],
                    contract_paths=public_fitted_state_paths(estimator),
                    capture_source='public save npz', model_class='.'.join(key),
                    excluded_non_fitted_state=dict(contract['excluded']),
                    normalized_metadata={'estimator': dict(saved=key[1], logical=estimator,
                        reason='public host route subclass uses the same versioned fitted-state format')})
    try:
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='mojolearn-typed-fitted-state-') as directory:
            path = Path(directory) / 'model.npz'
            model.save(path)
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
        metadata['complete_export_sha256'] = canonical_hash(arrays)
        state = normalize_public_fitted_state(arrays, estimator, key[1])
        return state, metadata
    except Exception as exc:
        return None, dict(metadata, status='UNAVAILABLE', missing_state=['validated complete fitted-state export'],
                          reason=type(exc).__name__ + ': ' + str(exc))


def canonical_hash(value):
    """SHA256 of typed, framed nested state (including array shape and dtype)."""
    import numpy as np
    h = hashlib.sha256(b"mojolearn-benchmark-state-v1\0")

    def frame(kind, raw):
        h.update(kind.encode() + b":" + str(len(raw)).encode() + b":" + raw)

    def visit(obj):
        if isinstance(obj, dict):
            frame("dict", str(len(obj)).encode())
            for key in sorted(obj):
                frame("key", str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            frame("sequence", str(len(obj)).encode())
            for item in obj:
                visit(item)
        elif isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError("object arrays cannot establish state identity")
            arr = np.ascontiguousarray(obj)
            frame("array", json.dumps([arr.dtype.str, list(arr.shape)], separators=(",", ":")).encode())
            frame("bytes", arr.tobytes())
        elif isinstance(obj, np.generic):
            visit(np.asarray(obj))
        elif obj is None or isinstance(obj, (str, bool, int, float)):
            frame(type(obj).__name__, json.dumps(obj, allow_nan=False).encode())
        else:
            raise TypeError("unsupported exported state type: " + type(obj).__name__)
    visit(value)
    return h.hexdigest()


def _model_receipt(model):
    """Hash a public snapshot; never fit, predict, or inspect opaque handles."""
    if model is None:
        return {"status": "unavailable", "reason": "runner does not retain an exportable model"}
    info = {"class": type(model).__name__, "schema": "mojolearn-benchmark-state-v1"}
    if not type(model).__module__.startswith("mojolearn"):
        return dict(info, status="unavailable", reason="opponent model export is not normalized")
    if callable(getattr(model, "state_dict", None)):
        state = model.state_dict()
        return dict(info, status="ok", source="public state_dict", sha256=canonical_hash(state))
    if callable(getattr(model, "save", None)):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="mojolearn-scored-state-") as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as saved:
                # Device routing metadata is not numerical state. Preserve
                # the complete export hash too so this exclusion is auditable.
                state = {key: saved[key] for key in saved.files}
                full = canonical_hash(state)
                omitted = [key for key in ("device",) if key in state]
                for key in omitted:
                    del state[key]
                return dict(info, status="ok", source="public save npz", sha256=canonical_hash(state),
                            complete_export_sha256=full, excluded_metadata=omitted)
    return dict(info, status="unavailable", reason="no public state_dict or save export")


def model_receipt(model):
    try:
        return _model_receipt(model)
    except Exception as exc:
        # Keep the scored measurement and its original diagnostic. Failed
        # exports are never an identity claim and can be repaired separately.
        return {"status": "error", "class": type(model).__name__,
                "error": type(exc).__name__ + ": " + str(exc)}


def runner_model(runner):
    callback = getattr(runner, "model_for_hash", None)
    if callback is not None:
        return callback()
    for name in ("est", "nn", "kd", "trainer", "model"):
        model = getattr(runner, name, None)
        if model is not None:
            return model
    return None


def scored_receipt(outputs, runner=None, model=None):
    present = outputs is not None and bool(outputs) and all(value is not None for value in outputs.values())
    return {"schema": "mojolearn-scored-receipt/1", "output_status": "ok" if present else "unavailable",
            "output_sha256": canonical_hash(outputs) if present else None,
            "model": model_receipt(model if model is not None else runner_model(runner))}
