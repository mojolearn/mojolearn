# SPDX-License-Identifier: Apache-2.0
"""The per-shard work of the neighbor and IVF multi-GPU drivers, run inside
the native shard workers (bindings/parallel_shard_native.mojo).

The drivers' partition, scheduling, merge and join are native. What a shard
worker runs for these three drivers is the plain estimators' own public
methods (NearestNeighbors.fit/kneighbors, the estimators' query methods, the
IVF partial search and the reference vote), which are not one native call
each yet: they are the neighbors family's own migration
(tools/public_runtime_inventory.py, section neighbors). Each function here
is exactly the old `_parallel_worker` operation of the same name, called
with the shard's decoded arguments; a shard model is rebuilt from its state
record without a constructor. No function here orchestrates a shard or a
merge."""

_SNAPSHOTS = {}


def _model(cls, state):
    # glue: the shard's model from its state record (the old worker unpickled it)
    model = object.__new__(cls)
    model.__dict__.update(state)
    model.__dict__['_resident_holder'] = {}
    return model


def _release(token):
    # glue: drop a stored snapshot and the resident handle it owns
    held = _SNAPSHOTS.pop(token, None)
    if held is not None:
        release = getattr(held, '_release_resident_index', getattr(held, '_release_resident_fit', None))
        if callable(release):
            release()


def reference_shard(args):
    """`neighbor_reference`: a fresh brute-force index over one block of
    reference rows, queried for its k nearest rows."""
    from .neighbors import NearestNeighbors
    params, index, query, k = args
    return NearestNeighbors(**params).fit(index).kneighbors(query, n_neighbors=k)


def reference_vote(args):
    """`neighbor_vote`: the estimator's vote over merged neighbors."""
    from .parallel_neighbors_reference import _vote
    from . import neighbors
    name, state, distances, indices, method = args
    return _vote(_model(getattr(neighbors, name), state), distances, indices, method)


def query_state(args):
    """`neighbor_state`: store (a state) or release (None) this worker's
    snapshot of the queried estimator under `token`."""
    from . import neighbors, density
    token, name, state = args
    if state is None:
        _release(token)
        return None
    cls = getattr(neighbors, name, None) or getattr(density, name)
    _SNAPSHOTS[token] = _model(cls, state)
    return None


def query_rows(args):
    """`neighbor_query`: one block of query rows through the stored
    estimator's query method, with its two diagnostics."""
    token, X, method, kwargs = args
    state = _SNAPSHOTS.get(token)
    if state is None:
        raise RuntimeError('neighbor worker has no snapshot for this query')
    result = getattr(state, method)(X, **kwargs)
    diagnostics = {name: getattr(state, name) for name in ('used_query_tile_', 'n_candidate_distances_')  # glue: two named diagnostic attributes
                   if hasattr(state, name)}
    return result, diagnostics


def ivf_store(args):
    """`ivf_store`: keep one IVF shard index in this worker under `token`."""
    from ._ivf_impl import IVFIndex
    token, state = args
    shard = _model(IVFIndex, state)
    shard._entry(shard._extension(), 'ivf_flat_partial_search')
    _SNAPSHOTS[token] = shard
    return shard.n_rows_


def ivf_search(args):
    """`ivf_search_stored`: the stored shard's partial search."""
    from .parallel_ivf import _partial_search
    token, queries = args
    shard = _SNAPSHOTS.get(token)
    if shard is None:
        raise RuntimeError('IVF shard is not stored in this worker')
    return _partial_search(shard, queries)


def release(args):
    """Drop this worker's snapshot under `token`."""
    _release(args[0])
    return None


#: The operations a shard worker may run, by name.
OPERATIONS = dict(reference_shard=reference_shard, reference_vote=reference_vote, query_state=query_state,
                  query_rows=query_rows, ivf_store=ivf_store, ivf_search=ivf_search, release=release)
