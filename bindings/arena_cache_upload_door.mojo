# SPDX-License-Identifier: Apache-2.0
# Whole public DeviceCache.id_of. Constructor/release/close remain separate.
#
# Lane arena (2026-11-21): UNPACK is the owner slot read and the buffer-key
# admission; RUN is the typed find (a hit, counted natively) or the put with
# the interpreter lock released and the publication
# (`bindings/arena_cache_run.mojo` `arena_cache_upload`; a lost publication
# race frees its own id and returns the id published first); PACK is the id.
# The instance dictionary, the Python counters (`PyNumber_InPlaceAdd`, the
# arbitrary-precision binary increment), the put / free builtins called with
# `PyObject_Call` and the double-failure exception wrapper are gone: the
# counters are native integers and a failed cleanup is reported in words.
from std.python import Python, PythonObject
from bindings.arena_cache_get_door import arena_cache_key, arena_cache_view
from bindings.arena_scope_door import arena_cache_slot
from bindings.optimizer_public_words_unpack import PENDING_PYTHON_ERROR, PYTHON_ERROR_PENDING
from core.arena_state import ARENA_CACHES, arena_handle


struct CachePendingError(Movable):
    # Own the fetched triple in fixed fields while cleanup can release the
    # GIL. Avoid allocating/copying a List after a resident slot needs cleanup.
    var kind: Int
    var value: Int
    var traceback: Int

    def __init__(out self):
        self.kind = 0
        self.value = 0
        self.traceback = 0


def _take_pending(error: Error) raises -> CachePendingError:
    var saved = CachePendingError()
    if String(error) == PYTHON_ERROR_PENDING:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        saved.kind = pending[].words[0]
        saved.value = pending[].words[1]
        saved.traceback = pending[].words[2]
        pending[].words[0] = 0
        pending[].words[1] = 0
        pending[].words[2] = 0
    return saved^


def _drop_pending(mut saved: CachePendingError) raises:
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    if saved.kind:
        release(saved.kind)
        saved.kind = 0
    if saved.value:
        release(saved.value)
        saved.value = 0
    if saved.traceback:
        release(saved.traceback)
        saved.traceback = 0


def _restore_pending(mut saved: CachePendingError) raises:
    if saved.kind:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        pending[].words[0] = saved.kind
        pending[].words[1] = saved.value
        pending[].words[2] = saved.traceback
        saved.kind = 0
        saved.value = 0
        saved.traceback = 0


def arena_cache_id_of(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var source = args[0]
    var view = arena_cache_view(model, context)
    var key = arena_cache_key(source, context)
    if key.words < 1:
        # No entry of an empty buffer can exist: the upload refuses it.
        raise Error("ValueError: DeviceCache: an empty buffer has no device copy")
    var lease = ARENA_CACHES.acquire(arena_handle(view.word))
    return PythonObject(arena_cache_slot(lease, view.word, view.store, source, key.address, key.words))
