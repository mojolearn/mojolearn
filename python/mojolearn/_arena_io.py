# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared transfer plumbing for arena-style bindings (lane py-shared).

Three pieces every arena family shares:

1. THE RANGES RUNNER: an arena program names its INPUT words and its OUTPUT
   words, and a binding entry built on `core/arena_io.mojo`
   (`upload_ranges`, `download_ranges`) moves only those: the other words
   are zeroed on the device (as the host arena's words arrive zeroed) and
   never come back. x_metrics (`x_metrics_run_ranges`) and x_prep
   (`x_prep_run_ranges`) run on it; `MOJOLEARN_ARENA_RANGES=0` sends both
   back to the whole-arena entries (the A/B arm). The range descriptors are
   laid out by the native program runner, `bindings/arena_prog_native.mojo`
   (lane migrate-prog-staging), which every `_Prog.run` calls once.

2. RESIDENT INPUTS (`DeviceCache`, `resident`): a device copy of a host
   buffer, uploaded once and named by an id in the binding's
   `core/device_store.mojo` store, keyed by BUFFER IDENTITY (its address and
   byte length; the cache holds the owner, so the address cannot be reused
   while the entry lives), and FREED DETERMINISTICALLY (`release`, `close`,
   or the end of the `with` block; never left to the garbage collector).
   A binding opts in by exporting `<prefix>_dev_put(addr, n_words) -> id`,
   `<prefix>_dev_free(id)` and `<prefix>_dev_live()`.

       with mojolearn._arena_io.resident():
           for params in grid:          # every x_metrics / x_prep program in
               score(model, X, y)       # the block uploads each big input once

   Inside `resident()` the caller promises not to write into a buffer it
   has already handed to a binding: the cache sees the same address and
   byte count and re-uses the device copy. The native runner creates the
   scope's per-binding caches. A family binding with its own state
   (optimizer moments, LM weights, an index) uses `DeviceCache` directly:
   `cache = DeviceCache(binding, "x_foo")`, `cache.id_of(arr)`,
   `cache.close()`.

3. DEVICE-ROWS INPUT (`DeviceRows`; lane cpu4-misc): a fold view of a host
   float32 C-contiguous 2-D Array (the base and an int64 index Array),
   handed by model_selection to an estimator that opts in (class attribute
   `_mojolearn_device_rows = True`). The native runner puts the base ONCE
   into its own binding's store (the active `resident()` scope's cache,
   identity-keyed, so every fold and candidate shares the upload) and
   gathers the fold rows on the device (`<prefix>_dev_take_rows`); the rows
   never come to the host. A runner without a device store (the host
   column, a CPU-only install) copies the fold rows' host snapshot.

Where a word travels moves no bit. The Python here only lays out Int32
range lists; a CPU-only install never reaches it (host bindings run the
arena in place).
"""
import threading

__all__ = ["DeviceCache", "DeviceRows", "resident", "RESIDENT_MIN_WORDS"]

#: Inputs below this many words always go up from the host arena: a
#: put costs one synchronized copy, which pays only on a big buffer.
RESIDENT_MIN_WORDS = 1 << 16


class DeviceCache:
    """Device copies of host buffers in one binding's store, by identity.

    `id_of(arr)` uploads `arr` (a C-contiguous Array of 4-byte items, or
    anything with `_addr`, `size` and `itemsize`) the first time and
    returns the same id while the entry lives. `release(arr)` frees one
    entry, `close()` all of them; both free on the device at once."""

    def __init__(self, binding, prefix):
        return _ROWS_CALL(self, "cache_init", (binding, prefix), _ROWS_CONTEXT)

    @staticmethod
    def supports(binding, prefix):
        return _ROWS_CALL(None, "cache_supports", (binding, prefix), _ROWS_CONTEXT)

    @staticmethod
    def _key(arr):
        if getattr(arr, "itemsize", 4) != 4:
            raise ValueError("DeviceCache: a store slot holds 4-byte words")
        return (int(arr._addr), int(arr.size))

    def id_of(self, arr):
        return _ROWS_CALL(self, "cache_id_of", (arr,), _ROWS_CONTEXT)

    def get(self, arr):
        """The id when `arr` is already resident, else None (no upload)."""
        return _ROWS_CALL(self, "cache_get", (arr,), _ROWS_CONTEXT)

    def release(self, arr):
        return _ROWS_CALL(self, "cache_release", (arr,), _ROWS_CONTEXT)

    def close(self):
        return _ROWS_CALL(self, "cache_close", (), _ROWS_CONTEXT)

    def live(self):
        """The binding store's live slot count (every cache on it)."""
        return _ROWS_CALL(self, "cache_live", (), _ROWS_CONTEXT)

    def __len__(self):
        return _ROWS_CALL(self, "cache_len", (), _ROWS_CONTEXT)

    def __enter__(self):
        return _ROWS_CALL(self, "cache_enter", (), _ROWS_CONTEXT)

    def __exit__(self, *exc):
        return _ROWS_CALL(self, "cache_exit", exc, _ROWS_CONTEXT)


class _Scope:
    def __init__(self, min_words):
        self.min_words = min_words
        self.caches = {}

    def close(self):
        return _ROWS_CALL(self, "scope_close", (), _ROWS_CONTEXT)


_LOCAL = threading.local()


class resident:
    """`with resident():` makes every arena program run in the block (on
    this thread) keep its big inputs on the device, keyed by identity, and
    frees every device copy when the block ends. Nested blocks share the
    outermost one's copies. `min_words` overrides RESIDENT_MIN_WORDS."""

    def __init__(self, min_words=None):
        self.min_words = RESIDENT_MIN_WORDS if min_words is None else int(min_words)
        self._scope = None

    def __enter__(self):
        return _ROWS_CALL(self, "scope_enter", (), _ROWS_CONTEXT)

    def __exit__(self, *exc):
        return _ROWS_CALL(self, "scope_exit", exc, _ROWS_CONTEXT)

    @staticmethod
    def stats():
        """{prefix: (uploads, hits)} of the active scope, or None."""
        return _ROWS_CALL(_LOCAL, "resident_stats", (), _ROWS_CONTEXT)


class DeviceRows:
    """Rows `indices` (a C-contiguous int64 Array) of `base` (a float32
    C-contiguous 2-D Array): a fold's X with no host gather. shape, dtype,
    ndim, size and len are known without the data. `materialize()` gives
    the rows as a host Array through `take(base, indices)` (model_selection's
    resident device gather) for a route that needs host words; it is called
    at most once."""

    def __init__(self, base, indices, take):
        _ROWS_CALL(self, "init", (base, indices, take), _ROWS_CONTEXT)

    def __len__(self):
        return _ROWS_CALL(self, "len", (), _ROWS_CONTEXT)

    def materialize(self):
        return _ROWS_CALL(self, "materialize", (), _ROWS_CONTEXT)
