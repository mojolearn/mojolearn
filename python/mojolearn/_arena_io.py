# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared transfer plumbing for arena-style bindings (lane py-shared).

Two APIs every family can call:

1. THE RANGES RUNNER (`input_ranges`, `output_ranges`, `pack_ins`,
   `pack_outs`, `complement`): an arena program names its INPUT words and
   its OUTPUT words, and a binding entry built on `core/arena_io.mojo`
   (`upload_ranges`, `download_ranges`) moves only those: the other words
   are zeroed on the device (as the host arena's words arrive zeroed) and
   never come back. x_metrics (`x_metrics_run_ranges`) and x_prep
   (`x_prep_run_ranges`) run on it; `MOJOLEARN_ARENA_RANGES=0` sends both
   back to the whole-arena entries (the A/B arm).

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
   byte count and re-uses the device copy. A family binding with its own
   state (optimizer moments, LM weights, an index) uses `DeviceCache`
   directly: `cache = DeviceCache(binding, "x_foo")`, `cache.id_of(arr)`,
   `cache.close()`.

3. DEVICE-ROWS INPUT (`DeviceRows`, `rows_slot`; lane cpu4-misc): a fold
   view of a host float32 C-contiguous 2-D Array (the base and an int64
   index Array), handed by model_selection to an estimator that opts in
   (class attribute `_mojolearn_device_rows = True`). An arena runner that
   meets one puts the base ONCE into its own binding's store (the active
   `resident()` scope's cache, identity-keyed, so every fold and candidate
   shares the upload) and gathers the fold rows on the device
   (`<prefix>_dev_take_rows`); the rows never come to the host. A runner
   without a device store (the host column, a CPU-only install) calls
   `materialize()`: the fold rows as a host Array, as before.

Where a word travels moves no bit. The Python here only lays out Int32
range lists; a CPU-only install never reaches it (host bindings run the
arena in place).
"""
import array
import os
import threading

__all__ = ["DeviceCache", "DeviceRows", "resident", "ranges_enabled", "input_ranges", "output_ranges",
           "complement", "pack_ins", "pack_outs", "RESIDENT_MIN_WORDS", "rows_slot"]

#: Inputs below this many words always go up from the host arena: a
#: put costs one synchronized copy, which pays only on a big buffer.
RESIDENT_MIN_WORDS = 1 << 16


def ranges_enabled():
    """False when MOJOLEARN_ARENA_RANGES=0 (the whole-arena A/B arm)."""
    return os.environ.get("MOJOLEARN_ARENA_RANGES", "1").strip() != "0"


def input_ranges(spans):
    """[(lo, hi, src)] ascending, disjoint, empty spans dropped, adjacent
    HOST-ARENA spans (src -1) merged (a G3 host-table span, src <= -2, is
    never merged: it names its own address). `spans` are (lo, hi, src) in any order;
    overlapping inputs are a layout bug and raise."""
    out = []
    for lo, hi, src in sorted((int(a), int(b), int(c)) for a, b, c in spans if int(b) > int(a)):  # glue: orders arena layout spans
        if out and lo < out[-1][1]:
            raise AssertionError(f"arena ranges: inputs overlap at [{lo}, {hi})")
        if out and src == -1 and out[-1][2] == -1 and lo == out[-1][1]:
            out[-1][1] = hi
        else:
            out.append([lo, hi, src])
    return out


def complement(ranges, size):
    """The [lo, hi) pairs of [0, size) outside `ranges` (pairs or longer rows)."""
    out = []
    at = 0
    for r in sorted((int(r[0]), int(r[1])) for r in ranges):  # glue: orders arena layout ranges
        if r[0] > at:
            out.append([at, r[0]])
        at = max(at, r[1])
    if at < size:
        out.append([at, size])
    return out


def output_ranges(pairs):
    """[lo, hi, -1, 1] quads (whole ranges, merged) for `pairs`."""
    merged = []
    for lo, hi in sorted((int(a), int(b)) for a, b in pairs if int(b) > int(a)):  # glue: orders arena layout ranges
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [[lo, hi, -1, 1] for lo, hi in merged]  # glue: arena range descriptor quads


def pack_ins(triples):
    return array.array("i", [v for t in triples for v in t] or [0, 0, -1])  # glue: packs arena range descriptors


def pack_outs(quads):
    return array.array("i", [v for q in quads for v in q] or [0, 0, -1, 1])  # glue: packs arena range descriptors


class DeviceCache:
    """Device copies of host buffers in one binding's store, by identity.

    `id_of(arr)` uploads `arr` (a C-contiguous Array of 4-byte items, or
    anything with `_addr`, `size` and `itemsize`) the first time and
    returns the same id while the entry lives. `release(arr)` frees one
    entry, `close()` all of them; both free on the device at once."""

    def __init__(self, binding, prefix):
        self._put = getattr(binding, prefix + "_dev_put")
        self._free = getattr(binding, prefix + "_dev_free")
        self._live = getattr(binding, prefix + "_dev_live", None)
        self._entries = {}
        self.uploads = 0
        self.hits = 0

    @staticmethod
    def supports(binding, prefix):
        return hasattr(binding, prefix + "_dev_put") and hasattr(binding, prefix + "_dev_free")

    @staticmethod
    def _key(arr):
        if getattr(arr, "itemsize", 4) != 4:
            raise ValueError("DeviceCache: a store slot holds 4-byte words")
        return (int(arr._addr), int(arr.size))

    def id_of(self, arr):
        key = self._key(arr)
        hit = self._entries.get(key)
        if hit is not None:
            self.hits += 1
            return hit[0]
        if key[1] < 1:
            raise ValueError("DeviceCache: an empty buffer has no device copy")
        slot = int(self._put(key[0], key[1]))
        self._entries[key] = (slot, arr)
        self.uploads += 1
        return slot

    def get(self, arr):
        """The id when `arr` is already resident, else None (no upload)."""
        hit = self._entries.get(self._key(arr))
        return None if hit is None else hit[0]

    def release(self, arr):
        hit = self._entries.pop(self._key(arr), None)
        if hit is not None:
            self._free(hit[0])

    def close(self):
        entries, self._entries = self._entries, {}
        for slot, _ in entries.values():  # glue: frees each cache slot
            self._free(slot)

    def live(self):
        """The binding store's live slot count (every cache on it)."""
        return None if self._live is None else int(self._live())

    def __len__(self):
        return len(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Scope:
    def __init__(self, min_words):
        self.min_words = min_words
        self.caches = {}

    def cache(self, binding, prefix):
        key = (id(binding), prefix)
        c = self.caches.get(key)
        if c is None:
            if not DeviceCache.supports(binding, prefix):
                return None
            c = self.caches[key] = (DeviceCache(binding, prefix), binding)
        return c[0]

    def close(self):
        caches, self.caches = self.caches, {}
        for c, _ in caches.values():  # glue: closes each arena cache
            c.close()


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
        if getattr(_LOCAL, "scope", None) is None:
            self._scope = _LOCAL.scope = _Scope(self.min_words)
        return self

    def __exit__(self, *exc):
        if self._scope is not None:
            _LOCAL.scope = None
            self._scope.close()
            self._scope = None
        return False

    @staticmethod
    def stats():
        """{prefix: (uploads, hits)} of the active scope, or None."""
        scope = getattr(_LOCAL, "scope", None)
        if scope is None:
            return None
        return {p: (c.uploads, c.hits) for (_, p), (c, _) in scope.caches.items()}  # glue: counters per arena cache


def active_cache(binding, prefix, n_words):
    """The active `resident()` scope's cache for `binding`, when there is a
    scope, the binding has a store and the input is big enough; else None."""
    scope = getattr(_LOCAL, "scope", None)
    if scope is None or n_words < scope.min_words:
        return None
    return scope.cache(binding, prefix)


class DeviceRows:
    """Rows `indices` (a C-contiguous int64 Array) of `base` (a float32
    C-contiguous 2-D Array): a fold's X with no host gather. shape, dtype,
    ndim, size and len are known without the data. `materialize()` gives
    the rows as a host Array through `take(base, indices)` (model_selection's
    resident device gather) for a route that needs host words; it is called
    at most once."""

    def __init__(self, base, indices, take):
        self.base, self.indices = base, indices
        self._take = take
        self._host = None
        self.shape = (int(indices.size),) + tuple(base.shape[1:])
        self.dtype = base.dtype
        self.itemsize = base.itemsize
        self.ndim = base.ndim
        n = 1
        for v in self.shape:  # glue: the product of shape dims
            n *= v
        self.size = n
        self.row_words = (base.size // base.shape[0]) if base.shape[0] else 0

    def __len__(self):
        return self.shape[0]

    def materialize(self):
        if self._host is None:
            self._host = self._take(self.base, self.indices)
        return self._host


def rows_slot(binding, prefix, rows, direct):
    """The store id (in `binding`'s own store) of `rows`' device gather, or
    None when the binding cannot gather (no `<prefix>_dev_take_rows` or no
    store): the caller then copies `rows.materialize()`. The base goes up
    once through the active `resident()` scope's cache (any size), else
    through `direct` (a DeviceCache the caller closes after its run). The
    returned slot belongs to the caller, who frees it with
    `<prefix>_dev_free` after the run."""
    try:
        take = getattr(binding, prefix + "_dev_take_rows")
    except (AttributeError, ImportError):  # an older build, or a host facade
        return None
    if take is None or not rows.size or not DeviceCache.supports(binding, prefix):
        return None
    scope = getattr(_LOCAL, "scope", None)
    cache = scope.cache(binding, prefix) if scope is not None else None
    if cache is None:
        cache = direct
    if cache is None:
        return None
    base_id = cache.id_of(rows.base)
    return int(take(base_id, rows.row_words, int(rows.indices._addr), int(rows.indices.size)))
