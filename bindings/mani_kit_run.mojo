# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the manifold estimators: the typed manifold kit (lane
entangled-mkit, 2026-11-12; was the Python-holding `MKit` of
bindings/mani_kit.mojo and its `ManifoldMatrix` / `ManifoldDevice` matrices).

One `MRun` arena serves one public operation. It holds the x_decomp images the
operation uses (`MImage`: the image's dispatch as an `EntryRef`, its key (the
binding module's address, read by UNPACK) and its capability words), the
matrices (`MMat`: a host store, an admitted buffer, or a pool id of one image)
and the pool ids the operation allocated. Every primitive is one call through
the image's dispatch (x_decomp/descriptor_schema.mojo and
x_decomp/descriptor_mani_schema.mojo), with the entries, operands, broadcast
modes, float32 scalars and order the Python-holding kit used, so the bits do
not move. No Python object is held or called here.

Kits: the operation's kit and the process-default helper kit (`_host_kit()`)
are two image indices into one arena (the same index when both name the same
binding), so a matrix made on one is read by the other exactly as the shared
`_M` matrices were: a device matrix of another image is downloaded through
its own image and uploaded to the reader's (it moves).

Fitted state (`MFitted`): a matrix a fit stored on the model is a typed
native owner (bindings/manifold_storage_native.mojo `ManifoldMatrix`). UNPACK
reads its words' address and its pool id into an `MFitted`; a read never moves
it, and a copy made for the other side is reported to PACK (`new_host`,
`new_device`), which keeps it on the owner for every later call (rows N291,
N301). In-place writers refuse a fitted matrix.

The door (bindings/mani_decomp_native.mojo, bindings/mani_umap_native.mojo)
releases the GIL around each RUN step and keeps every admitted owner alive
across it; PACK copies a matrix's words into a public Array and `close()`
returns every pool id the operation did not hand to a fitted owner.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.host_finite_scan import host_all_finite_f32
from x_decomp.descriptor_schema import (
    XD_CAP_RESIDENT, XD_CAP_DEV_SVD,
    XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_SQDIST, XD_MOVE,
    XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_GEMM, XD_DEV_COLSUM,
    XD_DEV_ROWSUM, XD_DEV_SQDIST, XD_DEV_MOVE, XD_DEV_ISO_GATHER_MIN, XD_DEV_UPLOAD_FINITE,
    XD_ORDER, XD_DEV_ORDER, XD_DEV_REDUCE, XD_REDUCE, XD_SVD, XD_DEV_SVD, XD_RAND, XD_DEV_RAND, XD_LU, XD_DEV_LU,
    XD_LU_AUX, XD_DEV_LU_AUX, XD_EIGH, XD_DEV_EIGH_TO, XD_ABSMAX_SIGN, XD_DEV_ABSMAX, XD_ARGMIN_ALL,
    XD_DEV_ARGMIN_ALL,
)
from x_decomp.descriptor_mani_schema import (
    XD_CAP_DEV_LU, XD_CAP_DEV_EIGH, XD_CAP_LLE_ORDERED,
    XD_TRISOLVE, XD_DSUM_SQ, XD_BARYCENTER, XD_LLE_APPLY, XD_LLE_LOCAL, XD_LLE_ITERATE, XD_MDS_FIT,
    XD_LANCZOS, XD_EIGH_TRIDIAG, XD_GRAPH_KNN, XD_GRAPH_KNN_DENSE, XD_GRAPH_RADIUS, XD_GRAPH_RADIUS_GEO,
    XD_GRAPH_LLE_IW, XD_GRAPH_COMPONENTS, XD_GRAPH_JOIN, XD_GRAPH_DIJKSTRA,
    XD_DEV_TRISOLVE, XD_DEV_DSUM_SQ, XD_DEV_LANCZOS, XD_DEV_EIGH_TRIDIAG, XD_DEV_GRAPH_KNN,
    XD_DEV_GRAPH_KNN_DENSE, XD_DEV_GRAPH_RADIUS, XD_DEV_GRAPH_RADIUS_GEO, XD_DEV_GRAPH_LLE_IW,
    XD_DEV_GRAPH_COMPONENTS, XD_DEV_GRAPH_JOIN, XD_DEV_GRAPH_DIJKSTRA, XD_LLE_GEMM_ORDERED, XD_LLE_LU_ORDERED,
)

comptime MK_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime MK_F64P = MutPointer[Float64, MutUntrackedOrigin]

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAXS = 5
comptime OP_SQRT = 7
comptime OP_SQ = 8
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_FMA = 15
comptime OP_RECIP = 16
comptime OP_ADDS = 22
comptime OP_GTS = 23
comptime OP_MAX = 30
comptime OP_MIN = 31
comptime OP_LE = 34
comptime OP_SELECT = 35

# x_decomp/moves.mojo move ops and bounds (`_MV_*` in _expansion_decomp.py)
comptime MV_TAKE_ROWS = 0
comptime MV_TRANSPOSE = 1
comptime MV_PLACE_COLS = 2
comptime MV_FILL0 = 3
comptime MV_TAKE_COLS = 4
comptime F32_INDEX_MAX = 1 << 24
comptime I32_MAX = (1 << 31) - 1

# `_Kit._SEL_*` (x_decomp/select_ops.mojo)
comptime SEL_MAXABS = 0
comptime SEL_MAX = 1
comptime SEL_MIN = 2


struct MImage(Copyable, Movable):
    """One x_decomp image as UNPACK admitted it (no Python object)."""
    var entry: EntryRef
    var key: Int  # the binding module's address: pool ids are keyed by it
    var resident: Bool  # `_res()`: the image has the device-resident operations
    var dev_svd: Bool  # `opt_dev("x_decomp_dev_svd")`
    var dev_lu: Bool  # `opt_dev("x_decomp_dev_lu")`
    var dev_eigh: Bool  # `opt_dev("x_decomp_dev_eigh_to")`
    var lle_ordered: Bool  # the Apple FAST image's ordered LLE cells exist and apply (mode fast, vendor metal)
    var lle_on: Bool  # `enable_lle_precision()` ran on this image
    var dev_one: Int  # `_dev_one(kit)`: the image's 1-word device zero, -1 until made
    var dev_one_made: Bool  # made by this operation (PACK caches it for the next one)

    def __init__(out self, entry: EntryRef, key: Int, capabilities: Int, lle_ordered: Bool, dev_one: Int):
        self.entry = entry
        self.key = key
        self.resident = (capabilities & XD_CAP_RESIDENT) != 0
        self.dev_svd = self.resident and (capabilities & XD_CAP_DEV_SVD) != 0
        self.dev_lu = self.resident and (capabilities & XD_CAP_DEV_LU) != 0
        self.dev_eigh = self.resident and (capabilities & XD_CAP_DEV_EIGH) != 0
        self.lle_ordered = lle_ordered and (capabilities & XD_CAP_LLE_ORDERED) != 0
        self.lle_on = False
        self.dev_one = dev_one
        self.dev_one_made = False


struct MStore(Movable):
    """A kit-owned host store (at least one word, so an empty matrix keeps a
    valid address, as `_M._one` did) and the matrices that view it."""
    var words: List[Float32]
    var references: Int

    def __init__(out self, cells: Int):
        self.words = List[Float32](length=max(cells, 1), fill=0.0)
        self.references = 0


struct MMat(Copyable, Movable):
    # `store`: the store index, -1 when none. `device` / `image`: the pool id
    # and the image index that owns it, -1 on the host. `borrowed`: an
    # admitted caller buffer read in place (kept alive by the door), 0 when
    # none. `fitted`: the MFitted index of a fitted matrix (frozen), else -1.
    var store: Int
    var device: Int
    var image: Int
    var rows: Int
    var cols: Int
    var borrowed: Int
    var fitted: Int

    def __init__(out self, store: Int, device: Int, image: Int, rows: Int, cols: Int):
        self.store = store
        self.device = device
        self.image = image
        self.rows = rows
        self.cols = cols
        self.borrowed = 0
        self.fitted = -1


struct MFitted(Copyable, Movable):
    """A fitted matrix's owner as UNPACK read it, and the copies this RUN made
    for the side it is not on (PACK keeps them on the owner)."""
    var rows: Int
    var cols: Int
    var host: Int  # the owner's host words, 0 when it has none
    var device: Int  # the owner's pool id, -1 when none
    var key: Int  # the image key of `device`
    var entry: EntryRef  # the image of `device` (its download)
    var on_device: Bool  # the side the fit left it on
    var new_host: Int  # a store this RUN downloaded for it, -1 when none
    var new_device: Int  # a pool id this RUN uploaded for it, -1 when none
    var new_image: Int

    def __init__(out self, rows: Int, cols: Int, host: Int, device: Int, key: Int, entry: EntryRef,
                 on_device: Bool):
        self.rows = rows
        self.cols = cols
        self.host = host
        self.device = device
        self.key = key
        self.entry = entry
        self.on_device = on_device
        self.new_host = -1
        self.new_device = -1
        self.new_image = -1


struct MExport(Movable):
    """A matrix handed to PACK for a fitted owner: its words when it is on
    the host, its pool id (no longer the arena's to free) when on a device."""
    var rows: Int
    var cols: Int
    var words: List[Float32]
    var device: Int
    var image: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.words = List[Float32]()
        self.device = -1
        self.image = -1


struct MRun(Movable):
    var images: List[MImage]
    var mats: List[MMat]
    var stores: List[MStore]
    var alloc_image: List[Int]
    var alloc_id: List[Int]
    var fitted: List[MFitted]
    var zero: Int  # `_M._one`, the shared host 1 x 1 zero, -1 until made

    def __init__(out self):
        self.images = List[MImage]()
        self.mats = List[MMat]()
        self.stores = List[MStore]()
        self.alloc_image = List[Int]()
        self.alloc_id = List[Int]()
        self.fitted = List[MFitted]()
        self.zero = -1

    def image(mut self, var value: MImage) -> Int:
        """Add an image; an image already present (same key) is reused, so the
        operation's kit and the helper kit share one when they name one
        binding."""
        for i in range(len(self.images)):  # driver-loop: over this operation's images (at most 2)
            if self.images[i].key == value.key:
                return i
        self.images.append(value^)
        return len(self.images) - 1

    # ---- the x_decomp dispatch
    def call(mut self, k: Int, op: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        return entry_ref_call(self.images[k].entry, op, addresses, ints, floats)

    def dcall(mut self, k: Int, op: Int, ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A device-resident operation on pool ids."""
        var floats = List[Float64]()
        if op == XD_DEV_EW or op == XD_DEV_SQDIST:
            floats.append(s)
        return self.call(k, op, List[Int](), ints, floats)

    def hcall(mut self, k: Int, op: Int, addresses: List[Int], ints: List[Int], s: Float64 = 0.0) raises -> Int:
        """A host-address operation; every address is a store of this arena,
        an admitted buffer or a fitted owner's words, alive through the call."""
        var floats = List[Float64]()
        if op == XD_EW or op == XD_SQDIST:
            floats.append(s)
        return self.call(k, op, addresses, ints, floats)

    def dmove(mut self, k: Int, src: Int, idx: Int, dst: Int, words: List[Int]) raises:
        """`x_decomp_dev_move` on pool ids; words = [op, count, a1, a2, a3,
        ist, ioff, nsrc, ndst, nidx]."""
        var ints: List[Int] = [src, idx, dst]
        for i in range(len(words)):  # driver-loop: native Mojo values, no CPython call: (params: the ten move words): 10
            ints.append(words[i])
        _ = self.dcall(k, XD_DEV_MOVE, ints)

    def hmove(mut self, k: Int, src: Int, idx: Int, dst: Int, words: List[Int]) raises:
        _ = self.hcall(k, XD_MOVE, [src, idx, dst], words)

    def alloc(mut self, k: Int, n: Int) raises -> Int:
        """`_devbuf(n)`: a pool id of image k, returned at close()."""
        var handle = self.dcall(k, XD_DEV_ALLOC, [max(n, 1)])
        self.alloc_image.append(k)
        self.alloc_id.append(handle)
        return handle

    def close(mut self):
        """Return every pool id this operation allocated and did not hand to a
        fitted owner, and a device zero PACK did not cache."""
        for i in range(len(self.alloc_id) - 1, -1, -1):  # driver-loop: over native pool allocation handles
            try:
                _ = self.dcall(self.alloc_image[i], XD_DEV_FREE, [self.alloc_id[i]])
            except:
                pass
        self.alloc_id.clear()
        self.alloc_image.clear()
        for k in range(len(self.images)):  # driver-loop: over this operation's images (at most 2)
            if self.images[k].dev_one_made and self.images[k].dev_one >= 0:
                try:
                    _ = self.dcall(k, XD_DEV_FREE, [self.images[k].dev_one])
                except:
                    pass
                self.images[k].dev_one = -1
                self.images[k].dev_one_made = False

    def _release_id(mut self, k: Int, handle: Int) -> Bool:
        """Take a pool id out of the arena's list (it now has another owner)."""
        for i in range(len(self.alloc_id)):  # driver-loop: over native pool allocation handles
            if self.alloc_id[i] == handle and self.alloc_image[i] == k:
                _ = self.alloc_id.pop(i)
                _ = self.alloc_image.pop(i)
                return True
        return False

    # ---- matrices
    def add(mut self, store: Int, rows: Int, cols: Int, device: Int = -1, image: Int = -1) -> Int:
        if store >= 0:
            self.stores[store].references += 1
        self.mats.append(MMat(store, device, image, rows, cols))
        return len(self.mats) - 1

    def _release_store(mut self, a: Int):
        var store = self.mats[a].store
        if store >= 0:
            self.stores[store].references -= 1
            if self.stores[store].references == 0:
                self.stores[store].words = List[Float32]()
            self.mats[a].store = -1

    def r(self, a: Int) -> Int:
        return self.mats[a].rows

    def c(self, a: Int) -> Int:
        return self.mats[a].cols

    def n(self, a: Int) -> Int:
        return self.mats[a].rows * self.mats[a].cols

    def frozen(self, a: Int) -> Bool:
        return self.mats[a].fitted >= 0

    def on_dev(self, a: Int) -> Bool:
        """`mk_on_dev`: the matrix is on a device (a fitted one: the side the
        fit left it on)."""
        if self.mats[a].fitted >= 0:
            return self.fitted[self.mats[a].fitted].on_device
        return self.mats[a].device >= 0

    def zeros(mut self, rows: Int, cols: Int) -> Int:
        """`_M.zeros(rows, cols)`."""
        self.stores.append(MStore(rows * cols))
        return self.add(len(self.stores) - 1, rows, cols)

    def of(mut self, values: List[Float64], rows: Int, cols: Int) raises -> Int:
        """`_M.of(values, r, c)`: each value rounded once to float32."""
        if len(values) != rows * cols:
            raise Error("x_decomp: matrix store does not match its shape")
        var out = self.zeros(rows, cols)
        var p = self.ptr(out)
        for i in range(len(values)):  # small-loop(values: a constant operand the caller spells out): bounded by the call site
            p[i] = Float32(values[i])
        return out

    def fill(mut self, value: Float64, rows: Int, cols: Int) raises -> Int:
        return self.of(List[Float64](length=rows * cols, fill=value), rows, cols)

    def shared_zero(mut self) -> Int:
        """`_M._one`: the host 1 x 1 zero (read-only), one per operation."""
        if self.zero < 0:
            self.zero = self.zeros(1, 1)
        return self.zero

    def fit(mut self, var value: MFitted) -> Int:
        """A fitted matrix as UNPACK read its owner (frozen)."""
        var rows = value.rows
        var cols = value.cols
        self.fitted.append(value^)
        var out = self.add(-1, rows, cols)
        self.mats[out].fitted = len(self.fitted) - 1
        return out

    def _image_of_key(self, key: Int) -> Int:
        for i in range(len(self.images)):  # driver-loop: over this operation's images (at most 2)
            if self.images[i].key == key:
                return i
        return -1

    def _frozen_host(mut self, a: Int) raises -> Int:
        """The host words of a fitted matrix without moving it: its words, or
        the copy of a device matrix, downloaded once (PACK keeps it)."""
        var fi = self.mats[a].fitted
        if self.fitted[fi].host != 0:
            return self.fitted[fi].host
        if self.fitted[fi].new_host < 0:
            var count = self.fitted[fi].rows * self.fitted[fi].cols
            self.stores.append(MStore(count))
            var store = len(self.stores) - 1
            self.stores[store].references = 1
            if count > 0:
                var words = List[Float64]()
                _ = entry_ref_call(self.fitted[fi].entry, XD_DEV_DOWNLOAD,
                                   [Int(self.stores[store].words.unsafe_ptr())], [self.fitted[fi].device, count],
                                   words)
            self.fitted[fi].new_host = store
        return Int(self.stores[self.fitted[fi].new_host].words.unsafe_ptr())

    def addr(mut self, a: Int) raises -> Int:
        """`m.addr` / `mk_store`: the host address of matrix a's words; a
        device matrix is downloaded once through its own image (it moves); a
        fitted matrix never moves."""
        if self.mats[a].fitted >= 0:
            return self._frozen_host(a)
        if self.mats[a].borrowed != 0 and self.mats[a].device < 0:
            return self.mats[a].borrowed
        if self.mats[a].store < 0:
            var count = self.n(a)
            self.stores.append(MStore(count))
            var store = len(self.stores) - 1
            if count > 0:
                _ = self.hcall(self.mats[a].image, XD_DEV_DOWNLOAD, [Int(self.stores[store].words.unsafe_ptr())],
                               [self.mats[a].device, count])
            self.stores[store].references = 1
            self.mats[a].store = store
        self.mats[a].device = -1
        self.mats[a].image = -1
        return Int(self.stores[self.mats[a].store].words.unsafe_ptr())

    def ptr(mut self, a: Int) raises -> MK_F32P:
        return MK_F32P(unsafe_from_address=self.addr(a))

    def at(mut self, a: Int, i: Int) raises -> Float64:
        """`float(m.s[i])`."""
        return Float64(self.ptr(a)[i])

    def device(mut self, k: Int, a: Int) raises -> Int:
        """`_did(M)`: M's pool id on image k, uploading a host matrix (it
        moves); a fitted matrix keeps its side and gets a copy, made once."""
        if self.mats[a].fitted >= 0:
            var fi = self.mats[a].fitted
            if self.fitted[fi].device >= 0 and self.fitted[fi].key == self.images[k].key:
                return self.fitted[fi].device
            if self.fitted[fi].new_device >= 0 and self.fitted[fi].new_image == k:
                return self.fitted[fi].new_device
            var count = self.n(a)
            var source = self._frozen_host(a)
            var handle = self.alloc(k, count)
            if count > 0:
                _ = self.hcall(k, XD_DEV_UPLOAD, [source], [handle, count])
            if self.fitted[fi].new_device < 0:
                self.fitted[fi].new_device = handle
                self.fitted[fi].new_image = k
            return handle
        if self.mats[a].device >= 0 and self.mats[a].image == k:
            return self.mats[a].device
        var count = self.n(a)
        var source = self.addr(a)
        var handle = self.alloc(k, count)
        if count > 0:
            _ = self.hcall(k, XD_DEV_UPLOAD, [source], [handle, count])
        self.mats[a].borrowed = 0
        self._release_store(a)
        self.mats[a].device = handle
        self.mats[a].image = k
        return handle

    def dout(mut self, k: Int, rows: Int, cols: Int) raises -> Int:
        var handle = self.alloc(k, rows * cols)
        return self.add(-1, rows, cols, handle, k)

    def did_of(self, a: Int) -> Int:
        return self.mats[a].device

    def dev_one(mut self, k: Int) raises -> Int:
        """`_dev_one(kit)`: the unused broadcast operand on the device, one per
        binding (UNPACK hands the cached one; PACK caches a new one)."""
        if self.images[k].dev_one < 0:
            var handle = self.dcall(k, XD_DEV_ALLOC, [1])
            var z = List[Float32](length=1, fill=0.0)
            _ = self.hcall(k, XD_DEV_UPLOAD, [Int(z.unsafe_ptr())], [handle, 1])
            _ = z^  # keep the word alive through the synchronous upload
            self.images[k].dev_one = handle
            self.images[k].dev_one_made = True
        return self.images[k].dev_one

    def writable(self, a: Int) raises:
        """In-place writers (fill0, place_strided, graph_join) refuse fitted
        state: its two copies would disagree afterwards."""
        if self.mats[a].fitted >= 0:
            raise Error("RuntimeError: a fitted manifold matrix is read-only")

    def copy(mut self, a: Int) raises -> Int:
        """`m.copy()`: a host copy."""
        var count = self.n(a)
        var src = self.addr(a)
        var out = self.zeros(self.r(a), self.c(a))
        if count > 0:
            unsafe_memcpy(dest=self.ptr(out), src=MK_F32P(unsafe_from_address=src), count=count)
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        """`m.reshape(r, c)`: a view of the same buffer or the same store."""
        if self.mats[a].fitted < 0 and self.mats[a].device >= 0 and rows * cols == self.n(a):
            return self.add(-1, rows, cols, self.mats[a].device, self.mats[a].image)
        if self.mats[a].fitted >= 0:
            var out = self.copy(a)
            self.mats[out].rows = rows
            self.mats[out].cols = cols
            return out
        _ = self.addr(a)
        var out = self.add(self.mats[a].store, rows, cols)
        self.mats[out].borrowed = self.mats[a].borrowed
        return out

    def res_moves(self, a: Int, count: Int) -> Bool:
        """`_res_moves(M, count)`."""
        return count > 0 and self.r(a) < F32_INDEX_MAX and self.c(a) < F32_INDEX_MAX

    def dev_gather(mut self, a: Int, op: Int, idx: List[Int]) raises -> Int:
        """`_dev_gather(M, op, idx)`: rows (TAKE_ROWS) or columns (TAKE_COLS)
        of a device matrix, on its own image."""
        var k = self.mats[a].image
        var d = self.mats[a].device
        var w = len(idx)
        var words = List[Float32](length=max(w, 1), fill=0.0)
        for t in range(w):  # small-loop(idx: the gathered row or column numbers): bounded by the caller's index list
            words[t] = Float32(Float64(idx[t]))
        var di = self.alloc(k, w)
        _ = self.hcall(k, XD_DEV_UPLOAD, [Int(words.unsafe_ptr())], [di, w])
        _ = words^  # keep the index words alive through the synchronous upload
        var r = self.r(a)
        var c = self.c(a)
        var n = r * c
        if op == MV_TAKE_ROWS:
            var count = w * c
            var out = self.dout(k, w, c)
            self.dmove(k, d, di, self.mats[out].device, [op, count, c, 0, 0, 1, 0, n, count, w])
            return out
        var count = r * w
        var out = self.dout(k, r, w)
        self.dmove(k, d, di, self.mats[out].device, [op, count, w, c, 0, 0, 0, n, count, w])
        return out

    def rows(mut self, a: Int, lo: Int, hi: Int) raises -> Int:
        """`m.rows(a, b)`."""
        var c = self.c(a)
        if (self.mats[a].fitted < 0 and self.mats[a].device >= 0 and 0 <= lo and lo < hi and hi <= self.r(a)
                and self.res_moves(a, (hi - lo) * c)):
            var idx = List[Int]()
            for i in range(lo, hi):  # native data loop (not a driver loop), no CPython call: (rows: the requested row range): bounded by the caller
                idx.append(i)
            return self.dev_gather(a, MV_TAKE_ROWS, idx)
        # `store[a * c:b * c]`, clamped as a slice clamps it
        var total = self.n(a)
        var start = min(max(lo * c, 0), total)
        var stop = min(max(hi * c, start), total)
        var src = self.addr(a)
        if stop - start != (hi - lo) * c:
            raise Error("x_decomp: matrix store does not match its shape")
        var out = self.zeros(hi - lo, c)
        if stop > start:
            unsafe_memcpy(dest=self.ptr(out), src=MK_F32P(unsafe_from_address=src) + start, count=stop - start)
        return out

    def take_cols(mut self, a: Int, idx: List[Int]) raises -> Int:
        """`m.take_cols(idx)`: on the device for a device matrix, else one
        host copy per selected column."""
        var w = len(idx)
        var r = self.r(a)
        var c = self.c(a)
        var in_range = True
        for t in range(w):  # small-loop(idx: the selected column numbers): bounded by the caller's index list
            if idx[t] < 0 or idx[t] >= c:
                in_range = False
        if self.mats[a].fitted < 0 and self.mats[a].device >= 0 and in_range and w > 0 and self.res_moves(a, r * w):
            return self.dev_gather(a, MV_TAKE_COLS, idx)
        if not in_range:
            raise Error("x_decomp: column index out of range")
        var sp = MK_F32P(unsafe_from_address=self.addr(a))
        var out = self.zeros(r, w)
        if r * w > 0:
            var dp = self.ptr(out)
            for t in range(w):  # small-loop(idx: one host column copy per selected column): bounded by the caller's index list
                for i in range(r):  # native data movement: the column's values of a host matrix
                    dp[i * w + t] = sp[i * c + idx[t]]
        return out

    def vstack_host(mut self, ms: List[Int]) raises -> Int:
        """`_vstack(*ms)` on the host: one native copy each into one store."""
        var total = 0
        var rows = 0
        for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
            total += self.n(ms[i])
            rows += self.r(ms[i])
        var out = self.zeros(rows, self.c(ms[0]))
        if total != self.n(out):
            raise Error("x_decomp: matrix store does not match its shape")
        var off = 0
        for i in range(len(ms)):  # small-loop(ms: one native copy per stacked matrix): bounded by the caller
            var cnt = self.n(ms[i])
            if cnt > 0:
                var src = self.addr(ms[i])
                unsafe_memcpy(dest=self.ptr(out) + off, src=MK_F32P(unsafe_from_address=src), count=cnt)
            off += cnt
        return out

    def export(mut self, a: Int) raises -> MExport:
        """PACK: matrix a for a fitted owner. A device matrix hands over its
        pool id (the arena no longer frees it); a host one its words."""
        var out = MExport(self.r(a), self.c(a))
        if self.mats[a].fitted < 0 and self.mats[a].device >= 0:
            if self._release_id(self.mats[a].image, self.mats[a].device):
                out.device = self.mats[a].device
                out.image = self.mats[a].image
                return out^
        var count = self.n(a)
        var src = self.addr(a)
        out.words = List[Float32](length=max(count, 1), fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=MK_F32P(unsafe_from_address=Int(out.words.unsafe_ptr())),
                          src=MK_F32P(unsafe_from_address=src), count=count)
        return out^

    def kept_host(mut self, fi: Int) -> List[Float32]:
        """PACK: the host copy this RUN made for fitted matrix fi (empty when
        none); the store gives its words up."""
        var store = self.fitted[fi].new_host
        if store < 0:
            return List[Float32]()
        var words = List[Float32]()
        swap(words, self.stores[store].words)
        self.fitted[fi].new_host = -1
        return words^

    def kept_device(mut self, fi: Int) -> Int:
        """PACK: the pool id this RUN uploaded for fitted matrix fi, taken out
        of the arena's list (-1 when none)."""
        var handle = self.fitted[fi].new_device
        if handle < 0:
            return -1
        if self._release_id(self.fitted[fi].new_image, handle):
            self.fitted[fi].new_device = -1
            return handle
        return -1

    # ---- placement
    def use(self, k: Int, a: Int, b: Int = -1, c: Int = -1) -> Bool:
        """`_use(a, b, c)` (-1 for an absent operand)."""
        if not self.images[k].resident:
            return False
        var big = False
        if a >= 0:
            if self.on_dev(a):
                return True
            if self.n(a) >= 1:
                big = True
        if b >= 0:
            if self.on_dev(b):
                return True
            if self.n(b) >= 1:
                big = True
        if c >= 0:
            if self.on_dev(c):
                return True
            if self.n(c) >= 1:
                big = True
        return big

    def resident_on(self, k: Int, a: Int) -> Bool:
        """The matrix's own buffer is a pool id of image k (`d is not None
        and device_binding(d) is self.raw`)."""
        if self.mats[a].fitted >= 0:
            var fi = self.mats[a].fitted
            return self.fitted[fi].on_device and self.fitted[fi].key == self.images[k].key
        return self.mats[a].device >= 0 and self.mats[a].image == k

    def host_snapshot(mut self, address: Int, rows: Int, cols: Int, ndim: Int, name: String) raises -> Int:
        """`mk_from_snapshot`: one host copy of the admitted C-order float32
        words, refused when empty or not finite."""
        if ndim != 2 or rows == 0 or cols == 0:
            raise Error(name + ": a nonempty two-dimensional input is required")
        var out = self.zeros(rows, cols)
        var src = MK_F32P(unsafe_from_address=address)
        unsafe_memcpy(dest=self.ptr(out), src=src, count=rows * cols)
        if not host_all_finite_f32(src, rows * cols):
            raise Error(name + ": input must be finite; NaN/inf are unsupported")
        return out

    def from_snapshot(mut self, k: Int, address: Int, rows: Int, cols: Int, ndim: Int, name: String) raises -> Int:
        """`from_snapshot` / `mk_from_snapshot`: the admitted C-order float32
        input, refused when it is empty or not finite. A resident image
        uploads straight from the admitted address (the finiteness test runs
        on the device over the uploaded words, lane a8d); a host image copies
        the words once and scans them."""
        if not self.images[k].resident:
            return self.host_snapshot(address, rows, cols, ndim, name)
        if ndim != 2 or rows == 0 or cols == 0:
            raise Error(name + ": a nonempty two-dimensional input is required")
        var handle = self.alloc(k, rows * cols)
        var flagged = self.hcall(k, XD_DEV_UPLOAD_FINITE, [address], [handle, rows * cols])
        if flagged != 0:
            raise Error(name + ": input must be finite; NaN/inf are unsupported")
        return self.add(-1, rows, cols, handle, k)

    def iso_gather_min(mut self, k: Int, d: Int, im: Int, sq: Int, q: Int, nn: Int) raises -> Int:
        """Isomap.transform's per-rank gather-and-min (row N295) as one native
        operation; -1 on a host image or a shape the per-rank path takes."""
        if not self.images[k].resident:
            return -1
        var dc = self.c(d)
        if q * dc <= 0 or nn <= 0 or self.r(sq) != q or self.c(sq) != nn or not self.res_moves(sq, q):
            return -1
        var idd = self.device(k, d)
        var iim = self.device(k, im)
        var isq = self.device(k, sq)
        var one = self.dev_one(k)
        var gid = self.dcall(k, XD_DEV_ISO_GATHER_MIN, [idd, iim, isq, q, dc, nn, one, self.n(d), self.n(im)])
        self.alloc_image.append(k)
        self.alloc_id.append(gid)
        return self.add(-1, q, dc, gid, k)

    # ---- elementwise
    def _mode_of(self, x: Int, a: Int) raises -> Int:
        if x < 0:
            return 3
        var xr = self.r(x)
        var xc = self.c(x)
        var ar = self.r(a)
        var ac = self.c(a)
        if xr == ar and xc == ac:
            return 0
        if xr == 1 and xc == ac:
            return 1
        if xc == 1 and xr == ar:
            return 2
        if xr == 1 and xc == 1:
            return 3
        raise Error("x_decomp: cannot broadcast " + String(xr) + "x" + String(xc) + " against "
                    + String(ar) + "x" + String(ac))

    def ew(mut self, k: Int, op: Int, a: Int, b: Int = -1, c: Int = -1, s: Float64 = 0.0) raises -> Int:
        """`ew(op, A, B, C, s)`."""
        var bmode = self._mode_of(b, a)
        var cmode = self._mode_of(c, a)
        var one = self.shared_zero()
        var bm = one if b < 0 else b
        var cm = one if c < 0 else c
        var n = self.n(a)
        if n > 0 and self.use(k, a, b, c):
            var out = self.dout(k, self.r(a), self.c(a))
            var one_id = -1
            if b < 0 or c < 0:
                one_id = self.dev_one(k)
            var ib = one_id if b < 0 else self.device(k, bm)
            var ic = one_id if c < 0 else self.device(k, cm)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_EW, [ia, ib, ic, self.mats[out].device, op, n, self.c(a), self.n(bm), bmode,
                                          self.n(cm), cmode], s)
            return out
        var out = self.zeros(self.r(a), self.c(a))
        if n > 0:
            var pa = self.addr(a)
            var pb = self.addr(bm)
            var pc = self.addr(cm)
            var po = self.addr(out)
            _ = self.hcall(k, XD_EW, [pa, pb, pc, po], [op, n, self.c(a), self.n(bm), bmode, self.n(cm), cmode], s)
        return out

    def neg_signs(mut self, k: Int, v: Int) raises -> Int:
        """-1 where V < 0, else +1: select(-V > 0)."""
        var neg = self.ew(k, OP_SCALE, v, s=-1.0)
        var lo = self.of([-1.0], 1, 1)
        var hi = self.of([1.0], 1, 1)
        return self.ew(k, OP_SELECT, neg, lo, hi)

    # ---- products and folds
    def mm(mut self, k: Int, a: Int, b: Int, ta: Bool, tb: Bool) raises -> Int:
        var m = self.c(a) if ta else self.r(a)
        var kk = self.r(a) if ta else self.c(a)
        var k2 = self.c(b) if tb else self.r(b)
        var n = self.r(b) if tb else self.c(b)
        if kk != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(kk) + " and " + String(k2) + " differ")
        var words: List[Int] = [m, kk, n, 1 if ta else 0, 1 if tb else 0]
        if self.images[k].lle_on:
            # the Apple FAST image's ordered GEMM (enable_lle_precision)
            var out = self.zeros(m, n)
            var pa = self.addr(a)
            var pb = self.addr(b)
            var po = self.addr(out)
            _ = self.call(k, XD_LLE_GEMM_ORDERED, [pa, pb, po], words, List[Float64]())
            return out
        if m * n > 0 and m * kk > 0 and kk * n > 0 and self.use(k, a, b):
            var out = self.dout(k, m, n)
            var ia = self.device(k, a)
            var ib = self.device(k, b)
            _ = self.dcall(k, XD_DEV_GEMM, [ia, ib, self.mats[out].device, words[0], words[1], words[2], words[3],
                                            words[4]])
            return out
        var out = self.zeros(m, n)
        if m * n > 0:
            var pa = self.addr(a)
            var pb = self.addr(b)
            var po = self.addr(out)
            _ = self.hcall(k, XD_GEMM, [pa, pb, po], words)
        return out

    def _sum(mut self, k: Int, a: Int, columns: Bool) raises -> Int:
        var r = self.r(a)
        var c = self.c(a)
        var nr = 1 if columns else r
        var nc = c if columns else 1
        if r * c > 0 and self.use(k, a):
            var out = self.dout(k, nr, nc)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_COLSUM if columns else XD_DEV_ROWSUM, [ia, self.mats[out].device, r, c])
            return out
        var out = self.zeros(nr, nc)
        var pa = self.addr(a)
        var po = self.addr(out)
        _ = self.hcall(k, XD_COLSUM if columns else XD_ROWSUM, [pa, po], [r, c])
        return out

    def colsum(mut self, k: Int, a: Int) raises -> Int:
        return self._sum(k, a, True)

    def rowsum(mut self, k: Int, a: Int) raises -> Int:
        return self._sum(k, a, False)

    def total(mut self, k: Int, a: Int) raises -> Int:
        """The sum of every entry (1 x 1): rows ascending, then the row sums."""
        var rs = self.rowsum(k, a)
        return self.colsum(k, rs)

    def pdist(mut self, k: Int, a: Int, b: Int, kind: Int, pw: Float64) raises -> Int:
        """kind 0 squared Euclidean (p 2), 1 manhattan, 2 chebyshev, 3
        minkowski pw, 4 cosine."""
        var ar = self.r(a)
        var br = self.r(b)
        var ac = self.c(a)
        if ar * br > 0 and ac > 0 and self.use(k, a, b):
            var out = self.dout(k, ar, br)
            var ia = self.device(k, a)
            var ib = self.device(k, b)
            _ = self.dcall(k, XD_DEV_SQDIST, [ia, ib, self.mats[out].device, ar, br, ac, kind], pw)
            return out
        var out = self.zeros(ar, br)
        var pa = self.addr(a)
        var pb = self.addr(b)
        var po = self.addr(out)
        _ = self.hcall(k, XD_SQDIST, [pa, pb, po], [ar, br, ac, kind], pw)
        return out

    def sqdist(mut self, k: Int, a: Int, b: Int) raises -> Int:
        return self.pdist(k, a, b, 0, 2.0)

    def rand(mut self, k: Int, r: Int, c: Int, seed: Int, stream: Int, kind: Int) raises -> Int:
        """`rand(r, c, seed, stream, kind)` (the Philox stream); `seed` is
        `int(seed) & 0xFFFFFFFF`, made by UNPACK."""
        var words: List[Int] = [r * c, seed, stream & 0xFFFFFFFF, kind]
        if r * c >= 1024 and self.images[k].resident:
            var out = self.dout(k, r, c)
            _ = self.dcall(k, XD_DEV_RAND, [self.mats[out].device, words[0], words[1], words[2], words[3]])
            return out
        var out = self.zeros(r, c)
        if r * c > 0:
            _ = self.hcall(k, XD_RAND, [self.addr(out)], words)
        return out

    # ---- small dense linear algebra
    def eigh(mut self, k: Int, a: Int) raises -> Tuple[Int, Int]:
        """Ascending eigenvalues (1 x n) and eigenvectors in COLUMNS (n x n)."""
        var n = self.r(a)
        if n >= 1 and self.c(a) == n and self.resident_on(k, a) and self.images[k].dev_eigh:
            # lane n9-decomp-eigh-families: w and V stay on the device
            var wd = self.dout(k, 1, n)
            var vd = self.dout(k, n, n)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_EIGH_TO, [ia, self.mats[wd].device, self.mats[vd].device, n, 0])
            return (wd, vd)
        var w = self.zeros(1, n)
        var v = self.zeros(n, n)
        var pa = self.addr(a)
        _ = self.hcall(k, XD_EIGH, [pa, self.addr(w), self.addr(v)], [n, 0])
        return (w, v)

    def lu(mut self, k: Int, a: Int) raises -> Tuple[Int, Int, Int]:
        """(lu, piv (n int32 words in an n x 1 store), info)."""
        var n = self.r(a)
        var piv = self.zeros(n, 1)
        var info = self.zeros(1, 1)
        if self.images[k].lle_on:
            var lu = self.copy(a)
            _ = self.call(k, XD_LLE_LU_ORDERED, [self.addr(lu), self.addr(piv), self.addr(info)], [n],
                          List[Float64]())
            return (lu, piv, Int(self.at(info, 0)))
        if n >= 1 and self.c(a) == n and self.images[k].dev_lu and self.use(k, a):
            var lu = self.dout(k, n, n)
            var ia = self.device(k, a)
            _ = self.call(k, XD_DEV_LU, [self.addr(piv), self.addr(info)], [ia, self.mats[lu].device, n],
                          List[Float64]())
            return (lu, piv, Int(self.at(info, 0)))
        var lu = self.copy(a)
        _ = self.call(k, XD_LU, [self.addr(lu), self.addr(piv), self.addr(info)], [n], List[Float64]())
        return (lu, piv, Int(self.at(info, 0)))

    def lu_aux(mut self, k: Int, lu: Int, piv: Int, clamp: Bool) raises -> Tuple[List[Float64], Int, Int, Int]:
        """(stats [max |u_ii|, zero pivots, negative pivots, swaps], diag
        1 x n, pm n x 1, im n x 1)."""
        var n = self.r(lu)
        var pm = self.zeros(n, 1)
        var im = self.zeros(n, 1)
        var diag = self.zeros(1, n)
        var st = self.zeros(1, 4)
        var cl = 1 if clamp else 0
        if n >= 1 and self.resident_on(k, lu) and self.images[k].dev_lu:
            var il = self.device(k, lu)
            _ = self.call(k, XD_DEV_LU_AUX, [self.addr(piv), self.addr(pm), self.addr(im), self.addr(diag),
                                             self.addr(st)], [il, n, cl], List[Float64]())
        else:
            var pl = self.addr(lu)
            _ = self.call(k, XD_LU_AUX, [pl, self.addr(piv), self.addr(pm), self.addr(im), self.addr(diag),
                                         self.addr(st)], [n, cl], List[Float64]())
        var stats = List[Float64]()
        for i in range(4):  # driver-loop: over (stats: the four lu_aux status words): 4
            stats.append(self.at(st, i))
        return (stats^, diag, pm, im)

    def trisolve(mut self, k: Int, lu: Int, idx: Int, b: Int, trans: Int) raises -> Int:
        var n = self.r(lu)
        var w = self.c(b)
        if n * w > 0 and self.use(k, lu, b):
            var out = self.dout(k, n, w)
            var il = self.device(k, lu)
            var ii = self.device(k, idx)
            var ib = self.device(k, b)
            _ = self.dcall(k, XD_DEV_TRISOLVE, [il, ii, ib, self.mats[out].device, n, w, trans])
            return out
        var out = self.zeros(n, w)
        if n * w > 0:
            var pl = self.addr(lu)
            var pi = self.addr(idx)
            var pb = self.addr(b)
            _ = self.hcall(k, XD_TRISOLVE, [pl, pi, pb, self.addr(out)], [n, w, trans])
        return out

    def svd(mut self, k: Int, a: Int, hk: Int) raises -> Tuple[Int, Int]:
        """(S 1 x n DESCENDING, Vt n x n) of a tall A: values sorted
        descending, ties to the lower index."""
        var m = self.r(a)
        var n = self.c(a)
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        if m >= n and n >= 1 and self.images[k].dev_svd and self.use(k, a):
            var ia = self.device(k, a)
            _ = self.call(k, XD_DEV_SVD, [self.addr(s), self.addr(v)], [ia, m, n], List[Float64]())
        else:
            var pa = self.addr(a)
            _ = self.call(k, XD_SVD, [pa, self.addr(s), self.addr(v)], [m, n], List[Float64]())
        var neg = self.ew(k, OP_SCALE, s, s=-1.0)
        var o = self.order(k, neg)
        var st = self.t(s, hk)
        var s_sorted = self.t(self.take_rows(k, st, o), hk)
        var vt = self.t(v, hk)
        var v_sorted = self.take_rows(k, vt, o)
        return (s_sorted, v_sorted)

    # ---- moves and orders
    def take_rows_at(mut self, k: Int, a: Int, idx: Int, m: Int, ist: Int, ioff: Int, radd: Int) raises -> Int:
        """`take_rows(A, idx, m, ist, ioff, radd)`: rows Int(idx[t * ist +
        ioff]) + radd of A for t < m."""
        var c = self.c(a)
        var count = m * c
        var words: List[Int] = [MV_TAKE_ROWS, count, c, radd, 0, ist, ioff, self.n(a), count, self.n(idx)]
        if count > 0 and self.use(k, a, idx):
            var out = self.dout(k, m, c)
            var ia = self.device(k, a)
            var ii = self.device(k, idx)
            self.dmove(k, ia, ii, self.mats[out].device, words)
            return out
        var out = self.zeros(m, c)
        if count > 0:
            var pa = self.addr(a)
            var pi = self.addr(idx)
            self.hmove(k, pa, pi, self.addr(out), words)
        return out

    def take_rows(mut self, k: Int, a: Int, idx: Int) raises -> Int:
        return self.take_rows_at(k, a, idx, self.n(idx), 1, 0, 0)

    def take_rows_list(mut self, a: Int, idx: List[Int], hk: Int) raises -> Int:
        """`m.take_rows(idx)` (a list of row numbers): the default-mode kit's
        move."""
        if self.r(a) >= F32_INDEX_MAX:
            raise Error("x_decomp: take_rows exceeds the float32 row-index bound (2^24 rows)")
        var values = List[Float64]()
        for t in range(len(idx)):  # native data loop (not a driver loop), no CPython call: (idx: the taken row numbers): bounded by the caller's index list
            values.append(Float64(idx[t]))
        var im = self.of(values, len(idx), 1)
        var count = len(idx)
        var c = self.c(a)
        if count * c > 0 and self.images[hk].resident:
            return self.take_rows(hk, a, im)
        var out = self.zeros(count, c)
        if count * c > 0:
            var pm = self.addr(a)
            var pi = self.addr(im)
            # cpu-route: host image only (CPU-only installs), the GPU image moved above
            self.hmove(hk, pm, pi, self.addr(out), [MV_TAKE_ROWS, count * c, c, 0, 0, 1, 0, self.n(a), count * c,
                                                    count])
        return out

    def t(mut self, a: Int, hk: Int) raises -> Int:
        """`m.T`. `hk` is `_host_kit()`, the default-mode image: a host matrix
        on a GPU image goes up once on it (it moves) and is transposed there."""
        var r = self.r(a)
        var c = self.c(a)
        var n = r * c
        if self.mats[a].fitted >= 0:
            # a fitted matrix stays where it is: transpose its host words
            var src = self.copy(a)
            return self.t(src, hk)
        if self.mats[a].device < 0 and n > 0 and r != 1 and c != 1 and n < I32_MAX:
            if self.images[hk].resident:
                _ = self.device(hk, a)
        if self.mats[a].device >= 0 and (r == 1 or c == 1):
            return self.add(-1, c, r, self.mats[a].device, self.mats[a].image)
        if self.mats[a].device >= 0 and n < I32_MAX:
            var j = self.mats[a].image
            var d = self.mats[a].device
            var out = self.dout(j, c, r)
            self.dmove(j, d, d, self.mats[out].device, [MV_TRANSPOSE, n, r, c, 0, 0, 0, n, n, 1])
            return out
        if r == 1 or c == 1:
            _ = self.addr(a)
            return self.add(self.mats[a].store, c, r)
        var out = self.zeros(c, r)
        var pm = self.addr(a)
        var one = self.shared_zero()
        # cpu-route: host image only (CPU-only installs); a GPU image transposed above
        self.hmove(hk, pm, self.addr(one), self.addr(out), [MV_TRANSPOSE, n, r, c, 0, 0, 0, n, n, 1])
        return out

    def fill0(mut self, k: Int, a: Int, start: Int, stride: Int, count: Int) raises:
        """A[start + t * stride] = 0 for t < count, in place."""
        if count == 0:
            return
        self.writable(a)
        var n = self.n(a)
        var words: List[Int] = [MV_FILL0, count, start, stride, 0, 0, 0, n, n, 1]
        if self.use(k, a):
            var ia = self.device(k, a)
            var one = self.dev_one(k)
            self.dmove(k, ia, one, ia, words)
            return
        var pa = self.addr(a)
        var po = self.addr(self.shared_zero())
        self.hmove(k, pa, po, pa, words)

    def reduce(mut self, k: Int, a: Int, op: Int) raises -> Int:
        """max |A| (0), max A (1) or min A (2) as a 1 x 1 matrix."""
        var n = self.n(a)
        if n == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        if self.use(k, a):
            var out = self.dout(k, 1, 1)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_REDUCE, [ia, self.mats[out].device, n, op])
            return out
        var out = self.zeros(1, 1)
        var pa = self.addr(a)
        _ = self.call(k, XD_REDUCE, [pa, self.addr(out)], [n, op], List[Float64]())
        return out

    def strided(mut self, k: Int, a: Int, count: Int, stride_in: Int, off: Int) raises -> Int:
        """A's values at off + t * stride (t < count) as 1 x count."""
        var stride = off + 1 if count == 1 else stride_in
        var idx = self.of([Float64(off)], 1, 1)
        var n = self.n(a)
        var words: List[Int] = [MV_TAKE_COLS, count, 1, stride, 0, 0, 0, n, count, 1]
        if count > 0 and self.use(k, a):
            var out = self.dout(k, 1, count)
            var ia = self.device(k, a)
            var ii = self.device(k, idx)
            self.dmove(k, ia, ii, self.mats[out].device, words)
            return out
        var out = self.zeros(1, count)
        if count > 0:
            var pa = self.addr(a)
            var pi = self.addr(idx)
            self.hmove(k, pa, pi, self.addr(out), words)
        return out

    def word(mut self, k: Int, a: Int, i: Int) raises -> Float64:
        """A's value at flat position i: one word down."""
        if self.on_dev(a) and self.n(a) > 1:
            return self.at(self.strided(k, a, 1, 1, i), 0)
        return self.at(a, i)

    def place_strided(mut self, k: Int, d: Int, v: Int, stride: Int, off: Int) raises:
        """D[off + t * stride] = V[t], in place."""
        var cnt = self.n(v)
        if cnt == 0:
            return
        self.writable(d)
        var words: List[Int] = [MV_PLACE_COLS, cnt, 1, stride, off, 0, 0, cnt, self.n(d), 1]
        if self.use(k, d, v):
            var iv = self.device(k, v)
            var one = self.dev_one(k)
            var id = self.device(k, d)
            self.dmove(k, iv, one, id, words)
            return
        var pv = self.addr(v)
        var po = self.addr(self.shared_zero())
        var pd = self.addr(d)
        self.hmove(k, pv, po, pd, words)

    def place_rows(mut self, k: Int, d: Int, v: Int, row: Int) raises:
        """V (r x D.c) into rows row .. row + r - 1 of D, in place."""
        if self.n(v) > 0:
            self.place_strided(k, d, self.reshape(v, 1, self.n(v)), 1, row * self.c(d))

    def vstack(mut self, k: Int, ms: List[Int]) raises -> Int:
        """`vstack(ms)`: device PLACE_COLS moves for a resident image."""
        var total = 0
        for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
            total += self.n(ms[i])
        if not (self.images[k].resident and total > 0):
            return self.vstack_host(ms)
        var c = self.c(ms[0])
        var rows = 0
        for i in range(len(ms)):  # small-loop(ms: the stacked matrices): bounded by the caller
            rows += self.r(ms[i])
        var out = self.dout(k, rows, c)
        var one = self.dev_one(k)
        var off = 0
        for i in range(len(ms)):  # driver-loop: over (ms: one Mojo move per stacked matrix): bounded by the caller
            var cnt = self.n(ms[i])
            if cnt > 0:
                var im = self.device(k, ms[i])
                self.dmove(k, im, one, self.mats[out].device, [MV_PLACE_COLS, cnt, c, c, off * c, 0, 0, cnt, rows * c, 1])
            off += self.r(ms[i])
        return out

    def absmax_signs(mut self, k: Int, a: Int, by_col: Bool) raises -> Int:
        """Per column (by_col, 1 x c) or row (r x 1): -1 where the largest-|.|
        entry (ties to the lower index) is negative, else +1."""
        var r = self.r(a)
        var c = self.c(a)
        var cnt = c if by_col else r
        var bc = 1 if by_col else 0
        var out: Int
        if cnt > 0 and r * c > 0 and self.use(k, a):
            out = self.dout(k, 1, cnt)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_ABSMAX, [ia, self.mats[out].device, r, c, bc])
        else:
            out = self.zeros(1, cnt)
            if cnt > 0 and r * c > 0:
                var pa = self.addr(a)
                _ = self.call(k, XD_ABSMAX_SIGN, [pa, self.addr(out)], [r, c, bc], List[Float64]())
        if by_col:
            return self.neg_signs(k, out)
        return self.neg_signs(k, self.reshape(out, cnt, 1))

    def refuse_nan(mut self, k: Int, a: Int) raises:
        var v = self.word(k, self.reduce(k, a, SEL_MAX), 0)
        if v != v:
            raise Error("x_decomp: a NaN has no order (refused)")

    def order(mut self, k: Int, a: Int) raises -> Int:
        """The stable ascending order of A's values (ties to the lower index)
        as n x 1 exact floats; NaN refused."""
        var n = self.n(a)
        if n > 0 and self.use(k, a):
            if n >= F32_INDEX_MAX:
                raise Error("x_decomp: order_f exceeds the float32 index bound")
            self.refuse_nan(k, a)
            var out = self.dout(k, n, 1)
            var cnt = self.dout(k, 1, 1)
            var ia = self.device(k, a)
            var sid = self.device(k, a)
            _ = self.dcall(k, XD_DEV_ORDER, [ia, sid, self.mats[out].device, self.mats[cnt].device, n, 0, 0])
            return out
        var out = self.zeros(n, 1)
        if n > 0:
            var pa = self.addr(a)
            _ = self.call(k, XD_ORDER, [pa, self.addr(out)], [n], List[Float64]())
        return out

    def argmin_all(mut self, k: Int, a: Int) raises -> Int:
        """Every position (exact floats, c x 1) whose value is the minimum."""
        var n = self.n(a)
        if n > 0 and self.use(k, a):
            if n >= F32_INDEX_MAX:
                raise Error("x_decomp: argmin_all exceeds the float32 index bound")
            var out = self.dout(k, n, 1)
            var cnt = self.dout(k, 1, 1)
            var ia = self.device(k, a)
            _ = self.dcall(k, XD_DEV_ARGMIN_ALL, [ia, self.mats[out].device, self.mats[cnt].device, n])
            var c = Int(self.word(k, cnt, 0))
            if c > 0:
                return self.rows(out, 0, c)
            return self.zeros(0, 1)
        var buf = self.zeros(max(n, 1), 1)
        var c = 0
        if n > 0:
            var pa = self.addr(a)
            c = self.call(k, XD_ARGMIN_ALL, [pa, self.addr(buf)], [n], List[Float64]())
        var out = self.zeros(c, 1)
        if c > 0:
            unsafe_memcpy(dest=self.ptr(out), src=self.ptr(buf), count=c)
        return out

    def pad_zero_row(mut self, k: Int, x: Int) raises -> Int:
        """[X; 0]: X over one zero row (on the device for a device X)."""
        var r = self.r(x)
        var c = self.c(x)
        if c > 0 and self.on_dev(x) and self.use(k, x):
            var tot = (r + 1) * c
            var out = self.dout(k, r + 1, c)
            var one = self.dev_one(k)
            var cnt = r * c
            var oid = self.mats[out].device
            if cnt > 0:
                var ix = self.device(k, x)
                self.dmove(k, ix, one, oid, [MV_PLACE_COLS, cnt, c, c, 0, 0, 0, cnt, tot, 1])
            self.dmove(k, oid, one, oid, [MV_FILL0, c, cnt, 1, 0, 0, 0, tot, tot, 1])
            return out
        # X's words, then one zero row (the store starts zeroed)
        var src = self.addr(x)
        var out = self.zeros(r + 1, c)
        if r * c > 0:
            unsafe_memcpy(dest=self.ptr(out), src=MK_F32P(unsafe_from_address=src), count=r * c)
        return out

    def dsum_sq(mut self, k: Int, m: Int) raises -> Float64:
        """`_dsum_sq(k, M)`: the binary64 sum of the squares of M's values."""
        var n = self.n(m)
        if n == 0:
            return 0.0
        if self.use(k, m):
            var out = self.dout(k, 1, 2)
            var im = self.device(k, m)
            _ = self.dcall(k, XD_DEV_DSUM_SQ, [im, self.mats[out].device, n])
            # the binary64 result's two float32 words, read as one double
            return MK_F64P(unsafe_from_address=self.addr(out))[0]
        # cpu-route: host image only (CPU-only installs); a GPU image sums on the device above
        var result = List[Float64](length=1, fill=0.0)
        var pm = self.addr(m)
        _ = self.call(k, XD_DSUM_SQ, [pm, Int(result.unsafe_ptr())], [n], List[Float64]())
        return result[0]

    # ---- neighbor graphs (x_decomp/graph_cells.mojo)
    def dijkstra(mut self, k: Int, w: Int) raises -> Int:
        var n = self.r(w)
        if self.images[k].resident:
            var dist = self.dout(k, n, n)
            var iw = self.device(k, w)
            _ = self.dcall(k, XD_DEV_GRAPH_DIJKSTRA, [iw, self.mats[dist].device, n])
            return dist
        var dist = self.zeros(n, n)
        var pw = self.addr(w)
        _ = self.hcall(k, XD_GRAPH_DIJKSTRA, [pw, self.addr(dist)], [n])
        return dist

    def graph_knn(mut self, k: Int, d: Int, nn: Int, exclude_self: Bool) raises -> Tuple[Int, Int]:
        var n = self.r(d)
        var m = self.c(d)
        var ex = 1 if exclude_self else 0
        if self.images[k].resident:
            var idx = self.dout(k, n, nn)
            var dst = self.dout(k, n, nn)
            if n * nn > 0:
                var id = self.device(k, d)
                _ = self.dcall(k, XD_DEV_GRAPH_KNN, [id, self.mats[idx].device, self.mats[dst].device, n, m, nn, ex])
            return (idx, dst)
        var idx = self.zeros(n, nn)
        var dst = self.zeros(n, nn)
        if n * nn > 0:
            var pd = self.addr(d)
            _ = self.hcall(k, XD_GRAPH_KNN, [pd, self.addr(idx), self.addr(dst)], [n, m, nn, ex])
        return (idx, dst)

    def graph_knn_dense(mut self, k: Int, idx: Int, w: Int, n: Int) raises -> Int:
        var nn = self.c(idx)
        if self.images[k].resident:
            var out = self.dout(k, n, n)
            var ii = self.device(k, idx)
            var iw = self.device(k, w)
            _ = self.dcall(k, XD_DEV_GRAPH_KNN_DENSE, [ii, iw, self.mats[out].device, n, nn])
            return out
        var out = self.zeros(n, n)
        var pi = self.addr(idx)
        var pw = self.addr(w)
        _ = self.hcall(k, XD_GRAPH_KNN_DENSE, [pi, pw, self.addr(out)], [n, nn])
        return out

    def graph_radius(mut self, k: Int, d: Int, r: Float64) raises -> Int:
        var n = self.r(d)
        var floats: List[Float64] = [r]
        if self.images[k].resident:
            var out = self.dout(k, n, n)
            var id = self.device(k, d)
            _ = self.call(k, XD_DEV_GRAPH_RADIUS, List[Int](), [id, self.mats[out].device, n], floats)
            return out
        var out = self.zeros(n, n)
        var pd = self.addr(d)
        _ = self.call(k, XD_GRAPH_RADIUS, [pd, self.addr(out)], [n], floats)
        return out

    def graph_radius_geo(mut self, k: Int, dq: Int, d: Int, r: Float64) raises -> Int:
        var nq = self.r(dq)
        var n = self.c(dq)
        var floats: List[Float64] = [r]
        if self.images[k].resident:
            var out = self.dout(k, nq, n)
            if nq * n > 0:
                var iq = self.device(k, dq)
                var id = self.device(k, d)
                _ = self.call(k, XD_DEV_GRAPH_RADIUS_GEO, List[Int](), [iq, id, self.mats[out].device, nq, n], floats)
            return out
        var out = self.zeros(nq, n)
        if nq * n > 0:
            var pq = self.addr(dq)
            var pd = self.addr(d)
            _ = self.call(k, XD_GRAPH_RADIUS_GEO, [pq, pd, self.addr(out)], [nq, n], floats)
        return out

    def graph_lle_iw(mut self, k: Int, idx: Int, wb: Int, n: Int) raises -> Int:
        var nn = self.c(idx)
        if self.images[k].resident:
            var out = self.dout(k, n, n)
            var ii = self.device(k, idx)
            var iw = self.device(k, wb)
            _ = self.dcall(k, XD_DEV_GRAPH_LLE_IW, [ii, iw, self.mats[out].device, n, nn])
            return out
        var out = self.zeros(n, n)
        var pi = self.addr(idx)
        var pw = self.addr(wb)
        _ = self.hcall(k, XD_GRAPH_LLE_IW, [pi, pw, self.addr(out)], [n, nn])
        return out

    def graph_components(mut self, k: Int, w: Int) raises -> Tuple[Int, Int]:
        var n = self.r(w)
        if self.images[k].resident:
            var comp = self.dout(k, n, 1)
            var iw = self.device(k, w)
            var c = self.dcall(k, XD_DEV_GRAPH_COMPONENTS, [iw, self.mats[comp].device, n])
            return (comp, c)
        var comp = self.zeros(n, 1)
        var pw = self.addr(w)
        var c = self.hcall(k, XD_GRAPH_COMPONENTS, [pw, self.addr(comp)], [n])
        return (comp, c)

    def graph_join(mut self, k: Int, w: Int, d: Int, comp: Int, count: Int) raises:
        """sklearn `_fix_connected_components`, W changed in place."""
        self.writable(w)
        var n = self.r(w)
        if self.images[k].resident:
            var iw = self.device(k, w)
            var id = self.device(k, d)
            var ic = self.device(k, comp)
            _ = self.dcall(k, XD_DEV_GRAPH_JOIN, [iw, id, ic, n, count])
            return
        var pw = self.addr(w)
        var pd = self.addr(d)
        var pc = self.addr(comp)
        _ = self.hcall(k, XD_GRAPH_JOIN, [pw, pd, pc], [n, count])

    # ---- LLE cells
    def barycenter(mut self, k: Int, x: Int, y: Int, nbr: Int, reg: Float64) raises -> Int:
        """sklearn barycenter_weights: (n x k) weights of each row of X on its
        k neighbors in Y."""
        var n = self.r(x)
        var kk = self.c(nbr)
        var w = self.zeros(n, kk)
        var flags = self.zeros(n, 1)
        var px = self.addr(x)
        var py = self.addr(y)
        var pi = self.addr(nbr)
        _ = self.call(k, XD_BARYCENTER, [px, py, pi, self.addr(w), self.addr(flags)], [n, self.r(y), self.c(x), kk],
                      [reg])
        return w

    def lle_apply(mut self, k: Int, wb: Int, idm: Int, e: Int) raises -> Int:
        """out (nq x nc) = sum_a Wb[i, a] E[idm[i, a]]."""
        var nq = self.r(wb)
        var nn = self.c(wb)
        var nc = self.c(e)
        var out = self.zeros(nq, nc)
        if nq * nc > 0:
            var pw = self.addr(wb)
            var pi = self.addr(idm)
            var pe = self.addr(e)
            _ = self.hcall(k, XD_LLE_APPLY, [pw, pi, pe, self.addr(out)], [nq, self.r(e), nn, nc])
        return out

    def lle_local(mut self, k: Int, m: Int, idm: Int, method: Int, nn: Int, nc: Int, tol: Float64) raises -> Int:
        """LTSA (0), Hessian (1) or modified (2) LLE's stacked factor B."""
        var n = self.r(m)
        var d = self.c(m)
        var rows = n * (nn - 1 - nc) if method == 1 else n * nn
        var bm = self.zeros(rows, n)
        var pm = self.addr(m)
        var pi = self.addr(idm)
        _ = self.call(k, XD_LLE_LOCAL, [pm, pi, self.addr(bm)], [method, n, d, nn, nc], [tol])
        return bm

    def enable_lle_precision(mut self, k: Int):
        """The Apple FAST image's ordered GEMM and LU for LLE's near-null
        solve (UNPACK decided `lle_ordered`: mode fast, vendor metal, the
        image has the cells); every other image keeps its entries."""
        if self.images[k].lle_ordered:
            self.images[k].lle_on = True
