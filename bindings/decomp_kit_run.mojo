# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the native decomposition owners: the typed x_decomp kit
(lane entangled-decompkit, 2026-11-12; was the Python-holding `DecompKit` of
bindings/decomp_kit_native.mojo, which held the connected entries dict, the
output context, one Python Array per matrix and the query model).

`DKit` is python/mojolearn/_expansion_decomp.py's `_Kit` / `_M`, operation for
operation, with the same x_decomp words in the same order under the same
route rules (a GPU image, one with the resident operations, runs every
non-empty operation on the device, as `_Kit._use` with `_RES_MIN` 1 does; a
host image takes the host-address operations). So the bits are the ones the
Python-holding kit produced.

A matrix keeps its host words in a kit-owned float32 store, or borrows an
admitted caller buffer by address (the door keeps its owner alive for the
call), and keeps its device words (an x_decomp pool id) once it has both, as
the Python-holding kit did. `on_device` follows the Python `_M` exactly; an
operation that writes a matrix in place drops the copy that went stale
(`wrote_device` / `wrote_host`). Pivots and membership words live in
kit-owned int32 stores (`istores`; were Python `array('i')` stores).

No Python object is held or called here. Every operation is one call
through the image's published dispatch (`EntryRef`, resolved by the door's
UNPACK, bindings/decomp_kit_native.mojo `decomp_kit_unpack`). The door
releases the GIL around a RUN sequence and keeps every borrowed owner alive
across it; PACK copies a matrix's words into a public Array.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.admitted_refusals import admitted_finite_refuse
from core.classical_tsqr_impl import tsqr_fill_constant, tsqr_fill_indices, tsqr_copy_columns, tsqr_fill_eye
from x_decomp.descriptor_schema import (
    XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_MOVE, XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD, XD_DEV_DOWNLOAD,
    XD_DEV_EW, XD_DEV_GEMM, XD_DEV_COLSUM, XD_DEV_ROWSUM, XD_DEV_MOVE, XD_DEV_UPLOAD_FINITE, XD_FLAGS, XD_ORDER,
    XD_DEV_REDUCE, XD_DEV_ORDER, XD_SVD, XD_DEV_SVD, XD_RANK, XD_ORTH_DIAG, XD_DEV_ORTH_DIAG, XD_GEQRF, XD_ORGQR,
    XD_RAND, XD_LU, XD_LU_SOLVE, XD_EIGH, XD_LASSO_ROWS, XD_LARS_ROWS, XD_OMP_ROWS, XD_LU_AUX, XD_ABSMAX_SIGN,
    XD_REDUCE, XD_SELECT_SMALLEST, XD_ARGMIN_ALL, XD_LU_RESID, XD_S_FLAGS, XD_QFIX_FLAGS, XD_DEV_RAND,
    XD_DEV_ABSMAX, XD_DEV_LU, XD_DEV_LU_AUX, XD_DEV_CODE_ROWS, XD_DEV_EIGH_TO, XD_DEV_ARGMIN_ALL,
    XD_DEV_SELECT_SMALLEST,
)

comptime DKitF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime DKitI32 = MutPointer[Int32, MutUntrackedOrigin]

# x_decomp/cells.mojo elementwise codes (`_expansion_decomp._OP`).
comptime DK_ADD = 0
comptime DK_SUB = 1
comptime DK_MUL = 2
comptime DK_DIV = 3
comptime DK_MAXS = 5
comptime DK_SQRT = 7
comptime DK_SQ = 8
comptime DK_LOGS = 10
comptime DK_ABS = 13
comptime DK_SCALE = 14
comptime DK_RECIP = 16
comptime DK_SOFT = 17
comptime DK_MINS = 19
comptime DK_ADDS = 22
comptime DK_GTS = 23
comptime DK_LE = 34
comptime DK_SELECT = 35

# x_decomp/moves.mojo move codes.
comptime DK_TAKE_ROWS = 0
comptime DK_TRANSPOSE = 1
comptime DK_PLACE_COLS = 2
comptime DK_TAKE_COLS = 4
comptime DK_F32_INDEX_MAX = 1 << 24
comptime DK_I32_MAX = (1 << 31) - 1

# The image's optional entries and route words, read once by the door's
# UNPACK from the connected entries dict (`_Kit._opt_dev` and `flags`).
comptime DK_HAS_DEV_SVD = 1          # x_decomp_dev_svd
comptime DK_HAS_DEV_LU = 2           # x_decomp_dev_lu
comptime DK_HAS_DEV_CODE_ROWS = 4    # x_decomp_dev_code_rows
comptime DK_HAS_DEV_EIGH_TO = 8      # x_decomp_dev_eigh_to
comptime DK_HAS_S_FLAGS = 16         # x_decomp_s_flags
comptime DK_HAS_QFIX_FLAGS = 32      # x_decomp_qfix_flags
comptime DK_HAS_IDN_FLAGS = 64       # x_decomp_idn_flags


struct DKitStore(Movable):
    """A kit-owned host store (at least one word, so an empty matrix keeps a
    valid address, as `_M._one` did). Never resized, so its address holds."""
    var words: List[Float32]

    def __init__(out self, cells: Int):
        self.words = List[Float32](length=max(cells, 1), fill=0.0)


struct DKitIStore(Movable):
    """A kit-owned int32 store (LU pivots, membership): was `array('i')`."""
    var words: List[Int32]

    def __init__(out self, cells: Int):
        self.words = List[Int32](length=max(cells, 1), fill=0)


struct DKitMatrix(Copyable, Movable):
    # `store`: the kit store holding the host words, -1 when none.
    # `borrowed`: the address of an admitted caller buffer the matrix reads
    # in place (kept alive by the door), 0 when none. A matrix has host
    # words when either is set.
    var store: Int
    var borrowed: Int
    var device: Int          # an x_decomp pool id, or -1
    var rows: Int
    var cols: Int
    # Whether the Python `_M` would be on the device (its `_d` set): every
    # route `_Kit` chose by residency reads this, never the buffers kept.
    var on_device: Bool

    def __init__(out self, store: Int, borrowed: Int, device: Int, rows: Int, cols: Int):
        self.store = store
        self.borrowed = borrowed
        self.device = device
        self.rows = rows
        self.cols = cols
        self.on_device = device >= 0


struct DKit(Movable):
    """`_Kit(mode)` over the x_decomp image the door's UNPACK resolved."""
    var entry: EntryRef
    var keyed: Bool      # the image published its table (else every call refuses)
    var resident: Bool   # a GPU image: it has the resident operations
    var present: Int     # DK_HAS_* bits of the image's optional entries
    var flag_read: Int   # DK_HAS_*_FLAGS bits already read into the words below
    var s_flags: Int
    var qfix_flags: Int
    var idn_flags: Int
    var matrices: List[DKitMatrix]
    var stores: List[DKitStore]
    var istores: List[DKitIStore]
    var allocations: List[Int]
    var one_device: Int
    var one_host: Int

    def __init__(out self, entry: EntryRef, keyed: Bool, resident: Bool, present: Int):
        self.entry = entry
        self.keyed = keyed
        self.resident = resident
        self.present = present
        self.flag_read = 0
        self.s_flags = 0
        self.qfix_flags = 0
        self.idn_flags = 0
        self.matrices = List[DKitMatrix]()
        self.stores = List[DKitStore]()
        self.istores = List[DKitIStore]()
        self.allocations = List[Int]()
        self.one_device = -1
        self.one_host = -1
        # `_M._one`: the unused broadcast operand, a 0.
        self.one_host = self.zeros(1, 1)

    def close(mut self):
        """Return every device buffer this call allocated to the pool (views
        share ids, so only the allocation ledger frees)."""
        for i in range(len(self.allocations) - 1, -1, -1):  # driver-loop: over native device allocation handles
            try:
                self.dev_free(self.allocations[i])
            except:
                pass
        self.allocations.clear()

    # ---- the x_decomp dispatch (x_decomp/descriptor_schema.mojo)
    def xd(self, op: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        """One x_decomp operation through the image's dispatch. The caller
        keeps every addressed buffer alive; the door releases the GIL."""
        if not self.keyed:
            raise Error("ImportError: mojolearn: the x_decomp binding publishes no native entry table")
        return entry_ref_call(self.entry, op, addresses, ints, floats)

    def dev_alloc(self, n: Int) raises -> Int:
        return self.xd(XD_DEV_ALLOC, List[Int](), [n], List[Float64]())

    def dev_free(self, handle: Int) raises:
        _ = self.xd(XD_DEV_FREE, List[Int](), [handle], List[Float64]())

    def dev_upload(self, handle: Int, address: Int, n: Int) raises:
        _ = self.xd(XD_DEV_UPLOAD, [address], [handle, n], List[Float64]())

    def dev_download(self, handle: Int, address: Int, n: Int) raises:
        _ = self.xd(XD_DEV_DOWNLOAD, [address], [handle, n], List[Float64]())

    def has(self, bit: Int) -> Bool:
        return (self.present & bit) != 0

    def opt_dev(self, bit: Int) -> Bool:
        # `_Kit._opt_dev`: an optional resident entry this image exports.
        return self.resident and self.has(bit)

    def flags(mut self, bit: Int) raises -> Int:
        """A compiled-route word (DK_HAS_S_FLAGS, DK_HAS_QFIX_FLAGS or
        DK_HAS_IDN_FLAGS): 0 when the image has no such entry, read once per
        kit (compile-time constants of the image)."""
        var op = XD_S_FLAGS if bit == DK_HAS_S_FLAGS else (XD_QFIX_FLAGS if bit == DK_HAS_QFIX_FLAGS else XD_FLAGS)
        if bit != DK_HAS_S_FLAGS and bit != DK_HAS_QFIX_FLAGS and bit != DK_HAS_IDN_FLAGS:
            raise Error("x_decomp: no compiled route word " + String(bit))
        if (self.flag_read & bit) == 0:
            var word = 0
            if self.has(bit):
                word = self.xd(op, List[Int](), List[Int](), List[Float64]())
            if bit == DK_HAS_S_FLAGS:
                self.s_flags = word
            elif bit == DK_HAS_QFIX_FLAGS:
                self.qfix_flags = word
            else:
                self.idn_flags = word
            self.flag_read |= bit
        if bit == DK_HAS_S_FLAGS:
            return self.s_flags
        if bit == DK_HAS_QFIX_FLAGS:
            return self.qfix_flags
        return self.idn_flags

    # ---- matrices
    def add(mut self, store: Int, rows: Int, cols: Int, device: Int = -1, borrowed: Int = 0) -> Int:
        var result = len(self.matrices)
        self.matrices.append(DKitMatrix(store, borrowed, device, rows, cols))
        return result

    def size(self, a: Int) -> Int:
        return self.matrices[a].rows * self.matrices[a].cols

    def nrows(self, a: Int) -> Int:
        return self.matrices[a].rows

    def ncols(self, a: Int) -> Int:
        return self.matrices[a].cols

    def _has_host(self, a: Int) -> Bool:
        return self.matrices[a].store >= 0 or self.matrices[a].borrowed != 0

    def _new_store(mut self, cells: Int) -> Int:
        var store = len(self.stores)
        self.stores.append(DKitStore(cells))
        return store

    def zeros(mut self, rows: Int, cols: Int) -> Int:
        # One spare word gives an empty matrix a safe address, as `_M._one`.
        return self.add(self._new_store(rows * cols), rows, cols)

    def borrow(mut self, address: Int, rows: Int, cols: Int) -> Int:
        """`input` / `input_snapshot`: a C-order float32 buffer the door
        admitted (and keeps alive), read in place, never written (every
        operation writes a fresh output)."""
        if rows * cols == 0 or address == 0:
            return self.zeros(rows, cols)
        return self.add(-1, rows, cols, -1, address)

    def defers_finite(self) -> Bool:
        """Whether this image tests an input on its upload (XD_DEV_UPLOAD_FINITE,
        compiled into every resident GPU image): a door may then admit a
        borrowed X with `defer_finite` and add it through `borrow_tested`."""
        return self.resident and self.keyed

    def borrow_tested(mut self, address: Int, rows: Int, cols: Int, est: String) raises -> Int:
        """`borrow` of an X the door admitted with `defer_finite`: the words go
        up now through XD_DEV_UPLOAD_FINITE, which tests them on the device,
        and a NaN or an infinity is refused with admit_x's wording. The
        matrix still starts as a host matrix (`on_device` False), so every
        route is the one `borrow` gave. Only where `defers_finite()` holds."""
        var index = self.borrow(address, rows, cols)
        var n = rows * cols
        if n < 1:
            return index
        var handle = self.dev_alloc(n)
        self.allocations.append(handle)
        var source: List[Int] = [self.at(index)]
        var word = self.xd(XD_DEV_UPLOAD_FINITE, source, [handle, n], List[Float64]())
        self.matrices[index].device = handle
        self.matrices[index].on_device = False
        admitted_finite_refuse(word, est)
        return index

    def add_resident(mut self, address: Int, handle: Int, rows: Int, cols: Int) -> Int:
        """A fitted operand whose device copy the door's UNPACK served from (or
        just gave to) the model's resident record: the host words borrowed at
        `address`, the device words `handle` (owned by the record, never by
        this kit). It starts as a host matrix (`on_device` False), exactly as
        `borrow` adds it, so no route or bit changes."""
        var index = self.add(-1, rows, cols, handle, address)
        self.matrices[index].on_device = False
        return index

    def add_device(mut self, handle: Int, rows: Int, cols: Int) -> Int:
        """A device matrix an earlier query computed and the model's record
        keeps (`resident_derived`): added as that call left it (`on_device`
        True); the record owns the buffer."""
        return self.add(-1, rows, cols, handle)

    def release_device(mut self, a: Int) -> Int:
        """`resident_keep`'s kit half: matrix a's pool id leaves this kit's
        allocation ledger (the door gives it to a record capsule in PACK);
        -1 when a is not on the device or the buffer is not this kit's (a
        view's or a resident input's)."""
        if not self.matrices[a].on_device or self.matrices[a].device < 0:
            return -1
        var handle = self.matrices[a].device
        for i in range(len(self.allocations)):  # driver-loop: over this kit's device allocation handles
            if self.allocations[i] == handle:
                _ = self.allocations.pop(i)
                return handle
        return -1

    def const(mut self, value: Float32, rows: Int = 1, cols: Int = 1) raises -> Int:
        # `_M.of([v] * (r * c), r, c)`: the value rounded once to float32.
        var out = self.zeros(rows, cols)
        tsqr_fill_constant(self.ptr(out), rows * cols, value)
        return out

    def indices(mut self, start: Int, stop: Int) raises -> Int:
        # `array.array("f", range(start, stop))`: row or column numbers as exact floats.
        var out = self.zeros(stop - start, 1)
        tsqr_fill_indices(self.ptr(out), start, stop)
        return out

    def device_output(mut self, rows: Int, cols: Int) raises -> Int:
        var handle = self.dev_alloc(max(rows * cols, 1))
        self.allocations.append(handle)
        return self.add(-1, rows, cols, handle)

    def host(mut self, a: Int) raises:
        # `_M.s`: a device-only matrix comes down once; the matrix then counts
        # as a host matrix (its device words stay valid and are reused by a
        # later upload, the same words).
        self.matrices[a].on_device = False
        if not self._has_host(a):
            var n = self.size(a)
            var store = self._new_store(n)
            if n:
                self.dev_download(self.matrices[a].device, Int(self.stores[store].words.unsafe_ptr()), n)
            self.matrices[a].store = store

    def at(mut self, a: Int) raises -> Int:
        """The host address of matrix a's words (`_M.s`: the matrix counts
        as a host matrix after)."""
        self.host(a)
        if self.matrices[a].borrowed != 0:
            return self.matrices[a].borrowed
        return Int(self.stores[self.matrices[a].store].words.unsafe_ptr())

    def ptr(mut self, a: Int) raises -> DKitF32:
        return DKitF32(unsafe_from_address=self.at(a))

    def device(mut self, a: Int) raises -> Int:
        # `_Kit._did`: a host matrix goes up once.
        if self.matrices[a].device < 0:
            var n = self.size(a)
            self.host(a)
            var handle = self.dev_alloc(max(n, 1))
            self.allocations.append(handle)
            if n:
                self.dev_upload(handle, self.at(a), n)
            self.matrices[a].device = handle
        self.matrices[a].on_device = True
        return self.matrices[a].device

    def dev_one(mut self) raises -> Int:
        # `_dev_one`: the unused operand on the device, uploaded once.
        if self.one_device < 0:
            var handle = self.dev_alloc(1)
            self.allocations.append(handle)
            self.dev_upload(handle, Int(self.stores[self.matrices[self.one_host].store].words.unsafe_ptr()), 1)
            self.one_device = handle
        return self.one_device

    def wrote_device(mut self, a: Int):
        # The device copy was written in place: the host words are stale.
        self.matrices[a].store = -1
        self.matrices[a].borrowed = 0

    def wrote_host(mut self, a: Int):
        # The host copy was written in place: the device words are stale.
        self.matrices[a].device = -1
        self.matrices[a].on_device = False

    def use(self, a: Int, b: Int = -1, c: Int = -1) -> Bool:
        # `_Kit._use` with `_RES_MIN` 1: a GPU image, and an operand on the
        # device or with at least one value.
        if not self.resident:
            return False
        var found = False
        for k in range(3):  # small-loop(operands): the three operand slots of one call
            var m = a if k == 0 else (b if k == 1 else c)
            if m < 0:
                continue
            if self.matrices[m].device >= 0 or self.size(m) >= 1:
                found = True
        return found

    def copy_host(mut self, a: Int) raises -> Int:
        # `_M.copy()`: a host copy (a device-only matrix's download is the copy).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var n = rows * cols
        self.matrices[a].on_device = False
        if not self._has_host(a):
            var store = self._new_store(n)
            if n:
                self.dev_download(self.matrices[a].device, Int(self.stores[store].words.unsafe_ptr()), n)
            return self.add(store, rows, cols)
        var out = self.zeros(rows, cols)
        if n:
            unsafe_memcpy(dest=self.ptr(out), src=self.ptr(a), count=n)
        return out

    def istore(mut self, cells: Int) -> Int:
        var index = len(self.istores)
        self.istores.append(DKitIStore(cells))
        return index

    def iaddr(self, index: Int) -> Int:
        return Int(self.istores[index].words.unsafe_ptr())

    # ---- elementwise
    def _mode_of(self, a: Int, x: Int) raises -> Int:
        if x < 0:
            return 3
        var ar = self.matrices[a].rows
        var ac = self.matrices[a].cols
        var xr = self.matrices[x].rows
        var xc = self.matrices[x].cols
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

    def ew(mut self, op: Int, a: Int, b: Int = -1, c: Int = -1, scalar: Float64 = 0.0) raises -> Int:
        var bm = self._mode_of(a, b)
        var cm = self._mode_of(a, c)
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nb = 1 if b < 0 else self.size(b)
        var nc = 1 if c < 0 else self.size(c)
        if rows * cols and self.use(a, b, c):
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var db = self.dev_one() if b < 0 else self.device(b)
            var dc = self.dev_one() if c < 0 else self.device(c)
            _ = self.xd(XD_DEV_EW, List[Int](), [da, db, dc, self.matrices[out].device, op, rows * cols, cols, nb, bm,
                        nc, cm], [scalar])
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var one = self.one_host
            var ib = one if b < 0 else b
            var ic = one if c < 0 else c
            var addresses: List[Int] = [self.at(a), self.at(ib), self.at(ic), self.at(out)]
            _ = self.xd(XD_EW, addresses, [op, rows * cols, cols, nb, bm, nc, cm], [scalar])
        return out

    # ---- products and sums
    def mm(mut self, a: Int, b: Int, ta: Bool = False, tb: Bool = False) raises -> Int:
        var m = self.matrices[a].cols if ta else self.matrices[a].rows
        var k = self.matrices[a].rows if ta else self.matrices[a].cols
        var k2 = self.matrices[b].cols if tb else self.matrices[b].rows
        var n = self.matrices[b].rows if tb else self.matrices[b].cols
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
        if m * n and m * k and k * n and self.use(a, b):
            var out = self.device_output(m, n)
            var da = self.device(a)
            var db = self.device(b)
            _ = self.xd(XD_DEV_GEMM, List[Int](), [da, db, self.matrices[out].device, m, k, n, Int(ta), Int(tb)],
                        List[Float64]())
            return out
        var out = self.zeros(m, n)
        if m * n:
            var addresses: List[Int] = [self.at(a), self.at(b), self.at(out)]
            _ = self.xd(XD_GEMM, addresses, [m, k, n, Int(ta), Int(tb)], List[Float64]())
        return out

    def _sum(mut self, a: Int, columns: Bool) raises -> Int:
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var nr = 1 if columns else rows
        var nc = cols if columns else 1
        if rows * cols and self.use(a):
            var out = self.device_output(nr, nc)
            var da = self.device(a)
            _ = self.xd(XD_DEV_COLSUM if columns else XD_DEV_ROWSUM, List[Int](),
                        [da, self.matrices[out].device, rows, cols], List[Float64]())
            return out
        var out = self.zeros(nr, nc)
        var addresses: List[Int] = [self.at(a), self.at(out)]
        _ = self.xd(XD_COLSUM if columns else XD_ROWSUM, addresses, [rows, cols], List[Float64]())
        return out

    def colsum(mut self, a: Int) raises -> Int:
        return self._sum(a, True)

    def rowsum(mut self, a: Int) raises -> Int:
        return self._sum(a, False)

    def colmean(mut self, a: Int) raises -> Int:
        # `_Kit.colmean`: the column sums scaled by 1.0 / rows (binary64, rounded by the cell).
        var sums = self.colsum(a)
        return self.ew(DK_SCALE, sums, scalar=1.0 / Float64(self.matrices[a].rows))

    # ---- moves
    def dev_move(self, src: Int, idx: Int, dst: Int, params: List[Int]) raises:
        # XD_DEV_MOVE: [src, idx, dst] then the ten move words, as dev_move_py.
        var words: List[Int] = [src, idx, dst]
        words.extend(params.copy())
        _ = self.xd(XD_DEV_MOVE, List[Int](), words, List[Float64]())

    def host_move(mut self, a: Int, idx: Int, dst: Int, params: List[Int]) raises:
        # XD_MOVE: the host addresses of matrices a, idx and dst, then the ten
        # move words, as move_py.
        var addresses: List[Int] = [self.at(a), self.at(idx), self.at(dst)]
        _ = self.xd(XD_MOVE, addresses, params, List[Float64]())

    def _move(mut self, a: Int, idx: Int, rows: Int, cols: Int, params: List[Int], resident: Bool) raises -> Int:
        if rows * cols and resident:
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var di = self.device(idx)
            self.dev_move(da, di, self.matrices[out].device, params)
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            self.host_move(a, idx, out, params)
        return out

    def take_rows(mut self, a: Int, idx: Int, radd: Int = 0) raises -> Int:
        # `_Kit.take_rows(A, idx, radd=radd)`: rows idx[t] + radd of A.
        var m = self.size(idx)
        var cols = self.matrices[a].cols
        var count = m * cols
        var params: List[Int] = [DK_TAKE_ROWS, count, cols, radd, 0, 1, 0, self.size(a), count, m]
        return self._move(a, idx, m, cols, params, count > 0 and self.use(a, idx))

    def _moves_on_device(self, a: Int, count: Int) -> Bool:
        # `_res_moves`: a device matrix's move runs on the device.
        return (self.matrices[a].on_device and count > 0 and self.matrices[a].rows < DK_F32_INDEX_MAX
                and self.matrices[a].cols < DK_F32_INDEX_MAX)

    def rows_of(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        # `_M.rows(a, b)`: gathered on the device, a slice of the host store otherwise.
        var cols = self.matrices[a].cols
        var count = (stop - start) * cols
        if 0 <= start and start < stop and stop <= self.matrices[a].rows and self._moves_on_device(a, count):
            var idx = self.indices(start, stop)
            var w = stop - start
            return self._move(a, idx, w, cols, [DK_TAKE_ROWS, count, cols, 0, 0, 1, 0, self.size(a), count, w], True)
        var lo = max(0, min(start, self.matrices[a].rows))
        var hi = max(lo, min(stop, self.matrices[a].rows))
        var out = self.zeros(hi - lo, cols)
        var source = self.ptr(a)      # `self.s[...]`: the store is read (A is a host matrix after)
        if (hi - lo) * cols:
            unsafe_memcpy(dest=self.ptr(out), src=source + lo * cols, count=(hi - lo) * cols)
        return out

    def cols_of(mut self, a: Int, start: Int, stop: Int) raises -> Int:
        # `_M.cols(a, b)` (`take_cols(range(a, b))`).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var width = stop - start
        if start < 0 or stop > cols:
            raise Error("x_decomp: column range " + String(start) + ".." + String(stop) + " exceeds "
                        + String(cols) + " columns")
        if width > 0 and self._moves_on_device(a, rows * width):
            var idx = self.indices(start, stop)
            return self._move(a, idx, rows, width, [DK_TAKE_COLS, rows * width, width, cols, 0, 0, 0,
                              rows * cols, rows * width, width], True)
        var out = self.zeros(rows, max(width, 0))
        if rows * width > 0:
            var source = self.ptr(a)
            tsqr_copy_columns(source, self.ptr(out), rows, cols, start, width)
        return out

    def transpose(mut self, a: Int) raises -> Int:
        # `_M.T`.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var n = rows * cols
        var vector = rows == 1 or cols == 1
        if not self.matrices[a].on_device and n and not vector and n < DK_I32_MAX and self.resident:
            _ = self.device(a)
        var d = self.matrices[a].device
        if vector:
            # The same words, a view: of the device words for a device
            # matrix, else of the host store.
            if self.matrices[a].on_device:
                return self.add(self.matrices[a].store, cols, rows, d, self.matrices[a].borrowed)
            self.host(a)
            var view = self.add(self.matrices[a].store, cols, rows, d, self.matrices[a].borrowed)
            self.matrices[view].on_device = False
            return view
        if self.matrices[a].on_device and n < DK_I32_MAX:
            var out = self.device_output(cols, rows)
            self.dev_move(d, d, self.matrices[out].device, [DK_TRANSPOSE, n, rows, cols, 0, 0, 0, n, n, 1])
            return out
        var out = self.zeros(cols, rows)
        self.host_move(a, self.one_host, out, [DK_TRANSPOSE, n, rows, cols, 0, 0, 0, n, n, 1])
        return out

    def reshape(mut self, a: Int, rows: Int, cols: Int) raises -> Int:
        # `_M.reshape`: a view of the same words (of the host store for a host matrix).
        if self.matrices[a].on_device:
            return self.add(self.matrices[a].store, rows, cols, self.matrices[a].device, self.matrices[a].borrowed)
        self.host(a)
        var view = self.add(self.matrices[a].store, rows, cols, self.matrices[a].device, self.matrices[a].borrowed)
        self.matrices[view].on_device = False
        return view

    def hstack(mut self, ms: List[Int]) raises -> Int:
        # `_Kit.hstack`: PLACE_COLS moves, copies only.
        var rows = self.matrices[ms[0]].rows
        var width = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            width += self.matrices[ms[i]].cols
        var resident = self.resident and rows * width > 0
        var out = self.device_output(rows, width) if resident else self.zeros(rows, width)
        var offset = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            var m = ms[i]
            var count = self.size(m)
            if count:
                var params: List[Int] = [DK_PLACE_COLS, count, self.matrices[m].cols, width, offset, 0, 0, count,
                                         rows * width, 1]
                if resident:
                    var dm = self.device(m)
                    var one = self.dev_one()
                    self.dev_move(dm, one, self.matrices[out].device, params)
                else:
                    self.host_move(m, self.one_host, out, params)
            offset += self.matrices[m].cols
        return out

    def vstack(mut self, ms: List[Int]) raises -> Int:
        # `_Kit.vstack`: PLACE_COLS moves on the device, the stores joined on the host.
        var cols = self.matrices[ms[0]].cols
        var rows = 0
        var values = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            rows += self.matrices[ms[i]].rows
            values += self.size(ms[i])
        if self.resident and values:
            var out = self.device_output(rows, cols)
            var offset = 0
            for i in range(len(ms)):  # driver-loop: over the stacked matrices
                var m = ms[i]
                var count = self.size(m)
                if count:
                    var dm = self.device(m)
                    var one = self.dev_one()
                    self.dev_move(dm, one, self.matrices[out].device,
                                  [DK_PLACE_COLS, count, cols, cols, offset * cols, 0, 0, count, rows * cols, 1])
                offset += self.matrices[m].rows
            return out
        var out = self.zeros(rows, cols)
        var at = 0
        for i in range(len(ms)):  # driver-loop: over the stacked matrices
            var count = self.size(ms[i])
            if count:
                var source = self.ptr(ms[i])
                unsafe_memcpy(dest=self.ptr(out) + at, src=source, count=count)
            at += count
        return out

    def strided(mut self, a: Int, count: Int, stride: Int, off: Int = 0) raises -> Int:
        # `_Kit.strided`: A[off + t * stride] for t < count, 1 x count.
        var step = off + 1 if count == 1 else stride
        var idx = self.const(Float32(off))
        var n = self.size(a)
        return self._move(a, idx, 1, count, [DK_TAKE_COLS, count, 1, step, 0, 0, 0, n, count, 1],
                          count > 0 and self.use(a))

    def place_strided(mut self, d: Int, v: Int, stride: Int, off: Int = 0) raises -> Int:
        # `_Kit.place_strided`: D[off + t * stride] = V[t], in place.
        var count = self.size(v)
        if count == 0:
            return d
        var params: List[Int] = [DK_PLACE_COLS, count, 1, stride, off, 0, 0, count, self.size(d), 1]
        if self.use(d, v):
            var dv = self.device(v)
            var dd = self.device(d)
            var one = self.dev_one()
            self.dev_move(dv, one, dd, params)
            self.wrote_device(d)
            return d
        self.host_move(v, self.one_host, d, params)
        self.wrote_host(d)
        return d

    def copy(mut self, a: Int) raises -> Int:
        # `_Kit.copy`: a device copy of a device matrix, else `_M.copy()`.
        var n = self.size(a)
        if n and self.matrices[a].on_device and self.use(a):
            var rows = self.matrices[a].rows
            var cols = self.matrices[a].cols
            var out = self.device_output(rows, cols)
            var da = self.device(a)
            var one = self.dev_one()
            self.dev_move(da, one, self.matrices[out].device, [DK_PLACE_COLS, n, cols, cols, 0, 0, 0, n, n, 1])
            return out
        return self.copy_host(a)

    def diag_add(mut self, a: Int, v: Int) raises -> Int:
        # `_Kit.diag_add`: a copy of square A with v added to its diagonal.
        var rows = self.matrices[a].rows
        var copied = self.copy(a)
        var diagonal = self.strided(a, rows, rows + 1, 0)
        var added = self.ew(DK_ADD, diagonal, v)
        return self.place_strided(copied, added, rows + 1, 0)

    # ---- orders and signs
    def word(mut self, a: Int, i: Int = 0) raises -> Float64:
        # `_Kit.word`: one value down.
        var index = a
        var at = i
        if self.matrices[a].on_device and self.size(a) > 1:
            index = self.strided(a, 1, 1, i)
            at = 0
        return Float64(self.ptr(index)[at])

    def order(mut self, a: Int) raises -> Int:
        # `_Kit.order`: the stable ascending order, NaN refused.
        var n = self.size(a)
        if n and self.use(a):
            if n >= DK_F32_INDEX_MAX:
                raise Error("x_decomp: order_f exceeds the float32 index bound")
            var maximum = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.xd(XD_DEV_REDUCE, List[Int](), [da, self.matrices[maximum].device, n, 1], List[Float64]())
            var v = self.word(maximum)
            if v != v:
                raise Error("x_decomp: a NaN has no order (refused)")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            _ = self.xd(XD_DEV_ORDER, List[Int](), [da, da, self.matrices[out].device, self.matrices[count].device,
                        n, 0, 0], List[Float64]())
            return out
        var out = self.zeros(n, 1)
        if n:
            var addresses: List[Int] = [self.at(a), self.at(out)]
            _ = self.xd(XD_ORDER, addresses, [n], List[Float64]())
        return out

    def neg_signs(mut self, v: Int) raises -> Int:
        # `_Kit.neg_signs`: -1 where V < 0, else +1.
        var negated = self.ew(DK_SCALE, v, scalar=-1.0)
        var minus = self.const(-1.0)
        var plus = self.const(1.0)
        return self.ew(DK_SELECT, negated, minus, plus, 0.0)

    def absmax_signs(mut self, a: Int, by_col: Bool) raises -> Int:
        # `_Kit.absmax_signs`.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var count = cols if by_col else rows
        var flag = 1 if by_col else 0
        var out: Int
        if count and rows * cols and self.use(a):
            out = self.device_output(1, count)
            var da = self.device(a)
            _ = self.xd(XD_DEV_ABSMAX, List[Int](), [da, self.matrices[out].device, rows, cols, flag], List[Float64]())
        else:
            out = self.zeros(1, count)
            if count and rows * cols:
                var addresses: List[Int] = [self.at(a), self.at(out)]
                _ = self.xd(XD_ABSMAX_SIGN, addresses, [rows, cols, flag], List[Float64]())
        return self.neg_signs(out if by_col else self.reshape(out, count, 1))

    def rand(mut self, rows: Int, cols: Int, seed: Int, stream: Int, kind: Int) raises -> Int:
        # `_Kit.rand` (seed already masked to 32 bits).
        if rows * cols >= 1024 and self.resident:
            var out = self.device_output(rows, cols)
            _ = self.xd(XD_DEV_RAND, List[Int](), [self.matrices[out].device, rows * cols, seed, stream, kind],
                        List[Float64]())
            return out
        var out = self.zeros(rows, cols)
        if rows * cols:
            var addresses: List[Int] = [self.at(out)]
            _ = self.xd(XD_RAND, addresses, [rows * cols, seed, stream, kind], List[Float64]())
        return out

    # ---- small dense linear algebra
    def svd(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.svd`: (S 1 x n descending, Vt n x n) of a tall A.
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var s = self.zeros(1, n)
        var v = self.zeros(n, n)
        if m >= n and n >= 1 and self.opt_dev(DK_HAS_DEV_SVD) and self.use(a):
            var da = self.device(a)
            var addresses: List[Int] = [self.at(s), self.at(v)]
            _ = self.xd(XD_DEV_SVD, addresses, [da, m, n], List[Float64]())
        else:
            var addresses: List[Int] = [self.at(a), self.at(s), self.at(v)]
            _ = self.xd(XD_SVD, addresses, [m, n], List[Float64]())
        var negated = self.ew(DK_SCALE, s, scalar=-1.0)
        var o = self.order(negated)
        var st = self.transpose(s)
        var sorted = self.take_rows(st, o)
        var s_desc = self.transpose(sorted)
        var vt = self.transpose(v)
        return (s_desc, self.take_rows(vt, o))

    def thin_svd(mut self, x: Int, nc: Int, u_based: Bool) raises -> Tuple[Int, Int, Int]:
        # `_thin_svd`: (U n x nc, S 1 x nc, Vt nc x d) with svd_flip.
        var n = self.matrices[x].rows
        var d = self.matrices[x].cols
        var u: Int
        var s: Int
        var vt: Int
        if d <= n:
            var sv = self.svd(x)
            s = self.cols_of(sv[0], 0, nc)
            vt = self.rows_of(sv[1], 0, nc)
            var xv = self.mm(x, vt, tb=True)
            u = self.ew(DK_DIV, xv, s)
        else:
            var xt = self.transpose(x)
            var su = self.svd(xt)
            s = self.cols_of(su[0], 0, nc)
            var ut = self.rows_of(su[1], 0, nc)
            u = self.transpose(ut)
            var ux = self.mm(u, x, ta=True)
            var st = self.transpose(s)
            vt = self.ew(DK_DIV, ux, st)
        if u_based:
            var sg = self.absmax_signs(u, True)
            var sgt = self.transpose(sg)
            var u2 = self.ew(DK_MUL, u, sg)
            var vt2 = self.ew(DK_MUL, vt, sgt)
            return (u2, s, vt2)
        var sg = self.absmax_signs(vt, False)
        var sgt = self.transpose(sg)
        var u2 = self.ew(DK_MUL, u, sgt)
        var vt2 = self.ew(DK_MUL, vt, sg)
        return (u2, s, vt2)

    def lu(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        # `_Kit.lu`: (the factor, its int32 pivot store (`istores` index), info).
        var n = self.matrices[a].rows
        var piv = self.istore(n)
        var info = self.zeros(1, 1)
        var pa = self.iaddr(piv)
        var lu: Int
        if n >= 1 and self.matrices[a].cols == n and self.opt_dev(DK_HAS_DEV_LU) and self.use(a):
            lu = self.device_output(n, n)
            var da = self.device(a)
            var addresses: List[Int] = [pa, self.at(info)]
            _ = self.xd(XD_DEV_LU, addresses, [da, self.matrices[lu].device, n], List[Float64]())
        else:
            lu = self.copy_host(a)
            var addresses: List[Int] = [self.at(lu), pa, self.at(info)]
            _ = self.xd(XD_LU, addresses, [n], List[Float64]())
            self.wrote_host(lu)
        return (lu, piv, Int(self.word(info)))

    def lu_resid(mut self, a: Int, x: Int, b: Int) raises -> Int:
        # `_Kit.lu_resid`: B - A X, folded in float-float.
        var out = self.zeros(self.matrices[b].rows, self.matrices[b].cols)
        var addresses: List[Int] = [self.at(a), self.at(x), self.at(b), self.at(out)]
        _ = self.xd(XD_LU_RESID, addresses, [self.matrices[a].rows, self.matrices[b].cols], List[Float64]())
        return out

    def lu_solve(mut self, lu: Int, piv: Int, b: Int) raises -> Int:
        # `_Kit.lu_solve(lu, piv, B)` (trans 0) with the kit's pivot store.
        return self.lu_solve_addr(lu, self.iaddr(piv), b, 0)

    def lu_solve_addr(mut self, lu: Int, piv_address: Int, b: Int, trans: Int) raises -> Int:
        # `_Kit.lu_solve(lu, piv, B, trans)`: on a host copy of B; the int32
        # pivots at `piv_address` (a kit store, or a caller Array the door
        # keeps alive); trans 0 is lu_solve_py's default word.
        var out = self.copy_host(b)
        var addresses: List[Int] = [self.at(lu), piv_address, self.at(out)]
        _ = self.xd(XD_LU_SOLVE, addresses, [self.matrices[lu].rows, self.matrices[b].cols, trans], List[Float64]())
        self.wrote_host(out)
        return out

    # ---- sparse coding rows
    def lasso_rows(mut self, g: Int, q: Int, w: Int, alpha: Float64, max_iter: Int, tol: Float64,
                   positive: Int) raises -> Int:
        # `_Kit.lasso_rows`: W the warm start, updated in place.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev(DK_HAS_DEV_CODE_ROWS) and self.use(g, q, w):
            var dg = self.device(g)
            var dq = self.device(q)
            var dw = self.device(w)
            _ = self.xd(XD_DEV_CODE_ROWS, List[Int](), [dg, dq, dw, 0, rows, cols, max_iter, positive], [alpha, tol])
            self.wrote_device(w)
            return w
        var its = self.zeros(rows, 1)
        var addresses: List[Int] = [self.at(g), self.at(q), self.at(w), self.at(its)]
        _ = self.xd(XD_LASSO_ROWS, addresses, [rows, cols, max_iter, positive], [alpha, tol])
        self.wrote_host(w)
        return w

    def lars_rows(mut self, g: Int, q: Int, m: Int, nnz: Int) raises -> Int:
        # `_Kit.lars_rows`.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev(DK_HAS_DEV_CODE_ROWS) and self.use(g, q):
            var w = self.device_output(rows, cols)
            var dg = self.device(g)
            var dq = self.device(q)
            _ = self.xd(XD_DEV_CODE_ROWS, List[Int](), [dg, dq, self.matrices[w].device, 1, rows, cols, m, nnz],
                        [Float64(0.0), Float64(0.0)])
            return w
        var w = self.zeros(rows, cols)
        var na = self.zeros(rows, 1)
        if rows * cols:
            var addresses: List[Int] = [self.at(g), self.at(q), self.at(w), self.at(na)]
            _ = self.xd(XD_LARS_ROWS, addresses, [rows, cols, m, nnz], List[Float64]())
        return w

    def omp_rows(mut self, g: Int, q: Int, nnz: Int) raises -> Int:
        # `_Kit.omp_rows`.
        var rows = self.matrices[q].rows
        var cols = self.matrices[q].cols
        if rows * cols and self.opt_dev(DK_HAS_DEV_CODE_ROWS) and self.use(g, q):
            var w = self.device_output(rows, cols)
            var dg = self.device(g)
            var dq = self.device(q)
            _ = self.xd(XD_DEV_CODE_ROWS, List[Int](), [dg, dq, self.matrices[w].device, 2, rows, cols, nnz, 0],
                        [Float64(0.0), Float64(0.0)])
            return w
        var w = self.zeros(rows, cols)
        var na = self.zeros(rows, 1)
        var addresses: List[Int] = [self.at(g), self.at(q), self.at(w), self.at(na)]
        _ = self.xd(XD_OMP_ROWS, addresses, [rows, cols, nnz], List[Float64]())
        return w

    # ---- reductions, selections and factor companions (MinCovDet)
    def reduce(mut self, a: Int, op: Int) raises -> Int:
        # `_Kit.reduce`: max |A| (0), max A (1) or min A (2) as 1 x 1.
        var n = self.size(a)
        if n == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        if self.use(a):
            var out = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.xd(XD_DEV_REDUCE, List[Int](), [da, self.matrices[out].device, n, op], List[Float64]())
            return out
        var out = self.zeros(1, 1)
        var addresses: List[Int] = [self.at(a), self.at(out)]
        _ = self.xd(XD_REDUCE, addresses, [n, op], List[Float64]())
        return out

    def total_word(mut self, a: Int) raises -> Float64:
        # `k.total(A).s[0]`: the sum of every entry, rows ascending, one word down.
        var rows = self.rowsum(a)
        var total = self.colsum(rows)
        return self.word(total)

    def eigh(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.eigh(A)` (uplo 0): ascending values (1 x n), vectors in columns.
        var n = self.matrices[a].rows
        if (n >= 1 and self.matrices[a].cols == n and self.matrices[a].on_device
                and self.opt_dev(DK_HAS_DEV_EIGH_TO)):
            # w and V stay on the device (`x_decomp_dev_eigh_to`, one status wait).
            var wd = self.device_output(1, n)
            var vd = self.device_output(n, n)
            _ = self.xd(XD_DEV_EIGH_TO, List[Int](), [self.matrices[a].device, self.matrices[wd].device,
                        self.matrices[vd].device, n, 0], List[Float64]())
            return (wd, vd)
        var w = self.zeros(1, n)
        var v = self.zeros(n, n)
        var addresses: List[Int] = [self.at(a), self.at(w), self.at(v)]
        _ = self.xd(XD_EIGH, addresses, [n, 0], List[Float64]())
        return (w, v)

    def refuse_nan(mut self, a: Int) raises:
        # `_Kit._refuse_nan`: a device operand's max is NaN when any value is.
        var maximum = self.reduce(a, 1)
        var v = self.word(maximum)
        if v != v:
            raise Error("x_decomp: a NaN has no order (refused)")

    def argmin_all(mut self, a: Int) raises -> Int:
        # `_Kit.argmin_all`: every position (exact floats, c x 1) of the minimum.
        var n = self.size(a)
        if n and self.use(a):
            if n >= DK_F32_INDEX_MAX:
                raise Error("x_decomp: argmin_all exceeds the float32 index bound")
            var out = self.device_output(n, 1)
            var count = self.device_output(1, 1)
            var da = self.device(a)
            _ = self.xd(XD_DEV_ARGMIN_ALL, List[Int](), [da, self.matrices[out].device, self.matrices[count].device, n],
                        List[Float64]())
            var c = Int(self.word(count))
            return self.rows_of(out, 0, c) if c else self.zeros(0, 1)
        var buffer = self.zeros(max(n, 1), 1)
        var c = 0
        if n:
            var addresses: List[Int] = [self.at(a), self.at(buffer)]
            c = self.xd(XD_ARGMIN_ALL, addresses, [n], List[Float64]())
        return self.rows_of(buffer, 0, c)

    def select_smallest(mut self, a: Int, h: Int) raises -> Tuple[Int, Int, Int]:
        # `_Kit.select_smallest`: (the h smallest positions ascending, h x 1;
        # the membership as an n x 1 0/1 device matrix, or -1; the host int32
        # membership store (`istores` index), or -1).
        var n = self.size(a)
        if n and self.use(a):
            if n >= DK_F32_INDEX_MAX or h < 0 or h > n:
                raise Error("x_decomp: select_smallest out of range")
            self.refuse_nan(a)
            var sel = self.device_output(h, 1)
            var mask = self.device_output(n, 1)
            var da = self.device(a)
            _ = self.xd(XD_DEV_SELECT_SMALLEST, List[Int](), [da, self.matrices[sel].device,
                        self.matrices[mask].device, n, h], List[Float64]())
            return (sel, mask, Int(-1))
        var sel = self.zeros(h, 1)
        var store = self.istore(n)
        if n:
            var addresses: List[Int] = [self.at(a), self.at(sel), self.iaddr(store)]
            _ = self.xd(XD_SELECT_SMALLEST, addresses, [n, h], List[Float64]())
        return (sel, Int(-1), store)

    def lu_aux(mut self, lu: Int, piv: Int) raises -> Tuple[Int, Int]:
        # `_Kit.lu_aux(lu, piv)` (clamp False): (stats 1 x 4 = [max |u_ii|,
        # zero pivots, negative pivots, swaps], the diagonal 1 x n), host.
        var n = self.matrices[lu].rows
        var pm = self.zeros(n, 1)
        var im = self.zeros(n, 1)
        var diag = self.zeros(1, n)
        var st = self.zeros(1, 4)
        var pa = self.iaddr(piv)
        if n >= 1 and self.matrices[lu].on_device and self.opt_dev(DK_HAS_DEV_LU):
            var dl = self.matrices[lu].device
            var addresses: List[Int] = [pa, self.at(pm), self.at(im), self.at(diag), self.at(st)]
            _ = self.xd(XD_DEV_LU_AUX, addresses, [dl, n, 0], List[Float64]())
            return (st, diag)
        var addresses: List[Int] = [self.at(lu), pa, self.at(pm), self.at(im), self.at(diag), self.at(st)]
        _ = self.xd(XD_LU_AUX, addresses, [n, 0], List[Float64]())
        return (st, diag)

    # ---- top-N orders (AlternatingLeastSquares)
    def order_dev(mut self, a: Int, neg: Int, skip: Int) raises -> Tuple[Int, Int]:
        # `_Kit._order_dev(A, neg, skip)` on a resident kit: (n x 1 order, 1 x 1 count).
        var n = self.size(a)
        if n >= DK_F32_INDEX_MAX:
            raise Error("x_decomp: order_f exceeds the float32 index bound")
        self.refuse_nan(a)
        var out = self.device_output(n, 1)
        var count = self.device_output(1, 1)
        var sid = self.device(skip) if skip >= 0 else self.device(a)
        var da = self.device(a)
        _ = self.xd(XD_DEV_ORDER, List[Int](), [da, sid, self.matrices[out].device, self.matrices[count].device, n,
                    neg, 1 if skip >= 0 else 0], List[Float64]())
        return (out, count)

    def take_cols_m(mut self, a: Int, idx: Int, w: Int) raises -> Int:
        # `_Kit.take_cols_m`: columns idx[0 .. w) of A.
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var params: List[Int] = [DK_TAKE_COLS, rows * w, w, cols, 0, 0, 0, rows * cols, rows * w, w]
        return self._move(a, idx, rows, w, params, rows * w > 0 and self.use(a, idx))

    # TOMBSTONE: DecompKit.orth, order_small and grown (used only by the per-step randomized_svd, now x_decomp/randomized_svd_owner.mojo) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
    # Restore: experiments/removed/A3_ADMISSION_HELPERS.patch.
    def count_gt(mut self, a: Int, s: Float64) raises -> Int:
        # `_Kit.count_gt`: how many values of A are > s.
        if not self.size(a):
            return 0
        var flags = self.ew(DK_GTS, a, scalar=s)
        return Int(self.total_word(flags))

    # ---- the tall SVD of `_linalg_impl._svd_tall` (lstsq)
    def orth_diag(mut self, a: Int) raises -> Tuple[Int, Int]:
        # `_Kit.orth_diag`: (orth(A), the passes' R-diagonal products 1 x c).
        var rows = self.matrices[a].rows
        var cols = self.matrices[a].cols
        var diag = self.zeros(1, cols)
        if rows * cols and self.use(a):
            var q = self.device_output(rows, cols)
            var da = self.device(a)
            var addresses: List[Int] = [self.at(diag)]
            _ = self.xd(XD_DEV_ORTH_DIAG, addresses, [da, self.matrices[q].device, rows, cols], List[Float64]())
            return (q, diag)
        var q = self.copy_host(a)
        var addresses: List[Int] = [self.at(q), self.at(diag)]
        _ = self.xd(XD_ORTH_DIAG, addresses, [rows, cols], List[Float64]())
        self.wrote_host(q)
        return (q, diag)

    def svd_tall(mut self, a: Int) raises -> Tuple[Int, Int, Int]:
        # `_linalg_impl._svd_tall(k, A, full=False)`: (U m x n, S 1 x n, Vt n x n).
        var m = self.matrices[a].rows
        var n = self.matrices[a].cols
        var sv = self.svd(a)
        var s = sv[0]
        var vt = sv[1]
        var r = 0
        if n:
            _ = self.word(s, 0)      # `S.s[0]`: S comes to the host, as before
            var addresses: List[Int] = [self.at(s)]
            r = self.xd(XD_RANK, addresses, [n], [Float64(0.00000095367431640625)])
        var av = self.mm(a, vt, tb=True)
        var left = self.cols_of(av, 0, r) if r < n else av
        var right = self.cols_of(s, 0, r) if r < n else s
        var ug = self.ew(DK_DIV, left, right)
        if r == n:
            var qd = self.orth_diag(ug)
            var absolute = self.ew(DK_ABS, qd[1])
            var zero = self.zeros(1, 1)
            var guarded = self.ew(DK_LE, absolute, zero)
            if self.total_word(guarded) == 0:
                var signs = self.neg_signs(qd[1])
                return (self.ew(DK_MUL, qd[0], signs), s, vt)
        if r == 0:
            var eye = self.zeros(m, n)
            tsqr_fill_eye(self.ptr(eye), m, n)
            return (eye, s, vt)
        # geqrf + orgqr of A V / s, Q's columns signed by R's diagonal.
        var h = self.copy_host(ug)
        var tau = self.zeros(1, min(self.matrices[ug].rows, self.matrices[ug].cols))
        var hq: List[Int] = [self.at(h), self.at(tau)]
        _ = self.xd(XD_GEQRF, hq, [self.matrices[ug].rows, self.matrices[ug].cols], List[Float64]())
        self.wrote_host(h)
        var qc = self.zeros(self.matrices[h].rows, n)
        var hq2: List[Int] = [self.at(h), self.at(tau), self.at(qc)]
        _ = self.xd(XD_ORGQR, hq2, [self.matrices[h].rows, self.matrices[h].cols, self.matrices[tau].cols, n],
                    List[Float64]())
        self.wrote_host(qc)
        var diagonal = self.strided(h, r, r + 1, 0)
        var sg = self.neg_signs(diagonal)
        if n > r:
            var ones = self.const(1.0, 1, n - r)
            sg = self.hstack([sg, ones])
        return (self.ew(DK_MUL, qc, sg), s, vt)
