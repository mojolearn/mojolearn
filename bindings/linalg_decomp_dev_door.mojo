# SPDX-License-Identifier: Apache-2.0
"""`solve`, `lstsq` and `randomized_svd` of TrainDeviceArray operands (lane vdf-linalg7, 2026-11-24), and the PACK
that `sparse_encode`'s device form shares (bindings/dict_learning_native_unpack.mojo).

A device array (bindings/linalg_dev_arrays_unpack.mojo: the training ops' one device-array convention) is read IN
PLACE: UNPACK admits it (its handle, shape and storage dtype; every matrix operand of the call a device array of the
same training binding, or the call refused in words), RUN adopts it into the x_decomp pool device to device and tests
it finite there (`DKit.adopt`: XD_DEV_FROM_TRAIN, XD_DEV_FINITE, refused in the host admission's words), then runs the
host operand's sequence on the same words, with the steps that took host addresses on pool ids:

- solve: gesv (the identical route, XD_DEV_LU_GESV) or the resident LU pair (XD_DEV_LU_FACTOR_IN /
  XD_DEV_LU_SOLVE_IN), else the kit's own factor and solve (`DKit.lu`, `ld_lu_solve_run`: the host route of an image
  with neither, whose solve is a host step: LU and B come down there for a device operand as for a host one).
- lstsq: the TSQR of [a | b] on pool ids (XD_DEV_TSQR_R), R kept on the device; or the kit's thin SVD.
- randomized_svd: the one driver (XD_RANDOMIZED_SVD) with M and the factors on pool ids.
- lu_factor (lane vdf-tail-b): the resident pair route (LU_FAST_RESIDENT / IDN_LU_RESIDENT_PAIR) copies A device to
  device into the pair's own pool id, tests it finite there and factors it there (XD_DEV_LU_FACTOR_IN); LU (float32)
  and the pivots (int32) are new device arrays, and the returned pair keeps the pool ids of A, LU and piv (`_dev`, as
  the host operand's pair does) for lu_solve. Else the kit's own factor on the adopted A (its host step), as for a
  host operand. Only info (one word) comes down on the pair route.
- lu_solve (lane vdf-tail-b): on lu_factor's pair (identity, trans 0) XD_DEV_LU_SOLVE_IN reads LU, piv, A and the
  adopted B in place and writes X on the device; else the kit's own solve (`ld_lu_solve_run`, a host step: the
  pivots come down, n words, and LU and B as for a host operand), LU_QFIX's refinement on the pair's device A copy.

PACK writes every array result into a new TrainDeviceArray (one training pool handle each, XD_DEV_TO_TRAIN); scalars
(lstsq's rank) are Python ints, as for host operands. Same route, same words: the bits are the host operand's.
device="cpu" with a device array is refused (it lives on the GPU).
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_item, py_type, py_setattr, py_contains
from core.function_device import FunctionDevice
from core.train_dev_pool import TRAIN_DEV_F32, TRAIN_DEV_I32
from bindings.linalg_dev_arrays_unpack import (
    LinDevOperand, lin_dev_device, lin_dev_entries, lin_dev_operand_of, lin_dev_new, lin_dev_handle,
    lin_is_device_array,
)
from bindings.decomp_kit_native_door import decomp_kit_unpack, decomp_float, decomp_int, decomp_seed, decomp_seed_word
from bindings.classical_arrays_door import classical_typed
from bindings.linalg_keys_unpack import linalg_keys
from bindings.decomp_kit_run import DKit, DK_HAS_S_FLAGS, DK_HAS_QFIX_FLAGS, DK_HAS_IDN_FLAGS
from bindings.linalg_decomp_run import (
    LstsqOut, ld_rsvd_run, ld_dev_gesv_run, ld_dev_lu_pair_run, ld_lu_solve_run, ld_lstsq_run,
    ld_dev_lu_factor_train_run, ld_dev_lu_solve_pair_run, ld_pool_to_train, ld_train_pivots_down, ld_pivots_to_train,
)
from bindings.linalg_decomp_native_unpack import _singular_warning, ld_entries, _decomp_module
from core.decomp_native_impl import decomp_piv_valid


comptime LD_F32_EPS = 1.1920928955078125e-07


def decomp_dev_out(mut kit: DKit, a: Int, dims: List[Int], tentries: PythonObject, held: PythonObject) raises -> PythonObject:
    """PACK: a new float32 TrainDeviceArray of `dims` holding kit matrix a's words (one device copy; a host matrix
    goes up once), allocated through the training owner the operands belong to (`lin_dev_new`)."""
    var out = lin_dev_new(tentries, held, dims, TRAIN_DEV_F32)
    var handle = lin_dev_handle(out)
    var py = Python()
    with GILReleased(py):
        kit.to_train(a, handle)
    return out


def decomp_dev_matrix(value: PythonObject, name: String, function: String, tentries: PythonObject,
                      held: PythonObject) raises -> LinDevOperand:
    """UNPACK a 2-D float32 device-array operand of `mojolearn.<function>` (refused in words otherwise)."""
    return lin_dev_operand_of(value, name, "mojolearn.", function, tentries, held, TRAIN_DEV_F32, 2)


def _dev_rhs(value: PythonObject, function: String, tentries: PythonObject, held: PythonObject) raises -> LinDevOperand:
    """UNPACK b: a 1-D or 2-D float32 device array (a vector is the n x 1 column of the same words)."""
    var b = lin_dev_operand_of(value, "b", "mojolearn.", function, tentries, held, TRAIN_DEV_F32, 0)
    if len(b.shape) != 1 and len(b.shape) != 2:
        raise Error("mojolearn." + function + ": b must be 1-D or 2-D, got " + String(len(b.shape)) + "-D")
    return b^


def decomp_dev_owner(args: PythonObject, count: Int, held: PythonObject) raises -> PythonObject:
    """The training owner entries of the first device array among `args[0:count]` (a host operand beside it is then
    refused in words by `lin_dev_operand_of`, never read as a device array)."""
    for i in range(count):  # small-loop(operands): the call's matrix operands (at most three)
        if lin_is_device_array(args[i], held):
            return lin_dev_entries(args[i], held)
    raise Error("TypeError: no device-array operand")


def _cols(b: LinDevOperand) -> Int:
    return 1 if len(b.shape) == 1 else b.dim(1)


def _x_dims(b: LinDevOperand, n: Int) -> List[Int]:
    if len(b.shape) == 1:
        return List[Int](length=1, fill=n)
    var dims: List[Int] = [n, b.dim(1)]
    return dims^


def solve_dev_call(args: PythonObject, device: FunctionDevice, held: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`solve(a, b)` of device arrays: X a new device array (b's rank)."""
    lin_dev_device(device, "mojolearn.solve")
    var tentries = decomp_dev_owner(args, 2, held)
    var a = decomp_dev_matrix(args[0], "a", "solve", tentries, held)
    var b = _dev_rhs(args[1], "solve", tentries, held)
    var n = a.dim(0)
    if n != a.dim(1):
        raise Error("expected a square matrix, got " + String(n) + " x " + String(a.dim(1)))
    if b.dim(0) != n:
        raise Error("b has " + String(b.dim(0)) + " rows, the factorization has " + String(n))
    var nrhs = _cols(b)
    var entries = ld_entries(device, args[2], context)
    var kit = decomp_kit_unpack(entries)
    try:
        var x = -1
        var singular = 0
        var py = Python()
        var idn = kit.flags(DK_HAS_IDN_FLAGS)
        var sf = kit.flags(DK_HAS_S_FLAGS)
        var qf = kit.flags(DK_HAS_QFIX_FLAGS)
        var stacklevel = 2
        with GILReleased(py):
            var am = kit.adopt(a.handle, n, n, "a")
            var bm = kit.adopt(b.handle, n, nrhs, "b")
            if (idn & 2) != 0:
                # gesv (the host route's first branch): one operation, the factor and pivots resident
                var solved = ld_dev_gesv_run(kit, am, bm)
                x = solved[0]
                singular = solved[1]
            elif ((sf & 4) != 0 and (qf & 4) != 0) or (sf & 8) != 0:
                # `_lu_factor_resident` + `_lu_solve_resident` (LU_FAST_RESIDENT refines, IDN_LU_RESIDENT_PAIR not)
                var pair = ld_dev_lu_pair_run(kit, am, bm, (sf & 4) != 0)
                x = pair[0]
                singular = pair[1]
                stacklevel = 3
                if x < 0:
                    # a singular factor: the host route warns, then solves on the downloaded factor (no refinement)
                    var lu = kit.add(-1, n, n, pair[2])
                    kit.matrices[lu].on_device = False
                    var piv = kit.istore(n)
                    kit.dev_download(pair[3], kit.iaddr(piv), n)
                    x = ld_lu_solve_run(kit, lu, kit.iaddr(piv), bm, 0, -1)
            else:
                # the kit's factor and solve, LU_QFIX's refinement on the kit's A when the factor is regular
                var factor = kit.lu(am)
                singular = factor[2]
                var refine = singular == 0 and (qf & 4) != 0
                x = ld_lu_solve_run(kit, factor[0], kit.iaddr(factor[1]), bm, 0, am if refine else -1)
        if singular != 0:
            _singular_warning(singular, stacklevel)
        return decomp_dev_out(kit, x, _x_dims(b, n), tentries, held)
    finally:
        kit.close()


def lstsq_dev_call(args: PythonObject, device: FunctionDevice, held: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`lstsq(a, b, rcond)` of device arrays: (solution, residues, rank, singular values), the arrays new device
    arrays (an empty residues vector is a host Array of shape (0,), as for host operands), the rank an int."""
    lin_dev_device(device, "mojolearn.lstsq")
    var tentries = decomp_dev_owner(args, 2, held)
    var a = decomp_dev_matrix(args[0], "a", "lstsq", tentries, held)
    var b = _dev_rhs(args[1], "lstsq", tentries, held)
    var m = a.dim(0)
    var nn = a.dim(1)
    if b.dim(0) != m:
        raise Error("Incompatible dimensions")
    var nrhs = _cols(b)
    var rcond = LD_F32_EPS * Float64(max(m, nn)) if args[2] is None else decomp_float(args[2])
    var entries = ld_entries(device, args[3], context)
    var kit = decomp_kit_unpack(entries)
    try:
        var out = LstsqOut(-1, -1, 0, -1)
        var py = Python()
        with GILReleased(py):
            var am = kit.adopt(a.handle, m, nn, "a")
            var bm = kit.adopt(b.handle, m, nrhs, "b")
            out = ld_lstsq_run(kit, am, bm, rcond)
        var resid = PythonObject(None)
        if out.resid >= 0:
            resid = decomp_dev_out(kit, out.resid, List[Int](length=1, fill=nrhs), tentries, held)
        else:
            # numpy's empty residues: no device array has zero elements, so the (0,) result stays a host Array
            var nk = linalg_keys()
            resid = classical_typed(Python.tuple(0), "<f4", py_item(context, nk[].k_output_context))
        var solution = decomp_dev_out(kit, out.x, _x_dims(b, nn), tentries, held)
        var values = decomp_dev_out(kit, out.s, List[Int](length=1, fill=kit.ncols(out.s)), tentries, held)
        return Python.tuple(solution, resid, out.rank, values)
    finally:
        kit.close()


def rsvd_params(args: PythonObject, rows: Int, cols: Int, devscan: Bool) raises -> List[Int]:
    """UNPACK randomized_svd's options (args: M, n_components, n_oversamples, n_iter, power_iteration_normalizer,
    transpose, flip_sign, random_state, ...) into XD_RANDOMIZED_SVD's fourteen words, refused in words:
    the ten option words, then the four pool ids of M, U, S and Vt, -1 (a device operand's call sets them). One body
    for host and device operands (bindings/linalg_decomp_native_pack.mojo `randomized_svd_call`)."""
    var b = native_loaded_module("builtins")
    var transpose_arg = args[5]
    var auto_transpose = Bool(py=py_call(b.isinstance, transpose_arg, b.str)) and String(py=transpose_arg) == "auto"
    var normalizer = args[4]
    var known = False
    if Bool(py=py_call(b.isinstance, normalizer, b.str)):
        var name = String(py=normalizer)
        known = name == "auto" or name == "QR" or name == "LU" or name == "none"
    if not known:
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var kc = decomp_int(args[1])
    var oversamples = decomp_int(args[2])
    var n_iter = args[3]
    var auto_iterations = Bool(py=py_call(b.isinstance, n_iter, b.str)) and String(py=n_iter) == "auto"
    var iterations = 0 if auto_iterations else decomp_int(n_iter)
    var transpose = (rows < cols) if auto_transpose else Bool(py=transpose_arg)
    if not (1 <= kc and kc <= min(rows, cols)):
        raise Error("n_components must be in [1, min(n_samples, n_features)]")
    var seed = decomp_seed_word(decomp_seed(args[7]))
    var params: List[Int] = [rows, cols, kc, oversamples, iterations, Int(auto_iterations), Int(transpose),
                             Int(Bool(py=args[6])), seed, Int(devscan), -1, -1, -1, -1]
    return params^


def randomized_svd_dev_call(args: PythonObject, device: FunctionDevice, held: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`randomized_svd(M, ...)` of a device array: (U, S, Vt) new device arrays."""
    lin_dev_device(device, "mojolearn.randomized_svd")
    var tentries = lin_dev_entries(args[0], held)
    var source = decomp_dev_matrix(args[0], "M", "randomized_svd", tentries, held)
    var rows = source.dim(0)
    var cols = source.dim(1)
    # M is tested finite on the device at its adoption: no deferred test (the last word 0)
    var params = rsvd_params(args, rows, cols, False)
    var kc = params[2]
    var entries = ld_entries(device, args[8], context)
    var kit = decomp_kit_unpack(entries)
    try:
        var u = -1
        var s = -1
        var vt = -1
        var py = Python()
        with GILReleased(py):
            var x = kit.adopt(source.handle, rows, cols, "M")
            u = kit.device_output(rows, kc)
            s = kit.device_output(1, kc)
            vt = kit.device_output(kc, cols)
            # the pool ids of M, U, S and Vt replace the four addresses (unread)
            params[10] = kit.device(x)
            params[11] = kit.matrices[u].device
            params[12] = kit.matrices[s].device
            params[13] = kit.matrices[vt].device
            ld_rsvd_run(kit, [0, 0, 0, 0], params)
        return Python.tuple(decomp_dev_out(kit, u, [rows, kc], tentries, held),
                            decomp_dev_out(kit, s, List[Int](length=1, fill=kc), tentries, held),
                            decomp_dev_out(kit, vt, [kc, cols], tentries, held))
    finally:
        kit.close()


# ---------------------------------------------------------------- lu_factor / lu_solve (lane vdf-tail-b)
# The pair's holders and its unpacking, shared with the host operand's door (bindings/linalg_decomp_native_pack.mojo
# `lu_factor_call`, `lu_solve_call`; moved here word for word).


def _device_buffer(kit: DKit, module: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    # `_DevBuf(raw, count)`: a pooled device buffer id, returned to the pool
    # when the holder dies; the holder is assembled from its slots. The id
    # comes from the image's table (XD_DEV_ALLOC; was a Python call of the
    # module's x_decomp_dev_alloc): the same pool the holder frees into.
    var bi = native_loaded_module("builtins")
    var holder = py_call(bi.object.__new__, py_item(context, nk[].k_lu_device_buffer))
    py_setattr(holder, "b", module)
    py_setattr(holder, "n", PythonObject(count))
    py_setattr(holder, "id", PythonObject(kit.dev_alloc(max(count, 1))))
    return holder


def _lu_pair(lu: PythonObject, piv: PythonObject, qfix_a: PythonObject, dev: PythonObject,
             context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    # `_LUPair((lu, piv), qfix_a, dev=dev)`, assembled from its slots.
    var bi = native_loaded_module("builtins")
    var pair = py_call(bi.tuple.__new__, py_item(context, nk[].k_lu_pair_class), Python.tuple(lu, piv))
    py_setattr(pair, "_qfix_a", qfix_a)
    py_setattr(pair, "_dev", dev)
    return pair




def lu_pair_items(pair: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`lu, piv = lu_and_piv`, with Python's unpacking refusals."""
    var bi = native_loaded_module("builtins")
    var iterable = (Bool(py=py_call(bi.hasattr, pair, PythonObject("__iter__")))
                    or Bool(py=py_call(bi.hasattr, pair, PythonObject("__getitem__"))))
    if not iterable:
        raise Error("TypeError: cannot unpack non-iterable " + String(py=py_type(pair).__name__) + " object")
    var items = py_call(bi.tuple, pair)
    if len(items) < 2:
        raise Error("not enough values to unpack (expected 2, got " + String(len(items)) + ")")
    if len(items) > 2:
        raise Error("too many values to unpack (expected 2)")
    # locals first: no PythonObject subscript in return position (AGENTS.md part 6)
    var lu = items[0]
    var piv = items[1]
    return (lu^, piv^)


def lu_pair_on_device(pair: PythonObject, held: PythonObject, context: PythonObject) raises -> Bool:
    """Whether lu_solve's (lu, piv) holds a device array as lu: read only for an exact tuple or lu_factor's pair class
    of two items (a C-level item read; any other object goes to the host door, which unpacks it)."""
    var kind = py_type(pair)
    if kind is not native_loaded_module("builtins").tuple and kind is not py_item(context, linalg_keys()[].k_lu_pair_class):
        return False
    if len(pair) != 2:
        return False
    var first = pair[0]
    var device = lin_is_device_array(first, held)
    return device


def lu_factor_dev_call(args: PythonObject, device: FunctionDevice, held: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`lu_factor(a)` of a device array: (lu, piv) new device arrays (float32 n x n, int32 n), in the pair the host
    operand's resident route returns when the factor is regular."""
    lin_dev_device(device, "mojolearn.lu_factor")
    var tentries = lin_dev_entries(args[0], held)
    var a = decomp_dev_matrix(args[0], "a", "lu_factor", tentries, held)
    var n = a.dim(0)
    if n != a.dim(1):
        raise Error("expected a square matrix, got " + String(n) + " x " + String(a.dim(1)))
    var entries = ld_entries(device, args[1], context)
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        var sf = kit.flags(DK_HAS_S_FLAGS)
        var qf = kit.flags(DK_HAS_QFIX_FLAGS)
        var lu_out = lin_dev_new(tentries, held, [n, n], TRAIN_DEV_F32)
        var piv_out = lin_dev_new(tentries, held, List[Int](length=1, fill=n), TRAIN_DEV_I32)
        var lh = lin_dev_handle(lu_out)
        var ph = lin_dev_handle(piv_out)
        if ((sf & 4) != 0 and (qf & 4) != 0) or (sf & 8) != 0:
            # the resident pair: A, LU and piv in pool ids the pair's holders own (they return to the pool with it)
            var module = _decomp_module(args[1], context)
            var da = _device_buffer(kit, module, n * n, context)
            var dlu = _device_buffer(kit, module, n * n, context)
            var dp = _device_buffer(kit, module, n, context)
            var ids: List[Int] = [Int(py=da.id), Int(py=dlu.id), Int(py=dp.id)]
            var singular = 0
            with GILReleased(py):
                singular = ld_dev_lu_factor_train_run(kit, a.handle, ids[0], ids[1], ids[2], n)
                ld_pool_to_train(kit, ids[1], lh, n * n)
                ld_pool_to_train(kit, ids[2], ph, n)
            if singular != 0:
                # the device buffers return to the pool with their holders
                _singular_warning(singular, 2)
                return Python.tuple(lu_out, piv_out)
            return _lu_pair(lu_out, piv_out, PythonObject(None), Python.tuple(da, dlu, dp, lu_out, piv_out), context)
        # the kit's own factor (an image without the resident pair): its host step on the adopted A
        var info = 0
        var qfix = -1
        with GILReleased(py):
            var am = kit.adopt(a.handle, n, n, "a")
            var factor = kit.lu(am)
            info = factor[2]
            kit.to_train(factor[0], lh)
            ld_pivots_to_train(kit, factor[1], ph, n)
            if info == 0 and (qf & 4) != 0:
                qfix = am
        if info != 0:
            _singular_warning(info, 2)
        if qfix >= 0:
            # LU_QFIX: the pair carries this call's own float32 copy of A (a new device array) so lu_solve can refine
            return _lu_pair(lu_out, piv_out, decomp_dev_out(kit, qfix, [n, n], tentries, held), PythonObject(None),
                            context)
        return Python.tuple(lu_out, piv_out)
    finally:
        kit.close()


def lu_solve_dev_call(args: PythonObject, device: FunctionDevice, held: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`lu_solve((lu, piv), b, trans)` of device arrays: X a new device array (b's rank)."""
    lin_dev_device(device, "mojolearn.lu_solve")
    var bi = native_loaded_module("builtins")
    var trans = args[2]
    if not py_contains(Python.tuple(0, 1, 2), trans):
        raise Error("trans must be 0, 1 or 2")
    var pair = args[0]
    var items = lu_pair_items(pair)
    var lu = items[0]
    var piv = items[1]
    var operands = Python.tuple(lu, piv, args[1])
    var tentries = decomp_dev_owner(operands, 3, held)
    var l = decomp_dev_matrix(lu, "lu", "lu_solve", tentries, held)
    var p = lin_dev_operand_of(piv, "piv", "mojolearn.", "lu_solve", tentries, held, TRAIN_DEV_I32, 1)
    var b = _dev_rhs(args[1], "lu_solve", tentries, held)
    var n = l.dim(0)
    if p.dim(0) != n:
        raise Error("piv must hold one row index in [0, n) per row")
    if b.dim(0) != n:
        raise Error("b has " + String(b.dim(0)) + " rows, the factorization has " + String(n))
    var nrhs = _cols(b)
    var transposed = Bool(py=py_call(bi.bool, trans))
    var entries = ld_entries(device, args[3], context)
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        var sf = kit.flags(DK_HAS_S_FLAGS)
        var qf = kit.flags(DK_HAS_QFIX_FLAGS)
        var dev = py_call(bi.getattr, pair, PythonObject("_dev"), PythonObject(None))
        var x = -1
        if dev is not None and not transposed and lu is dev[3] and piv is dev[4] and (sf & 12) != 0:
            # lu_factor's pair: LU, piv and A where the factor left them (the pair's holders own them; `dev` is
            # named below past the solve)
            var ids: List[Int] = [Int(py=dev[0].id), Int(py=dev[1].id), Int(py=dev[2].id)]
            with GILReleased(py):
                var bm = kit.adopt(b.handle, n, nrhs, "b")
                x = ld_dev_lu_solve_pair_run(kit, ids[0], ids[1], ids[2], bm, (sf & 4) != 0)
            _ = dev  # keep the pair's device holders alive through the solve above
            return decomp_dev_out(kit, x, _x_dims(b, n), tentries, held)
        # the kit's own solve (the host operand's route for any other pair): the pivots come down (n words)
        var a0 = py_call(bi.getattr, pair, PythonObject("_qfix_a"), PythonObject(None))
        var refine = (a0 is not None and not transposed and lin_is_device_array(a0, held)
                      and (qf & 4) != 0)
        var qa = LinDevOperand(-1, List[Int](), TRAIN_DEV_F32)
        if refine:
            qa = decomp_dev_matrix(a0, "a", "lu_solve", tentries, held)
            refine = qa.dim(0) == n and qa.dim(1) == n
        var bad = False
        with GILReleased(py):
            var lm = kit.adopt(l.handle, n, l.dim(1), "lu")
            var store = ld_train_pivots_down(kit, p.handle, n)
            if not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.iaddr(store)), n):
                bad = True
            else:
                var bm = kit.adopt(b.handle, n, nrhs, "b")
                var am = -1
                if refine:
                    am = kit.adopt(qa.handle, n, n, "a")
                x = ld_lu_solve_run(kit, lm, kit.iaddr(store), bm, 1 if transposed else 0, am)
        if bad:
            raise Error("piv must hold one row index in [0, n) per row")
        return decomp_dev_out(kit, x, _x_dims(b, n), tentries, held)
    finally:
        kit.close()
