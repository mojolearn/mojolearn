# SPDX-License-Identifier: Apache-2.0
# UNPACK and PACK of the typed manifold kit (lane entangled-mkit, 2026-11-12).
# The kit itself is RUN, with no Python object (bindings/mani_kit_run.mojo
# `MRun`). This file is its boundary:
#   UNPACK  mk_image   an exact x_decomp binding module -> an MImage (its
#                      dispatch as an EntryRef, its key, its capability word,
#                      the cached device zero), connected once per binding
#           mk_fitted  a fitted `_ManifoldMatrix` owner -> an MFitted
#   PACK    mk_out / mk_out2   a matrix's words -> a public float32 Array
#           mk_owner   a matrix -> a new `_ManifoldMatrix` fitted owner
#           mk_keep    the copies a RUN made of fitted matrices -> their owners
#           mk_finish  cache the device zero, return every other pool id
#
# TOMBSTONE: the Python-holding MKit (struct MKit: b/raw/ctx/table/handles/
# optional/one_host/one_dev PythonObject fields; `fn(name)` + `family_call`
# per primitive; mk_* helpers over PythonObject matrices) deleted 2026-11-12
# by lane entangled-mkit; code recoverable at the lane's base e561a66d6b.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_address
from bindings.family_common import family_array
from bindings.kit_call import kit_connect, kit_capabilities, kit_entry_ref, kit_provider, kit_call
from bindings.manifold_storage_native import ManifoldMatrix
from bindings.mani_kit_run import MRun, MImage, MFitted, MExport, MK_F32P
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES, XD_VENDOR

comptime XD_VENDOR_METAL = 1  # x_decomp/descriptor_schema.mojo xd_vendor_code("metal")


def mk_cache_home(ctx: PythonObject) raises -> PythonObject:
    """Where the kit's cross-call device zero lives (row N290): the long-lived
    shared connection context bindings/manifold_native.mojo hands down as
    `family_mani_shared`; a caller without it keeps its own."""
    var home = py_get(ctx, "family_mani_shared")
    return ctx if home is None else home


def _mk_dev_one_cache(ctx: PythonObject) raises -> PythonObject:
    var home = mk_cache_home(ctx)
    var cache = py_get(home, "family_mani_native_dev_one")
    if cache is None:
        cache = Python.dict()
        py_setitem(home, "family_mani_native_dev_one", cache)
    return cache


def mk_image(mut run: MRun, mode: String, binding: PythonObject, ctx: PythonObject) raises -> Int:
    """UNPACK `_Kit(mode, binding)`: the exact native x_decomp module's
    dispatch, capability word and cached device zero as an image of `run`."""
    # Proxy resolution is import-time work. A Python __getattr__ or object
    # descriptor must never select a numerical entry at runtime.
    if py_type(binding) is not py_type(native_loaded_module("builtins")):
        raise Error("TypeError: manifold kit requires an exact native module")
    var snapshot = kit_connect(binding, "x_decomp_native_api_v1", String(XD_ABI_NAME),
                               XD_OPERATION_COUNT, XD_MODE, XD_CAPABILITIES)
    var entry = kit_entry_ref(snapshot, XD_OPERATION_COUNT)
    var capabilities = kit_capabilities(snapshot)
    # `enable_lle_precision`: the FAST image on Metal only (the vendor word
    # read once here; was the `x_decomp_vendor` entry's string)
    var lle_ordered = False
    if mode == "fast":
        lle_ordered = kit_call(kit_provider(snapshot), XD_VENDOR, List[Int](), List[Int](), List[Float64](),
                               False) == XD_VENDOR_METAL
    _ = snapshot  # keep the snapshot alive through the reads above
    var key = Int(binding._obj_ptr)
    var dev_one = -1
    var cached = py_get(_mk_dev_one_cache(ctx), PythonObject(key))
    if cached is not None:
        dev_one = Int(py=cached)
    return run.image(MImage(entry, key, capabilities, lle_ordered, dev_one))


def mk_fitted(mut run: MRun, owner: PythonObject, name: String) raises -> Int:
    """UNPACK a fitted `_ManifoldMatrix` owner (the door keeps `owner` alive
    through the RUN): its words' address and pool id, never moved."""
    if owner is None:
        raise Error("RuntimeError: fitted manifold matrix " + name + " is missing")
    var p = owner.downcast_value_ptr[ManifoldMatrix]()
    return run.fit(MFitted(p[].rows, p[].cols, p[].words_address(), p[].device, p[].device_key,
                           p[].device_entry, p[].on_device))


def mk_keep(mut run: MRun, a: Int, owner: PythonObject) raises:
    """PACK the copies the RUN made of fitted matrix `a` for the side it is
    not on: the owner keeps them for every later call. A copy another thread
    kept first wins; ours goes back to the pool at mk_finish."""
    var fi = run.mats[a].fitted
    if fi < 0:
        return
    var p = owner.downcast_value_ptr[ManifoldMatrix]()
    if not p[].has_words:
        var words = run.kept_host(fi)
        if len(words) > 0:
            p[].words = words^
            p[].has_words = True
    if p[].device < 0 and run.fitted[fi].new_device >= 0:
        var k = run.fitted[fi].new_image
        var handle = run.kept_device(fi)
        if handle >= 0:
            p[].device = handle
            p[].device_key = run.images[k].key
            p[].device_entry = run.images[k].entry


def mk_owner(mut run: MRun, a: Int) raises -> PythonObject:
    """PACK matrix a as a new fitted `_ManifoldMatrix` owner, on the side the
    fit left it."""
    var ex = run.export(a)
    var owner = ManifoldMatrix(ex.rows, ex.cols)
    if ex.device >= 0:
        owner.device = ex.device
        owner.device_key = run.images[ex.image].key
        owner.device_entry = run.images[ex.image].entry
        owner.on_device = True
    else:
        owner.words = ex.words^
        owner.has_words = True
    return PythonObject(alloc=owner^)


def mk_out(mut run: MRun, a: Int, shape: PythonObject, ctx: PythonObject) raises -> PythonObject:
    """PACK `m.out(shape)`: a public float32 Array (C order) holding a copy; a
    device matrix is downloaded once (it moves), the GIL released."""
    var size = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        size *= Int(py=shape[k])
    if size != run.n(a):
        raise Error("RuntimeError: an array store does not match its shape")
    var out = family_array(shape, "<f4", ctx)
    if size > 0:
        var src = 0
        with GILReleased(Python()):
            src = run.addr(a)
        unsafe_memcpy(dest=MK_F32P(unsafe_from_address=Int(py=classical_address(out))),
                      src=MK_F32P(unsafe_from_address=src), count=size)
    _ = out  # keep the Python object alive through the copy above
    return out


def mk_out2(mut run: MRun, a: Int, ctx: PythonObject) raises -> PythonObject:
    """PACK `m.out()`: shape (r, c)."""
    return mk_out(run, a, Python.tuple(run.r(a), run.c(a)), ctx)


def mk_finish(mut run: MRun, ctx: PythonObject) raises:
    """PACK the end of an operation: a device zero this RUN made is cached
    for the next operation on its binding (one per binding, row N290); every
    pool id the RUN did not hand to a fitted owner goes back to the pool."""
    for k in range(len(run.images)):  # driver-loop: over this operation's images (at most 2)
        if run.images[k].dev_one_made:
            var cache = _mk_dev_one_cache(ctx)
            var key = PythonObject(run.images[k].key)
            if py_get(cache, key) is None:
                py_setitem(cache, key, PythonObject(run.images[k].dev_one))
                run.images[k].dev_one_made = False
    run.close()


def mk_seed_word(seed: PythonObject) raises -> Int:
    """UNPACK `int(seed) & 0xFFFFFFFF`, the Philox seed word."""
    var b = native_loaded_module("builtins")
    return Int(py=py_call(native_loaded_module("operator").and_, py_call(b.int, seed), PythonObject(0xFFFFFFFF)))
