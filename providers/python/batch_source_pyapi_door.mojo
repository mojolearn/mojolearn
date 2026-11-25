# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `mojolearn.ops.BatchSource` and of `Program.fit(source=)` (lane batch-source, 2026-11-25).

UNPACK reads the handle once into a typed `BatchSourceSpec` (autograd/batch_source.mojo): the exact connected
`BatchSource` class only (a Python iterable, generator, data loader or a subclass is refused before anything is
called: no call back into Python during a fit), its `_spec` tuple and its `_position` pair. RUN trains on the
handle's opened source (autograd/program_entries.mojo `program_fit_source`). PACK writes the consumed position
back to the handle and builds the returned dicts.

THE OWNER (lane fr2-followups; autograd/batch_source.mojo THE OWNER). The opened source (index, record table, epoch
plan, shard descriptors, host buffers, read-ahead batch) lives on a native `BatchSourceOwner` held in the handle's
`_owner`, built by the first fit and reused by every later one while it stands at the handle's `_position`. A
position written from outside a fit (`load_state_dict`, `Session.load_training`) or a reopen drops it; an owner of
another binding image (another mode or executor) is replaced. The owner is a cache of what `_spec` and
`_position` determine, so a handle rebuilt in a fresh process from the same directory and a saved state is still
the same source. Releasing the handle closes the descriptors (the owner's destructor, no Python `__del__`).

`_spec` = (path, batch_size, input field codes, target code, shuffle, seed, bucket_batches, fingerprint, sealed);
`_position` = (epoch, consumed batches of that epoch).

THE RESUME DOORS (lane fable-finalize, 2026-11-25): `Program.save_training(source=)`, `Session.load_training(source=)`
and `load_state_dict` bind a position to the source's DATA (the fingerprint, which holds the shard index's content
digests: autograd/batch_source.mojo THE SEAL), its PLAN settings and its ordered input and target FIELDS, through
the one comparison `program_training_source_refuse`. A source whose index is not sealed is refused by name at
each of them (`_need_sealed`): its fingerprint is record lengths only, which other data can share.
`BatchSource.seal(path)` ("seal") writes the digests, natively, once.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_setitem, py_setattr, py_item
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_json_door import native_path
from providers.python.batch_source_class_door import (
    batch_source_class_connect, batch_source_class_ready, batch_source_is_handle,
)
from providers.python.autograd_program_arrays_pyapi_door import is_integer_input, uint64_input
from autograd.batch_source import (
    BatchSource, BatchSourceSpec, BATCH_SOURCE_FORMAT, BS_FIELD_TARGETS, batch_source_open, batch_source_field,
    batch_source_field_name, batch_source_seal, ProgramTrainingSource, program_training_source_refuse,
    batch_source_part_of,
)
from autograd.session_lifetime import SESSION_MODE


# DEVICE and MODE are part of the type on purpose, as SessionLifetime's (autograd/session_lifetime.mojo): the
# Python type registry is one per process and `import mojolearn` loads the device and host x_sequence images in
# both numeric modes; each registers its own owner type (bindings/_mojolearn_x_sequence(_host).mojo add_type).
struct BatchSourceOwner[DEVICE: Bool, MODE: Int = SESSION_MODE](Movable, Writable):
    """A BatchSource handle's opened source, kept across fit calls (module note THE OWNER)."""

    var src: BatchSource

    def __init__(out self, var src: BatchSource):
        self.src = src^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_BatchSourceOwner(records=", self.src.n, ", epoch=", self.src.epoch, ", consumed=",
                     self.src.consumed, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_BatchSourceOwner(records=", self.src.n, ", epoch=", self.src.epoch, ", consumed=",
                     self.src.consumed, ")")

    def __deinit__(deinit self):
        self.src.release_descriptors()


comptime BatchSourcePtr = UnsafePointer[BatchSource, MutUntrackedOrigin]


def batch_source_held[DEVICE: Bool](source: PythonObject, var spec: BatchSourceSpec) raises -> BatchSourcePtr:
    """RUN's opened source for the handle `source` whose words are `spec` (module note THE OWNER): the handle's
    owner when it is this image's and stands at the handle's position, else a source opened once from `spec` and
    kept on the handle. The source lives in the owner the handle holds; the caller keeps `source` alive while it
    uses the pointer."""
    var held = source._owner
    if held is not None:
        try:
            var owner = held.downcast_value_ptr[BatchSourceOwner[DEVICE]]()
            if owner[].src.epoch == spec.epoch and owner[].src.consumed == spec.consumed:
                var address = Int(UnsafePointer(to=owner[].src))
                _ = held
                return BatchSourcePtr(unsafe_from_address=address)
        except:
            pass  # another binding image's owner (another mode or executor): replaced below
    var made = PythonObject(alloc=BatchSourceOwner[DEVICE](batch_source_open(spec^)))
    py_setattr(source, "_owner", made)
    var owner = made.downcast_value_ptr[BatchSourceOwner[DEVICE]]()
    var address = Int(UnsafePointer(to=owner[].src))
    _ = made
    return BatchSourcePtr(unsafe_from_address=address)


def _drop_owner(source: PythonObject) raises:
    """Forget the handle's opened source (its descriptors close when the owner is released)."""
    py_setattr(source, "_owner", PythonObject(None))




def _int(value: PythonObject, what: String, low: Int) raises -> Int:
    if not is_integer_input(value):
        raise Error("TypeError: BatchSource: " + what + " must be an integer")
    var v = Int(py=value)
    if v < low:
        raise Error("ValueError: BatchSource: " + what + " must be >= " + String(low))
    return v


def _flag(value: PythonObject, what: String) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.bool:
        raise Error("TypeError: BatchSource: " + what + " must be True or False")
    return Bool(py=value)


def _word_out(v: UInt64) raises -> PythonObject:
    return PythonObject(Int(v >> 32)).__lshift__(32).__or__(Int(v & UInt64(0xFFFFFFFF)))


def batch_source_admit(source: PythonObject) raises:
    """Refuse anything but the exact connected BatchSource (module note)."""
    if not batch_source_class_ready():
        raise Error("TypeError: BatchSource is not connected (mojolearn.ops connects it at import)")
    if not batch_source_is_handle(source):
        raise Error("TypeError: Program.fit(source=) takes a mojolearn.ops.BatchSource; Python iterables, "
                    + "generators, data loaders and batch callbacks are not accepted (a fit never calls back into "
                    + "Python). Write the data as batch shards and open them with BatchSource(path, ...)")


def batch_source_spec(source: PythonObject) raises -> BatchSourceSpec:
    """UNPACK: the handle's typed words."""
    batch_source_admit(source)
    var spec = BatchSourceSpec()
    var words = source._spec
    var at = source._position
    spec.path = String(py=words[0])
    spec.batch = Int(py=words[1])
    var codes = words[2]
    for k in range(len(codes)):  # small-loop(codes: the source's input fields)
        spec.inputs.append(Int(py=codes[k]))
    spec.target = Int(py=words[3])
    spec.shuffle = Bool(py=words[4])
    spec.seed = uint64_input(words[5], "seed")
    spec.bucket = Int(py=words[6])
    spec.fingerprint = uint64_input(words[7], "fingerprint")
    spec.sealed = Bool(py=words[8])
    spec.epoch = Int(py=at[0])
    spec.consumed = Int(py=at[1])
    return spec^


def batch_source_position_out(source: PythonObject, epoch: Int, consumed: Int) raises:
    """PACK of a fit: the consumed position back on the handle (its owner stands there and is kept)."""
    py_setattr(source, "_position", Python.tuple(PythonObject(epoch), PythonObject(consumed)))


def batch_source_reposition(source: PythonObject, epoch: Int, consumed: Int) raises:
    """A position set from outside a fit (a loaded state or checkpoint): written to the handle and its owner
    dropped, so the next fit rebuilds the opened source once at that position."""
    _drop_owner(source)
    py_setattr(source, "_position", Python.tuple(PythonObject(epoch), PythonObject(consumed)))


def _need_sealed(spec: BatchSourceSpec, who: String) raises:
    """Refuse a resume door on a source whose shard index records no content digest (module note THE RESUME
    DOORS)."""
    if not spec.sealed:
        raise Error("ValueError: " + who + ": the shard index under " + spec.path + " records no content digest, "
                    + "so a resume could not tell these shards from other data of the same shape; call "
                    + "BatchSource.seal(path) once after writing the shards, then open the source again")


def _part_of(spec: BatchSourceSpec) -> ProgramTrainingSource:
    """The words a resume binds: position, data fingerprint, plan settings and the ordered field mapping."""
    return batch_source_part_of(spec)


def batch_source_training_part(source: PythonObject) raises -> ProgramTrainingSource:
    """The source part of a training checkpoint (autograd/program_training_checkpoint.mojo), from the handle. A
    source that is not sealed is refused here, for the save and for the load alike."""
    var out = ProgramTrainingSource()
    if source is None:
        return out^
    var spec = batch_source_spec(source)
    _need_sealed(spec, "a training checkpoint with source=")
    return _part_of(spec)


def batch_source_restore(source: PythonObject, part: ProgramTrainingSource, who: String) raises:
    """A saved position onto the handle (`load_state_dict`), after checking it is the same sealed data, the same
    plan settings and the same input and target fields (`program_training_source_refuse`)."""
    if source is None:
        return
    if not part.present:
        raise Error("ValueError: " + who + ": the state was saved without a batch source")
    var spec = batch_source_spec(source)
    _need_sealed(spec, who)
    program_training_source_refuse(_part_of(spec), part, who)
    batch_source_reposition(source, part.epoch, part.consumed)


def _state_dict(source: PythonObject) raises -> PythonObject:
    var spec = batch_source_spec(source)
    var out = native_loaded_module("builtins").dict()
    py_setitem(out, PythonObject("format"), PythonObject(BATCH_SOURCE_FORMAT))
    py_setitem(out, PythonObject("epoch"), PythonObject(spec.epoch))
    py_setitem(out, PythonObject("consumed"), PythonObject(spec.consumed))
    py_setitem(out, PythonObject("fingerprint"), _word_out(spec.fingerprint))
    py_setitem(out, PythonObject("seed"), _word_out(spec.seed))
    py_setitem(out, PythonObject("batch_size"), PythonObject(spec.batch))
    py_setitem(out, PythonObject("bucket_batches"), PythonObject(spec.bucket))
    py_setitem(out, PythonObject("shuffle"), PythonObject(spec.shuffle))
    var names = Python.list()
    for k in range(len(spec.inputs)):  # small-loop(inputs: the source's input fields)
        _ = py_call(names.append, PythonObject(batch_source_field_name(spec.inputs[k])))
    py_setitem(out, PythonObject("inputs"), py_call(native_loaded_module("builtins").tuple, names))
    if spec.target != 0:
        py_setitem(out, PythonObject("target"), PythonObject(batch_source_field_name(spec.target)))
    else:
        py_setitem(out, PythonObject("target"), PythonObject(None))
    py_setitem(out, PythonObject("sealed"), PythonObject(spec.sealed))
    return out


def _load_state_dict(source: PythonObject, state: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(state) is not b.dict or len(state) != 11:
        raise Error("TypeError: BatchSource.load_state_dict takes the dict state_dict returns")
    var spec = batch_source_spec(source)
    var fmt = py_item(state, PythonObject("format"))
    if py_type(fmt) is not b.str or String(py=fmt) != BATCH_SOURCE_FORMAT:
        raise Error("ValueError: BatchSource.load_state_dict: the state's format words are not '"
                    + BATCH_SOURCE_FORMAT + "'")
    var part = ProgramTrainingSource()
    part.present = True
    part.epoch = _int(py_item(state, PythonObject("epoch")), "epoch", 0)
    part.consumed = _int(py_item(state, PythonObject("consumed")), "consumed", 0)
    part.fingerprint = uint64_input(py_item(state, PythonObject("fingerprint")), "fingerprint")
    part.seed = uint64_input(py_item(state, PythonObject("seed")), "seed")
    part.batch = _int(py_item(state, PythonObject("batch_size")), "batch_size", 1)
    part.bucket = _int(py_item(state, PythonObject("bucket_batches")), "bucket_batches", 1)
    part.shuffle = _flag(py_item(state, PythonObject("shuffle")), "shuffle")
    var names = py_item(state, PythonObject("inputs"))
    if py_type(names) is not b.tuple or len(names) < 1 or len(names) > 64:
        raise Error("TypeError: BatchSource.load_state_dict: the state's inputs must be a tuple of field names")
    for k in range(len(names)):  # small-loop(names: the saved input fields)
        if py_type(names[k]) is not b.str:
            raise Error("TypeError: BatchSource.load_state_dict: a field name must be a str")
        part.inputs.append(batch_source_field(String(py=names[k])))
    var saved_target = py_item(state, PythonObject("target"))
    if saved_target is not None:
        if py_type(saved_target) is not b.str or batch_source_field(String(py=saved_target)) != BS_FIELD_TARGETS:
            raise Error("ValueError: BatchSource.load_state_dict: the state's target must be 'targets' or None")
        part.target = BS_FIELD_TARGETS
    if not _flag(py_item(state, PythonObject("sealed")), "sealed"):
        raise Error("ValueError: BatchSource.load_state_dict: the state was taken on shards whose index records no "
                    + "content digest; a position is bound to sealed shards (BatchSource.seal(path))")
    if part.consumed > Int(py=source.n_batches):
        raise Error("ValueError: BatchSource.load_state_dict: consumed is past the epoch's batches")
    batch_source_restore(source, part, "BatchSource.load_state_dict")
    _ = spec^


def _open(source: PythonObject, args: PythonObject) raises:
    """BatchSource(path, batch_size, inputs, target, shuffle, seed, bucket_batches): admit, open once (every
    header and offset table checked, the fingerprint taken), close, and write the handle's words."""
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple or len(args) != 7:
        raise Error("TypeError: BatchSource takes path, batch_size, inputs, target, shuffle, seed, bucket_batches")
    batch_source_admit(source)
    var spec = BatchSourceSpec()
    spec.path = String(py=native_path(args[0]))
    spec.batch = _int(args[1], "batch_size", 1)
    var names = args[2]
    if py_type(names) is not b.tuple and py_type(names) is not b.list:
        raise Error("TypeError: BatchSource: inputs must be a tuple of field names")
    if len(names) < 1:
        raise Error("ValueError: BatchSource: inputs names at least one field")
    var codes = Python.list()
    for k in range(len(names)):  # small-loop(names: the requested fields)
        if py_type(names[k]) is not b.str:
            raise Error("TypeError: BatchSource: a field name must be a str")
        var code = batch_source_field(String(py=names[k]))
        spec.inputs.append(code)
        _ = py_call(codes.append, PythonObject(code))
    if args[3] is not None:
        if py_type(args[3]) is not b.str:
            raise Error("TypeError: BatchSource: target must be 'targets' or None")
        if batch_source_field(String(py=args[3])) != BS_FIELD_TARGETS:
            raise Error("ValueError: BatchSource: target must be 'targets' or None")
        spec.target = BS_FIELD_TARGETS
    spec.shuffle = _flag(args[4], "shuffle")
    spec.seed = uint64_input(args[5], "seed")
    spec.bucket = _int(args[6], "bucket_batches", 1)
    var path = spec.path
    var batch = spec.batch
    var target = spec.target
    var shuffle = spec.shuffle
    var seed = spec.seed
    var bucket = spec.bucket
    var src = batch_source_open(spec^)
    src.close()
    _drop_owner(source)  # a reopened handle: an owner of its former words is not its source
    py_setattr(source, "_spec", Python.tuple(PythonObject(path), PythonObject(batch), py_call(b.tuple, codes),
                                            PythonObject(target), PythonObject(shuffle), _word_out(seed),
                                            PythonObject(bucket), _word_out(src.fingerprint),
                                            PythonObject(src.sealed)))
    py_setattr(source, "_position", Python.tuple(PythonObject(0), PythonObject(0)))
    py_setattr(source, "n_records", PythonObject(src.n))
    py_setattr(source, "n_batches", PythonObject(src.n_batches))
    py_setattr(source, "n_features", PythonObject(src.width))
    py_setattr(source, "max_length", PythonObject(src.max_len))
    py_setattr(source, "target_width", PythonObject(src.twidth))  # lane stream-fits: 0 without targets
    py_setattr(source, "sealed", PythonObject(src.sealed))


def _next_records(source: PythonObject) raises -> PythonObject:
    """The record ids (in row order) of the batch the next fit trains first."""
    var src = batch_source_open(batch_source_spec(source))
    var at = src.pending()
    var ids = src.records(at[0], at[1])
    src.close()
    var out = Python.list()
    for k in range(len(ids)):  # small-loop(ids: one batch's record ids)
        _ = py_call(out.append, PythonObject(ids[k]))
    return out


def batch_source_call_py(op: PythonObject, source: PythonObject, args: PythonObject) raises -> PythonObject:
    """The one binding entry of BatchSource: "connect" (import time: the exact class), "open", "state_dict",
    "load_state_dict", "next_records", "seal" (args (path,); `source` is the class, not a handle)."""
    var b = native_loaded_module("builtins")
    if py_type(op) is not b.str:
        raise Error("TypeError: BatchSource operation must be a str")
    var name = String(py=op)
    if name == "connect":
        batch_source_class_connect(source)
        return PythonObject(None)
    if name == "open":
        _open(source, args)
        return PythonObject(None)
    if name == "state_dict":
        return _state_dict(source)
    if name == "load_state_dict":
        _load_state_dict(source, args[0])
        return PythonObject(None)
    if name == "next_records":
        return _next_records(source)
    if name == "seal":
        if py_type(args) is not b.tuple or len(args) != 1:
            raise Error("TypeError: BatchSource.seal takes the shard directory")
        return PythonObject(batch_source_seal(String(py=native_path(args[0]))))
    raise Error("ValueError: unknown BatchSource operation " + name)
