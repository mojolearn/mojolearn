# SPDX-License-Identifier: Apache-2.0
# Public Transformer decode-session admission/ownership. The Python class is a
# thin connection; GPU execution uses the common native provider and CPU copies
# stay native. No Python session methods or block hooks are invoked here.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from std.ffi import _Global
from std.memory import memcpy
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_setitem, py_type, py_contains
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.causal_lm_construct import _model_shape
from bindings.neural_block_public import _neural_input
from bindings.neural_block_admission import neural_record_fields, neural_block_fields, neural_block_snapshot, neural_array_snapshot, neural_metadata_snapshot, neural_string_keys
from bindings.transformer_public_native import _transformer_fields, _transformer_state_words, _transformer_configuration, _transformer_same_words
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.generation_metadata_pyapi import GenerationLayerSnapshot
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_module_fields
from providers.python.transformer_resident_pyapi import transformer_resident_decode_handle, transformer_resident_decode_call_handle
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME
from core.hostptr import f32_ptr
from transformer.block_options import BlockOptions
from bindings.cpython_table import cpython_table
from bindings.native_model_records import records_stampable, model_generation, ContentRecord, content_value, content_same
from std.ffi import _Global


struct TransformerDecodeHandle(Movable, Writable):
    var binding: PythonObject
    var block: PythonObject
    var state: PythonObject
    var configuration: List[Int]
    var state_shape: List[Int]
    var provider: PythonObject
    var device: PythonObject
    var open: Bool
    var weights: List[List[Float32]]
    var optional: List[List[Float32]]
    var k_cache: List[Float32]
    var v_cache: List[Float32]
    # N186: the provider's native handle of `device`, resolved once at open
    # (0 when no provider route). `device` retains its owner.
    var native_handle: Int
    # N181: the compiled entries a step, state transfer or close calls,
    # resolved once at open by _TRANSFORMER_DECODE_* index (None when the
    # binding lacks one; the use raises the same refusal as before).
    var entries: List[PythonObject]
    # Row N177: the words a step needs, checked once at a full admission and
    # kept while the block's and the state's generation words are unchanged
    # (records module, dictionary watcher). 0 = no proof: full admission.
    var block_generation: Int
    var state_generation: Int
    var width: Int
    var batch: Int
    var params: List[Int]
    var live_state: PythonObject
    # The block's option tail is a list the dictionary watcher cannot see
    # edited in place: its item words are checked natively per step.
    var tail_content: ContentRecord

    def __init__(out self, binding: PythonObject, block: PythonObject, state: PythonObject):
        self.binding = binding
        self.block = block
        self.state = state
        self.configuration = List[Int]()
        self.state_shape = List[Int]()
        self.provider = PythonObject(None)
        self.device = PythonObject(None)
        self.open = False
        self.weights = List[List[Float32]]()
        self.optional = List[List[Float32]]()
        self.k_cache = List[Float32]()
        self.v_cache = List[Float32]()
        self.native_handle = 0
        self.entries = List[PythonObject]()
        self.block_generation = 0
        self.state_generation = 0
        self.width = 0
        self.batch = 0
        self.params = List[Int]()
        self.live_state = PythonObject(None)
        self.tail_content = ContentRecord()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerDecodeHandle(open=", self.open, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _transformer_decode_module(module: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(module) is bi.str:
        raise Error("TransformerDecodeSession: " + String(module))
    if py_type(module) is not context["admission_context"]["native_module_type"]:
        raise Error("TypeError: Transformer decode requires its connected native module")
    # N192: the module's live namespace with its key walk done once per module
    # identity (the shared neural_descriptor admission), instead of a copy of
    # the whole dictionary and a walk of every key three times per session
    # open. Readers only look entries up; nothing writes this namespace.
    return neural_descriptor_module_fields(module)


def _transformer_decode_entry(entries: PythonObject, name: String) raises -> PythonObject:
    var entry = py_get(entries, name)
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("NotImplementedError: Transformer decode lacks its connected compiled " + name + " entry")
    return entry


struct TransformerDecodeWords(Defaultable, Movable):
    # Compiled vendor and numeric-mode read-backs per connected entry pair
    # (N192). A compiled binding's vendor and mode are fixed when it is built;
    # the entries are retained, so an identity compare is never fooled by a
    # reused address, and a replaced entry reads its words again.
    var vendor_entries: List[PythonObject]
    var mode_entries: List[PythonObject]
    var vendors: List[String]
    var modes: List[Int]

    def __init__(out self):
        self.vendor_entries = List[PythonObject]()
        self.mode_entries = List[PythonObject]()
        self.vendors = List[String]()
        self.modes = List[Int]()


# Bump the name's version whenever TransformerDecodeWords' layout changes.
comptime _TRANSFORMER_DECODE_WORDS = _Global[StorageType=TransformerDecodeWords,
    name="MojolearnTransformerDecodeWordsV1", init_fn=TransformerDecodeWords.__init__]


def _transformer_decode_words(entries: PythonObject, prefix: String, mut vendor: String, mut mode: Int) raises:
    var vendor_entry = _transformer_decode_entry(entries, prefix + "_vendor")
    var mode_entry = _transformer_decode_entry(entries, prefix + "_numeric_mode")
    var g = _TRANSFORMER_DECODE_WORDS.get_or_create_ptr()
    for i in range(len(g[].vendor_entries)):  # small-loop(connected decode bindings): a few compiled modules per process, never data
        if g[].vendor_entries[i] is vendor_entry and g[].mode_entries[i] is mode_entry:
            vendor = g[].vendors[i]
            mode = g[].modes[i]
            return
    vendor = String(vendor_entry())
    mode = Int(py=mode_entry())
    g = _TRANSFORMER_DECODE_WORDS.get_or_create_ptr()
    if len(g[].vendor_entries) >= 64:
        g[].vendor_entries.clear()
        g[].mode_entries.clear()
        g[].vendors.clear()
        g[].modes.clear()
    g[].vendor_entries.append(vendor_entry)
    g[].mode_entries.append(mode_entry)
    g[].vendors.append(vendor)
    g[].modes.append(mode)


comptime _TRANSFORMER_DECODE_STEP = 0
comptime _TRANSFORMER_DECODE_FORWARD = 1
comptime _TRANSFORMER_DECODE_LOAD = 2
comptime _TRANSFORMER_DECODE_EXPORT = 3
comptime _TRANSFORMER_DECODE_CLOSE = 4
comptime _TRANSFORMER_DECODE_HOST_STEP = 5
comptime _TRANSFORMER_DECODE_HOST_FORWARD = 6
comptime _TRANSFORMER_DECODE_DECODE = 7


def _transformer_decode_entry_names() -> List[String]:
    # Indexed by the _TRANSFORMER_DECODE_* words above.
    return [String("transformer_decode_session_step"), String("transformer_decode_session_forward"),
            String("transformer_decode_session_load_state"), String("transformer_decode_session_export_state"),
            String("transformer_decode_session_close"), String("transformer_decode_step"), String("transformer_forward"),
            String("transformer_decode_session_decode")]


def _transformer_decode_keep_entries(mut handle: TransformerDecodeHandle) raises:
    # N181: resolve once at open what every step looked up by name. A missing
    # or non-compiled entry is kept as None and refused at its use, with the
    # message _transformer_decode_entry gives, as before.
    var function_type = py_type(native_loaded_module("builtins").len)
    var names = _transformer_decode_entry_names()
    handle.entries.clear()
    for name in names:  # small-loop(eight fixed decode entry names)
        var entry = py_get(handle.binding, PythonObject(name))
        handle.entries.append(entry if py_type(entry) is function_type else PythonObject(None))


def _transformer_decode_kept(handle: TransformerDecodeHandle, index: Int) raises -> PythonObject:
    var entry = handle.entries[index]
    if entry is None:
        raise Error("NotImplementedError: Transformer decode lacks its connected compiled " + _transformer_decode_entry_names()[index] + " entry")
    return entry
struct _TransformerDecodeBindings(Defaultable, Movable):
    # Row N192: the last admitted binding resolution. A session open with the
    # same connected binding object and mode reuses the checked entries
    # dictionary instead of copying three module dictionaries and calling
    # the four compiled vendor/mode queries again (their answers are fixed
    # for a compiled module).
    var selected: PythonObject
    var mode: String
    var binding: PythonObject

    def __init__(out self):
        self.selected = PythonObject(None)
        self.mode = String("")
        self.binding = PythonObject(None)


comptime _TRANSFORMER_DECODE_BINDINGS = _Global[
    StorageType=_TransformerDecodeBindings, name="MojolearnTransformerDecodeBindingsV1",
    init_fn=_TransformerDecodeBindings.__init__,
]


def _transformer_decode_default_mode(module: PythonObject, context: PythonObject) raises -> PythonObject:
    # One key of the backend module: read from its dictionary in place
    # (formerly a copy of the whole module dictionary per session open).
    var bi = native_loaded_module("builtins")
    if py_type(module) is bi.str:
        raise Error("TransformerDecodeSession: " + String(module))
    if py_type(module) is not context["admission_context"]["native_module_type"]:
        raise Error("TypeError: Transformer decode requires its connected native module")
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyModule_GetDict(module._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_borrowed=raw)
    if not py_contains(fields, "_DEFAULT_MODE"):
        raise Error("KeyError: _DEFAULT_MODE")
    return fields["_DEFAULT_MODE"]


def _transformer_decode_binding(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var route = String(fields["_native_route"])
    if route != "cpu" and route != "gpu":
        raise Error("TypeError: Transformer decode route must be cpu or gpu")
    var mode = fields["numeric_mode"]
    if route == "cpu":
        mode = PythonObject("identical")
    elif mode is None:
        mode = _transformer_decode_default_mode(context["backend"], context)
    if py_type(mode) is not bi.str:
        raise Error("TypeError: Transformer decode mode requires an ordinary string")
    var mode_name = String(mode.strip().lower())
    if mode_name != "identical" and mode_name != "fast":
        raise Error("Transformer decode mode must be identical or fast")
    var selected = py_get(context["bindings"], PythonObject(route + ":" + mode_name))
    if selected is None:
        raise Error("Transformer decode native binding connection is unavailable")
    var held = _TRANSFORMER_DECODE_BINDINGS.get_or_create_ptr()
    if held[].binding is not None and held[].selected is selected and held[].mode == route + ":" + mode_name:
        return held[].binding
    var binding = _transformer_decode_module(selected["transformer"], context)
    var prefix = "neural_host" if route == "cpu" else "transformer"
    var vendor = String()
    var mode_code = 0
    _transformer_decode_words(binding, prefix, vendor, mode_code)
    if (route == "cpu" and vendor != "cpu") or (route == "gpu" and vendor != "metal" and vendor != "cuda" and vendor != "hip"):
        raise Error("Transformer decode native vendor differs from its selected route")
    if mode_code != (1 if mode_name == "identical" else 0):
        raise Error("Transformer decode native mode differs from its selected mode")
    var arithmetic = selected["linalg"]
    if py_type(arithmetic) is not bi.str:
        var math_entries = _transformer_decode_module(arithmetic, context)
        var math_vendor = String()
        var math_mode = 0
        _transformer_decode_words(math_entries, "linalg", math_vendor, math_mode)
        if math_vendor != vendor or math_mode != mode_code:
            raise Error("Transformer decode packed binding vendor/mode differs from the model")
    held = _TRANSFORMER_DECODE_BINDINGS.get_or_create_ptr()
    held[].selected = selected
    held[].mode = route + ":" + mode_name
    held[].binding = binding
    return binding


def _transformer_decode_weights(fields: PythonObject) raises:
    var configuration = _transformer_configuration(fields)
    var d = configuration[0]; var kw = configuration[2] * configuration[3]
    var hd = configuration[3]; var it = configuration[4]
    var opts = BlockOptions.from_params(configuration, 6)
    opts.validate(hd)
    var expected = List[List[Int]]()
    expected.append([d]); expected.append([d]); expected.append([d, d])
    expected.append([kw, d]); expected.append([kw, d]); expected.append([d, d])
    expected.append([it, d]); expected.append([it, d]); expected.append([d, it])
    var weights = fields["_w"]
    if len(weights) != 9:
        raise Error("Transformer decode requires nine base weight slots")
    for i in range(9):  # driver-loop: original projection layouts before raw-address dispatch
        if i == 6 and not opts.gated():
            if weights[i] is not None:
                raise Error("Transformer decode ungated MLP cannot carry a gate weight")
        else:
            if weights[i] is None:
                raise Error("Transformer decode required base weight is missing")
            _model_shape(weights[i], expected[i], "Transformer decode weight")
    var optional = fields["_wopt"]
    if len(optional) != 11:
        raise Error("Transformer decode requires eleven optional weight slots")
    var widths: List[Int] = [d, kw, kw, d, d, d, it, d, it, hd, hd]
    var needed: List[Bool] = [opts.qkv_bias, opts.qkv_bias, opts.qkv_bias, opts.o_bias,
        opts.norm_bias, opts.norm_bias, opts.mlp_bias, opts.mlp_bias,
        opts.mlp_bias and opts.gated(), opts.qk_norm, opts.qk_norm]
    for i in range(11):  # driver-loop: optional weight shapes match the captured option record
        if needed[i]:
            if optional[i] is None:
                raise Error("Transformer decode required optional weight is missing")
            _model_shape(optional[i], [widths[i]], "Transformer decode optional weight")
        elif optional[i] is not None:
            raise Error("Transformer decode optional weight is present with its option disabled")
    var packed = py_get(fields, "_int15")
    if packed is not None:
        if len(packed) != 7:
            raise Error("Transformer decode requires seven packed projection slots")
        for i in range(7):  # driver-loop: retained fixed15 plane extents
            var triplet = packed[i]
            if i == 4 and not opts.gated():
                if triplet is not None:
                    raise Error("Transformer decode ungated MLP cannot carry packed gate planes")
                continue
            if triplet is None or len(triplet) != 3:
                raise Error("Transformer decode packed projections require high/low/exponent planes")
            for plane in range(3):  # small-loop(three integer plane formats)
                var array = triplet[plane]
                if array is None or String(array.order) != "C" or String(array.dtype) != ("<i4" if plane == 2 else "<i1"):
                    raise Error("TypeError: Transformer decode packed planes require contiguous integer storage")
                _model_shape(array, [expected[i + 2][0]] if plane == 2 else expected[i + 2].copy(), "Transformer decode packed plane")



def _transformer_decode_fields(session: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(session) is not context["transformer_session_class"]:
        raise Error("TypeError: TransformerDecodeSession requires its exact native public owner")
    return neural_record_fields(session, Python.tuple("_block", "_state", "_handle"))


def _transformer_decode_owner(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _transformer_decode_fields(session, context)
    var hobj = fields["_handle"]
    var owner = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if fields["_block"] is not owner[].block or fields["_state"] is not owner[].state:
        raise Error("TypeError: TransformerDecodeSession owner fields were replaced")
    return hobj


def _transformer_decode_block(block: PythonObject, context: PythonObject, weights: Bool = False) raises -> PythonObject:
    if weights:
        return neural_block_snapshot(block, context)
    # Retained decode sessions do not reread changed weights. Capture only
    # current configuration/profile metadata, preserving that ownership rule.
    var fields = neural_block_fields(block, context)
    var snapshot = py_call(native_loaded_module("builtins").dict, fields)
    py_setitem(snapshot, "_opts_tail", neural_metadata_snapshot(fields["_opts_tail"]))
    return snapshot


def _transformer_decode_state_fields(state: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var names = Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "cached_tokens", "window", "k_cache", "v_cache", "arithmetic_profile", "_resident_session")
    var live = _transformer_fields(state, context["generation_classes"][2]["transformer"], names)
    var slots = py_call(native_loaded_module("builtins").dict, live)
    # Old serialized states may omit window when no sliding window is used.
    if not py_contains(slots, "window"):
        py_setitem(slots, "window", PythonObject(0))
    var words = _transformer_state_words(slots)
    var profile = py_get(slots, "arithmetic_profile")
    if profile is not None and py_type(profile) is not native_loaded_module("builtins").str:
        raise Error("TypeError: Transformer cache arithmetic profile requires an ordinary string")
    var configuration = _transformer_configuration(fields)
    if words[1] != configuration[2] or words[2] != configuration[3] or words[5] != configuration[5]:
        raise Error("TransformerDecodeSession: state attention/cache dimensions differ from the block")
    py_setitem(slots, "_decode_live_fields", live)
    return slots


def _transformer_decode_publish(slots: PythonObject) raises:
    var live = slots["_decode_live_fields"]
    for key in [String("cached_tokens"), String("arithmetic_profile"), String("_resident_session")]:  # driver-loop: admitted native state bookkeeping only
        if py_contains(slots, key):
            py_setitem(live, key, slots[key])


def _transformer_decode_shape(slots: PythonObject) raises -> List[Int]:
    var words = _transformer_state_words(slots)
    return [words[0], words[1], words[2], words[3], words[5]]


def _transformer_decode_compatible(owner: TransformerDecodeHandle, fields: PythonObject, slots: PythonObject) raises:
    if not _transformer_same_words(owner.configuration, _transformer_configuration(fields)) or not _transformer_same_words(owner.state_shape, _transformer_decode_shape(slots)):
        raise Error("TransformerDecodeSession: configuration/cache dimensions changed; close and reopen the session")


struct TransformerDecodeCache(Movable):
    var original: PythonObject
    var captured: PythonObject
    var pin: Optional[AdmittedBorrowedBuffer]
    var copied: List[Float32]
    var address: Int
    var count: Int

    def __init__(out self, value: PythonObject, count: Int, context: PythonObject) raises:
        self.original = value
        self.captured = PythonObject(None)
        self.pin = None
        self.copied = List[Float32]()
        self.address = 0
        self.count = count
        if py_type(value) is context["array_class"]:
            self.captured = neural_array_snapshot(value, context)
            var array = self.captured
            if String(array.dtype) != "<f4" or String(array.order) != "C" or Bool(py=array._readonly) or Bool(py=array._store.readonly):
                raise Error("TypeError: Transformer decode cache requires writable contiguous float32 storage")
            if len(array.shape) != 1 or Int(py=array.shape[0]) != count:
                raise Error("Transformer decode cache shape differs from its declared capacity")
            self.address = Int(py=array._addr)
        else:
            admit_native_buffer_exporter(value)
            self.pin = AdmittedBorrowedBuffer(value, True)
            var layout = self.pin.value().layout.value().copy()
            if self.pin.value().format != "f" or layout.itemsize != 4 or not layout.contiguous() or len(layout.shape) != 1 or layout.shape[0] != count:
                raise Error("TypeError: Transformer decode cache requires writable flat contiguous float32 storage")
            # Preserve the legacy foreign-buffer copy/writeback boundary. It
            # publishes completed exports only after successful execution.
            self.copied = List[Float32](length=count, fill=0)
            self.address = Int(self.copied.unsafe_ptr())
            layout.copy_to(self.address)

    def publish(self) raises:
        if self.pin:
            self.pin.value().layout.value().copy_from(self.address)



def _transformer_decode_profile(fields: PythonObject, slots: PythonObject) raises:
    var profile = py_get(fields, "_arithmetic_profile")
    var saved = py_get(slots, "arithmetic_profile")
    if saved is None and Int(py=slots["cached_tokens"]) != 0 and Bool(py=py_get(fields, "_arithmetic_profile_changed", PythonObject(False))):
        raise Error("TransformerState with cached tokens needs arithmetic profile metadata under an experimental version")
    if saved is not None and profile is not None and Bool(py=saved != profile):
        raise Error("TransformerState arithmetic profile differs from this block")
    if profile is not None:
        py_setitem(slots, "arithmetic_profile", profile)


def _transformer_decode_params(fields: PythonObject, slots: PythonObject, batch: Int, length: Int) raises -> List[Int]:
    var p: List[Int] = [batch, length, Int(py=fields["d_model"]), Int(py=fields["n_heads"]),
        Int(py=fields["n_kv_heads"]), Int(py=fields["head_dim"]), Int(py=fields["intermediate"]),
        Int(py=slots["max_tokens"]), Int(py=slots["cached_tokens"]), Int(py=fields["window"])]
    var tail = fields["_opts_tail"]
    for i in range(len(tail)):  # driver-loop: admitted fixed block-option words
        p.append(Int(py=tail[i]))
    return p^


def _transformer_decode_copy(value: PythonObject, context: PythonObject) raises -> List[Float32]:
    if value is None:
        return List[Float32]()
    var array = value  # native-owned contiguous snapshot captured once at session admission
    var result = List[Float32](length=Int(py=array.size), fill=0)
    if len(result) != 0:
        memcpy(dest=result.unsafe_ptr(), src=f32_ptr(Int(py=array._addr)), count=len(result))
    _ = array
    return result^


def _transformer_decode_remember(hobj: PythonObject, fields: PythonObject, slots: PythonObject, context: PythonObject) raises:
    # Row N177: after a full admission (init or a step that re-admitted),
    # keep the checked step words and the generation words that prove them,
    # plus a native content record of the option tail list (its in-place
    # edits change no dictionary). An object with no generation, or a tail
    # the content record cannot check, keeps the full admission every step.
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    var block = h[].block
    var state = h[].state
    var live_block = neural_block_fields(block, context)
    var tail = ContentRecord()
    content_value(tail, py_get(live_block, "_opts_tail"), 0, 2, False)
    var block_word = model_generation(block) if tail.ok and records_stampable(block) else 0
    var state_word = model_generation(state) if records_stampable(state) else 0
    var width = Int(py=fields["d_model"])
    var batch = Int(py=slots["batch_size"])
    var params = _transformer_decode_params(fields, slots, batch, 0)
    var live = slots["_decode_live_fields"]
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    h[].width = width
    h[].batch = batch
    h[].params = params^
    h[].live_state = live
    h[].tail_content = tail^
    h[].block_generation = block_word
    h[].state_generation = state_word


def _transformer_decode_unchanged(hobj: PythonObject) raises -> Bool:
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if h[].block_generation == 0 or h[].state_generation == 0:
        return False
    var block = h[].block
    var state = h[].state
    var block_word = h[].block_generation
    var state_word = h[].state_generation
    if model_generation(block) != block_word or model_generation(state) != state_word:
        return False
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    return content_same(h[].tail_content)


def transformer_decode_init(session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var values = _transformer_decode_fields(session, context)
    var fields = _transformer_decode_block(block, context, True)
    var binding = _transformer_decode_binding(fields, context)
    _transformer_decode_weights(fields)
    var device = py_contains(binding, "transformer_decode_session_create")
    var packed = py_get(fields, "_int15") is not None
    if packed and (not device or not py_contains(binding, "transformer_decode_session_open_int15")):
        raise Error("NotImplementedError: TransformerDecodeSession fixed15_v1 needs its GPU resident session entry")
    if not device and not py_contains(binding, "transformer_decode_step"):
        raise Error("NotImplementedError: TransformerDecodeSession binding has neither resident nor per-call decode")
    if state is None:
        raise Error("TransformerDecodeSession: state is required (allocate_state)")
    var slots = _transformer_decode_state_fields(state, fields, context)
    _transformer_decode_profile(fields, slots)
    if py_get(slots, "_resident_session") is not None:
        raise Error("TransformerDecodeSession: the state belongs to an open resident session")
    var batch = Int(py=slots["batch_size"])
    if batch < 1:
        raise Error("TransformerDecodeSession: the state holds no rows")
    if Int(py=py_get(slots, "window", PythonObject(0))) != Int(py=fields["window"]):
        raise Error("TransformerDecodeSession: state window differs from this block")
    var keep = List[PythonObject]()
    var words = _transformer_state_words(slots)
    var keys = TransformerDecodeCache(slots["k_cache"], words[6], context)
    var vals = TransformerDecodeCache(slots["v_cache"], words[6], context)
    var pieces: List[Int] = [keys.address, vals.address, words[6]]
    var handle = TransformerDecodeHandle(binding, block, state)
    _transformer_decode_keep_entries(handle)
    handle.configuration = _transformer_configuration(fields)
    handle.state_shape = _transformer_decode_shape(slots)
    var cached = Int(py=slots["cached_tokens"])
    py_setitem(values, "_block", block)
    py_setitem(values, "_state", state)
    if device:
        var snapshot = GenerationLayerSnapshot(fields, "transformer", packed)
        var a = snapshot.weights.resolve(keep)
        a.append(pieces[0]); a.append(pieces[1])
        var optional = snapshot.optional.resolve(keep)
        for address in optional:  # driver-loop: optional resident weight addresses
            a.append(address)
        var planes = snapshot.packed.resolve(keep)
        for address in planes:  # driver-loop: packed projection plane addresses
            a.append(address)
        var p = snapshot.words.transformer_params(batch, 0, Int(py=slots["max_tokens"]), cached)
        var open_p = List[Int]()
        for i in range(len(p)):  # driver-loop: resident shape words omit sequence length
            if i != 1:
                open_p.append(p[i])
        handle.device = _transformer_decode_entry(binding, "transformer_decode_session_create")()
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1" and String(fields["numeric_mode"]) == "identical":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _transformer_decode_entry(binding, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
            handle.native_handle = transformer_resident_decode_handle(handle.provider, handle.device)
            cached = transformer_resident_decode_call_handle(handle.provider, handle.device, handle.native_handle,
                19 if packed else 18, a, open_p)
        else:
            cached = Int(py=py_call(_transformer_decode_entry(binding, "transformer_decode_session_open_int15" if packed else "transformer_decode_session_open"),
                handle.device, py_ints(a), py_ints(open_p)))
    else:
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _transformer_decode_entry(binding, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
        var weights = fields["_w"]
        var optional = fields["_wopt"]
        for i in range(len(weights)):  # driver-loop: native session-owned weight copies
            handle.weights.append(_transformer_decode_copy(weights[i], context))
        for i in range(len(optional)):  # driver-loop: native session-owned optional weight copies
            handle.optional.append(_transformer_decode_copy(optional[i], context))
        handle.k_cache = List[Float32](length=pieces[2], fill=0)
        handle.v_cache = List[Float32](length=pieces[2], fill=0)
        memcpy(dest=handle.k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
        memcpy(dest=handle.v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
    handle.open = True
    py_setitem(values, "_handle", PythonObject(alloc=handle^))
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    py_setitem(slots, "_resident_session", session)
    _transformer_decode_publish(slots)
    # The step words are those of this admission; `fields` here is the
    # weight snapshot, whose configuration words equal the block's.
    _transformer_decode_remember(values["_handle"], _transformer_decode_block(block, context), slots, context)
    _ = keys^
    _ = vals^
    _ = keep^
    return PythonObject(None)


def _transformer_decode_host_tokens(handle: TransformerDecodeHandle, mut a: List[Int], p: List[Int], px: Int, py: Int,
                                    batch: Int, length: Int, width: Int, cached0: Int) raises -> Int:
    """`decode` on the host arm: the host step entry once per token on the
    session's own weights and caches, token t copied into and out of
    (B, d_model) buffers. Bytes equal `length` host `step` calls."""
    var out_slot = 1 + len(handle.weights) + 2
    var row = batch * width
    var xt = List[Float32](length=row, fill=Float32(0.0))
    var yt = List[Float32](length=row, fill=Float32(0.0))
    a[0] = Int(xt.unsafe_ptr())
    a[out_slot] = Int(yt.unsafe_ptr())
    var src = f32_ptr(px)
    var dst = f32_ptr(py)
    var entry = _transformer_decode_kept(handle, _TRANSFORMER_DECODE_HOST_STEP)
    var cached = cached0
    for t in range(length):  # driver-loop: over decode tokens, one host step each
        for r in range(batch):  # driver-loop: over the session's batch rows, a host-arm copy of token t
            memcpy(dest=xt.unsafe_ptr() + r * width, src=src + (r * length + t) * width, count=width)
        # The host step's shape words omit the sequence length (index 1).
        var words = List[Int](capacity=len(p))
        for i in range(len(p)):  # driver-loop: host decode shape words omit sequence length
            if i == 1:
                continue
            words.append(cached if i == 8 else p[i])
        cached = Int(py=py_call(entry, py_ints(a), py_ints(words)))
        for r in range(batch):  # driver-loop: over the session's batch rows, a host-arm copy of token t
            memcpy(dest=dst + (r * length + t) * width, src=yt.unsafe_ptr() + r * width, count=width)
    _ = xt^
    _ = yt^
    return cached


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    return transformer_decode_run(session, input, step, False, context)


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, decode: Bool, context: PythonObject) raises -> PythonObject:
    # `decode` (D2): x is (B, N, d_model) and the N tokens run as N steps in
    # one call; `step` is the one-token case, neither is a forward of L tokens.
    var hobj = _transformer_decode_owner(session, context)
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        raise Error("TransformerDecodeSession: the session is closed")
    # Row N177: while the block's and the state's generation words are those
    # of the last full admission, the kept words stand for the block, state
    # and configuration records (no re-admission per token); the step reads
    # and writes only cached_tokens.
    var kept = _transformer_decode_unchanged(hobj)
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    var fields = PythonObject(None)
    var slots = PythonObject(None)
    var width = h[].width
    var rows = h[].batch
    if not kept:
        fields = _transformer_decode_block(h[].block, context)
        slots = _transformer_decode_state_fields(h[].state, fields, context)
        h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
        _transformer_decode_compatible(h[], fields, slots)
        width = Int(py=fields["d_model"])
        rows = Int(py=slots["batch_size"])
    var x = _neural_input(input, width, step, context)
    var shape = x.shape
    var batch = Int(py=shape[0])
    var length = Int(py=shape[1])
    if batch != rows:
        raise Error("TransformerDecodeSession: input rows differ from the state")
    var y = generation_array([batch, length, width])
    var keep = List[PythonObject]()
    var px = generation_address(x, keep)
    var py = generation_address(y, keep)
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    var live = h[].live_state
    var cached = Int(py=live["cached_tokens"]) if kept else Int(py=slots["cached_tokens"])
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if h[].device is not None and decode:
        # The provider descriptor wraps the same session object; the decode
        # entry is called on it directly.
        cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_DECODE),
            h[].device, py_ints([px, py]), py_ints([length, cached])))
    elif h[].device is not None:
        var p: List[Int] = [cached] if step else [length, cached]
        if h[].provider is not None:
            cached = transformer_resident_decode_call_handle(h[].provider, h[].device, h[].native_handle,
                20 if step else 21, [px, py], p)
        else:
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_STEP if step else _TRANSFORMER_DECODE_FORWARD),
                h[].device, py_ints([px, py]), py_ints(p)))
    else:
        var a: List[Int] = [px]
        for i in range(len(h[].weights)):  # driver-loop: native session-owned weight addresses
            a.append(0 if len(h[].weights[i]) == 0 else Int(h[].weights[i].unsafe_ptr()))
        a.append(Int(h[].k_cache.unsafe_ptr())); a.append(Int(h[].v_cache.unsafe_ptr())); a.append(py)
        var p: List[Int]
        if kept:
            p = h[].params.copy()
            p[1] = length
            p[8] = cached
        else:
            p = _transformer_decode_params(fields, slots, batch, length)
        for i in range(len(h[].optional)):  # driver-loop: native optional-weight address slots
            a.append(0 if len(h[].optional[i]) == 0 else Int(h[].optional[i].unsafe_ptr()))
        if decode:
            cached = _transformer_decode_host_tokens(h[], a, p, px, py, batch, length, width, cached)
        elif h[].provider is not None:
            cached = neural_descriptor_execute(h[].provider, 33, a, p, [])
        else:
            if step:
                var shortened = List[Int]()
                for i in range(len(p)):  # driver-loop: host decode shape words omit sequence length
                    if i != 1:
                        shortened.append(p[i])
                p = shortened^
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_HOST_STEP if step else _TRANSFORMER_DECODE_HOST_FORWARD), py_ints(a), py_ints(p)))
    if kept:
        # The only state word a step changes; the publish of the full path
        # writes arithmetic_profile and _resident_session unchanged.
        py_setitem(live, "cached_tokens", PythonObject(cached))
        # Fold this step's own write into the kept state word.
        var state = hobj.downcast_value_ptr[TransformerDecodeHandle]()[].state
        var word = model_generation(state)
        hobj.downcast_value_ptr[TransformerDecodeHandle]()[].state_generation = word
    else:
        py_setitem(slots, "cached_tokens", PythonObject(cached))
        _transformer_decode_publish(slots)
        _transformer_decode_remember(hobj, fields, slots, context)
    _ = x^
    _ = keep^
    _ = hobj
    return y


def transformer_decode_state(session: PythonObject, load: Bool, context: PythonObject) raises -> PythonObject:
    var hobj = _transformer_decode_owner(session, context)
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        raise Error("TransformerDecodeSession: the session is closed")
    var fields = _transformer_decode_block(h[].block, context)
    var state = h[].state
    var slots = _transformer_decode_state_fields(state, fields, context)
    _transformer_decode_compatible(h[], fields, slots)
    if load:
        _transformer_decode_profile(fields, slots)
    var words = _transformer_state_words(slots)
    var keys = TransformerDecodeCache(slots["k_cache"], words[6], context)
    var vals = TransformerDecodeCache(slots["v_cache"], words[6], context)
    var pieces: List[Int] = [keys.address, vals.address, words[6]]
    var cached = Int(py=slots["cached_tokens"])
    if h[].device is not None:
        var a: List[Int] = [pieces[0], pieces[1]]
        var p: List[Int] = [cached] if load else []
        if h[].provider is not None:
            cached = transformer_resident_decode_call_handle(h[].provider, h[].device, h[].native_handle,
                23 if load else 22, a, p)
        elif load:
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_LOAD), h[].device, py_ints(a), py_ints(p)))
        else:
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_EXPORT), h[].device, py_ints(a)))
    else:
        if pieces[2] != len(h[].k_cache) or pieces[2] != len(h[].v_cache):
            raise Error("TransformerDecodeSession: cache dimensions differ from the retained native session")
        if load:
            memcpy(dest=h[].k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
            memcpy(dest=h[].v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
        else:
            memcpy(dest=f32_ptr(pieces[0]), src=h[].k_cache.unsafe_ptr(), count=pieces[2])
            memcpy(dest=f32_ptr(pieces[1]), src=h[].v_cache.unsafe_ptr(), count=pieces[2])
    if not load:
        keys.publish()
        vals.publish()
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    _transformer_decode_publish(slots)
    _ = keys^
    _ = vals^
    _ = hobj
    return state


def transformer_decode_release(session: PythonObject, context: PythonObject) raises:
    var values = _transformer_decode_fields(session, context)
    var hobj = py_get(values, "_handle")
    if hobj is None:
        return
    hobj = _transformer_decode_owner(session, context)
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        return
    var state_fields = neural_record_fields(h[].state, Python.tuple("_resident_session"))
    h[].open = False
    py_setitem(state_fields, "_resident_session", PythonObject(None))
    if h[].device is not None:
        if h[].provider is not None:
            _ = transformer_resident_decode_call_handle(h[].provider, h[].device, h[].native_handle, 24, [], [])
        else:
            _ = py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_CLOSE), h[].device)
    h[].weights.clear(); h[].optional.clear(); h[].k_cache.clear(); h[].v_cache.clear()


def transformer_decode_native_call(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "new":
        var session = py_call(native_loaded_module("builtins").object.__new__, context["transformer_session_class"])
        _ = transformer_decode_init(session, values[0], values[1], context)
        return session
    var session = values[0]
    if py_type(session) is not context["transformer_session_class"]:
        raise Error("TypeError: TransformerDecodeSession requires its exact native public owner")
    if operation == "init":
        return transformer_decode_init(session, values[1], values[2], context)
    if operation == "step" or operation == "forward":
        return transformer_decode_run(session, values[1], operation == "step", context)
    if operation == "decode":
        return transformer_decode_run(session, values[1], False, True, context)
    if operation == "sync" or operation == "load":
        return transformer_decode_state(session, operation == "load", context)
    if operation == "state":
        var hobj = _transformer_decode_owner(session, context)
        var state = hobj.downcast_value_ptr[TransformerDecodeHandle]()[].state
        _ = hobj
        return state
    if operation == "enter":
        return session
    if operation == "close" or operation == "exit":
        var hobj = py_get(_transformer_decode_fields(session, context), "_handle")
        if hobj is None or not hobj.downcast_value_ptr[TransformerDecodeHandle]()[].open:
            return PythonObject(False) if operation == "exit" else PythonObject(None)
        try:
            _ = transformer_decode_state(session, False, context)
        finally:
            transformer_decode_release(session, context)
        return PythonObject(False) if operation == "exit" else PythonObject(None)
    if operation == "drop":
        try:
            transformer_decode_release(session, context)
        except:
            pass
        return PythonObject(None)
    if operation == "discard":
        transformer_decode_release(session, context)
        return PythonObject(None)
    var hobj = _transformer_decode_owner(session, context)
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if operation == "is_open":
        return PythonObject(h[].open)
    if operation == "native":
        return h[].device
    if operation == "repr":
        var fields = _transformer_decode_block(h[].block, context)
        var slots = _transformer_decode_state_fields(h[].state, fields, context)
        var cached = Int(py=slots["cached_tokens"])
        return PythonObject("TransformerDecodeSession(open=" + String("True" if h[].open else "False") + ", cached_tokens=" + String(cached) + ")")
    raise Error("Transformer decode: unknown operation " + operation)


def transformer_decode_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var bi = native_loaded_module("builtins")
        if len(args) != 3 or py_type(args[0]) is not bi.str or py_type(args[1]) is not bi.tuple or py_type(args[2]) is not bi.dict:
            raise Error("TypeError: Transformer decode operation requires native connection arguments")
        var result = transformer_decode_native_call(String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
