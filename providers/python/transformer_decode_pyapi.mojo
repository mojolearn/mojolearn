# SPDX-License-Identifier: Apache-2.0
# Public Transformer decode-session admission/ownership. The Python class is a
# thin connection; GPU execution uses the common native provider and CPU copies
# stay native. No Python session methods or block hooks are invoked here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
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
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute
from providers.python.transformer_resident_pyapi import transformer_resident_decode_call
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME
from core.hostptr import f32_ptr
from transformer.block_options import BlockOptions


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

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TransformerDecodeHandle(open=", self.open, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _transformer_decode_module(module: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(module) is bi.str:
        raise Error("TransformerDecodeSession: " + String(module))
    if py_type(module) is not context["admission_context"]["native_module_type"]:
        raise Error("TypeError: Transformer decode requires its connected native module")
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_borrowed=raw)
    neural_string_keys(fields)
    return py_call(bi.dict, fields)


def _transformer_decode_entry(entries: PythonObject, name: String) raises -> PythonObject:
    var entry = py_get(entries, name)
    if py_type(entry) is not py_type(Python.import_module("builtins").len):
        raise Error("NotImplementedError: Transformer decode lacks its connected compiled " + name + " entry")
    return entry


def _transformer_decode_binding(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var route = String(fields["_native_route"])
    if route != "cpu" and route != "gpu":
        raise Error("TypeError: Transformer decode route must be cpu or gpu")
    var mode = fields["numeric_mode"]
    if route == "cpu":
        mode = PythonObject("identical")
    elif mode is None:
        mode = _transformer_decode_module(context["backend"], context)["_DEFAULT_MODE"]
    if py_type(mode) is not bi.str:
        raise Error("TypeError: Transformer decode mode requires an ordinary string")
    var mode_name = String(mode.strip().lower())
    if mode_name != "identical" and mode_name != "fast":
        raise Error("Transformer decode mode must be identical or fast")
    var selected = py_get(context["bindings"], PythonObject(route + ":" + mode_name))
    if selected is None:
        raise Error("Transformer decode native binding connection is unavailable")
    var binding = _transformer_decode_module(selected["transformer"], context)
    var prefix = "neural_host" if route == "cpu" else "transformer"
    var vendor = String(_transformer_decode_entry(binding, prefix + "_vendor")())
    var mode_code = Int(py=_transformer_decode_entry(binding, prefix + "_numeric_mode")())
    if (route == "cpu" and vendor != "cpu") or (route == "gpu" and vendor != "metal" and vendor != "cuda" and vendor != "hip"):
        raise Error("Transformer decode native vendor differs from its selected route")
    if mode_code != (1 if mode_name == "identical" else 0):
        raise Error("Transformer decode native mode differs from its selected mode")
    var arithmetic = selected["linalg"]
    if py_type(arithmetic) is not bi.str:
        var math_entries = _transformer_decode_module(arithmetic, context)
        if String(_transformer_decode_entry(math_entries, "linalg_vendor")()) != vendor or Int(py=_transformer_decode_entry(math_entries, "linalg_numeric_mode")()) != mode_code:
            raise Error("Transformer decode packed binding vendor/mode differs from the model")
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
    var snapshot = py_call(Python.import_module("builtins").dict, fields)
    py_setitem(snapshot, "_opts_tail", neural_metadata_snapshot(fields["_opts_tail"]))
    return snapshot


def _transformer_decode_state_fields(state: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var names = Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "cached_tokens", "window", "k_cache", "v_cache", "arithmetic_profile", "_resident_session")
    var live = _transformer_fields(state, context["generation_classes"][2]["transformer"], names)
    var slots = py_call(Python.import_module("builtins").dict, live)
    # Old serialized states may omit window when no sliding window is used.
    if not py_contains(slots, "window"):
        py_setitem(slots, "window", PythonObject(0))
    var words = _transformer_state_words(slots)
    var profile = py_get(slots, "arithmetic_profile")
    if profile is not None and py_type(profile) is not Python.import_module("builtins").str:
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


def transformer_decode_init(session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
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
            cached = transformer_resident_decode_call(handle.provider, handle.device, 19 if packed else 18, a, open_p)
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
    _ = keys^
    _ = vals^
    _ = keep^
    return PythonObject(None)


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    var hobj = _transformer_decode_owner(session, context)
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open:
        raise Error("TransformerDecodeSession: the session is closed")
    var fields = _transformer_decode_block(h[].block, context)
    var slots = _transformer_decode_state_fields(h[].state, fields, context)
    _transformer_decode_compatible(h[], fields, slots)
    var width = Int(py=fields["d_model"])
    var x = _neural_input(input, width, step, context)
    var shape = x.shape
    var batch = Int(py=shape[0])
    var length = Int(py=shape[1])
    if batch != Int(py=slots["batch_size"]):
        raise Error("TransformerDecodeSession: input rows differ from the state")
    var y = generation_array([batch, length, width])
    var keep = List[PythonObject]()
    var px = generation_address(x, keep)
    var py = generation_address(y, keep)
    var cached = Int(py=slots["cached_tokens"])
    if h[].device is not None:
        var p: List[Int] = [cached] if step else [length, cached]
        if h[].provider is not None:
            cached = transformer_resident_decode_call(h[].provider, h[].device, 20 if step else 21, [px, py], p)
        else:
            cached = Int(py=py_call(_transformer_decode_entry(h[].binding, "transformer_decode_session_step" if step else "transformer_decode_session_forward"),
                h[].device, py_ints([px, py]), py_ints(p)))
    else:
        var a: List[Int] = [px]
        for i in range(len(h[].weights)):  # driver-loop: native session-owned weight addresses
            a.append(0 if len(h[].weights[i]) == 0 else Int(h[].weights[i].unsafe_ptr()))
        a.append(Int(h[].k_cache.unsafe_ptr())); a.append(Int(h[].v_cache.unsafe_ptr())); a.append(py)
        var p = _transformer_decode_params(fields, slots, batch, length)
        for i in range(len(h[].optional)):  # driver-loop: native optional-weight address slots
            a.append(0 if len(h[].optional[i]) == 0 else Int(h[].optional[i].unsafe_ptr()))
        if h[].provider is not None:
            cached = neural_descriptor_execute(h[].provider, 33, a, p, [])
        else:
            if step:
                var shortened = List[Int]()
                for i in range(len(p)):  # driver-loop: host decode shape words omit sequence length
                    if i != 1:
                        shortened.append(p[i])
                p = shortened^
            cached = Int(py=py_call(_transformer_decode_entry(h[].binding, "transformer_decode_step" if step else "transformer_forward"), py_ints(a), py_ints(p)))
    py_setitem(slots, "cached_tokens", PythonObject(cached))
    _transformer_decode_publish(slots)
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
            cached = transformer_resident_decode_call(h[].provider, h[].device, 23 if load else 22, a, p)
        elif load:
            cached = Int(py=py_call(_transformer_decode_entry(h[].binding, "transformer_decode_session_load_state"), h[].device, py_ints(a), py_ints(p)))
        else:
            cached = Int(py=py_call(_transformer_decode_entry(h[].binding, "transformer_decode_session_export_state"), h[].device, py_ints(a)))
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
            _ = transformer_resident_decode_call(h[].provider, h[].device, 24, [], [])
        else:
            _ = py_call(_transformer_decode_entry(h[].binding, "transformer_decode_session_close"), h[].device)
    h[].weights.clear(); h[].optional.clear(); h[].k_cache.clear(); h[].v_cache.clear()


def transformer_decode_native_call(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "new":
        var session = py_call(Python.import_module("builtins").object.__new__, context["transformer_session_class"])
        _ = transformer_decode_init(session, values[0], values[1], context)
        return session
    var session = values[0]
    if py_type(session) is not context["transformer_session_class"]:
        raise Error("TypeError: TransformerDecodeSession requires its exact native public owner")
    if operation == "init":
        return transformer_decode_init(session, values[1], values[2], context)
    if operation == "step" or operation == "forward":
        return transformer_decode_run(session, values[1], operation == "step", context)
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
        var bi = Python.import_module("builtins")
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
