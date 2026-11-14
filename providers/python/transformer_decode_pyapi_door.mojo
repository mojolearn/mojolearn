# SPDX-License-Identifier: Apache-2.0
# Public Transformer decode-session admission/ownership. The Python class is a
# thin connection; GPU execution uses the common native provider and CPU copies
# stay native. No Python session methods or block hooks are invoked here.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from std.memory import unsafe_memcpy
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_setitem, py_type, py_contains, py_fields
from bindings.generation_buffers_door import generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.causal_lm_construct_door import _model_shape
from std.sys.compile import is_defined
from bindings.native_loaded_module_door import NativeModules
from bindings.neural_block_public_door import _neural_input, _neural_input_rows
from bindings.neural_block_admission_door import neural_record_fields, neural_block_fields, neural_block_snapshot, neural_array_snapshot, neural_metadata_snapshot, neural_string_keys
from bindings.transformer_public_native_door import _transformer_fields, _transformer_state_words, _transformer_configuration, _transformer_same_words
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.generation_metadata_pyapi_door import GenerationLayerSnapshot
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_module_fields
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME
from core.hostptr import f32_ptr
from transformer.block_options import BlockOptions
from bindings.py_cpython_table import cpython_table
from bindings.py_model_records import records_stampable, model_generation, ContentRecord, content_value, content_same
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor
from core.causal_lm_entry_table_sessions import (
    causal_sessions_ref, CAUSAL_SESSIONS_TRANSFORMER, CAUSAL_SESSION_DECODE_CREATE, CAUSAL_SESSION_DECODE_OPEN,
    CAUSAL_SESSION_DECODE_OPEN15, CAUSAL_SESSION_DECODE_EXPORT, CAUSAL_SESSION_DECODE_LOAD,
    CAUSAL_SESSION_DECODE_CLOSE, CAUSAL_SESSION_DECODE_PIN, CAUSAL_SESSION_DECODE_UNPIN,
    CAUSAL_SESSION_DECODE_PINNED, CAUSAL_DECODE_PINNED_STEP, CAUSAL_DECODE_PINNED_RUN,
    CAUSAL_DECODE_PINNED_FORWARD,
)


#: Lane restore-neural-call-paths (2026-11-13; AGENTS.md "A measured winner is
#: never deleted on reasoning"): the per-token door of a decode step. Default
#: (A): a (B, d_model) step input is admitted as is and its address handed to
#: the session (`_neural_input_rows`), as 0.8.55's per-token path did (one
#: upload, one llama_decoder_layer_forward, one download, one wait per token;
#: python/_transformer_impl.py at 0e4de0f46e:1788). Board on record (NV L40S):
#: transformer-decode 1105 ms (0.8.55) -> 1455 (0.8.66, native step identical
#: to 0.8.55, so the door) -> 1406 (candidate 4c2340386abc); MI300X 2445 ->
#: 2699 at 0.8.66. `-D MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR` keeps the
#: door of d37be875fc (B arm of the owed A/B, owed-pre0867.json row
#: wave2-transformer-decode-loop, formerly restore-transformer-decode-door): a (B, 1, d_model) Array view rebuilt per
#: token (`_neural_input`). Same bytes at the same address in both arms, so no
#: output bit moves.
#:
#: Lane fix-transformer-decode-door (2026-11-13, owed row
#: wave2-transformer-decode-loop): the same define now decides the whole
#: per-token door of the GPU session. Default (A): the session is a native
#: registry decode session (`transformer/causal_session_entry.mojo`
#: DECODE_SESSIONS, the CausalLM layer sessions' registry) and every step,
#: forward, decode, export, load and close is one native call through the
#: CausalLM session table (`causal_sessions_ref`, resolved once at open; GIL
#: released; Array envelopes, no heap), with no Python call into
#: `_mojolearn_transformer` and no Python argument lists per token; the step
#: op's position word is written lazily (once before any operation that can
#: rebuild the resident entry, not once per token:
#: transformer/attn_step_session.mojo `_flush_word`). B
#: (`-D MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR`): the former door, a
#: Python-owned `TransformerDecodeSession` reached through the binding entries
#: `transformer_decode_session_*` with `py_ints` argument lists, the (B, 1,
#: d_model) Array view per token and the per-token word fill. Both arms run
#: `_session_step_run` / `_session_decode_run` / `_session_forward_run` on the
#: same `AttnStepArm` (same `llama_decoder_layer_forward`, same order, same
#: stages), so no output bit moves on any vendor.
comptime TRANSFORMER_DECODE_PER_TOKEN_DOOR = is_defined["MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR"]()

#: Lane fix-transformer-decode-step-admission (2026-11-13, owed row
#: fix-transformer-decode-step-admission; best-ever history audit row 3):
#: what a GPU step admits per token. Default (A), the 0.8.55 shape
#: (python/mojolearn/_transformer_impl.py at 0e4de0f46e:1788-1813: one
#: binding call per token, no per-token admission): when the session's
#: `_state` is the state object admitted at init and the handle holds its
#: registry word, a step / decode / forward reads the handle, admits only x,
#: allocates y, reads and writes `state.cached_tokens` (exact int; the
#: resident session refuses a word that differs from its cache) and makes one
#: native call on the pinned session (`_transformer_decode_run_admitted`).
#: The owner record read, the block/state generation words, the option-tail
#: content check, the per-token NativeModules, the address re-snapshots of x
#: and y and the state-word re-stamp are not run per token; the owner and
#: configuration checks still run at sync / load / close / repr, and a
#: replaced `_state` takes the full path (refused). B
#: (`-D MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION`): the per-token
#: admission of lanes N177 / fix-transformer-decode-door (owner record,
#: generation words, tail content, a new NativeModules per token). Same native
#: call on the same AttnStepArm in both arms: no output bit moves on any vendor.
#: Board on record NV L40S 1105 ms (0.8.55) -> 1406 (candidate 4c2340386abc).
comptime TRANSFORMER_DECODE_PER_TOKEN_ADMISSION = is_defined["MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION"]()


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
    # Lane fix-transformer-decode-door (default arm): the CausalLM session
    # table entry of this binding's image, resolved once at open, and the
    # registry handle word of the GPU decode session (0: none; the host route
    # and the per-token-door arm keep 0).
    var entry: EntryRef
    var word: Int
    # Hot-path audit 2026-11-13: the door's lease on that session
    # (CAUSAL_SESSION_DECODE_PIN), taken once right after create and dropped
    # only when this handle is destroyed (an in-flight call on another thread
    # holds this handle, so the lease outlives it). Every step / decode /
    # forward runs on it: no registry lookup per token. 0: none.
    var pin: Int
    # Lane fix-transformer-decode-step-admission (default arm): the module
    # admissions of one step's x admission, built once at init and cleared at
    # the start of every step (never held across a GIL window: the step
    # releases the GIL only after admission), so a token allocates no map.
    var mods: NativeModules
    # The address of the exact `int` class, read once at init: the admitted
    # step reads `state.cached_tokens` only when it is an exact int (no
    # conversion hook can run), as the full admission requires.
    var int_class: Int

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
        self.entries = List[PythonObject]()
        self.block_generation = 0
        self.state_generation = 0
        self.width = 0
        self.batch = 0
        self.params = List[Int]()
        self.live_state = PythonObject(None)
        self.tail_content = ContentRecord()
        self.entry = entry_ref_unresolved()
        self.word = 0
        self.pin = 0
        self.mods = NativeModules()
        self.int_class = 0

    def __deinit__(deinit self):
        # A registry session the Python owner never closed (an open that
        # failed after create, or a dropped owner) is closed here, as the
        # Python-owned session object released its device state when
        # collected. A refused close (busy) cannot be reported.
        if self.word != 0:
            try:
                _transformer_decode_native_close(self.entry, self.word)
            except:
                pass
        if self.pin != 0:
            try:
                var words: List[Int] = [self.pin]
                _ = _transformer_decode_native_lists(self.entry, CAUSAL_SESSION_DECODE_UNPIN, List[Int](), words)
            except:
                pass

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
comptime _TRANSFORMER_DECODE_WORDS = SharedSlot[TransformerDecodeWords, "MojolearnTransformerDecodeWordsV2"]


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
comptime _TRANSFORMER_DECODE_HOST_DECODE = 8


def _transformer_decode_entry_names() -> List[String]:
    # Indexed by the _TRANSFORMER_DECODE_* words above.
    return [String("transformer_decode_session_step"), String("transformer_decode_session_forward"),
            String("transformer_decode_session_load_state"), String("transformer_decode_session_export_state"),
            String("transformer_decode_session_close"), String("transformer_decode_step"), String("transformer_forward"),
            String("transformer_decode_session_decode"), String("transformer_decode_steps")]


def _transformer_decode_keep_entries(mut handle: TransformerDecodeHandle) raises:
    # N181: resolve once at open what every step looked up by name. A missing
    # or non-compiled entry is kept as None and refused at its use, with the
    # message _transformer_decode_entry gives, as before.
    var function_type = py_type(native_loaded_module("builtins").len)
    var names = _transformer_decode_entry_names()
    handle.entries.clear()
    for name in names:  # small-loop(nine fixed decode entry names)
        var entry = py_get(handle.binding, PythonObject(name))
        handle.entries.append(entry if py_type(entry) is function_type else PythonObject(None))


def _transformer_decode_kept(handle: TransformerDecodeHandle, index: Int) raises -> PythonObject:
    var entry = handle.entries[index]
    if entry is None:
        raise Error("NotImplementedError: Transformer decode lacks its connected compiled " + _transformer_decode_entry_names()[index] + " entry")
    return entry


def _transformer_decode_native_key(binding: PythonObject) raises -> CausalEntryKey:
    # The (tier, vendor) key of the connected GPU image: its compiled vendor
    # and mode words, read once per connected entry pair (N192 cache).
    var vendor = String()
    var mode = 0
    _transformer_decode_words(binding, "transformer", vendor, mode)
    return CausalEntryKey(mode, causal_entry_vendor(vendor))


def _transformer_decode_native_lists(entry: EntryRef, op: Int, a: List[Int], words: List[Int]) raises -> Int:
    # One session-table call for the admission-time operations (create, open,
    # export, load): GIL released around the device work, as the former
    # binding entries released it. The caller pins every address.
    var floats = List[Float64]()
    var result = 0
    with GILReleased(Python()):
        result = entry_ref_call(entry, op, a, words, floats)
    return result


def _transformer_decode_native_close(entry: EntryRef, word: Int) raises:
    var words: List[Int] = [word]
    _ = _transformer_decode_native_lists(entry, CAUSAL_SESSION_DECODE_CLOSE, List[Int](), words)


def _transformer_decode_native_run(entry: EntryRef, kind: Int, px: Int, py: Int, pin: Int, count: Int, cached: Int) raises -> Int:
    # The per-token call (CAUSAL_DECODE_PINNED_STEP: count 0) and the
    # N-token / L-token calls (RUN / FORWARD) on the pinned session: one
    # indirect call over envelopes on this frame, GIL released. The session
    # image reads them in place (no native List, no registry lookup, no lock:
    # transformer/causal_session_entry.mojo `_decode_pinned`).
    var a: Array[Int, 2] = [px, py]
    var f: Array[Float64, 0] = []
    var p: Array[Int, 4] = [pin, kind, count, cached]
    var result = 0
    with GILReleased(Python()):
        result = entry_ref_call(entry, CAUSAL_SESSION_DECODE_PINNED, a, p, f)
    _ = p
    _ = a
    _ = f
    return result


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


comptime _TRANSFORMER_DECODE_BINDINGS = SharedSlot[_TransformerDecodeBindings, "MojolearnTransformerDecodeBindingsV2"]


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
        unsafe_memcpy(dest=result.unsafe_ptr(), src=f32_ptr(Int(py=array._addr)), count=len(result))
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


def _transformer_decode_admitted(session: PythonObject) raises -> PythonObject:
    # Default arm (TRANSFORMER_DECODE_PER_TOKEN_ADMISSION off): the session's
    # handle when a step may skip the per-token admission, else None (the
    # full path below runs and raises what it raised before). Two words
    # decide it: the handle holds its registry session (word and lease taken
    # at init) and the session's `_state` is the state object admitted at
    # init. The session's exact class was checked by the caller; its
    # instance dictionary is read natively (`py_fields`, no attribute hook).
    var fields = py_fields(session)
    var hobj = py_get(fields, "_handle")
    if hobj is None:
        return PythonObject(None)
    var state = py_get(fields, "_state")
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    if not h[].open or h[].word == 0 or h[].pin == 0 or state is not h[].state:
        return PythonObject(None)
    return hobj


def _transformer_decode_run_admitted(hobj: PythonObject, input: PythonObject, step: Bool, decode: Bool, context: PythonObject) raises -> PythonObject:
    # The admitted per-token path (0.8.55 shape, _transformer_impl.py at
    # 0e4de0f46e:1788-1813): x admitted (the same admit_block_f32 BLOCK_READ
    # refusals), one y Array, x and y addresses read from the admitted
    # Arrays (both C-order, held below past the call), cached_tokens read
    # from and written to the state's fields, one native call on the pinned
    # session (GIL released): upload, llama_decoder_layer_forward, download,
    # one wait. Nothing else per token.
    var h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    var width = h[].width
    var rows = h[].batch
    var batch = 0
    var length = 0
    # A kept map never outlives a GIL window: cleared before this admission.
    h[].mods.released()
    var x = _neural_input_rows(h[].mods, input, width, step, context, batch, length)
    if batch != rows:
        raise Error("TransformerDecodeSession: input rows differ from the state")
    var y = generation_array([batch, length, width])
    var px = Int(py=x._addr)
    var py = Int(py=y._addr)
    h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
    var live = h[].live_state
    var int_class = h[].int_class
    var entry = h[].entry
    var pin = h[].pin
    var word = py_get(live, "cached_tokens")
    var word_class = py_type(word)
    if Int(word_class._obj_ptr) != int_class:
        raise Error("TypeError: TransformerState cached_tokens requires an ordinary int")
    # The resident session refuses a word that differs from its cache.
    var cached = Int(py=word)
    var kind = CAUSAL_DECODE_PINNED_RUN if decode else (CAUSAL_DECODE_PINNED_STEP if step else CAUSAL_DECODE_PINNED_FORWARD)
    cached = _transformer_decode_native_run(entry, kind, px, py, pin, length, cached)
    py_setitem(live, "cached_tokens", PythonObject(cached))
    _ = x^
    _ = word_class^
    _ = word^
    _ = hobj
    return y


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
    var int_class = py_type(PythonObject(0))
    handle.int_class = Int(int_class._obj_ptr)
    _ = int_class^
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
        # TOMBSTONE: TransformerDecodeHandle.native_handle (the public decode session's descriptor-provider arm under MOJOLEARN_NEURAL_NATIVE_DISPATCH, SUPERSEDED by the step-op session) deleted 2026-11-09 by lane x4-transformer-on-executor; code recoverable at f59d913b0b.
        # Restore: git apply experiments/removed/TRANSFORMER_DECODE_PROVIDER_ARM.patch
        # One device open: a float32 session runs on the attention step op
        # (`_session_open_run`); fixed15_v1 keeps its planes (`_session_open_run_int15`).
        comptime if TRANSFORMER_DECODE_PER_TOKEN_DOOR:
            handle.device = _transformer_decode_entry(binding, "transformer_decode_session_create")()
            cached = Int(py=py_call(_transformer_decode_entry(binding, "transformer_decode_session_open_int15" if packed else "transformer_decode_session_open_step_op"),
                handle.device, py_ints(a), py_ints(open_p)))
        else:
            # A registry decode session of this binding's image (the CausalLM
            # layer sessions' registry), the same two opens. A failed open
            # closes the handle when `handle` is destroyed (its __deinit__).
            handle.entry = causal_sessions_ref(CAUSAL_SESSIONS_TRANSFORMER, _transformer_decode_native_key(binding))
            handle.word = _transformer_decode_native_lists(handle.entry, CAUSAL_SESSION_DECODE_CREATE, List[Int](), List[Int]())
            # Resolve the session once (hot-path audit 2026-11-13): the lease
            # every per-token call runs on, dropped by the handle's __deinit__.
            var pin_words: List[Int] = [handle.word]
            handle.pin = _transformer_decode_native_lists(handle.entry, CAUSAL_SESSION_DECODE_PIN, List[Int](), pin_words)
            var open_words: List[Int] = [handle.word]
            for i in range(len(open_p)):  # driver-loop: resident shape words after the handle
                open_words.append(open_p[i])
            cached = _transformer_decode_native_lists(handle.entry,
                CAUSAL_SESSION_DECODE_OPEN15 if packed else CAUSAL_SESSION_DECODE_OPEN, a, open_words)
            # The device route's marker (`native` reads it; `device is not
            # None` selects the device route below): the handle word.
            handle.device = PythonObject(handle.word)
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
        unsafe_memcpy(dest=handle.k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
        unsafe_memcpy(dest=handle.v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
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


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    return transformer_decode_run(session, input, step, False, context)


def transformer_decode_run(session: PythonObject, input: PythonObject, step: Bool, decode: Bool, context: PythonObject) raises -> PythonObject:
    # `decode` (D2): x is (B, N, d_model) and the N tokens run as N steps in
    # one call; `step` is the one-token case, neither is a forward of L tokens.
    comptime if not TRANSFORMER_DECODE_PER_TOKEN_ADMISSION and not TRANSFORMER_DECODE_PER_TOKEN_DOOR:
        var admitted = _transformer_decode_admitted(session)
        if admitted is not None:
            return _transformer_decode_run_admitted(admitted, input, step, decode, context)
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
    var batch = 0
    var length = 0
    var x = PythonObject(None)
    comptime if TRANSFORMER_DECODE_PER_TOKEN_DOOR:
        x = _neural_input(input, width, step, context)
        var shape = x.shape
        batch = Int(py=shape[0])
        length = Int(py=shape[1])
    else:
        var mods = NativeModules()
        x = _neural_input_rows(mods, input, width, step, context, batch, length)
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
    var native_word = h[].word
    if native_word != 0:
        # Default arm: one native call through the session table per step
        # (or per decode / forward call) on the pinned session; no Python
        # call, no Python lists, no registry lookup.
        var entry = h[].entry
        var kind = CAUSAL_DECODE_PINNED_RUN if decode else (CAUSAL_DECODE_PINNED_STEP if step else CAUSAL_DECODE_PINNED_FORWARD)
        cached = _transformer_decode_native_run(entry, kind, px, py, h[].pin, length, cached)
    elif h[].device is not None and decode:
        # The provider descriptor wraps the same session object; the decode
        # entry is called on it directly.
        cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_DECODE),
            h[].device, py_ints([px, py]), py_ints([length, cached])))
    elif h[].device is not None:
        var p: List[Int] = [cached] if step else [length, cached]
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
            # TOMBSTONE: _transformer_decode_host_tokens (the door's per-token loop of host step calls, SUPERSEDED by transformer_decode_steps) deleted 2026-11-09 by lane x4-transformer-on-executor; code recoverable at f59d913b0b.
            # Restore: git apply experiments/removed/TRANSFORMER_DECODE_HOST_TOKEN_LOOP.patch
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_HOST_DECODE), py_ints(a), py_ints(p)))
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
    if h[].word != 0:
        var a: List[Int] = [pieces[0], pieces[1]]
        var words: List[Int] = [h[].word]
        if load:
            words.append(cached)
        var entry = h[].entry
        cached = _transformer_decode_native_lists(entry, CAUSAL_SESSION_DECODE_LOAD if load else CAUSAL_SESSION_DECODE_EXPORT, a, words)
    elif h[].device is not None:
        var a: List[Int] = [pieces[0], pieces[1]]
        var p: List[Int] = [cached] if load else []
        if load:
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_LOAD), h[].device, py_ints(a), py_ints(p)))
        else:
            cached = Int(py=py_call(_transformer_decode_kept(h[], _TRANSFORMER_DECODE_EXPORT), h[].device, py_ints(a)))
    else:
        if pieces[2] != len(h[].k_cache) or pieces[2] != len(h[].v_cache):
            raise Error("TransformerDecodeSession: cache dimensions differ from the retained native session")
        if load:
            unsafe_memcpy(dest=h[].k_cache.unsafe_ptr(), src=f32_ptr(pieces[0]), count=pieces[2])
            unsafe_memcpy(dest=h[].v_cache.unsafe_ptr(), src=f32_ptr(pieces[1]), count=pieces[2])
        else:
            unsafe_memcpy(dest=f32_ptr(pieces[0]), src=h[].k_cache.unsafe_ptr(), count=pieces[2])
            unsafe_memcpy(dest=f32_ptr(pieces[1]), src=h[].v_cache.unsafe_ptr(), count=pieces[2])
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
    if h[].word != 0:
        var entry = h[].entry
        var word = h[].word
        # The word is cleared after the close succeeds; a refused (busy)
        # close keeps it, so the handle's destruction retries it.
        _transformer_decode_native_close(entry, word)
        h = hobj.downcast_value_ptr[TransformerDecodeHandle]()
        h[].word = 0
    elif h[].device is not None:
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
