"""Source contracts only; public session runtime validation is pending."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_transformer_session_public_methods_are_single_native_connections():
    tree=ast.parse((ROOT/'python/mojolearn/_transformer_impl.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='TransformerDecodeSession')
    methods={node.name:node for node in cls.body if isinstance(node,ast.FunctionDef)}
    for name in ('__init__','step','forward','sync_state','load_state','close','discard','state','is_open','__enter__','__exit__','__repr__','__del__'):
        body=methods[name].body
        if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant):
            body=body[1:]
        assert len(body)==1, name
        assert isinstance(body[0],ast.Return), name
        assert ast.unparse(body[0].value.func)=='_native_neural_bridge.RUNTIME.transformer_decode_native_call', name
    for retired in ('_host_call','_require_open','_release'):
        assert retired not in methods
    assert not any(isinstance(node,(ast.For,ast.While,ast.ListComp)) for node in ast.walk(cls))


def test_native_session_owner_preserves_cpu_copies_gpu_provider_and_cleanup():
    source=(ROOT/'providers/python/transformer_decode_pyapi_door.mojo').read_text()
    assert 'var weights: List[List[Float32]]' in source
    assert 'var k_cache: List[Float32]' in source
    assert 'snapshot.packed.resolve(keep)' in source
    # The descriptor-provider arm (ops 18-24) was deleted from the door by 8c1cbf86b5; the
    # session runs its one device open and the decode program.
    assert 'finally:\n            transformer_decode_release(session, context)' in source
    assert 'keys.publish()' in source and 'vals.publish()' in source
    assert 'if load:\n        _transformer_decode_profile(fields, slots)' in source
    # Registered by the public neural host binding since the 2026-11-07 host
    # split; the tokenizer host keeps a native refusal under the name.
    assert 'transformer_decode_native_c, "transformer_decode_native_call"' in (ROOT/'bindings/_mojolearn_runtime_neural_host.mojo').read_text()
    assert 'runtime_classical_unavailable_c, "transformer_decode_native_call"' in (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert '"transformer_session_class": TransformerDecodeSession' in (ROOT/'python/mojolearn/models/causal_lm.py').read_text()


def test_resident_whole_model_uses_original_run_and_atomic_group_reservation():
    from check_generation_native_boundary import audit
    _, failures=audit(ROOT,('transformer.native_lm_resident_dispatch','core.native_session_group'))
    assert not failures, failures
    source=(ROOT/'transformer/native_lm_resident_dispatch.mojo').read_text()
    reserve=source.split('def _lm_group_reserve(')[1].split('def _lm_group_release')[0]
    assert reserve.index('_require_session_open(layer[])') < reserve.index('].busy = True')
    assert '_lm_run(owner[], group.layers' in source
    assert '_lm_forward_run(owner[], group.layers' in source
    assert 'owner[].usable = False' in source
    assert '].usable = False' in source
    # The provider boundary (transformer_resident_pyapi.mojo, ops 30-32) and the public door's
    # provider-session calls were deleted (9b26007213, 58fbf80607): sessions are native handles.
    public=(ROOT/'bindings/causal_lm_public_door.mojo').read_text()
    assert 'transformer_resident_lm_run(provider,' not in public


def test_decode_captures_fields_and_uses_retained_native_entry_dictionary():
    source=(ROOT/'providers/python/transformer_decode_pyapi_door.mojo').read_text()
    # The one instance-dictionary read is the admitted per-token check (ce964f069e), native and
    # hook-free; no other function reads owner fields that way.
    assert source.count('py_fields(') == 1
    assert 'py_fields(' in source.split('def _transformer_decode_admitted(')[1].split('\ndef ')[0]
    assert '_native_function(' not in source
    assert 'py_call(bi.hasattr' not in source
    # N192: the module namespace is admitted once per module identity by the
    # shared neural_descriptor helper (no whole-dictionary copy per open).
    assert 'return neural_descriptor_module_fields(module)' in source
    assert 'py_call(bi.dict, fields)' not in source.split('def _transformer_decode_module(')[1].split('def _transformer_decode_entry(')[0]
    assert 'self.block = block' in source and 'self.state = state' in source
    assert 'owner fields were replaced' in source
    assert 'configuration/cache dimensions changed' in source
    cache=source.split('struct TransformerDecodeCache',1)[1].split('def _transformer_decode_profile',1)[0]
    assert cache.index('admit_native_buffer_exporter(value)') < cache.index('AdmittedBorrowedBuffer(value, True)')
    assert 'layout.copy_to(self.address)' in cache
    assert 'layout.value().copy_from(self.address)' in cache
    # N181: step, state and close entries are resolved once at open and kept.
    assert '_transformer_decode_entry(h[].binding' not in source
    assert '_transformer_decode_keep_entries(handle)' in source
    assert '_transformer_decode_kept(h[], _TRANSFORMER_DECODE_STEP if step else _TRANSFORMER_DECODE_FORWARD)' in source
    assert 'len(args) != 3 or py_type(args[0]) is not bi.str' in source


def test_decode_session_keeps_binding_words():
    # N192: vendor and mode words are read once per compiled entry pair. The provider
    # native_handle (N186) and the value-handle cache went with the provider arm
    # (8c1cbf86b5, 58fbf80607).
    source=(ROOT/'providers/python/transformer_decode_pyapi_door.mojo').read_text()
    assert 'transformer_resident_decode_call(' not in source
    words=source.split('def _transformer_decode_words(')[1].split('comptime _TRANSFORMER_DECODE_STEP')[0]
    assert 'g[].vendor_entries[i] is vendor_entry and g[].mode_entries[i] is mode_entry' in words
    provider=(ROOT/'providers/python/neural_descriptor_pyapi_door.mojo').read_text()
    assert 'def neural_descriptor_value_handle(' not in provider
