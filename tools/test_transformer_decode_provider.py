"""Source contracts only; public session runtime validation is pending."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_transformer_session_public_methods_are_single_native_connections():
    tree=ast.parse((ROOT/'python/mojolearn/_transformer_impl.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='TransformerDecodeSession')
    methods={node.name:node for node in cls.body if isinstance(node,ast.FunctionDef)}
    for name in ('__init__','step','forward','sync_state','load_state','close','discard','state','is_open','__enter__','__exit__','__repr__'):
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
    source=(ROOT/'providers/python/transformer_decode_pyapi.mojo').read_text()
    assert 'var weights: List[List[Float32]]' in source
    assert 'var k_cache: List[Float32]' in source
    assert 'snapshot.packed.resolve(keep)' in source
    for operation in ('19 if packed else 18','20 if step else 21','23 if load else 22'):
        assert operation in source
    assert 'finally:\n            transformer_decode_release(session)' in source
    assert '_writeback(copies)' in source
    assert 'if load:\n        _transformer_decode_profile(fields, slots)' in source
    assert 'transformer_decode_native_c, "transformer_decode_native_call"' in (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
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
    boundary=(ROOT/'providers/python/transformer_resident_pyapi.mojo').read_text()
    assert 'List[NativeSessionGroup]()' in boundary
    assert '30, 31, 32, [], [], []' in boundary
    public=(ROOT/'bindings/causal_lm_public.mojo').read_text()
    for name in ('transformer_resident_lm_run','transformer_resident_lm_open','transformer_resident_lm_close'):
        assert name+'(provider,' in public
