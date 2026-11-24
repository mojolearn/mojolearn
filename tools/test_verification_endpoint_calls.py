import ast
from verification_endpoint_calls import Calls


def rows():
    return [dict(endpoint='mojolearn.ops.Session.'+op, op=op, kind='method',
                 aliases=['mojolearn.ops.Session']) for op in ('__init__','mul','backward')]


def test_with_receiver_and_argument_helper_reach_the_public_endpoint():
    tree = ast.parse('''
def helper(session, x):
    return session.mul(x, x)
def fixture(ml):
    ops = ml.ops
    with ops.Session() as s:
        loss = helper(s, 1)
        s.backward(loss)
''')
    helper, fixture = tree.body
    visitor = Calls(rows(), {'helper':helper})
    visitor.function(fixture, {'ml':{'mojolearn'}})
    assert visitor.seen == {r['endpoint'] for r in rows()}


def test_unrelated_receiver_does_not_gain_coverage_from_a_method_name():
    node = ast.parse('''
def fixture(ml):
    unused = ml.ops.Session
    something_else.mul(1, 2)
''').body[0]
    visitor = Calls(rows())
    visitor.function(node, {'ml':{'mojolearn'}})
    assert not visitor.seen


def test_relative_import_in_installed_verifier_resolves_public_class():
    node = ast.parse('''
def fixture():
    from . import ops
    with ops.Session() as s:
        s.backward(s.mul(1, 2))
''').body[0]
    visitor = Calls(rows())
    visitor.function(node)
    assert visitor.seen == {r['endpoint'] for r in rows()}


def test_nested_audited_callable_records_execution_not_just_passing():
    node = ast.parse('''
def fixture():
    from . import ops
    from ._verify_native_runtime import audit_native_call
    def native(label, operation, *args):
        return audit_native_call(operation, *args).assert_native()
    def opened(label, operation, *args):
        yield native(label, operation, *args)
    with ops.Session() as s:
        with opened('test', s.mul, 1, 2):
            pass
''').body[0]
    visitor = Calls(rows())
    visitor.function(node)
    assert visitor.seen == {'mojolearn.ops.Session.__init__', 'mojolearn.ops.Session.mul'}


def _rows(cls, ops, kind='method'):
    return [dict(endpoint=cls + '.' + op, op=op, kind=kind, aliases=[cls]) for op in ops]


def test_instance_call_is_dunder_call_and_operators_reach_their_dunders():
    rows = _rows('mojolearn.models.HFConfig', ('__init__', '__call__', '__contains__', '__getitem__'))
    node = ast.parse('''
def fixture(ml):
    cfg = ml.models.HFConfig({})
    cfg(1)
    assert 'a' in cfg and 'b' not in cfg
    width = cfg['hidden_size']
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_calling_the_class_is_init_not_call():
    rows = _rows('mojolearn.LayerNorm', ('__init__', '__call__'))
    node = ast.parse('''
def fixture(ml):
    ml.LayerNorm(4)
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {'mojolearn.LayerNorm.__init__'}


def test_loader_and_import_model_results_are_typed_instances():
    rows = (_rows('mojolearn.models.CausalLM', ('generate', '__call__'))
            + _rows('mojolearn.importers._program.ImportedProgram', ('run',))
            + [dict(endpoint='mojolearn.importers.import_model', op='call', kind='function',
                    aliases=['mojolearn.importers.import_model'])])
    node = ast.parse('''
def fixture(ml, path):
    model = ml.models.CausalLM.load(path)
    model.generate([1], 2)
    model([1])
    program = ml.importers.import_model(path)
    program.run([1.0])
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_a_container_of_instances_does_not_reach_an_unrelated_dunder():
    rows = _rows('mojolearn.KMeans', ('__init__', 'fit'))
    node = ast.parse('''
def fixture(ml, X):
    models = [ml.KMeans(2)]
    models[0].fit(X)
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {'mojolearn.KMeans.__init__', 'mojolearn.KMeans.fit'}
