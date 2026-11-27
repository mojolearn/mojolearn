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


def test_side_view_helper_returns_either_branch_of_a_conditional():
    rows = _rows('mojolearn.ops.Session', ('__init__', 'cos'))
    tree = ast.parse('''
def _ops():
    from . import ops, _verify_device as vd
    side = vd.current_side()
    return ops if side is None else vd.view(ops, side)
def case():
    ops = _ops()
    with ops.Session() as s:
        s.cos(1)
''')
    helper, case = tree.body
    visitor = Calls(rows, {'_ops': helper})
    visitor.function(case)
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_partial_of_the_shared_audited_helpers_executes_the_operation():
    rows = _rows('mojolearn.ops.Session', ('__init__', 'load_partitioned', 'mul'))
    node = ast.parse('''
def verify(path):
    from functools import partial
    from . import ops
    from ._verify_native_runtime import audited, audited_context
    native = partial(audited, [])
    opened = partial(audited_context, [])
    with ops.Session() as s:
        native("mul", s.mul, 1, 2)
        with opened("load", s.load_partitioned, path) as restored:
            pass
''').body[0]
    visitor = Calls(rows)
    visitor.function(node)
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_passing_a_method_to_a_partial_of_another_helper_is_not_a_call():
    rows = _rows('mojolearn.ops.Session', ('__init__', 'mul'))
    node = ast.parse('''
def verify():
    from functools import partial
    from . import ops
    keep = partial(print, 1)
    with ops.Session() as s:
        keep("mul", s.mul)
''').body[0]
    visitor = Calls(rows)
    visitor.function(node)
    assert visitor.seen == {'mojolearn.ops.Session.__init__'}


def test_attribute_assignment_deletion_equality_and_hash_reach_their_endpoints():
    rows = (_rows('mojolearn.RMSprop', ('__init__', 'lr_schedule:get', 'lr_schedule:set', 'lr_schedule:delete'))
            + _rows('mojolearn.LanguageModelConfig', ('__init__', '__eq__', '__hash__'))
            + _rows('mojolearn.LanguageModelConfig', ('__delattr__',), kind='installed_method'))
    node = ast.parse('''
def fixture(ml, sched):
    opt = ml.RMSprop([1])
    opt.lr_schedule = sched
    assert opt.lr_schedule is not None
    del opt.lr_schedule
    cfg = ml.LanguageModelConfig()
    same = cfg == ml.LanguageModelConfig()
    h = hash(cfg)
    delattr(cfg, "batch")
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_reading_an_installed_attribute_runs_its_descriptor():
    rows = _rows('mojolearn.Array', ('__init__',)) + _rows('mojolearn.Array', ('flags',), kind='installed_method')
    node = ast.parse('''
def fixture(ml):
    a = ml.Array((2,))
    f = a.flags
''').body[0]
    visitor = Calls(rows)
    visitor.function(node, {'ml': {'mojolearn'}})
    assert visitor.seen == {r['endpoint'] for r in rows}


def test_kernel_infix_call_is_attributed_to_its_left_operand():
    endpoints = (_rows('mojolearn.RBF', ('__init__', '__add__', '__mul__'))
                 + _rows('mojolearn.Matern', ('__init__', '__add__', '__mul__')))
    fixture = ast.parse('''
def fixture(ml):
    r = ml.RBF()
    m = ml.Matern()
    unused = r + (m * r)
''').body[0]
    visitor = Calls(endpoints)
    visitor.function(fixture, {'ml': {'mojolearn'}})
    assert {x for x in visitor.seen if not x.endswith('__init__')} == {
        'mojolearn.RBF.__add__', 'mojolearn.Matern.__mul__'}


def test_shipped_fixture_uses_its_own_nested_helpers():
    fixture, inner = ast.parse('''
def light_case(session):
    return nested(session)
def nested(session):
    return session.mul(1, 2)
''').body
    outer, shadowed = ast.parse('''
def lane(ml):
    light_case(ml.ops.Session())
def nested(session):
    session.backward(4)
''').body
    visitor = Calls(rows(), {'light_case': fixture, 'nested': shadowed}, {fixture: {'nested': inner}})
    visitor.function(outer, {'ml': {'mojolearn'}})
    assert 'mojolearn.ops.Session.mul' in visitor.seen
    assert 'mojolearn.ops.Session.backward' not in visitor.seen
