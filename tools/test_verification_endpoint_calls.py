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
