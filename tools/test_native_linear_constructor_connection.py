"""Constructor connection/source guards; these do not claim native runtime success."""
import ast
import inspect
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
CONNECTOR=ROOT/'python/mojolearn/_native_linear_constructors.py'


def selected():
    tree=ast.parse((ROOT/'python/mojolearn/_native_params_setter.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='_CLASSES')


def fixture():
    source=ast.parse((ROOT/'python/mojolearn/_expansion_linear.py').read_text())
    helpers=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_glm_init']
    namespace={}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<constructor oracle>','exec'),namespace)
    classes=[]; signatures={}
    for node in source.body:
        if not isinstance(node,ast.ClassDef) or node.name not in selected():continue
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<constructor oracle>','exec'),namespace)
        original=namespace['__init__']
        signature=inspect.signature(original)
        signature=signature.replace(parameters=[*signature.parameters.values(),inspect.Parameter('numeric_mode',inspect.Parameter.KEYWORD_ONLY,default=None)])
        original.__signature__=signature
        cls=type(node.name,(),{'__init__':original});classes.append(cls);signatures[cls]=signature
    order=selected();classes.sort(key=lambda cls:order.index(cls.__name__))
    body=[n for n in ast.parse(CONNECTOR.read_text()).body if isinstance(n,ast.FunctionDef)]
    calls=[];context={'plans':{},'ready':False}
    linear=ModuleType('constructor_fixture._expansion_linear')
    for cls in classes:setattr(linear,cls.__name__,cls)
    ns={'linear':linear,'_CLASSES':tuple(classes),'_CONTEXT':context,'_ENTRY':lambda *args:calls.append(args),'__package__':'constructor_fixture'}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(CONNECTOR),'exec'),ns)
    return ns,signatures,calls


def test_install_preserves_exact_signature_defaults_and_only_calls_binding():
    ns,signatures,calls=fixture();ns['install']()
    for cls,expected in signatures.items():
        assert inspect.signature(cls.__init__)==expected
        assert not hasattr(cls.__init__,'__wrapped__')
        obj=cls(alpha=object(),numeric_mode='identical')
        assert len(calls[-1])==4 and calls[-1][0] is obj
        assert calls[-1][3] is ns['_CONTEXT']
        assert ns['_CONTEXT']['plans'][cls]['names']==tuple(expected.parameters)[1:]
    assert len(calls)==19


def test_connection_uses_final_setter_layout_and_does_not_construct():
    ns,signatures,calls=fixture();ns['install']()
    setters={'plans':{cls:{'constructor':dict(init=None,forest=None,initial_fields={},
        parameters=tuple(signature.parameters)[1:],mro=cls.__mro__)} for cls,signature in signatures.items()}}
    fake=ModuleType('constructor_fixture._native_params_setter');fake._CLASSES=selected()
    with patch.dict('sys.modules',{fake.__name__:fake}):ns['connect'](setters)
    assert not calls and ns['_CONTEXT']['ready'] and ns['_CONTEXT']['setters'] is setters
    for cls,plan in ns['_CONTEXT']['plans'].items():
        assert all(value is vars(base).get(name) for base,name,value in plan['field_hooks'])


def test_additional_constructor_work_requires_explicit_new_schema():
    ns,signatures,calls=fixture();ns['install']()
    setters={'plans':{cls:{'constructor':dict(init='extra-work',forest=None,parameters=tuple(signature.parameters)[1:])} for cls,signature in signatures.items()}}
    fake=ModuleType('constructor_fixture._native_params_setter');fake._CLASSES=selected()
    with patch.dict('sys.modules',{fake.__name__:fake}),pytest.raises(TypeError,match='field-only'):
        ns['connect'](setters)
    assert not calls and not ns['_CONTEXT']['ready']


def test_installer_precedes_all_layout_snapshots():
    text=(ROOT/'python/mojolearn/__init__.py').read_text()
    calls=['_install_linear_constructors(', '_connect_mode_metadata_classes(', '_connect_public_attributes(',
           '_connect_native_cv(', '_connect_params_setter(', '_connect_params_metadata_classes(', '_connect_linear_constructors(']
    positions=[text.index(call) for call in calls];assert positions==sorted(positions)


def test_constructor_has_one_native_call_and_no_python_lifecycle_fallback():
    tree=ast.parse(CONNECTOR.read_text());factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_initializer')
    wrapper=factory.body[0];assert isinstance(wrapper,ast.FunctionDef)
    assert len(wrapper.body)==1 and ast.unparse(wrapper.body[0])=='return _ENTRY(self, args, kwargs, _CONTEXT)'
    native=(ROOT/'bindings/linear_constructor_native.mojo').read_text()
    for forbidden in ('Python.import_module','.__init__(','.__setattr__(','.bind(', 'fields.clear', 'fields.pop', 'native_storage_allocate'):
        assert forbidden not in native
    assert native.index('_release_admit(fields[name]') < native.index('py_setitem(fields, order[i]')
    assert '_setter_plan(model, fields, context["setters"])' in native
    assert 'py_get(_class_fields(hooks[i][0]), hooks[i][1]) is not hooks[i][2]' in native
    assert 'values["numeric_mode"] is None and py_contains(fields, "numeric_mode")' in native
    registration=(ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert '\n        module.def_py_c_function(linear_constructor_c, "linear_constructor_native_call")\n' in registration


def test_field_order_matches_each_original_constructor_body():
    ns,_,_=fixture();ns['install']()
    tree=ast.parse((ROOT/'python/mojolearn/_expansion_linear.py').read_text())
    glm=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_glm_init')
    def fields(function):
        found=[]
        def target(node):
            if isinstance(node,ast.Tuple):
                for value in node.elts:target(value)
            else:
                assert isinstance(node,ast.Attribute) and ast.unparse(node.value)=='self'
                found.append(node.attr)
        for statement in function.body:
            if isinstance(statement,ast.Assign):
                for node in statement.targets:target(node)
            elif isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
                assert ast.unparse(statement.value.func)=='_glm_init'
                found.extend(fields(glm))
        return found
    originals={node.name:node for node in tree.body if isinstance(node,ast.ClassDef)}
    for cls,plan in ns['_CONTEXT']['plans'].items():
        function=next(n for n in originals[cls.__name__].body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assert plan['order']==tuple(fields(function))+('numeric_mode',)
