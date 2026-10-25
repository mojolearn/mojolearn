# SPDX-License-Identifier: Apache-2.0
"""Prep availability connection tests; no product imports or native builds."""
import ast
import functools
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import test_native_public_attribute_connection as public_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    package, mode, ns, context, modules, original = public_fixture.PublicAttributeConnection().fixture()
    mode_tree = ast.parse((ROOT / 'python/mojolearn/_mode.py').read_text())
    declaration = next(node for node in mode_tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'available_if')
    mode.functools = functools
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), 'available_if fixture', 'exec'), vars(mode))
    prep = ModuleType(package.__name__ + '._expansion_prep')
    prep.available_if = mode.available_if
    prep.SklearnTagsMixin = mode.PublicOutputsMixin
    tree = ast.parse((ROOT / 'python/mojolearn/_expansion_prep.py').read_text())
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            body = []
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    # Copy source descriptor kinds and Python method identities;
                    # no body, default expression or constructor is executed.
                    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')],
                        kwonlyargs=[], kw_defaults=[], defaults=[])
                    body.append(ast.FunctionDef(name=member.name, args=args,
                        body=[ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError', ctx=ast.Load()),
                            args=[ast.Constant(value='product method executed')], keywords=[]))],
                        decorator_list=[d for d in member.decorator_list if (isinstance(d, ast.Name) and d.id in ('classmethod', 'staticmethod', 'property')) or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'available_if')], returns=None, type_comment=None))
            declarations.append(ast.ClassDef(name=node.name, bases=node.bases, keywords=[],
                body=body or [ast.Pass()], decorator_list=[]))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
            declarations.append(node)
    # Source class inheritance and descriptor declarations, without imports,
    # native binding construction or execution of any product method body.
    exec(compile(ast.fix_missing_locations(ast.Module(body=declarations, type_ignores=[])),
                 'prep declarations fixture', 'exec'), vars(prep))
    for name in prep.__all__:
        vars(prep).setdefault(name, object())
    modules[prep.__name__] = prep
    return package, mode, ns, context, modules, original, prep


def test_all_24_declared_prep_slots_connect_with_exact_availability_plans():
    package, mode, ns, context, modules, original, prep = fixture()
    with patch.dict(sys.modules, modules):
        selected = ns['_prep_availability'](package, context['missing'])
        ns['connect_classes'](package)
    assert len(selected) == 24
    assert set(cls.__name__ for cls in selected) == set(ast.literal_eval(next(
        node.value for node in ast.parse((ROOT / 'python/mojolearn/tests/test_native_prep_params.py').read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLASSES' for t in node.targets))))
    for cls in selected:
        assert vars(cls)['__getattribute__'] is context['descriptor']
        assert cls in context['plans'] and cls not in context['pending']
        plan = context['plans'][cls]['availability'].get('fit_transform')
        if plan is not None:
            assert plan['check'] in selected[cls]
            assert plan['code'] is plan['check'].__code__
            assert plan['function'] is vars(plan['member'])['fn']
    newly_admitted = {cls.__name__ for cls in selected if context['plans'][cls]['availability']}
    assert newly_admitted == {'BernoulliNB', 'CategoricalNB', 'ComplementNB', 'GaussianNB',
                              'MultinomialNB', 'QuadraticDiscriminantAnalysis'}
    assert vars(prep._PrepBase)['__getattribute__'] is original
    assert vars(prep._Classifier)['__getattribute__'] is original


def test_cloned_descriptor_supported_but_unrelated_predicate_stays_pending():
    package, mode, ns, context, modules, original, prep = fixture()
    member = ns['_resolved'](prep.GaussianNB, 'fit_transform', context['missing'])
    # Public output/fit tracking installers copy available_if with the same
    # check and a wrapped fn, without calling __set_name__ on the copy.
    def wrapped(self):
        raise AssertionError('wrapped method executed')
    prep.GaussianNB.fit_transform = mode.available_if(member.check)(wrapped)
    prep.CategoricalNB.fit_transform = mode.available_if(lambda self: True)(wrapped)
    with patch.dict(sys.modules, modules):
        ns['connect_classes'](package)
    plan = context['plans'][prep.GaussianNB]['availability']['fit_transform']
    assert plan['attribute_name'] is context['missing']
    assert plan['function'] is wrapped
    assert vars(prep.CategoricalNB)['__getattribute__'] is original
    assert 'unconnected availability predicate' in context['pending'][prep.CategoricalNB]


def test_unknown_descriptor_and_other_families_are_not_blessed():
    package, mode, ns, context, modules, original, prep = fixture()
    calls = []
    class Descriptor:
        def __get__(self, instance, owner):
            calls.append(instance)
            raise AssertionError('descriptor executed during connection')
    prep.Normalizer.fit_transform = Descriptor()
    prep.RobustScaler.fit_transform = vars(prep._PrepBase)['fit_transform']
    with patch.dict(sys.modules, modules):
        ns['connect_classes'](package)
    assert not calls
    assert vars(prep.Normalizer)['__getattribute__'] is original
    assert 'unconnected descriptor' in context['pending'][prep.Normalizer]
    assert prep.RobustScaler not in context['plans']


def test_native_availability_never_calls_predicate_or_descriptor():
    source = (ROOT / 'bindings/public_attributes_native.mojo').read_text()
    assert '"PyFunction_GetCode"' in source
    assert '"availability"' in source
    assert '"transform"' in source
    code = '\n'.join(line.split('#', 1)[0] for line in source.splitlines())
    for forbidden in ('.check(', '.fn(', 'member.__get__', '.__code__'):
        assert forbidden not in code
