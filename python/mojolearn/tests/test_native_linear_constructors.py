"""Actual public constructor/lifetime probes; require the integrated native build."""
import array
import inspect
import weakref

import pytest

from mojolearn import _expansion_linear as linear
from mojolearn._native_linear_constructors import _CONTEXT
from mojolearn._native_params_setter import _CLASSES
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


@pytest.mark.parametrize('name', _CLASSES)
def test_public_constructor_borrows_exact_defaults_and_signatures(name):
    cls = getattr(linear, name)
    signature = inspect.signature(cls)
    assert tuple(signature.parameters) == _CONTEXT['plans'][cls]['names']
    assert signature.parameters['numeric_mode'].kind is inspect.Parameter.KEYWORD_ONLY
    model = clean(cls)
    assert vars(model).keys() == signature.parameters.keys()
    assert tuple(vars(model)) == _CONTEXT['plans'][cls]['order']
    assert all(vars(model)[key] is parameter.default for key, parameter in signature.parameters.items())
    assert clean(model.get_params, False).keys() == signature.parameters.keys()


@pytest.mark.parametrize('name', _CLASSES)
def test_public_constructor_retains_parameter_reference_identity(name):
    cls = getattr(linear, name)
    # Parameter admission belongs to fit; constructors have always stored
    # ordinary references without coercion or callbacks.
    values = {key: object() for key in inspect.signature(cls).parameters}
    model = clean(cls, **values)
    assert all(vars(model)[key] is value for key, value in values.items())


@pytest.mark.parametrize('name', _CLASSES)
def test_public_reinitialization_keeps_fitted_fields_and_existing_none_mode_rule(name):
    cls = getattr(linear, name)
    model = clean(cls, numeric_mode='identical')
    fitted = array.array('f', [1.0]); user = object()
    vars(model).update(coef_=fitted, user_metadata=user, user_note_=user)
    assert clean(model.__init__) is None
    assert vars(model)['coef_'] is fitted and vars(model)['user_metadata'] is vars(model)['user_note_'] is user
    assert vars(model)['numeric_mode'] == 'identical'
    assert clean(model.__init__, numeric_mode=None) is None
    assert vars(model)['numeric_mode'] == 'identical'
    marker = 'validation-remains-owned-by-fit'
    assert clean(model.__init__, numeric_mode=marker) is None
    assert vars(model)['numeric_mode'] is marker


@pytest.mark.parametrize('name', _CLASSES)
def test_argument_errors_precede_mutation(name):
    cls = getattr(linear, name); model = clean(cls, numeric_mode='identical')
    before = dict(vars(model)); signature = inspect.signature(cls)
    positional = [p for p in signature.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    clean(model.__init__, unexpected_argument=True, error=TypeError)
    clean(model.__init__, *([1] * (len(positional) + 1)), error=TypeError)
    if positional:
        clean(model.__init__, 1, **{positional[0].name: 2}, error=TypeError)
    assert vars(model) == before


@pytest.mark.parametrize('name', ('SGDClassifier', 'SGDRegressor', 'LassoLars', 'RidgeClassifier', 'RidgeCV'))
def test_supported_positional_parameter_keeps_identity(name):
    cls=getattr(linear,name); value=object()
    first=next(iter(inspect.signature(cls).parameters))
    model=clean(cls,value,numeric_mode='identical')
    assert vars(model)[first] is value


def test_lifecycle_refusal_does_not_release_or_reinitialize_any_field():
    calls=[]
    class Finalizer:
        def __del__(self): calls.append('finalizer')
    held=Finalizer()
    model=clean(linear.RidgeCV,alphas=held,numeric_mode='identical')
    before=dict(vars(model))
    clean(model.__init__,alphas=(1.0,),numeric_mode='fast',error=TypeError)
    assert vars(model)==before and not calls


def test_weakref_callback_owner_cannot_be_overwritten():
    calls=[]; held=array.array('f',[1.0]); ref=weakref.ref(held,lambda value:calls.append('weakref'))
    model=clean(linear.RidgeCV,alphas=held)
    clean(model.__init__,alphas=(1.0,),error=TypeError)
    assert vars(model)['alphas'] is ref() is held and not calls


def test_changed_parameter_descriptor_is_refused_without_hook(monkeypatch):
    calls=[]; model=clean(linear.SGDRegressor)
    class Descriptor:
        def __get__(self,owner,cls):calls.append('get');return 1
        def __set__(self,owner,value):calls.append('set')
    monkeypatch.setattr(linear.SGDRegressor,'alpha',Descriptor(),raising=False)
    clean(linear.SGDRegressor.__init__,model,alpha=0.2,error=TypeError)
    assert not calls


def test_instance_protocol_override_is_refused_without_callback():
    calls=[]; model=clean(linear.SGDRegressor)
    vars(model)['__setattr__']=lambda *args:calls.append('setattr')
    clean(linear.SGDRegressor.__init__,model,error=TypeError)
    assert not calls


def test_keyword_string_subclass_protocol_is_refused_without_call():
    calls=[]
    class Key(str):
        def __hash__(self):calls.append('hash');return str.__hash__(self)
        def __eq__(self,other):calls.append('eq');return str.__eq__(self,other)
    key=Key('alpha'); values={key:0.2}; calls.clear()
    # Python's ** expansion happens in this test controller. Exercise the
    # public constructor's envelope without asking that caller expansion to
    # implement the product's own admission policy.
    model=linear.SGDRegressor()
    from mojolearn._native_linear_constructors import _ENTRY
    clean(_ENTRY,model,(),values,_CONTEXT,error=TypeError)
    assert not calls


@pytest.mark.parametrize('args,keywords,message', (
    ((1,2), {}, 'takes from 1 to 2 positional arguments but 3 were given'),
    ((1,2), {'alpha':0.2}, 'takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given'),
    ((1,2), {'numeric_mode':'identical'}, 'takes from 1 to 2 positional arguments but 3 were given'),
    ((1,2), {'loss':'hinge'}, "got multiple values for argument 'loss'"),
    ((1,2), {'unknown':1}, "got an unexpected keyword argument 'unknown'"),
))
def test_python_argument_error_order_and_mode_wrapper_arity_are_preserved(args,keywords,message):
    failure=clean(linear.SGDRegressor,*args,error=TypeError,**keywords)
    assert str(failure)=='SGDRegressor.__init__() '+message


def test_keyword_only_constructor_arity_and_missing_self_are_preserved():
    failure=clean(linear.PoissonRegressor,1,error=TypeError)
    assert str(failure)=='PoissonRegressor.__init__() takes 1 positional argument but 2 were given'
    clean(linear.SGDRegressor.__init__,error=TypeError)


@pytest.mark.parametrize('name', _CLASSES)
def test_passive_subclasses_keep_inherited_native_constructor_behavior(name):
    cls=getattr(linear,name)
    class Passive(cls):
        pass
    model=clean(Passive,numeric_mode='identical')
    assert type(model) is Passive and vars(model)['numeric_mode']=='identical'
    assert set(vars(model)) == set(inspect.signature(cls).parameters)
    assert clean(model.__init__,numeric_mode=None) is None
    assert vars(model)['numeric_mode']=='identical'


def test_passive_mixin_ancestry_keeps_inherited_constructor_behavior():
    class Label:
        label='test-label'
    class Passive(linear.SGDRegressor,Label):
        pass
    model=clean(Passive,alpha=0.25)
    assert type(model) is Passive and vars(model)['alpha']==0.25
    assert vars(model)['numeric_mode'] is None


def test_passive_subclass_class_mode_hasattr_rule_preserves_absent_instance_field():
    class Passive(linear.SGDRegressor):
        numeric_mode='identical'
    model=clean(Passive)
    assert 'numeric_mode' not in vars(model) and model.numeric_mode=='identical'
    assert clean(model.__init__,numeric_mode=None) is None
    assert 'numeric_mode' not in vars(model) and model.numeric_mode=='identical'
    assert clean(model.__init__,numeric_mode='explicit-mode') is None
    assert vars(model)['numeric_mode']=='explicit-mode'
    assert clean(model.__init__,numeric_mode=None) is None
    assert vars(model)['numeric_mode']=='explicit-mode'


def test_passive_subclass_changed_storage_hook_refuses_without_callback():
    calls=[]
    class Changed(linear.SGDRegressor):
        def __setattr__(self,name,value):calls.append((name,value))
    model=object.__new__(Changed)
    clean(linear.SGDRegressor.__init__,model,error=TypeError)
    assert not calls and not vars(model)


def test_passive_subclass_mode_descriptor_refuses_without_hasattr_callback():
    calls=[]
    class Mode:
        def __get__(self,model,owner):calls.append('get');return 'identical'
    class Changed(linear.SGDRegressor):
        numeric_mode=Mode()
    model=object.__new__(Changed)
    clean(linear.SGDRegressor.__init__,model,error=TypeError)
    assert not calls and not vars(model)


def test_passive_subclass_delete_only_descriptor_refuses_without_callback():
    calls=[]
    class Parameter:
        def __delete__(self,model):calls.append('delete')
    class Changed(linear.SGDRegressor):
        alpha=Parameter()
    model=object.__new__(Changed)
    clean(linear.SGDRegressor.__init__,model,alpha=0.2,error=TypeError)
    assert not calls and not vars(model)
