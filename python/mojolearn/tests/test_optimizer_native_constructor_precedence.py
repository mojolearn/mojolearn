"""Public constructor refusals keep their order after typed native unpacking."""
import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("kind,options,category,message", [
    ("SGD", {"accumulation_steps": 3, "lr": "bad"}, ValueError, "accumulation_steps"),
    ("Adam", {"accumulation_steps": 3, "betas": ()}, ValueError, "accumulation_steps"),
    ("AdamW", {"accumulation_steps": 3, "maximize": 1}, ValueError, "accumulation_steps"),
    ("SGD", {"maximize": 1, "lr": "bad"}, TypeError, "maximize must be a bool"),
    ("Adam", {"maximize": 1, "betas": ()}, TypeError, "maximize must be a bool"),
    ("SGD", {"nesterov": True, "momentum": 0, "dampening": "bad", "lr": "bad"},
     ValueError, "nesterov=True needs momentum"),
    ("SGD", {"nesterov": True, "momentum": .5, "dampening": 1, "lr": "bad"},
     ValueError, "nesterov=True needs momentum"),
    ("AdamW", {"betas": (), "lr": "bad"}, ValueError, "betas must be a pair"),
    ("Adam", {"weight_decay": "bad decay", "eps": "bad eps"}, ValueError, "bad decay"),
])
def test_native_constructor_refusal_precedence(kind, options, category, message):
    parameters = np.array([1., -.5, .25], dtype="f4")
    before = parameters.tobytes()
    result = audit_native_call(getattr(T, kind), parameters,
                               numeric_mode="identical", resident=False, **options)
    assert type(result.exception) is category, result.report()
    assert message in str(result.exception), result.report()
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[field], result.report()
    assert parameters.tobytes() == before
