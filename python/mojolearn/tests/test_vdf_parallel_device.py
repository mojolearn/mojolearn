# SPDX-License-Identifier: Apache-2.0
"""Boundary words of `device=` on the host-only inference classes (lane
vdf-parallel): LanguageModelInference, MLPInference and SambaInference run
compiled CPU code on every installation, so None and 'cpu' keep today's
route and 'gpu' is refused naming the class. Words and exception types only;
the numbers are verify's. The word is admitted before any file or weight is
read, so a missing checkpoint path never reaches its own error here."""
import inspect

import pytest

import mojolearn

CLASSES = [
    ("LanguageModelInference", lambda: mojolearn.LanguageModelInference),
    ("MLPInference", lambda: mojolearn.MLPInference),
    ("SambaInference", lambda: mojolearn.SambaInference),
]
MISSING = "vdf-parallel-no-such-checkpoint.ckpt"


@pytest.mark.parametrize("name,cls", CLASSES)
def test_constructors_take_keyword_only_device_default_none(name, cls):
    for method in (cls().__init__, cls().from_checkpoint):
        parameter = inspect.signature(method).parameters["device"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is None, name


@pytest.mark.parametrize("name,cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_unknown_device_is_refused_in_the_shared_words(name, cls, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got "):
        cls().from_checkpoint(MISSING, device=bad)


@pytest.mark.parametrize("name,cls", CLASSES)
def test_gpu_is_refused_naming_the_class(name, cls):
    with pytest.raises(ValueError, match="device='gpu' is not available for " + name + ": it has no GPU implementation"):
        cls().from_checkpoint(MISSING, device="gpu")


def test_gpu_is_refused_by_the_constructors_before_the_weights():
    with pytest.raises(ValueError, match="device='gpu' is not available for MLPInference"):
        mojolearn.MLPInference(None, None, None, None, device="gpu")
    with pytest.raises(ValueError, match="device='gpu' is not available for LanguageModelInference"):
        mojolearn.LanguageModelInference(None, device="gpu")
