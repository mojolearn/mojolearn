# SPDX-License-Identifier: Apache-2.0
"""Package connection records absent optional modes without changing routes."""
from mojolearn.models import causal_lm


def test_absent_fast_family_keeps_identical_connection_and_named_refusal(monkeypatch):
    backend = causal_lm._backend
    entries = {}

    def binding(name, mode):
        if name == "_mojolearn_mamba" and mode == "fast":
            raise AttributeError(name)
        return entries.setdefault((name, mode), object())

    monkeypatch.setattr(backend, "vendor", lambda: "metal")
    monkeypatch.setattr(backend, "binding", binding)
    _, context = causal_lm._connect_causal_native()
    selected = context["bindings"]
    assert selected["gpu:identical"]["mamba1"] is entries[("_mojolearn_mamba", "identical")]
    assert selected["gpu:identical"]["mamba2"] is entries[("_mojolearn_mamba", "identical")]
    assert selected["gpu:fast"]["transformer"] is entries[("_mojolearn_transformer", "fast")]
    for family in ("mamba1", "mamba2"):
        failure = selected["gpu:fast"][family]
        assert "_mojolearn_mamba" in failure and "fast" in failure and "unavailable" in failure
        assert isinstance(failure, str)
