"""The device identity probes must consume outputs and reject moved state."""
import hashlib
import sys
import types

import numpy as np
import pytest

import identity_break as identity


@pytest.mark.parametrize("name", ["adafactor", "lamb"])
@pytest.mark.parametrize("failure", [None, "parameter", "state"])
def test_device_identity_lane_consumes_native_download_and_detects_moves(monkeypatch, name, failure):
    calls = dict(upload=0, download=0, device_step=0, state=0)

    class Tensor:
        def __init__(self, value):
            self.value = value.copy()

        def numpy(self):
            calls["download"] += 1
            return self.value.copy()

        def __array__(self, *args, **kwargs):
            raise AssertionError("lane used the unqualified dtype conversion")

    def to_device(value, numeric_mode=None):
        assert numeric_mode == "identical"
        calls["upload"] += 1
        return Tensor(value)

    class Optimizer:
        def __init__(self, params, numeric_mode=None, **settings):
            assert numeric_mode == "identical"
            self.params = params
            self.resident = isinstance(params[0], Tensor)
            self.t = 0

        def step(self, grads):
            assert all(isinstance(g, Tensor) == self.resident for g in grads)
            calls["device_step"] += int(self.resident)
            self.t += 1
            for p, g in zip(self.params, grads):
                if self.resident:
                    p.value -= g.value
                else:
                    p -= g
            if self.resident and failure == "parameter":
                self.params[0].value[0, 0] += np.float32(1)

        @property
        def state(self):
            calls["state"] += 1
            value = np.float32(1 if self.resident and failure == "state" else 0)
            if name == "adafactor":
                return [dict(row_var=np.full(32, value, dtype="f4"),
                             col_var=np.zeros(8, dtype="f4")),
                        dict(variance=np.zeros(8, dtype="f4"))]
            return [np.full(264, value, dtype="f4"), np.zeros(264, dtype="f4")]

    device = types.ModuleType("mojolearn._x_sequence_device")
    device.SequenceDeviceTensor = Tensor
    device.to_device = to_device
    monkeypatch.setitem(sys.modules, device.__name__, device)
    monkeypatch.setattr(identity, "_coverage_digester",
                        lambda ml: lambda a: hashlib.sha256(a.tobytes()).hexdigest())
    ml = types.SimpleNamespace(Adafactor=Optimizer, LAMB=Optimizer)
    x = np.random.RandomState(41).normal(size=(512, 16)).astype("f4")
    run = identity.LANES["sequence-" + name + "-device"]
    if failure:
        with pytest.raises(ValueError, match="resident.*host-array"):
            run(ml, x, None, None)
        return
    result = run(ml, x, None, None)
    assert len(result) == (12 if name == "adafactor" else 10)
    assert calls == dict(upload=28, download=4, device_step=12, state=4)
    assert identity.BATCH["sequence-" + name + "-device"].startswith("n/a:coverage-probe")
