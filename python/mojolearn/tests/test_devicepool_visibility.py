# SPDX-License-Identifier: Apache-2.0
"""Device-mask mapping/refusal contracts; no accelerator execution."""
import pytest
from mojolearn import _backend
from mojolearn._parallel_pool import DevicePool


@pytest.mark.parametrize('vendor,variable', [('cuda', 'CUDA_VISIBLE_DEVICES'),
                                            ('hip', 'HIP_VISIBLE_DEVICES'),
                                            ('hip', 'ROCR_VISIBLE_DEVICES')])
@pytest.mark.parametrize('mask,message', [('GPU-a,GPU-a', 'repeat an identifier'),
                                          ('GPU-a, GPU-a ', 'repeat an identifier'),
                                          ('GPU-a', 'outside'), ('GPU-a,', 'outside')])
def test_bad_visible_mask_fails_before_any_worker(monkeypatch, vendor, variable, mask, message):
    from mojolearn import _parallel_pool
    monkeypatch.setattr(_backend, 'vendor', lambda: vendor)
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, mask)
    started = []
    monkeypatch.setattr(_parallel_pool.subprocess, 'Popen', lambda *a, **kw: started.append(kw))
    pool = DevicePool((0, 1))
    with pytest.raises(ValueError, match=message):
        pool._start()
    assert not started and pool._threads is None


@pytest.mark.parametrize('vendor,variable', [('cuda', 'CUDA_VISIBLE_DEVICES'),
                                            ('hip', 'HIP_VISIBLE_DEVICES'),
                                            ('hip', 'ROCR_VISIBLE_DEVICES')])
def test_visible_device_order_and_hip_filter_are_preserved(monkeypatch, vendor, variable):
    import io
    from mojolearn import _parallel_pool
    monkeypatch.setattr(_backend, 'vendor', lambda: vendor)
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, 'GPU-a, GPU-b ')
    if variable == 'ROCR_VISIBLE_DEVICES':
        monkeypatch.setenv('HIP_VISIBLE_DEVICES', '7,8')
    started = []
    class Child:
        def __init__(self, *args, **kw):
            started.append(kw['env'])
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(_parallel_pool.subprocess, 'Popen', Child)
    pool = DevicePool((1, 0))
    try:
        pool._start()
        assert [env[variable] for env in started] == ['GPU-b', 'GPU-a']
        if vendor == 'hip':
            other = 'HIP_VISIBLE_DEVICES' if variable.startswith('ROCR') else 'ROCR_VISIBLE_DEVICES'
            assert all(other not in env for env in started)
    finally:
        pool.close()


@pytest.mark.parametrize('vendor,devices,cooperative', [
    ('cuda', (1, 0), False), ('hip', (1, 0), False),
    ('cpu', (1, 0), False), ('metal', (0,), False),
    ('cuda', (1, 0), True), ('hip', (1, 0), True),
    ('cpu', (0,), True), ('metal', (0,), True),
])
def test_worker_native_counts_match_its_group_not_parent(
        monkeypatch, vendor, devices, cooperative):
    import io
    import os
    from mojolearn import _parallel_pool

    monkeypatch.setattr(_backend, 'vendor', lambda: vendor)
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    for name in _parallel_pool.DEVICE_COUNT_VARIABLES:
        monkeypatch.setenv(name, '8')
    started = []

    class Child:
        def __init__(self, *args, **kw):
            started.append(kw['env'])
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        def poll(self): return 0
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(_parallel_pool.subprocess, 'Popen', Child)
    pool = DevicePool(devices, cooperative=cooperative)
    try:
        pool._start()
        expected = len(devices) if cooperative else 1
        assert len(started) == (1 if cooperative else len(devices))
        for env in started:
            # These concrete inherited settings previously reached workers
            # unchanged despite their single-device visibility mask.
            assert env['MOJOLEARN_GBDT_DEVICE_COUNT'] == str(expected)
            assert env['MOJOLEARN_GP_DEVICE_COUNT'] == str(expected)
            assert all(env[name] == str(expected)
                       for name in _parallel_pool.DEVICE_COUNT_VARIABLES)
        assert os.environ['MOJOLEARN_GP_DEVICE_COUNT'] == '8'
    finally:
        pool.close()
