# SPDX-License-Identifier: Apache-2.0
"""Driver ABI and placement refusal tests with fake C calls, no GPU claim."""
import copy
import ctypes as C

import pytest

from mojolearn import _gpu_witness as witness


class Function:
    def __init__(self, action):
        self.action = action
    def __call__(self, *args):
        return self.action(*args)


def driver(vendor, count=2):
    class Library:
        _name = 'fixture-library'
    lib = Library()
    def scalar(pointer, value):
        C.cast(pointer, C.POINTER(C.c_int))[0] = value
        return 0
    def uuid(pointer, device):
        C.memmove(pointer, bytes([device + 1]) * 16, 16)
        return 0
    def name(buffer, size, device):
        buffer.value = f'fixture GPU {device}'.encode()
        return 0
    def pci(buffer, size, device):
        buffer.value = f'0000:{device + 1:02x}:00.0'.encode()
        return 0
    functions = ({'cuInit': lambda flags: 0, 'cuDeviceGetCount': lambda p: scalar(p, count),
                  'cuDeviceGet': lambda p, i: scalar(p, i), 'cuDeviceGetUuid_v2': uuid,
                  'cuDeviceGetName': name, 'cuDeviceGetPCIBusId': pci} if vendor == 'cuda' else
                 {'hipInit': lambda flags: 0, 'hipGetDeviceCount': lambda p: scalar(p, count),
                  'hipDeviceGetUuid': uuid, 'hipDeviceGetName': name, 'hipDeviceGetPCIBusId': pci})
    for key, fn in functions.items():
        setattr(lib, key, Function(fn))
    return lib


@pytest.mark.parametrize('vendor', ['cuda', 'hip'])
def test_driver_inventory_uses_visible_ordinals_and_retains_identity(monkeypatch, vendor):
    lib = driver(vendor)
    monkeypatch.setattr(witness, '_library', lambda v: lib)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES' if vendor == 'cuda' else 'HIP_VISIBLE_DEVICES', '2,0')
    record = witness.visible_gpu_inventory(vendor)
    assert record['kind'] == 'visible-device-inventory'
    assert record['vendor'] == vendor and record['driver_library'] == 'fixture-library'
    assert record['pid'] > 0
    assert record['devices'] == [dict(ordinal=i, uuid=bytes([i + 1] * 16).hex(),
                                    name=f'fixture GPU {i}', pci_bus_id=f'0000:{i + 1:02x}:00.0')
                                 for i in range(2)]
    assert '2,0' in record['visibility'].values()


def test_legacy_cuda_uuid_fallback(monkeypatch):
    lib = driver('cuda')
    lib.cuDeviceGetUuid = lib.cuDeviceGetUuid_v2
    del lib.cuDeviceGetUuid_v2
    monkeypatch.setattr(witness, '_library', lambda vendor: lib)
    assert len(witness.visible_gpu_inventory('cuda')['devices']) == 2


@pytest.mark.parametrize('failure', ['init', 'count', 'zero-count', 'uuid', 'zero-uuid', 'pci'])
def test_driver_errors_never_become_successful_inventory(monkeypatch, failure):
    lib = driver('cuda', count=0 if failure == 'zero-count' else 2)
    symbol = {'init': 'cuInit', 'count': 'cuDeviceGetCount', 'uuid': 'cuDeviceGetUuid_v2',
              'zero-uuid': 'cuDeviceGetUuid_v2', 'pci': 'cuDeviceGetPCIBusId'}.get(failure)
    if symbol:
        setattr(lib, symbol, Function(lambda *args: 0 if failure == 'zero-uuid' else 7))
    monkeypatch.setattr(witness, '_library', lambda vendor: lib)
    with pytest.raises(RuntimeError):
        witness.visible_gpu_inventory('cuda')


def records():
    return [dict(kind='visible-device-inventory', vendor='cuda', pid=100 + i,
                 devices=[dict(ordinal=0, uuid=f'{i + 1:032x}', pci_bus_id=f'0000:{i + 1:02x}:00.0')])
            for i in range(2)]


def test_distinct_worker_inventory_is_only_placement_evidence():
    witness.require_distinct_workers(records(), 'cuda', 2)


@pytest.mark.parametrize('fault', ['same-uuid', 'same-pci', 'same-pid', 'extra-gpu',
                                  'wrong-vendor', 'missing-worker', 'wrong-kind',
                                  'bad-uuid', 'empty-uuid', 'bad-pci', 'bad-ordinal'])
def test_duplicate_aliases_and_incomplete_witnesses_refused(fault):
    value = records()
    if fault.startswith('same-'):
        key = {'same-uuid': 'uuid', 'same-pci': 'pci_bus_id', 'same-pid': 'pid'}[fault]
        left, right = (value[0], value[1]) if key == 'pid' else (value[0]['devices'][0], value[1]['devices'][0])
        right[key] = left[key]
    elif fault == 'extra-gpu':
        value[0]['devices'].append(copy.deepcopy(value[1]['devices'][0]))
    elif fault == 'wrong-vendor': value[0]['vendor'] = 'cpu'
    elif fault == 'missing-worker': value.pop()
    elif fault == 'wrong-kind': value[0]['kind'] = 'requested-device-list'
    elif fault == 'bad-uuid': value[0]['devices'][0]['uuid'] = 'unknown'
    elif fault == 'empty-uuid': value[0]['devices'][0]['uuid'] = '0' * 32
    elif fault == 'bad-pci': value[0]['devices'][0]['pci_bus_id'] = 'unknown'
    elif fault == 'bad-ordinal': value[0]['devices'][0]['ordinal'] = 1
    with pytest.raises(RuntimeError):
        witness.require_distinct_workers(value, 'cuda', 2)
