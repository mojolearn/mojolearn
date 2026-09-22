# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.models.ParallelCausalLM`: the unit contracts beside its lane.

WHAT CHANGED ON 2026-09-20 (lane/cpu-routes-gpu-only-four). This file used to
open by explaining why `par-causal-lm` could not be an identity lane: `load`
and `__init__` both raised NotImplementedError unless `_backend.vendor()` was
'cuda' or 'hip', so on every column this project runs there was no arithmetic
to hash. That is no longer true. On a CPU-only install a "device index" is one
WORKER PROCESS, each layer is built from `CausalLM`'s own CPU route and the
class's claim -- "Output and state mathematics are the ordinary CausalLM
path" -- is held byte for byte against a plain in-process `CausalLM` by the
`par-causal-lm` lane in `tools/identity_break.py`, which takes a CPU column.

SO THE DIVISION OF LABOUR IS NOW THE ORDINARY ONE. The lane hashes the
arithmetic; this file pins the contracts a hash cannot see: the refusal for a
GPU install of the wrong vendor before any checkpoint read, the CPU route's
admission through `_parallel_pool.CPU_OPERATIONS`, the layer-map validation
before a worker is started, the worker-index arithmetic and the closed-model
refusal, the remote state's ownership and idempotent release, and -- in the
last test -- the CPU numerics driven through emulated RPC, held to the plain
model's digest. That last one is NOT physical two-GPU evidence and says so;
the transport is mocked.

What is still owed is a run on two real CUDA or HIP devices.
A one-process-per-index CPU column is the DEGENERATE case of the device axis,
exactly as `identity_break._par_devices`'s docstring says of every `par-*`
lane, and it is AGREEMENT with the plain path rather than a claim that either
is right.
"""
from types import SimpleNamespace
import pytest
from mojolearn.models import parallel_causal_lm as mod


@pytest.mark.parametrize('vendor', ['cpu'])
def test_gpu_install_of_the_wrong_vendor_rejected_before_checkpoint_read(monkeypatch, vendor):
    """A GPU INSTALL whose vendor is not CUDA or HIP still refuses, before a
    file is opened. `_CPU_ONLY is None` here: a GPU install reporting
    `cpu` is NOT the CPU-only host route, which
    `test_cpu_only_install_takes_the_host_route` covers. The two are
    different facts and the vocabulary keeps them apart."""
    monkeypatch.setattr(mod._backend, 'vendor', lambda: vendor)
    monkeypatch.setattr(mod._backend, '_CPU_ONLY', None)
    with pytest.raises(NotImplementedError, match='CUDA or HIP'):
        mod.ParallelCausalLM.load('/does/not/exist', layer_devices=(0,1))


def test_cpu_only_install_takes_the_host_route(monkeypatch):
    """lane/cpu-routes-gpu-only-four (2026-09-20). On a CPU-only install the
    route is `cpu` and `load` gets as far as reading the checkpoint, which is
    what the FileNotFoundError here proves: the vendor refusal no longer
    stands in front of it."""
    monkeypatch.setattr(mod._backend, 'vendor', lambda: 'cpu')
    monkeypatch.setattr(mod._backend, '_CPU_ONLY', 'no identical binding on this box')
    assert mod._admit_route() == 'cpu'
    with pytest.raises(FileNotFoundError):
        mod.ParallelCausalLM.load('/does/not/exist', layer_devices=(0,1))


def test_the_layer_operation_is_gated_by_cpu_operations():
    """`causal_lm_layer` is admitted on the CPU route, a neighbour is not.
    `_rpc` addresses one worker and so never passes through `pool.map`'s
    admission, which is why it calls `_cpu_refusal` itself; this asserts the
    set it consults says yes to this operation and no in general."""
    from mojolearn._parallel_pool import _cpu_refusal
    assert _cpu_refusal([('causal_lm_layer', 'run', ())], False) is None
    assert _cpu_refusal([('gbdt_fit', None, ())], False) is not None


@pytest.mark.parametrize('devices', [(0,), (0,-1), (0,True)])
def test_invalid_map_precedes_worker_creation(devices):
    with pytest.raises(ValueError, match='per layer'):
        mod.ParallelCausalLM(SimpleNamespace(n_layers=2), {}, layer_devices=devices)


def test_explicit_worker_index_and_closed_refusal():
    model=object.__new__(mod.ParallelCausalLM)
    requests=[]
    pool=SimpleNamespace(_workers=['worker2','worker0'],
                         _call=lambda w,r: requests.append((w,r)))
    model._closed=False; model._pools={2:pool,0:pool}; model._worker_index={2:0,0:1}
    model._rpc(0,'head','activation')
    assert requests==[('worker0',('causal_lm_layer','head','activation'))]
    model._closed=True
    with pytest.raises(RuntimeError, match='closed'): model._rpc(0,'head',None)


def test_remote_state_refuses_wrong_layer_and_release_is_idempotent():
    calls=[]
    a=object.__new__(mod._RemoteBlock); b=object.__new__(mod._RemoteBlock)
    a.model=SimpleNamespace(_closed=False); a.call=lambda *args: calls.append(args)
    state=mod._RemoteState(a,'token')
    with pytest.raises(ValueError,match='another layer'): b._run('step',None,state)
    state.close(); state.close()
    assert calls==[('release','token')]
    with pytest.raises(ValueError,match='released'): a._run('step',None,state)


@pytest.mark.parametrize('vendor,owners', [('cuda', (2, 0)), ('metal', (0, 0))])
def test_layer_transport_matches_cpu_with_isolated_mock_workers(monkeypatch, tmp_path, vendor, owners):
    """Real CPU numerics through emulated RPC; not physical GPU evidence."""
    import pickle
    from mojolearn import Array
    from mojolearn import _causal_lm_worker as worker
    from mojolearn import _causal_lm_fixtures as fixtures
    from mojolearn.models import causal_lm as base
    from mojolearn._verify_causal_lm import digest, state_digest
    try:
        base._CpuPrimitives()
    except ImportError as exc:
        pytest.skip(str(exc))
    cfg=fixtures._llama_config()
    fixtures._write_checkpoint(tmp_path, cfg, fixtures._llama_tensors(cfg))
    plain=base.CausalLM.load(tmp_path, device='cpu')
    cpu_classes=base._block_classes('cpu')
    monkeypatch.setattr(base,'_block_classes', lambda route: cpu_classes)
    monkeypatch.setattr(base,'_GpuPrimitives',base._CpuPrimitives)
    monkeypatch.setattr(mod._backend,'vendor',lambda:vendor)
    worlds={}
    calls=[]
    class Pool:
        def __init__(self,devices): self._workers=list(devices)
        def _start(self): pass
        def close(self): pass
        @staticmethod
        def _call(device,request):
            world=worlds.setdefault(device, ({},{},{}))
            worker._blocks,worker._states,worker._tensors=world
            _,operation,args=pickle.loads(pickle.dumps(request))
            calls.append((device,operation))
            return pickle.loads(pickle.dumps(worker.execute(operation,args)))
    monkeypatch.setattr(mod,'DevicePool',Pool)
    ids=Array.from_list([[1,3,7],[2,4,8]],'<i4')
    with mod.ParallelCausalLM.load(tmp_path, layer_devices=iter(owners)) as split:
        assert digest(plain.forward(ids))==digest(split.forward(ids))
        a=plain.allocate_state(2,8); b=split.allocate_state(2,8)
        assert digest(plain.forward(ids,a))==digest(split.forward(ids,b))
        token=Array.from_list([5,6],'<i4')
        assert digest(plain.step(token,a))==digest(split.step(token,b))
        assert state_digest(a)==state_digest(b)
        split.reset_state(b)
        assert digest(split.forward(ids,b))==digest(plain.forward(ids))
        assert digest(split.generate(ids,2))==digest(plain.generate(ids,2))
        assert set(split.parameters())==set(plain.parameters())
    if vendor == 'metal':
        assert set(worlds) == {0}
        assert set(worlds[0][0]) == {0, 1}
    else:
        assert set(worlds[2][0])=={0} and set(worlds[0][0])=={1}
    assert (owners[0],'embedding') in calls and (owners[-1],'head') in calls


@pytest.mark.parametrize('devices', [(0, 1), (1, 1), (True, 0), ()])
def test_metal_owner_validation_precedes_checkpoint_read(monkeypatch, devices):
    monkeypatch.setattr(mod._backend, 'vendor', lambda: 'metal')
    monkeypatch.setattr(mod._backend, '_CPU_ONLY', None)
    with pytest.raises(ValueError):
        mod.ParallelCausalLM.load('/does/not/exist', layer_devices=devices)


def test_metal_zero_owners_use_gpu_route_and_reach_checkpoint(monkeypatch):
    monkeypatch.setattr(mod._backend, 'vendor', lambda: 'metal')
    monkeypatch.setattr(mod._backend, '_CPU_ONLY', None)
    assert mod._admit_route((0, 0)) == 'gpu'
    with pytest.raises(FileNotFoundError):
        mod.ParallelCausalLM.load('/does/not/exist', layer_devices=iter((0, 0)))


def test_metal_nonzero_owner_refuses_before_pool(monkeypatch):
    monkeypatch.setattr(mod._backend, 'vendor', lambda: 'metal')
    monkeypatch.setattr(mod._backend, '_CPU_ONLY', None)
    monkeypatch.setattr(mod, 'DevicePool', lambda *a, **k: pytest.fail('pool must not start'))
    with pytest.raises(ValueError, match='device 0'):
        mod.ParallelCausalLM(SimpleNamespace(n_layers=2), {}, layer_devices=(0, 1))
