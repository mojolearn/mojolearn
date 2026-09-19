# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.models.ParallelCausalLM`. THIS IS A TEST AND NOT AN IDENTITY
LANE, and the reason is the class's own first line.

`tools/verification_matrix.py` reported `models.ParallelCausalLM` with no
lane on 2026-09-19 and lane/models-namespace-lanes gave the rest of the
`mojolearn.models` namespace three (`hf-checkpoint`, `hf-tokenizer`,
`hf-causal-lm`). This one deliberately got none. `ParallelCausalLM.load` and
`__init__` both raise NotImplementedError unless `_backend.vendor()` is
'cuda' or 'hip' -- the refusal is by name and comes before a file is opened
-- so on the CPU column, on Apple and in that harness there is no arithmetic
to hash at all. A lane would record REFUSED on every fixture of every column
this project actually runs, and a column total cannot tell a REFUSED from a
build that did not run. Writing one would have added a row that cannot fail.

What CAN be pinned without two CUDA devices is pinned here: the vendor
refusal before any checkpoint read, the layer-map validation before a worker
is started, the worker-index arithmetic and the closed-model refusal, the
remote state's ownership and idempotent release, and -- in the last test --
the real CPU numerics of the ordinary `CausalLM` path driven through
emulated RPC, held to the plain model's digest. That last one is NOT
physical two-GPU evidence and says so; the transport is mocked.

What is still owed is a run on two real CUDA or HIP devices
(docs/lanes/LANE_STATUS_causal-lm-distributed-proof.md's outstanding item).
The day this box has one, the question is whether `ParallelCausalLM` belongs
in `identity_break` as a `par-*` lane, which `tools/lane_applicability.py`
already refuses on a zero-device column, rather than here.
"""
from types import SimpleNamespace
import pytest
from mojolearn.models import parallel_causal_lm as mod


def test_cpu_rejected_before_checkpoint_read(monkeypatch):
    monkeypatch.setattr(mod._backend, 'vendor', lambda: 'cpu')
    with pytest.raises(NotImplementedError, match='CUDA or HIP'):
        mod.ParallelCausalLM.load('/does/not/exist', layer_devices=(0,1))


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


def test_layer_transport_matches_cpu_with_isolated_mock_workers(monkeypatch, tmp_path):
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
    monkeypatch.setattr(mod._backend,'vendor',lambda:'cuda')
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
    with mod.ParallelCausalLM.load(tmp_path, layer_devices=(2,0)) as split:
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
    assert set(worlds[2][0])=={0} and set(worlds[0][0])=={1}
    assert (2,'embedding') in calls and (0,'head') in calls
