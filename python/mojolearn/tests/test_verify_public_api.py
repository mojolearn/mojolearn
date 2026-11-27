"""Cheap regression checks for completeness, fault admission and default integration."""
import copy
import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
from mojolearn import _verify_api_runner as api
from mojolearn import _verify_public as public


def row(value=1.0):
    raw=struct.pack('<f',value)
    return {'status':'captured','kind':'numerical','output':{
        'shape':[1],'dtype':'<f4','bytes':raw.hex(),'sha256':hashlib.sha256(raw).hexdigest()}}


def report():
    return {'expected':['example.call'],'endpoints':{'example.call':row()},'pending':[]}


def test_every_original_gap_has_an_executable_fixture_route():
    routes=api.registry()
    expected={r['endpoint'] for r in api.TARGETS}
    # 1275 less the deleted TransformerBlockInference tape pair (TOMBSTONE TRANSFORMER_FORWARD_TAPE_API) = 1273;
    # 1240 at the integration base; less the 24 rows of the four *BlockInference classes neural-consolidate-2
    # removed (1216); + the 27 Sequential layer methods of 2026-11-22 + log_softmax and the six `compile:loss=`
    # rows of lanes nn-losses and nn-losses-2 (2026-11-23); less the 3 Mamba3Block owned-weights methods
    # lane/owned-weights-deletion removed (2026-11-25) = 1247
    assert len(expected)==1247
    assert routes.keys()==expected
    from mojolearn._verify_api_faults import FAULTS
    assert set(FAULTS.values())<=expected


def test_bitwise_comparison_detects_one_bit_and_rejects_incomplete_evidence():
    clean=report()
    assert not api.differences(clean,clean)
    bad=copy.deepcopy(clean);entry=bad['endpoints']['example.call']['output']
    raw=bytearray.fromhex(entry['bytes']);raw[0]^=1
    entry.update(bytes=raw.hex(),sha256=hashlib.sha256(raw).hexdigest())
    assert api.differences(bad,clean)==['output differs: example.call']
    entry['shape']=[2]
    assert any(x.startswith('invalid output evidence:') for x in api.differences(bad,clean))
    bad=copy.deepcopy(clean);bad['endpoints']['example.call']={'status':'failed','error':'injected crash'}
    assert api.differences(bad,clean)==['execution failed: example.call']
    bad=copy.deepcopy(clean);bad['pending']=['example.call']
    assert any(x.startswith('unexecuted endpoints:') for x in api.differences(bad,clean))


@pytest.mark.parametrize('part',['tape','endpoints'])
@pytest.mark.parametrize('status',['FAILED','OWED'])
def test_normal_public_checks_include_faults_and_propagate_failures(monkeypatch,part,status):
    from mojolearn import _verify_light_command
    calls=[]
    def tape(reference,*,self_test):
        calls.append(('tape',self_test));return {'status':status if part=='tape' else 'MATCH'}
    def endpoints(*,self_test,scope):
        assert scope=='all'
        calls.append(('endpoints',self_test));return {'status':status if part=='endpoints' else 'MATCH'}
    monkeypatch.setattr(_verify_light_command,'verify',tape)
    monkeypatch.setattr(api,'verify',endpoints)
    result=public.run()
    assert calls==[('tape',True),('endpoints',True)]
    assert result['status']==status
    assert not result['release_qualified']


@pytest.mark.parametrize("side", ["cpu", "gpu"])
def test_installed_endpoint_faults_change_outputs_and_restore_them(side):
    from mojolearn import _verify_api_faults as controls

    if side not in api.sides():
        pytest.skip("GPU plugin unavailable on this installation")
    clean = api.capture(endpoints=controls.fault_endpoints(), side=side)
    assert not clean["pending"]
    retained = {}
    faults = controls.check_faults(clean, retain=retained.__setitem__)
    assert faults.keys() == controls.FAULTS.keys()
    for name, endpoint in controls.FAULTS.items():
        row = clean["endpoints"][endpoint]
        if row["status"] == "not applicable":
            assert side == "cpu" and row["kind"] in {"device only", "installation side"}
            assert faults[name]["status"] == "NOT_APPLICABLE"
        else:
            assert faults[name]["status"] == "CAUGHT_AND_RESTORED", (side, name, faults[name])
    for name, result in faults.items():
        if result["status"] == "NOT_APPLICABLE":
            continue
        endpoints = set(controls.fault_endpoints(name)) - result["not_applicable"].keys()
        assert set(result["differences"]) == {"output differs: " + e for e in endpoints}
        restored = retained[name + "-restored"]
        assert retained[name + "-faulty"]["side"] == restored["side"] == side
        assert restored["endpoints"] == {e: clean["endpoints"][e] for e in endpoints}
    assert faults["estimator-regularization"]["status"] == "CAUGHT_AND_RESTORED"


@pytest.mark.parametrize('cpu_status', ['OWED', 'UNTESTED', 'FAILED'])
def test_gpu_match_preserves_the_cpu_verdict(monkeypatch, cpu_status):
    monkeypatch.setattr(api, 'sides', lambda: ['cpu', 'gpu'])
    monkeypatch.setattr(api, 'capture', lambda *, endpoints, side: {'side': side})
    monkeypatch.setattr(api, '_judge_side', lambda clean, ref, scope:
                        dict(status=cpu_status if clean['side']=='cpu' else 'MATCH',
                             failures={'cpu-op': 'error'} if cpu_status=='FAILED' and clean['side']=='cpu' else {},
                             errors=[], pending=[]))
    result = api.verify(self_test=False, reference={})
    assert result['status'] == cpu_status
    assert result['sides']['gpu']['failures'] == {}


def test_fault_controls_report_each_side_and_fail_on_cpu_only_miss(monkeypatch):
    from mojolearn import _verify_api_faults as controls

    captured = []
    def check(clean, *, retain):
        side = clean['side']
        retain('probe-faulty', clean)
        return {'probe': {'status': 'FAILED' if side == 'cpu' else 'CAUGHT_AND_RESTORED'}}
    monkeypatch.setattr(api, 'sides', lambda: ['cpu', 'gpu'])
    monkeypatch.setattr(api, 'selected_endpoints', lambda scope: [])
    monkeypatch.setattr(api, 'capture', lambda *, endpoints, side: {'side': side})
    monkeypatch.setattr(api, '_judge_side', lambda clean, ref, scope:
                        dict(status='MATCH', evidence=clean, errors=[], failures={}, pending=[]))
    monkeypatch.setattr(controls, 'check_faults', check)
    result = api.verify(reference={}, retain=lambda name, data: captured.append(name))
    assert result['status'] == 'FAILED'
    assert result['sides']['cpu']['faults']['probe']['status'] == 'FAILED'
    assert result['faults'] == result['sides']['gpu']['faults']
    assert {'api-cpu-probe-faulty', 'api-gpu-probe-faulty'} <= set(captured)


def test_fault_requires_changed_consumed_output_and_clean_restoration(monkeypatch):
    from mojolearn import _verify_api_faults as faults
    from contextlib import contextmanager
    endpoint=next(iter(faults.FAULTS.values()))
    name=next(iter(faults.FAULTS))
    clean={'expected':[endpoint],'endpoints':{endpoint:row()},'pending':[], 'bindings':[{'sha256':'already-hashed'}]}
    monkeypatch.setattr(faults,'FAULTS',{name:endpoint})
    @contextmanager
    def inject(_):yield
    monkeypatch.setattr(faults,'inject',inject)
    def captures(fault,restored):
        results=iter([fault,restored])
        def capture(**kwargs):
            assert kwargs['artifacts'] is clean['bindings']
            return next(results)
        monkeypatch.setattr(api,'capture',capture)
        return faults.check_faults(clean)[name]['status']
    changed=copy.deepcopy(clean);changed['endpoints'][endpoint]=row(2.)
    assert captures(changed,clean)=='CAUGHT_AND_RESTORED'
    assert captures(clean,clean)=='FAILED'
    assert captures(changed,changed)=='FAILED'
    crash=copy.deepcopy(clean);crash['endpoints'][endpoint]={'status':'failed'}
    assert captures(crash,clean)=='FAILED'
    malformed=copy.deepcopy(changed);malformed['endpoints'][endpoint]['output']['shape']=[2]
    assert captures(malformed,clean)=='FAILED'


def test_default_cli_routes_to_main_verifier_without_light_flag(monkeypatch):
    from mojolearn import __main__ as cli
    observed=[]
    monkeypatch.setattr(cli._verify_all,'cmd_verify_all',lambda args:observed.append(args) or 0)
    args=SimpleNamespace()
    assert cli._verify_dispatch(args)==0
    assert observed==[args] and args.routine


@pytest.mark.parametrize('existing',[0,1])
def test_missing_public_reference_cannot_mask_an_existing_mismatch(existing):
    from mojolearn import _verify_all as all_checks
    result={'public_api':{'status':'OWED'},'detail':'test'}
    code,headline=all_checks._public_check_verdict(result,existing,'MISMATCH' if existing else 'VERIFIED')
    assert code==(all_checks.EXIT_MISMATCH if existing else all_checks.EXIT_NO_REFERENCE)
    assert headline==('MISMATCH' if existing else 'INCOMPLETE')


def test_reference_admission_requires_complete_matching_current_release(monkeypatch):
    monkeypatch.setattr(api,'TARGETS',[{'endpoint':'example.call'}])
    monkeypatch.setattr(api,'sources',lambda:{'fixture':'current'})
    clean=report()
    clean.update(vendor='cuda',numeric_mode='identical',version='test-release',
                 sources={'fixture':'current'},bindings=[{'sha256':'native'}])
    apple=copy.deepcopy(clean);apple['vendor']='metal'
    accepted=api.admit({'metal':apple,'cuda':clean})
    assert accepted['untested_vendors']==['cpu','hip'] and not accepted['release_qualified']
    amd=copy.deepcopy(clean);amd['vendor']='hip'
    assert api.admit({'metal':apple,'cuda':clean,'hip':amd})['untested_vendors']==['cpu']
    amd['endpoints']['example.call']=row(2.)
    with pytest.raises(ValueError):api.admit({'metal':apple,'cuda':clean,'hip':amd})
    for field,value in [('version','old-release'),('sources',{}),('bindings',[]),
                        ('expected',[]),('numeric_mode','fast')]:
        invalid=copy.deepcopy(apple);invalid[field]=value
        with pytest.raises(ValueError):api.admit({'metal':invalid,'cuda':clean})
    invalid=copy.deepcopy(apple);invalid['endpoints']['example.call']=row(2.)
    with pytest.raises(ValueError):api.admit({'metal':invalid,'cuda':clean})
    only_nvidia=api.admit({'cuda':clean})
    assert only_nvidia['validated_vendors']==['cuda']
    assert only_nvidia['untested_vendors']==['cpu','hip','metal']
    partial=copy.deepcopy(apple)
    partial['endpoints']['example.call']={'status':'failed','error':'native build pending'}
    partial['pending']=['example.call']
    accepted=api.admit({'cuda':clean,'metal':partial})
    assert accepted['validated_vendors']==['cuda']
    assert accepted['device_validation']['metal']['compared']==0
    assert accepted['device_validation']['metal']['failures']=={'example.call':'native build pending'}
    broken=copy.deepcopy(clean);broken['pending']=['example.call']
    with pytest.raises(ValueError):api.admit({'cuda':broken})


def test_reference_admits_a_cpu_column_that_reproduces_the_nvidia_bytes(monkeypatch):
    # The CPU column is the CPU side (device="cpu") of a capture; its device-only endpoints
    # read not applicable and are not missing from it; any output it disagrees on refuses.
    monkeypatch.setattr(api,'TARGETS',[{'endpoint':'example.call'},{'endpoint':'example.device'}])
    monkeypatch.setattr(api,'sources',lambda:{'fixture':'current'})
    clean=report()
    clean.update(vendor='cuda',side='gpu',numeric_mode='identical',version='test-release',
                 expected=['example.call','example.device'],sources={'fixture':'current'},
                 bindings=[{'sha256':'native'}])
    clean['endpoints']['example.device']=row(3.)
    cpu=copy.deepcopy(clean);cpu.update(vendor='cpu',side='cpu')
    cpu['endpoints']['example.device']={'status':'not applicable','kind':'device only',
                                        'reason':'the CPU side (device="cpu")'}
    accepted=api.admit({'cpu':cpu,'cuda':clean})
    assert accepted['validated_vendors']==['cpu','cuda']
    assert accepted['untested_vendors']==['hip','metal']
    assert accepted['device_validation']['cpu']['not_applicable']==['example.device']
    assert accepted['device_validation']['cpu']['failures']=={}
    moved=copy.deepcopy(cpu);moved['endpoints']['example.call']=row(2.)
    with pytest.raises(ValueError):api.admit({'cpu':moved,'cuda':clean})
    gpu_side=copy.deepcopy(cpu);gpu_side['side']='gpu'
    with pytest.raises(ValueError):api.admit({'cpu':gpu_side,'cuda':clean})
    failed=copy.deepcopy(cpu);failed['endpoints']['example.call']={'status':'failed','error':'refused'}
    partial=api.admit({'cpu':failed,'cuda':clean})
    assert partial['validated_vendors']==['cuda']
    assert partial['device_validation']['cpu']['incomplete']==['example.call']


def test_recorder_never_overwrites_an_earlier_failed_operation(tmp_path):
    from mojolearn._verify_api import Recorder
    recorder=Recorder(tmp_path/'capture.json',['example.call'])
    def fail():raise ValueError('original failure')
    recorder.check('example.call',fail)
    with pytest.raises(ValueError,match='overwrite evidence'):
        recorder.check('example.call',lambda:1)
    assert recorder.rows['example.call']['status']=='failed'


def test_neural_scope_includes_its_api_and_faults_without_classical_estimators():
    all_endpoints=set(api.selected_endpoints('all'))
    neural=set(api.selected_endpoints('neural'))
    assert neural < all_endpoints
    for endpoint in ('mojolearn.TransformerBlock.set_params','mojolearn.Sequential.from_checkpoint',
        'mojolearn.Adam.state_dict','mojolearn.LanguageModelTrainer.evaluate',
        'mojolearn.importers._program.ImportedProgram.run','mojolearn.models.ParallelCausalLM.step',
        'mojolearn.RNNRegressor.load_state_dict','mojolearn.Embedding.to_device'):
        assert endpoint in neural
    assert not any(e.startswith(('mojolearn.Ridge.','mojolearn.BaggingClassifier.')) for e in neural)
    from mojolearn._verify_api_faults import FAULTS
    assert {name for name,endpoint in FAULTS.items() if endpoint in neural} == {
        'optimizer-rate','block-weight','trainer-checkpoint','random-stream','program-product'}
    with pytest.raises(ValueError):api.selected_endpoints('light')


def test_neural_command_routes_public_checks_to_neural_scope(monkeypatch):
    from mojolearn import _verify_light_command
    monkeypatch.setattr(_verify_light_command,'verify',lambda *a,**k:{'status':'MATCH'})
    seen=[]
    def endpoints(**kwargs):seen.append(kwargs);return {'status':'MATCH'}
    monkeypatch.setattr(api,'verify',endpoints)
    assert public.run(scope='neural')['status']=='MATCH'
    assert seen==[{'self_test':True,'scope':'neural'}]


def test_encoding_preserves_scalar_rank_layout_and_signed_zero():
    import numpy as np
    from mojolearn._verify_api import encode
    from mojolearn._verify_light import digest
    scalar=np.array(-0.,dtype=np.float32)
    vector=scalar.reshape(1)
    for encoding in (encode,digest):
        a,b=encoding(scalar),encoding(vector)
        assert a['shape']==[] and b['shape']==[1]
        assert a['sha256']==b['sha256']
        assert a['sha256']!=encoding(np.array(0.,dtype=np.float32))['sha256']
    x=np.arange(12,dtype=np.float32).reshape(3,4)[:,::2]
    assert encode(x)==encode(x.copy())


def test_inventory_regeneration_cannot_hide_a_new_unmapped_api(monkeypatch):
    # tools/verification_endpoints.py is imported by name: put tools/ on sys.path here
    # rather than rely on another test file having done it in the same process.
    import pathlib
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[3] / 'tools'))
    from verification_endpoints import require_complete_mapping
    old=[{'endpoint':'existing.call'}]
    routes={'existing.call':object()}
    require_complete_mapping([{'endpoint':'existing.call','status':'mapped'}],old,routes)
    with pytest.raises(ValueError,match='new.call'):
        require_complete_mapping([{'endpoint':'new.call','status':'unmapped'}],old,routes)
    with pytest.raises(ValueError,match='existing.call'):
        require_complete_mapping([],old,{})


def test_release_label_reuses_only_unchanged_executed_native_inputs():
    clean={'vendor':'cuda','version':'new','families':{'example.call':'metadata'},
        'sources':{'metadata':'fixture'},'bindings':[{'module':'native','sha256':'binary'}]}
    prior=copy.deepcopy(clean);prior['version']='old'
    reference={'witnesses':{'cuda':prior}}
    assert api.unchanged_reference_inputs(clean,reference)
    changed=copy.deepcopy(clean);changed['bindings'][0]['sha256']='changed'
    assert not api.unchanged_reference_inputs(changed,reference)
    changed=copy.deepcopy(clean);changed['sources']['metadata']='changed'
    assert not api.unchanged_reference_inputs(changed,reference)
    changed=copy.deepcopy(clean);changed['bindings']=[]
    assert not api.unchanged_reference_inputs(changed,reference)
    assert not api.unchanged_reference_inputs(clean,{'witnesses':{}})


def test_changed_native_bytes_are_judged_by_their_outputs(monkeypatch):
    row={'status':'captured','kind':'value','output':{'sha256':'a'*64}}
    clean={'vendor':'cuda','version':'new','families':{'example.call':'metadata'},'sources':{'metadata':'fixture'},
        'bindings':[{'module':'native','sha256':'changed'}],'expected':['example.call'],'pending':[],
        'endpoints':{'example.call':dict(row)}}
    witness=copy.deepcopy(clean);witness['version']='old';witness['bindings'][0]['sha256']='binary'
    reference={'validated_vendors':['cuda'],'witness':witness,'witnesses':{'cuda':witness}}
    monkeypatch.setattr(api,'output_contract',lambda value:'contract')
    monkeypatch.setattr(api,'selected_endpoints',lambda scope:None)
    monkeypatch.setattr(api,'capture',lambda **kw:copy.deepcopy(clean))
    result=api.verify(self_test=False,reference=reference)
    assert result['status']=='MATCH' and result['evidence_reuse'].startswith('changed native bytes')
    clean['endpoints']['example.call']['output']={'sha256':'b'*64}
    result=api.verify(self_test=False,reference=reference)
    assert result['status']=='OWED' and 'output differs: example.call' in result['errors']


@pytest.mark.parametrize('vendor',['cuda','cpu'])
def test_a_stale_endpoint_reference_reads_owed_on_every_vendor(monkeypatch,vendor):
    # r6-proving (2026-11-23): the fixture-source staleness test sat below the UNTESTED return, so a stale
    # reference passed the CPU column's self-test while the L40S self-test of the same wheel read OWED.
    row={'status':'captured','kind':'value','output':{'sha256':'a'*64}}
    clean={'vendor':vendor,'version':'new','families':{'example.call':'neural'},'sources':{'neural':'changed'},
        'bindings':[{'module':'native','sha256':'binary'}],'expected':['example.call'],'pending':[],
        'endpoints':{'example.call':dict(row)}}
    witness=copy.deepcopy(clean);witness['vendor']='cuda';witness['sources']={'neural':'fixture'}
    reference={'validated_vendors':['cuda'],'witness':witness,'witnesses':{'cuda':witness}}
    monkeypatch.setattr(api,'output_contract',lambda value:'contract')
    monkeypatch.setattr(api,'selected_endpoints',lambda scope:None)
    monkeypatch.setattr(api,'capture',lambda **kw:copy.deepcopy(clean))
    result=api.verify(self_test=False,reference=reference)
    assert result['status']=='OWED'
    assert result['errors']==['endpoint fixtures changed; current reference is stale (neural)']
    clean['sources']['neural']='fixture'
    result=api.verify(self_test=False,reference=reference)
    assert result['status']==('MATCH' if vendor=='cuda' else 'UNTESTED') and not result['errors']


@pytest.mark.parametrize('cpu_only',[True,False])
def test_device_only_endpoints_are_not_applicable_only_where_the_build_declares_no_device(monkeypatch,cpu_only):
    # w3f5-e (2026-11-24): the CPU self-test listed the optimizers' device I/O, Embedding.to_device and
    # every ParallelByteLanguageModelTrainer method as execution failures. A build that declares no device
    # (runtime_report cpu_only) records them 'not applicable' without running them; a device build runs them.
    import mojolearn
    device,host='mojolearn.Adam.grads_to_device','mojolearn.Adam.state_dict'
    ran=[]
    module=SimpleNamespace(optimizer_checks=lambda recorder,public,methods:[ran.append(m) or recorder.check(
        public+'.'+m,lambda:1.0) for m in methods])
    routes={e:('optimizers',module,'mojolearn.Adam',e.rsplit('.',1)[1]) for e in (device,host)}
    monkeypatch.setattr(api,'registry',lambda:routes)
    monkeypatch.setattr(mojolearn,'runtime_report',lambda:{'cpu_only':cpu_only,'vendor':'cpu' if cpu_only
        else 'cuda','default_mode':'identical'})
    report=api.capture(endpoints=[device,host],artifacts=[])
    assert report['endpoints'][host]['status']=='captured' and not report['pending']
    if cpu_only:
        assert ran==['state_dict'] and report['endpoints'][device]['status']=='not applicable'
    else:
        assert sorted(ran)==['grads_to_device','state_dict'] and report['endpoints'][device]['status']=='captured'
    assert api.device_only('optimizers','mojolearn.SGD','step_device')
    assert api.device_only('training','mojolearn.ParallelByteLanguageModelTrainer','checkpoint')
    assert api.device_only('neural','mojolearn.Embedding','to_device')
    assert not api.device_only('neural','mojolearn.LayerNorm','to_device')
    assert not api.device_only('metadata','mojolearn.ParallelByteLanguageModelTrainer','get_params')


@pytest.mark.parametrize('cpu_only',[True,False])
@pytest.mark.parametrize(('family','public','method','route'),[
    ('neural','mojolearn.SambaStack','load_weights','samba_route_unpack.mojo'),
    ('models','mojolearn.models.ParallelCausalLM','generate','parallel_causal_native_run.mojo'),
])
def test_installation_side_cpu_endpoints(monkeypatch,cpu_only,family,public,method,route):
    import mojolearn
    endpoint=public+'.'+method
    ran=[]
    def checks(recorder,public,methods):
        for op in methods:
            ran.append(op)
            recorder.check(public+'.'+op,lambda:1.0)
    module=SimpleNamespace(neural_checks=checks,model_checks=checks)
    monkeypatch.setattr(api,'registry',lambda:{endpoint:(family,module,public,method)})
    monkeypatch.setattr(mojolearn,'runtime_report',lambda:{'cpu_only':cpu_only,'vendor':'cpu' if cpu_only
        else 'cuda','default_mode':'identical'})
    report=api.capture(endpoints=[endpoint],artifacts=[],side='cpu')
    if cpu_only:
        assert ran==[method] and report['endpoints'][endpoint]['status']=='captured'
    else:
        row=report['endpoints'][endpoint]
        assert ran==[] and row['status']=='not applicable' and row['kind']=='installation side'
        assert route in row['reason']


@pytest.mark.parametrize('side',['cpu','gpu'])
def test_language_model_fixtures_forward_the_capture_device(monkeypatch,tmp_path,side):
    import mojolearn
    from mojolearn import _causal_lm_fixtures, _verify_api_models, _verify_api_training, _verify_device
    from mojolearn._verify_api import Recorder

    class Trainer:
        @staticmethod
        def parameter_registry(shape):
            return []

        def __init__(self,parameters,*,shape,data_schedule,lr,device=None):
            self.device=device

    loaded=[]
    class Model:
        @classmethod
        def load(cls,path,*,device=None):
            loaded.append(device)
            return lambda ids: ids

    monkeypatch.setattr(mojolearn,'LanguageModelTrainer',Trainer)
    monkeypatch.setattr(mojolearn.models,'CausalLM',Model)
    monkeypatch.setattr(_causal_lm_fixtures,'_write_checkpoint',lambda *args,**kwargs:None)
    public_model='mojolearn.models.CausalLM'
    endpoint=public_model+'.__call__'
    recorder=Recorder(tmp_path/'capture.json',[endpoint])
    with _verify_device.side(side):
        trainer,_,_=_verify_api_training.tiny_lm()
        _verify_api_models.model_checks(recorder,public_model,['__call__'])
    assert trainer.device==side
    assert loaded==[side]
    assert recorder.rows[endpoint]['status']=='captured'


def test_unchanged_vjp_derivative_fails_even_when_forward_changes(monkeypatch):
    from contextlib import nullcontext
    from mojolearn import _verify_api_faults as faults

    endpoint = faults.FAULTS['program-product'].rsplit('.', 1)[0] + '.vjp_if'

    def capture(output, derivative):
        return dict(expected=[endpoint], pending=[], endpoints={
            endpoint: dict(status='captured', kind='numerical', output={
                'outputs': output, 'input_gradients': derivative, 'parameter_gradient': derivative,
            }),
        })

    clean = capture(1, 2)
    replays = iter((capture(3, 2), clean))
    monkeypatch.setattr(faults, 'inject', lambda name: nullcontext())
    monkeypatch.setattr(api, 'capture', lambda **kwargs: next(replays))
    result = faults.check_faults(clean)['program-product']
    assert result['status'] == 'FAILED'
    assert result['differences'] == ['output differs: ' + endpoint]
    assert result['observation_errors'] == [
        endpoint + ' did not detect the fault in input_gradients',
        endpoint + ' did not detect the fault in parameter_gradient',
    ]
