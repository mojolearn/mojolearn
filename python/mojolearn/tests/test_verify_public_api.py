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
    assert len(expected)==1275
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
    assert accepted['untested_vendors']==['hip'] and not accepted['release_qualified']
    amd=copy.deepcopy(clean);amd['vendor']='hip'
    assert api.admit({'metal':apple,'cuda':clean,'hip':amd})['untested_vendors']==[]
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
    assert only_nvidia['untested_vendors']==['hip','metal']
    partial=copy.deepcopy(apple)
    partial['endpoints']['example.call']={'status':'failed','error':'native build pending'}
    partial['pending']=['example.call']
    accepted=api.admit({'cuda':clean,'metal':partial})
    assert accepted['validated_vendors']==['cuda']
    assert accepted['device_validation']['metal']['compared']==0
    assert accepted['device_validation']['metal']['failures']=={'example.call':'native build pending'}
    broken=copy.deepcopy(clean);broken['pending']=['example.call']
    with pytest.raises(ValueError):api.admit({'cuda':broken})


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


def test_inventory_regeneration_cannot_hide_a_new_unmapped_api():
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
