# SPDX-License-Identifier: Apache-2.0
import copy
import pytest
from mojolearn import _verify_distributed as v


def receipt():
    value = dict(protocol=v.PROTOCOL, status='NUMERICAL_MATCH_EXECUTION_TRACE_OWED',
                 vendor='cuda', repeats=2, devices=[0,1], cells=[], controls=[],
                 worker_calls=[], worker_groups=[], source_files={'profile.py':'a'*64},
                 bindings={name:dict(vendor='cuda',sha256='b'*64) for name in
                    ('_mojolearn_arima','_mojolearn_tsa','_mojolearn_gp','_mojolearn_ivf')})
    parts=[dict(shape=[2,3],dtype='<f4',sha256='c'*64)]
    for layout,devices in enumerate(([0],[0,1],[1,0])):
        for repeat in range(2):
            for case in v.CASES:
                group=len(value['worker_groups'])
                inventory=[]
                for i,device in enumerate(devices):
                    pid=100+group*2+i
                    inventory.append(dict(kind='visible-device-inventory',vendor='cuda',pid=pid,
                        devices=[dict(ordinal=0,uuid=f'{device+1:032x}',pci_bus_id=f'0000:{device+1:02x}:00.0')]))
                    value['worker_calls'].append(dict(group=group,pid=pid,operation={'arima':'forecast_predict','holtwinters':'forecast_predict','gpc':'gpc_class_predict','gpc_fit':'gpc_class_fit','ivf':'ivf_search_stored'}[case]))
                value['worker_groups'].append(dict(group=group,devices=list(devices),inventory=inventory))
                value['cells'].append(dict(case=case,layout=layout,devices=list(devices),repeat=repeat,
                    worker_groups=[group],actual=copy.deepcopy(parts),expected=copy.deepcopy(parts),match=True))
    value['controls']=[dict(case=case,fault=fault,kind='transport',triggered=True,detected=True)
                      for case in v.CASES for fault in ('drop_result','reverse_results')]
    return value


def test_complete_receipt_and_exact_comparison():
    a=receipt(); assert v.validate_receipt(a); assert v.compare(a,copy.deepcopy(a))
    b=copy.deepcopy(a)
    for cell in b['cells']:
        if cell['case']=='ivf':
            cell['actual'][0]['sha256']='d'*64
            cell['expected'][0]['sha256']='d'*64
    assert not v.compare(a,b)


@pytest.mark.parametrize('fault',['missing_cell','duplicate_cell','unused_worker','alias_gpu',
                                  'unobserved_control','missing_control','bad_hash','missing_binding',
                                  'changed_baseline','mismatch','wrong_devices','source','wrong_operation'])
def test_incomplete_or_false_pass_evidence_rejected(fault):
    a=receipt(); b=copy.deepcopy(a)
    if fault=='missing_cell': b['cells'].pop()
    if fault=='duplicate_cell': b['cells'].append(copy.deepcopy(b['cells'][0]))
    if fault=='unused_worker': b['worker_calls'].pop()
    if fault=='alias_gpu':
        group=next(g for g in b['worker_groups'] if len(g['inventory'])==2)
        group['inventory'][1]['devices']=copy.deepcopy(group['inventory'][0]['devices'])
    if fault=='unobserved_control': b['controls'][0]['triggered']=False
    if fault=='missing_control': b['controls'].pop()
    if fault=='bad_hash': b['cells'][0]['actual'][0]['sha256']='not-a-hash'
    if fault=='missing_binding': b['bindings'].pop('_mojolearn_gp')
    if fault=='changed_baseline':
        b['cells'][0]['expected'][0]['sha256']='d'*64
        b['cells'][0]['actual'][0]['sha256']='d'*64
    if fault=='mismatch': b['cells'][0]['match']=False
    if fault=='wrong_devices': b['cells'][0]['devices']=[1]
    if fault=='wrong_operation': b['worker_calls'][0]['operation']='device_inventory'
    if fault=='source': b['source_files']['profile.py']='different'
    with pytest.raises((ValueError,RuntimeError)):
        v.compare(a,b)


@pytest.mark.parametrize('fault,expected',[('drop_result',[1,2]),('reverse_results',[3,2,1])])
def test_transport_controls_change_actual_result_batches_and_restore(fault,expected):
    class Pool:
        def map(self,requests): return [1,2,3]
    original=Pool.map; control={'triggered':False}
    with v.transport_fault(Pool,fault,control):
        assert Pool().map([('ivf_store',None,None)])==[1,2,3]
        assert not control['triggered']
        assert Pool().map([('ivf_search_stored',None,None)])==expected
        assert Pool().map([('ivf_search_stored',None,None)])==[1,2,3]
    assert control['triggered'] and Pool.map is original


def test_transport_restored_after_exception():
    class Pool:
        def map(self,requests): return [1,2]
    original=Pool.map
    with pytest.raises(RuntimeError):
        with v.transport_fault(Pool,'drop_result',{'triggered':False}):
            raise RuntimeError('stop')
    assert Pool.map is original


def test_invalid_device_arguments_do_not_start_workers(tmp_path):
    with pytest.raises(SystemExit):
        v.main(['--devices','bad,1','--out',str(tmp_path/'out.json')])
    assert not (tmp_path/'out.json').exists()
