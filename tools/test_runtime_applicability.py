"""Exact applicability judgment; synthetic rows are tooling fixtures, not runs."""
import copy
import pytest
import qualify_native_runtime_installed as q
import runtime_test_applicability as policy


PY311 = dict(implementation='cpython', version='3.11.16 (main, fixture) [Clang]')


def summary(filename, node, reason):
    case=dict(name=node, classname='mojolearn.tests.'+filename[:-3], exact_node='',
              outcome='skipped', skip_reason=reason)
    return dict(total=1,passed=0,failures=[],skipped=[dict(name=node,classname=case['classname'],reason=reason)],cases=[case])


@pytest.mark.parametrize('filename,node,reason',[(f,n,r) for f,rows in policy.CPU.items() for n,r in rows.items()])
def test_only_exact_cpu_device_case_is_not_applicable(filename,node,reason):
    tests=summary(filename,node,reason)
    legacy=q.test_scope_summary(tests,[filename],'cpu',supplemental_policy=False)
    assert not legacy['complete']
    for vendor in ('cpu','metal','cuda','hip'):
        result=q.test_scope_summary(tests,[filename],vendor,interpreter=PY311)
        assert result['complete'] == (vendor=='cpu')
        assert result['applicable_passed']==0
        assert result['not_applicable_count']==(1 if vendor=='cpu' else 0)
    assert legacy['not_applicable_count']==0  # No original judgment was mutated.


@pytest.mark.parametrize('vendor',['cpu','metal','cuda','hip'])
def test_exporter_requires_recorded_old_cpython_on_every_vendor(vendor):
    filename='test_samba_native_admission.py';node,reason=next(iter(policy.PRE_PY312[filename].items()))
    tests=summary(filename,node,reason)
    assert q.test_scope_summary(tests,[filename],vendor,interpreter=PY311)['complete']
    for identity in (None, {}, dict(PY311,implementation='pypy'), dict(PY311,version='3.12.0'),
                     dict(PY311,version='3.14.0'),dict(PY311,version='unknown'),dict(PY311,version='3.9.19')):
        assert not q.test_scope_summary(tests,[filename],vendor,interpreter=identity)['complete']


@pytest.mark.parametrize('which',['reason','node','file','failure'])
def test_close_but_unproven_limitations_are_refused(which):
    filename='test_samba_native_admission.py';node,reason=next(iter(policy.PRE_PY312[filename].items()))
    if which=='reason':reason+=' (different)'
    if which=='node':node+='[other]'
    if which=='file':filename='test_samba_step_native.py'
    tests=summary(filename,node,reason)
    if which=='failure':
        tests['cases'][0]['outcome']='failed';tests['failures']=[dict(name=node)];tests['skipped']=[]
    assert not q.test_scope_summary(tests,[filename],'cpu',interpreter=PY311)['complete']


def test_added_applicability_does_not_hide_missing_whole_file():
    filename='test_samba_native_admission.py';node,reason=next(iter(policy.PRE_PY312[filename].items()))
    tests=summary(filename,node,reason)
    result=q.test_scope_summary(tests,[filename,'test_samba_step_native.py'],'cpu',interpreter=PY311)
    assert result['not_applicable_count']==1 and not result['complete']


@pytest.mark.parametrize('filename,node,reason',[(f,n,r) for f,rows in policy.CPU.items() for n,r in rows.items()])
@pytest.mark.parametrize('changed',['node','reason'])
def test_cpu_device_near_matches_are_refused(filename,node,reason,changed):
    if changed=='node':node+='[unknown]'
    else:reason+=' unknown'
    assert not q.test_scope_summary(summary(filename,node,reason),[filename],'cpu',interpreter=PY311)['complete']


@pytest.mark.parametrize('vendor',['cpu','metal','cuda','hip','unknown'])
def test_only_cpu_cooperative_refusal_is_inapplicable_to_gpu(vendor):
    filename='test_native_process_pool.py';node,reason=next(iter(policy.GPU[filename].items()))
    result=q.test_scope_summary(summary(filename,node,reason),[filename],vendor,interpreter=PY311)
    assert result['complete']==(vendor in ('metal','cuda','hip'))
    assert result['applicable_passed']==0
    for bad_node,bad_reason in ((node+'[other]',reason),(node,reason+' other')):
        assert not q.test_scope_summary(summary(filename,bad_node,bad_reason),[filename],vendor,interpreter=PY311)['complete']


@pytest.mark.parametrize('filename,node,reason',[(f,n,r) for f,rows in policy.PRE_PY312.items() for n,r in rows.items()])
@pytest.mark.parametrize('vendor',['cpu','metal','cuda','hip'])
def test_each_buffer_exporter_limitation_is_exact(filename,node,reason,vendor):
    tests=summary(filename,node,reason)
    assert q.test_scope_summary(tests,[filename],vendor,interpreter=PY311)['complete']
    for identity in (None,dict(PY311,version='3.12.0'),dict(PY311,implementation='pypy')):
        assert not q.test_scope_summary(tests,[filename],vendor,interpreter=identity)['complete']
    for n,r in ((node+'[other]',reason),(node,reason+' other')):
        assert not q.test_scope_summary(summary(filename,n,r),[filename],vendor,interpreter=PY311)['complete']
    tests['cases'][0]['outcome']='failed';tests['failures']=[dict(name=node)];tests['skipped']=[]
    assert not q.test_scope_summary(tests,[filename],vendor,interpreter=PY311)['complete']
