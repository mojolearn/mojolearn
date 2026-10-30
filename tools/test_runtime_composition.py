"""Tooling fixtures only: never claim the synthetic cases executed product code."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

import qualify_native_runtime_installed as q
import release_runtime_composition as c
import release_runtime_installed as r
import release_installed_evidence as adapter

COMMIT = 'a' * 40
INTERPRETER = dict(implementation='cpython', version='fixture 3.11', cache_tag='cpython-311',
                   soabi='fixture', platform='fixture', machine='fixture', executable_sha256='b' * 64)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def make_wheel(path):
    files = {'mojolearn/__init__.py': b'# fixture\n',
             'mojolearn/host/binding.so': b'not an executable: tooling fixture',
             'mojolearn/identity_columns/COMMIT': COMMIT.encode()}
    record = 'mojolearn_core-0.8.63.dist-info/RECORD'
    stream = io.StringIO(); rows = csv.writer(stream)
    for name, body in files.items():
        rows.writerow((name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip('='), len(body)))
    rows.writerow((record, '', '')); files[record] = stream.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, body in files.items(): archive.writestr(name, body)


def component(out, number, selected, payload, wheels, tests, *, na=None):
    directory = out / 'components' / str(number); directory.mkdir(parents=True)
    cases = []
    for name in selected:
        case, skip = 'test_fixture', ''
        if name == na:
            case, reason = next(iter(q.CPU_NOT_APPLICABLE[name].items()))
            import html
            skip = '<skipped message="' + html.escape(reason, quote=True) + '"/>'
        cases.append('<testcase classname="mojolearn.tests.%s" name="%s">%s</testcase>' % (Path(name).stem, case, skip))
    write(directory / 'tests.xml', '<testsuite>' + ''.join(cases) + '</testsuite>')
    summary = q.junit_summary(directory / 'tests.xml')
    scope = q.test_scope_summary(summary, selected, 'cpu')
    prefix = '/fixture/env'; base = prefix + '/lib/python3.11/site-packages/'
    modules = {name: dict(path=base + name, sha256=row['sha256'], native=name.endswith('.so'))
               for name, row in payload.items() if name.endswith(('.py', '.so'))}
    native = payload['mojolearn/host/binding.so']
    installed = dict(prefix=prefix, vendor='cpu', numeric_mode='identical', interpreter=INTERPRETER,
                     inventory_errors=[], worker_error=None, pytest_exit=0, modules=modules,
                     binding_provenance_protocol='loaded-native-v2', generation_test_device='cpu',
                     bindings=[dict(module='mojolearn._host.binding', file=base+'mojolearn/host/binding.so',
                                    sha256=native['sha256'], size=native['bytes'])])
    write(directory / 'installed-modules.json', installed)
    write(directory / 'controller.py', '# fixture controller\n')
    receipt = dict(source_commit=COMMIT, test_commit=COMMIT, vendor='cpu', numeric_mode='identical',
                   wheels=wheels, native_assets={'mojolearn/host/binding.so': native},
                   package_source_hashes={'mojolearn/__init__.py': payload['mojolearn/__init__.py']['sha256']},
                   test_hashes=tests, controller_sha256=r.sha256(directory/'controller.py'),
                   interpreter=INTERPRETER, test_selection_kind='whole-files', selected_test_nodes=None,
                   selected_public_scope={n:q.PUBLIC_TESTS[n] for n in selected},
                   test_dependencies=q.test_dependencies(selected), selected_test_applicability=q.test_applicability(selected,'cpu'),
                   test_device_overrides={'MOJOLEARN_GENERATION_DEVICE':'cpu'}, tests=summary,
                   test_scope_summary=scope, worker_exit=0, pytest_exit=0,
                   status=q.selected_test_status(0,summary,scope,True),
                   requested_native_source_hashes={'fixture.mojo':'c'*64}, archive_omitted_historical_native_sources=[])
    write(directory/'receipt.json',receipt)
    write(directory/'request.json',dict(payload=payload,test_files=selected,test_nodes=None,vendor='cpu',
                                       numeric_mode='identical',generation_device='cpu'))
    write(directory/'tests.log','fixture log\n')
    write(directory/'pip-freeze.log','pytest==9.2.0\nnumpy==2.4.6\n')
    return directory.relative_to(out).as_posix()


@pytest.fixture
def composed(tmp_path):
    out=tmp_path/'composition'; out.mkdir()
    make_wheel(out/'core.whl')
    payload,wheels=q.wheel_inventory([out/'core.whl'])
    write(out/'coverage.json',json.loads(r.COVERAGE_PATH.read_text()))
    tests={}
    for name in q.PUBLIC_TESTS:
        write(out/'frozen-tests'/name,'# fixture\n')
        tests[name]=r.sha256(out/'frozen-tests'/name)
    names=list(q.PUBLIC_TESTS)
    paths=[component(out,0,names[:35],payload,wheels,tests),component(out,1,names[35:],payload,wheels,tests)]
    write(out/c.MANIFEST,dict(schema=c.SCHEMA,vendor='cpu',source_commit=COMMIT,test_commit=COMMIT,
                             package_source_hashes={'mojolearn/__init__.py':payload['mojolearn/__init__.py']['sha256']},
                             wheels=['core.whl'],test_hashes=tests,components=paths))
    return out


def change(out, file, fn):
    path=out/file; data=json.loads(path.read_text()); fn(data); write(path,data)


def test_complete_union_rederived_by_actual_ledger_adapter(composed):
    report=r.summarize(composed)
    assert report['status']=='passed' and report['completed_files']==73
    assert report['component_count']==2 and report['cases']==73
    assert not (composed/'tests.xml').exists()
    report['qualifier_exit']=0; write(composed/r.REPORT,report)
    facts=adapter._facts('runtime')(composed,{'report':str(composed/r.REPORT)})
    assert len(facts['coverage'])==5
    assert all(x['completed']==x['expected'] for x in facts['coverage'])
    (composed/'components/1/tests.xml').write_text('<testsuite/>')
    with pytest.raises(ValueError,match='JUnit'): adapter._facts('runtime')(composed,{'report':str(composed/r.REPORT)})


def test_partial_is_explicit_and_cannot_enter_ledger(composed):
    change(composed,c.MANIFEST,lambda d:d['components'].pop())
    report=r.summarize(composed)
    assert report['status']=='incomplete' and report['missing_files']
    report['qualifier_exit']=1; write(composed/r.REPORT,report)
    with pytest.raises(ValueError,match='did not pass'):
        adapter._facts('runtime')(composed,{'report':str(composed/r.REPORT)})


@pytest.mark.parametrize('file,change_fn,match',[
    (c.MANIFEST,lambda d:d['components'].append(d['components'][0]),'overlapping'),
    (c.MANIFEST,lambda d:d['components'].__setitem__(0,'../escape'),'escaped'),
    ('components/0/receipt.json',lambda d:d.update(source_commit='b'*40),'source drift'),
    ('components/0/receipt.json',lambda d:d['wheels'][0].update(sha256='0'*64),'wheel identity'),
    ('components/0/receipt.json',lambda d:d.update(test_selection_kind='exact-nodes'),'whole-file'),
    ('components/0/receipt.json',lambda d:d['test_hashes'].update(helper='0'*64),'test/helper'),
    ('components/0/receipt.json',lambda d:d.update(worker_exit=124),'failed or timed out'),
    ('components/0/receipt.json',lambda d:d.update(test_dependencies=['pytest']),'dependency policy'),
    ('components/0/receipt.json',lambda d:d.update(test_device_overrides={}),'device policy'),
    ('components/0/installed-modules.json',lambda d:d.pop('interpreter'),'interpreter identity'),
    ('components/0/installed-modules.json',lambda d:d.update(inventory_errors=['error']),'guard failed'),
    ('components/0/installed-modules.json',lambda d:d['bindings'][0].update(sha256='0'*64),'witness differs'),
    ('components/0/installed-modules.json',lambda d:d['modules']['mojolearn/__init__.py'].update(sha256='0'*64),'differs from retained'),
    ('components/0/installed-modules.json',lambda d:d.update(vendor='hip'),'backend/mode'),
    ('components/0/receipt.json',lambda d:d.update(status='failed-or-incomplete'),'failed or timed out'),
])
def test_drift_refused(composed,file,change_fn,match):
    change(composed,file,change_fn)
    with pytest.raises(ValueError,match=match): r.summarize(composed)


def test_identical_version_label_is_insufficient_for_other_interpreter(composed):
    for file in ('receipt.json','installed-modules.json'):
        change(composed,'components/1/'+file,lambda d:d['interpreter'].update(executable_sha256='0'*64))
    with pytest.raises(ValueError,match='interpreter/native-source'): r.summarize(composed)


def test_dependency_versions_must_agree(composed):
    (composed/'components/1/pip-freeze.log').write_text('pytest==9.3.0\n')
    with pytest.raises(ValueError,match='dependency version'): r.summarize(composed)


def test_original_direct_wheel_dependency_is_hash_bound(composed):
    path=composed/'components/0/pip-freeze.log'
    path.write_text(path.read_text()+'mojolearn @ file:///original/input/core.whl#sha256='+r.sha256(composed/'core.whl')+'\n')
    assert r.summarize(composed)['status']=='passed'
    path.write_text(path.read_text().replace('#sha256=','#sha256=0'))
    with pytest.raises(ValueError,match='direct dependency'):r.summarize(composed)


def test_declared_cpu_na_retained_but_not_counted_as_pass(composed):
    manifest=c.read(composed/c.MANIFEST); payload,wheels=q.wheel_inventory([composed/'core.whl'])
    name=next(iter(q.CPU_NOT_APPLICABLE))
    for rel in manifest['components']:
        selected=list(c.read(composed/rel/'receipt.json')['selected_public_scope'])
        if name in selected:
            import shutil
            number=Path(rel).name;shutil.rmtree(composed/rel)
            component(composed,number,selected,payload,wheels,manifest['test_hashes'],na=name)
    report=r.summarize(composed)
    assert report['status']=='passed'
    assert sum(x['not_applicable'] for x in report['coverage'])==1
    assert sum(x['completed'] for x in report['coverage'])==72


def test_wholly_inapplicable_component_retains_truthful_status(composed):
    manifest=c.read(composed/c.MANIFEST); payload,wheels=q.wheel_inventory([composed/'core.whl'])
    name=next(iter(q.CPU_NOT_APPLICABLE))
    import shutil
    shutil.rmtree(composed/'components')
    paths=[component(composed,0,[n for n in q.PUBLIC_TESTS if n!=name],payload,wheels,manifest['test_hashes']),
           component(composed,1,[name],payload,wheels,manifest['test_hashes'],na=name)]
    manifest['components']=paths;write(composed/c.MANIFEST,manifest)
    assert c.read(composed/'components/1/receipt.json')['status']=='selected-tests-not-applicable'
    assert r.summarize(composed)['status']=='passed'


@pytest.mark.parametrize('body',['<failure message="original failure"/>','<skipped message="arbitrary skip"/>'])
def test_consistent_failing_receipt_cannot_be_composed(composed,body):
    directory=composed/'components/0'
    path=directory/'tests.xml'; text=path.read_text(); text=text.replace('</testcase>',body+'</testcase>',1);path.write_text(text)
    receipt=c.read(directory/'receipt.json'); selected=list(receipt['selected_public_scope'])
    summary=q.junit_summary(path); scope=q.test_scope_summary(summary,selected,'cpu')
    receipt.update(tests=summary,test_scope_summary=scope,status='failed-or-incomplete')
    write(directory/'receipt.json',receipt)
    with pytest.raises(ValueError,match='coverage incomplete'):r.summarize(composed)


def test_modified_original_controller_refused(composed):
    (composed/'components/0/controller.py').write_text('different controller\n')
    with pytest.raises(ValueError,match='controller bytes'):r.summarize(composed)


def test_helper_change_refused_even_if_selected_file_unchanged(composed):
    write(composed/'frozen-tests/shared_helper.py','changed helper\n')
    with pytest.raises(ValueError,match='frozen tests differ'):r.summarize(composed)
