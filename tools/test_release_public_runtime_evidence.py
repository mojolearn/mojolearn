"""Retrospective admission uses original executions, never invented begin records."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

import release_public_runtime_evidence as runtime
from release_requirement_receipts import validate_receipt
from release_requirements import RequirementsLedger, requirements


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def retained(tmp_path, monkeypatch):
    """Synthetic producer artifacts; no product import, execution or compilation."""
    root = tmp_path / 'repo'; root.mkdir()
    git(root, 'init', '-q'); git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    spec = runtime.profile()
    monkeypatch.setattr(runtime, 'profile', lambda: spec)
    paths = {node.split('::')[0] for cases in spec['suites'].values() for node in cases}
    paths |= {'python/mojolearn/_array.py', 'bindings/native.mojo', 'bindings/merged_build.sh', 'packaging/support.c'}
    for name in paths:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture\n')
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'fixture source')
    revision = git(root, 'rev-parse', 'HEAD')
    cache = tmp_path / 'cache'; groups = []; binaries = {}
    for index in range(79):
        key = f'group-{index}'; stem = f'_fixture_{index}'
        fields = dict(repo=str(root), platform='darwin', source=dict(scope='closure', unresolved=[],
            files={'bindings/native.mojo': runtime.evidence.sha256(root/'bindings/native.mojo')}),
            builders=dict(scripts={'bindings/merged_build.sh': runtime.evidence.sha256(root/'bindings/merged_build.sh')}))
        row = dict(key=key, fields=fields, mode='host' if index else 'identical', stem=stem)
        groups.append(row)
        binary = root/'python/mojolearn'/row['mode']/'.development'/key/(stem+'.so')
        binary.parent.mkdir(parents=True, exist_ok=True); binary.write_bytes(f'fixture binary {index}'.encode())
        binaries[str(binary)] = runtime.evidence.sha256(binary)
        put(cache/'artifacts'/key/'receipt.json', dict(key=key, fields=fields, exit_code=0, sha256=binaries[str(binary)]))
    support = dict(key='support', name='portable_math', install='.dylibs/libfixture.dylib',
        fields=dict(platform='darwin', inputs={'packaging/support.c': runtime.evidence.sha256(root/'packaging/support.c')}))
    binary = root/'python/mojolearn'/support['install']; binary.parent.mkdir(); binary.write_bytes(b'support fixture')
    binaries[str(binary)] = runtime.evidence.sha256(binary)
    put(cache/'artifacts/support/receipt.json', dict(key='support', fields=support['fields'], exit_code=0, sha256=binaries[str(binary)]))
    build = tmp_path/'build.json'
    put(build, dict(schema='mojolearn.development-build.v1', state='activated', source_commit=revision,
        activated=['identical','host'], missing=[], results=[], plan=dict(repo=str(root), cache=str(cache), groups=groups, support=[support])))
    suites = {}
    for label, nodes in spec['suites'].items():
        directory = tmp_path/label; suites[label] = str(directory)
        collection = [dict(node=node, file=node.split('::')[0].rsplit('/',1)[-1]) for node in nodes]
        counts = runtime.Counter(row['file'] for row in collection)
        manifest_counts = dict(counts)
        if label == 'existing': manifest_counts['test_native_params_metadata.py'] += 1
        manifest = dict(source_commit=revision, expected_cases=len(nodes), total=168 if label=='existing' else 58,
            files=[dict(file=file, cases=count, sha256=runtime.evidence.sha256(root/'python/mojolearn/tests'/file))
                for file,count in sorted(manifest_counts.items())])
        put(directory/'collection-manifest.json', manifest)
        put(directory/'run/collection.json', collection)
        put(directory/'run/summary.json', dict(source_commit=revision, status='passed', pytest_exit=0,
            runtime_executed=True, release_qualified=False, expected=len(nodes), collected=len(nodes),
            outcomes={'passed':len(nodes)}, bad=[], loaded_native={'module':next(iter(binaries))}))
        put(directory/'run/preflight.json', dict(source_commit=revision, scope=runtime.SCOPE,
            dependencies={'numpy':'2.5.3','pytest':'9.1.1','pandas':'3.0.5'}, python='/fixture/bin/python',
            expected_cases=len(nodes), build_report=str(build), build_report_sha256=runtime.evidence.sha256(build), binaries=binaries))
        controller = f"from pathlib import Path\nREV={revision!r}\nBUILD=Path({str(build)!r})\nimport os\nos.environ.update(MOJOLEARN_VENDOR='metal', MOJOLEARN_NUMERIC_MODE='identical')\n"
        (directory/'run_selected.py').write_text(controller)
        spec['controller_sha256'][label] = runtime.evidence.sha256(directory/'run_selected.py')
        (directory/'controller.log').write_text('all synthetic fixture cases passed\n')
        (directory/'controller.rc').write_text('0\n')
        xml = ET.Element('testsuites'); suite = ET.SubElement(xml,'testsuite', tests=str(len(nodes)), errors='0',failures='0',skipped='0',timestamp='2026-10-25T11:42:53-04:00')
        audit_index = 0
        for node in nodes:
            path,name = node.split('::',1)
            case = ET.SubElement(suite,'testcase', classname=path.removesuffix('.py').replace('/','.'), name=name)
            if name.startswith('test_public_fit_predict_score_have_no_python_lifecycle_frames['):
                reports=[]
                for _ in range(3):
                    strict = audit_index < 12; audit_index += 1
                    reports.append(dict(schema='mojolearn.native-runtime-audit.v1',passed=strict,compatibility_qualified=True,
                        exception=None,forbidden_native_calls=[],unresolved_native_calls=[],wrapper_work=[],unobserved_existing_threads=[],
                        approved_python_fallbacks=[] if strict else [dict(id='array-length-metadata',callee='mojolearn._array.Array.__len__',calls=1)],
                        python_frames=[] if strict else [dict(function='Array.__len__',file='/fixture/python/mojolearn/_array.py',calls=1)]))
                ET.SubElement(ET.SubElement(case,'properties'),'property',name='runtime_qualification',value=json.dumps(reports))
        ET.ElementTree(xml).write(directory/'run/junit.xml',encoding='unicode')
    return dict(root=root, revision=revision, suites=suites, out=tmp_path/'review', build=build)


def prepare(retained):
    return runtime.prepare(retained['root'],retained['revision'],retained['suites'],retained['out'])


def test_exact_retained_profile_and_strict_compatibility_split(retained):
    review=prepare(retained)
    assert (review['distinct_cases'],review['executions']) == (203,235)
    assert review['reusable'] and not review['broad_runtime_gate_qualified'] and not review['installed_wheels_qualified']
    qualified=review['runtime_qualification']
    assert qualified['strict_native_passed']==12 and qualified['compatibility_qualified']==24
    assert qualified['strict_native_complete'] is False
    report=runtime.load(retained['out']/'report.json')
    assert [n['original_manifest_total'] for n in report['facts']['accounting_notes']] == [168,58]
    proof=runtime.load(retained['out']/'reconciliation.json')
    assert proof['mode']=='retrospective' and proof['original_run_id'] is None
    assert not validate_receipt(runtime.profile()['id'],review['original_requirement'],runtime.load(retained['out']/'receipt.json'))


@pytest.mark.parametrize('mutation', ['failed','skipped','collection','source','binary','dependency','controller','build','cache'])
def test_reject_incomplete_or_changed_producer_artifacts(retained,mutation):
    target=Path(retained['suites']['targeted'])
    if mutation in ('failed','skipped'):
        path=target/'run/junit.xml'; tree=ET.parse(path)
        ET.SubElement(tree.find('.//testcase'),'failure' if mutation=='failed' else 'skipped');tree.write(path)
    elif mutation=='collection':
        path=target/'run/collection.json'; rows=runtime.load(path);put(path,rows[:-1])
    elif mutation=='source':
        path=target/'collection-manifest.json'; raw=runtime.load(path);raw['files'][0]['sha256']='0'*64;put(path,raw)
    elif mutation=='binary':
        raw=runtime.load(target/'run/preflight.json');Path(next(iter(raw['binaries']))).write_bytes(b'changed')
    elif mutation=='dependency':
        path=target/'run/preflight.json';raw=runtime.load(path);raw['dependencies']['numpy']='3.0';put(path,raw)
    elif mutation=='controller':
        path=target/'run_selected.py';path.write_text(path.read_text().replace("MOJOLEARN_VENDOR='metal'","MOJOLEARN_VENDOR='cuda'"))
    elif mutation=='build': retained['build'].write_text('{}')
    else:
        path=retained['root'].parent/'cache/artifacts/group-0/receipt.json';raw=runtime.load(path);raw['fields']['source']['files']['bindings/native.mojo']='0'*64;put(path,raw)
    with pytest.raises(ValueError): prepare(retained)


@pytest.mark.parametrize('mutation',['strict-label','unknown-fallback','extra-frame','audit-case'])
def test_refuse_misclassified_compatibility_audits(retained,mutation):
    path=Path(retained['suites']['targeted'])/'run/junit.xml';tree=ET.parse(path)
    props=tree.findall('.//property')
    if mutation=='audit-case':
        tree.findall('.//testcase')[-1].set('name','different_case')
    else:
        prop=props[-1];reports=json.loads(prop.get('value'));report=reports[-1]
        if mutation=='strict-label': report['passed']=True
        elif mutation=='unknown-fallback':report['approved_python_fallbacks'][0]['callee']='arbitrary.callback'
        else: report['python_frames'].append(dict(function='other',file='caller.py',calls=1))
        prop.set('value',json.dumps(reports))
    tree.write(path)
    with pytest.raises(ValueError):prepare(retained)


def test_receipt_cannot_claim_broad_gate_or_fabricated_begin(retained):
    review=prepare(retained);wanted=review['original_requirement'];receipt=runtime.load(retained['out']/'receipt.json')
    broad=copy.deepcopy(wanted);broad['required']=True;broad['inputs']['contract']['id']='runtime-apple'
    assert validate_receipt('runtime-apple',broad,receipt)
    for key,value in [('execution_count',203),('runtime_qualification',{'strict_native_complete':True})]:
        altered=copy.deepcopy(receipt);altered[key]=value
        assert validate_receipt(runtime.profile()['id'],wanted,altered)
    path=retained['out']/'reconciliation.json';proof=runtime.load(path);proof['original_run_id']='invented-pre-run';put(path,proof)
    with pytest.raises(ValueError,match='retrospective'):runtime.producer_payload(runtime.profile()['id'],wanted,retained['out']/'report.json',proof)


def test_admission_is_retrospective_optional_idempotent_and_syncable(retained):
    review=prepare(retained);root=retained['root'];name=runtime.profile()['id']
    ledger=RequirementsLedger(root.parent/'ledger.json',source_root=root)
    catalog={'source_inputs':{},'requirements':[dict(id='runtime-apple',kind='runtime',scope='whole runtime',coverage=['whole runtime'])]}
    manifest=requirements(root,retained['revision'],'0.8.57',catalog)
    ledger.sync(manifest)
    first=ledger.admit_retained_runtime(retained['out']/'report.json',retained['revision'])
    assert first['run_id'] is None and first['origin']=='retained-execution' and 'started_at' not in first
    ledger.admit_retained_runtime(retained['out']/'report.json',retained['revision'])
    doc=ledger.read();assert len(doc['attempts'])==1 and doc['runs']==[]
    assert doc['status'][name]['status']=='passed' and doc['status']['runtime-apple']['status']=='untested'
    assert ledger.gate()[0] is False
    (root/'NOTES.md').write_text('documentation only\n');git(root,'add','NOTES.md');git(root,'commit','-qm','docs')
    later=git(root,'rev-parse','HEAD');manifest=requirements(root,later,'0.8.57',catalog,doc['configuration'])
    status=ledger.sync(manifest)
    assert status[name]['action']=='reuse' and status[name]['tested_revision']==retained['revision']
    (root/'python/mojolearn/_array.py').write_text('changed product\n');git(root,'add','.');git(root,'commit','-qm','product')
    status=ledger.sync(requirements(root,git(root,'rev-parse','HEAD'),'0.8.57',catalog,doc['configuration']))
    assert status[name]['status']=='untested' and any('_array.py' in reason for reason in status[name]['reasons'])


@pytest.mark.parametrize('field',['dependencies','binaries','build_inputs','interpreter'])
def test_target_environment_changes_invalidate_retained_execution(retained,field):
    review=prepare(retained);report=runtime.load(retained['out']/'report.json')
    environment=runtime.frozen_environment(report['facts'])
    if field=='interpreter':environment[field]='/another/bin/python'
    else:environment[field][next(iter(environment[field]))]='changed'
    declaration=dict(id=runtime.profile()['id'],report=runtime.reference(retained['out']/'report.json'),environment=environment)
    current=runtime.requirement_from_declaration(retained['root'],retained['revision'],declaration)
    assert current['fingerprint'] != review['original_requirement']['fingerprint']


def test_no_admission_into_ledger_for_another_revision(retained):
    prepare(retained)
    ledger=RequirementsLedger(retained['root'].parent/'ledger.json',source_root=retained['root'])
    with pytest.raises(ValueError,match='synchronize'):
        ledger.admit_retained_runtime(retained['out']/'report.json',retained['revision'])
    assert not ledger.path.exists()


def test_readmission_does_not_reset_changed_target_environment(retained):
    prepare(retained);root=retained['root'];name=runtime.profile()['id']
    ledger=RequirementsLedger(root.parent/'ledger.json',source_root=root)
    catalog={'source_inputs':{},'requirements':[]}
    ledger.sync(requirements(root,retained['revision'],'0.8.57',catalog))
    ledger.admit_retained_runtime(retained['out']/'report.json',retained['revision'])
    config=ledger.read()['configuration']
    config[runtime.CONFIG_KEY][0]['environment']['dependencies']['numpy']='future-version'
    ledger.sync(requirements(root,retained['revision'],'0.8.57',catalog,config))
    ledger.admit_retained_runtime(retained['out']/'report.json',retained['revision'])
    assert ledger.read()['status'][name]['status']=='untested'
