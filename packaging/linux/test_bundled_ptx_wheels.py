"""Synthetic metadata only; these tests create no production admission."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest
from test_split_wheels import make_sets, members, pw, ROOT, wheel_api_audit
import check_linux_release_qualification as release_gate


def inputs(root, include_byte_lm=False):
    dirs = make_sets(root, sets=(("cuda", "sm_89"), ("cuda", "sm_90a"), ("cuda", "sm_80")), include_byte_lm=include_byte_lm)
    baseline = root / 'sets/cuda/sm_80'
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = []
    for path in sorted(baseline.rglob('_mojolearn*.so')):
        rel = path.relative_to(baseline).as_posix()
        if rel.startswith('host/'):
            continue
        files.append(dict(file=rel, numeric_mode=rel.split('/')[0] if '/' in rel else 'fast',
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          ptx_modules=[dict(target='sm_80', sha256='b'*64)]))
    manifest = dict(schema='mojolearn.ptx-baseline.v1', code_format='ptx-baseline', vendor='cuda',
                    target='sm_80', min_compute_capability=[8, 0], source_commit=source, source_dirty=False,
                    experimental=True, identical_qualified=False, qualification_required=True, errors=[], files=files)
    manifest_path = baseline / pw.gpu_plugins.BASELINE_MANIFEST
    manifest_path.write_text(json.dumps(manifest))
    shared = ['train','infer','model','batchgrad','batchscale','ragged','stepfull']
    admission = dict(schema='mojolearn.ptx-identity-admission.v1', qualified=True, numeric_mode='identical',
        source_commit=source, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        coverage_contract='mojolearn.cross-vendor-identical.v1',
        coverage=dict(inventory_sha256='c'*64,harness_sha256='d'*64,
            shared=dict(lanes=['synthetic'],fixtures=['base','denormal','odd'],parts=shared,
                        vendors=['cuda','hip','metal'],comparison_sha256='e'*64),
            nvidia=dict(lanes=['synthetic'],fixtures=['base','ties','hashed','wide','denormal','denormal_ftz','dupes','odd','negative'],
                        parts=shared+['batch','rlpair'],comparison_sha256='f'*64)),
        configurations=[dict(device_name='Synthetic GPU', compute_capability=[8,0],driver_version='580.1.2',cuda_driver_version=13000)])
    admission_path = root / 'synthetic-admission.json'
    admission_path.write_text(json.dumps(admission))
    return dirs, baseline, admission_path


def pack(root, dirs, admission, wheels='nvidia', extra=()):
    """`admission=None` packs the manifest-only bundle (--bundle-ptx)."""
    args = [arg for directory in dirs for arg in ('--set', directory)]
    args += ['--wheels', wheels, '--out', str(root/'out'), *extra]
    args += ['--bundle-ptx'] if admission is None else ['--bundle-ptx-admission', str(admission)]
    assert pw.main(args, _gates=False) == 0
    return next((root/'out').glob('mojolearn_nvidia-*.whl'))


def rewrite(path, files):
    with zipfile.ZipFile(path,'w') as archive:
        for member,data in files.items():archive.writestr(member,data)


def test_vendor_bundle_preserves_all_bytes_and_native_marker_arches(tmp_path):
    dirs,baseline,admission=inputs(tmp_path)
    wheel=pack(tmp_path,dirs,admission)
    files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    for row in json.loads((baseline/pw.gpu_plugins.BASELINE_MANIFEST).read_text())['files']:
        assert files[prefix+row['file']]==(baseline/row['file']).read_bytes()
    assert files[prefix+pw.gpu_plugins.BASELINE_ADMISSION]==admission.read_bytes()
    marker=json.loads(next(data for name,data in files.items() if name.endswith('/gpu_plugin.json')))
    assert marker['arches']==['sm_89','sm_90a']
    assert marker['bundled_ptx']['admission_sha256']==hashlib.sha256(admission.read_bytes()).hexdigest()
    assert not wheel_api_audit.split_audit([wheel])['problems']
    assert not any('nvidia_ptx80-' in path.name for path in (tmp_path/'out').glob('*.whl'))


@pytest.mark.parametrize('field,value', [('qualified',False),('source_commit','0'*40),('manifest_sha256','0'*64),('configurations',[])])
def test_unqualified_or_unbound_admission_refused(tmp_path,field,value):
    dirs,baseline,admission=inputs(tmp_path);doc=json.loads(admission.read_text());doc[field]=value;admission.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,match='unqualified bundled PTX'):pack(tmp_path,dirs,admission)


def test_incomplete_coverage_and_dirty_source_refused(tmp_path):
    dirs,baseline,admission=inputs(tmp_path);doc=json.loads(admission.read_text());doc['coverage']['shared']['vendors']=['cuda','metal'];admission.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,match='cross-vendor references incomplete'):pack(tmp_path,dirs,admission)


@pytest.mark.parametrize('mutation', ['binary','admission','manifest','extra','marker'])
def test_audit_detects_bundle_tampering(tmp_path,mutation):
    dirs,baseline,admission=inputs(tmp_path);wheel=pack(tmp_path,dirs,admission);files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    if mutation=='binary':files[next(n for n in files if n.startswith(prefix) and n.endswith('.so'))]+=b'tamper'
    elif mutation=='extra':files[prefix+'unexpected.txt']=b'undeclared'
    elif mutation=='marker':
        name=next(n for n in files if n.endswith('/gpu_plugin.json'));doc=json.loads(files[name]);del doc['bundled_ptx'];files[name]=json.dumps(doc).encode()
    else:files[prefix+(pw.gpu_plugins.BASELINE_ADMISSION if mutation=='admission' else pw.gpu_plugins.BASELINE_MANIFEST)]+=b' '
    rewrite(wheel,files)
    assert wheel_api_audit.split_audit([wheel])['problems']


def test_bundle_cannot_also_emit_experimental_owner(tmp_path):
    dirs,baseline,admission=inputs(tmp_path)
    with pytest.raises(SystemExit,match='excludes the separate'):pack(tmp_path,dirs,admission,wheels='nvidia,nvidia-ptx80')


def test_release_bundle_provenance_is_checked_separately(tmp_path):
    dirs,baseline,admission=inputs(tmp_path);wheel=pack(tmp_path,dirs,admission);files=members(wheel)
    prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    hashes={n:hashlib.sha256(data).hexdigest() for n,data in files.items() if n.startswith(prefix) and n.endswith('.so')}
    marker=json.loads(next(data for name,data in files.items() if name.endswith('/gpu_plugin.json')))
    descriptor=marker['bundled_ptx'];source=json.loads(admission.read_text())['source_commit']
    payload=dict(source_commit=source,bundled_ptx=dict(descriptor,root=prefix[:-1],source_commit=source),
                 binding_origin={name:dict(origin='qualified-ptx-bundle',**descriptor) for name in hashes})
    with zipfile.ZipFile(wheel) as archive:
        result=release_gate.inspect_bundled_ptx(archive,payload,hashes)
        assert result['extension_hashes']==hashes
        for key in ('bundled_ptx','binding_origin'):
            bad=copy.deepcopy(payload);del bad[key]
            with pytest.raises(ValueError):release_gate.inspect_bundled_ptx(archive,bad,hashes)
        bad=copy.deepcopy(payload);bad['source_commit']='0'*40
        with pytest.raises(ValueError):release_gate.inspect_bundled_ptx(archive,bad,hashes)


def test_dirty_baseline_manifest_refused_even_with_matching_admission_digest(tmp_path):
    dirs,baseline,admission=inputs(tmp_path)
    path=baseline/pw.gpu_plugins.BASELINE_MANIFEST;doc=json.loads(path.read_text());doc['source_dirty']=True;path.write_text(json.dumps(doc))
    doc=json.loads(admission.read_text());doc['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();admission.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,match='clean source manifest'):pack(tmp_path,dirs,admission)


def test_separate_experimental_wheel_cannot_share_installed_bundle_files(tmp_path):
    dirs,baseline,admission=inputs(tmp_path);vendor=pack(tmp_path,dirs,admission)
    args=[arg for directory in dirs for arg in ('--set',directory)]
    assert pw.main(args+['--wheels','nvidia-ptx80','--out',str(tmp_path/'experimental')],_gates=False)==0
    experimental=next((tmp_path/'experimental').glob('*.whl'))
    problems=wheel_api_audit.split_audit([vendor,experimental])['problems']
    assert any('is in both' in problem for problem in problems)


def test_release_routes_only_native_sets_to_native_build_proof_verifier(tmp_path, monkeypatch):
    dirs,baseline,admission=inputs(tmp_path, include_byte_lm=True)
    class NativeProofReached(Exception):
        pass
    def native_proof(sets, proofs, version, **kwargs):
        assert {(s.vendor,s.arch) for s in sets} == {('cuda','sm_89'),('cuda','sm_90a')}
        assert kwargs['required'] == {('cuda','sm_89'),('cuda','sm_90')}
        raise NativeProofReached()
    monkeypatch.setattr(pw,'release_inventory',native_proof)
    args=[arg for directory in dirs for arg in ('--set',directory)]
    with pytest.raises(NativeProofReached):
        pw.main(args+['--profile','release-split','--wheels','nvidia','--bundle-ptx-admission',str(admission),
                     '--out',str(tmp_path/'out')],_gates=False)


# ---- manifest-only bundle: the FAST/DETERMINISTIC fallback, no admission ----

def test_descriptor_accepts_manifest_only_and_nothing_else():
    gp = pw.gpu_plugins
    gp.validate_bundle_descriptor(dict(manifest_sha256='a'*64))
    gp.validate_bundle_descriptor(dict(manifest_sha256='a'*64, admission_sha256='b'*64))
    assert gp.bundled_ptx_origin(dict(manifest_sha256='a'*64)) == 'ptx-fallback-bundle'
    assert gp.bundled_ptx_origin(dict(manifest_sha256='a'*64, admission_sha256='b'*64)) == 'qualified-ptx-bundle'
    for bad in ({}, dict(admission_sha256='b'*64), dict(manifest_sha256='A'*64), dict(manifest_sha256='a'*63),
                dict(manifest_sha256='a'*64, admission_sha256=None), dict(manifest_sha256='a'*64, other='b'*64), None):
        with pytest.raises(ValueError):
            gp.validate_bundle_descriptor(bad)


def test_bundle_bytes_and_descriptor_must_agree_about_the_admission(tmp_path):
    dirs,baseline,admission=inputs(tmp_path);gp=pw.gpu_plugins
    manifest=(baseline/gp.BASELINE_MANIFEST).read_bytes()
    files={row['file']:row['sha256'] for row in json.loads(manifest)['files']}
    only=dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
    assert gp.validate_bundled_ptx(only,manifest,None,files) is None
    with pytest.raises(ValueError,match='disagree'):gp.validate_bundled_ptx(only,manifest,admission.read_bytes(),files)
    both=dict(only,admission_sha256=hashlib.sha256(admission.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='disagree'):gp.validate_bundled_ptx(both,manifest,None,files)
    with pytest.raises(ValueError):gp.validate_bundled_ptx(only,manifest+b' ',None,files)


def test_manifest_only_bundle_ships_ptx_with_no_admission(tmp_path):
    dirs,baseline,admission=inputs(tmp_path)
    wheel=pack(tmp_path,dirs,None)
    files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    manifest=(baseline/pw.gpu_plugins.BASELINE_MANIFEST).read_bytes()
    for row in json.loads(manifest)['files']:
        assert files[prefix+row['file']]==(baseline/row['file']).read_bytes()
    assert files[prefix+pw.gpu_plugins.BASELINE_MANIFEST]==manifest
    assert prefix+pw.gpu_plugins.BASELINE_ADMISSION not in files
    marker=json.loads(next(data for name,data in files.items() if name.endswith('/gpu_plugin.json')))
    assert marker['arches']==['sm_89','sm_90a']
    assert marker['bundled_ptx']==dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
    report=wheel_api_audit.split_audit([wheel])
    assert not report['problems']
    assert not any('nvidia_ptx80-' in path.name for path in (tmp_path/'out').glob('*.whl'))


def test_bundle_flags_are_exclusive_and_keep_the_other_refusals(tmp_path):
    dirs,baseline,admission=inputs(tmp_path)
    with pytest.raises(SystemExit,match='not both'):pack(tmp_path,dirs,admission,extra=('--bundle-ptx',))
    with pytest.raises(SystemExit,match='excludes the separate'):pack(tmp_path,dirs,None,wheels='nvidia,nvidia-ptx80')
    path=baseline/pw.gpu_plugins.BASELINE_MANIFEST;doc=json.loads(path.read_text());doc['source_dirty']=True;path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,match='clean source manifest'):pack(tmp_path,dirs,None)


@pytest.mark.parametrize('mutation', ['binary','manifest','extra','stray-admission','marker','marker-claims-admission'])
def test_audit_detects_manifest_only_bundle_tampering(tmp_path,mutation):
    dirs,baseline,admission=inputs(tmp_path);wheel=pack(tmp_path,dirs,None);files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    name=next(n for n in files if n.endswith('/gpu_plugin.json'));doc=json.loads(files[name])
    if mutation=='binary':files[next(n for n in files if n.startswith(prefix) and n.endswith('.so'))]+=b'tamper'
    elif mutation=='manifest':files[prefix+pw.gpu_plugins.BASELINE_MANIFEST]+=b' '
    elif mutation=='extra':files[prefix+'unexpected.txt']=b'undeclared'
    elif mutation=='stray-admission':files[prefix+pw.gpu_plugins.BASELINE_ADMISSION]=admission.read_bytes()
    elif mutation=='marker':del doc['bundled_ptx'];files[name]=json.dumps(doc).encode()
    else:
        doc['bundled_ptx']['admission_sha256']=hashlib.sha256(admission.read_bytes()).hexdigest();files[name]=json.dumps(doc).encode()
    rewrite(wheel,files)
    assert wheel_api_audit.split_audit([wheel])['problems']


def test_release_gate_checks_a_manifest_only_bundle_and_never_calls_it_qualified(tmp_path):
    dirs,baseline,admission=inputs(tmp_path);wheel=pack(tmp_path,dirs,None);files=members(wheel)
    prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    hashes={n:hashlib.sha256(data).hexdigest() for n,data in files.items() if n.startswith(prefix) and n.endswith('.so')}
    marker=json.loads(next(data for name,data in files.items() if name.endswith('/gpu_plugin.json')))
    descriptor=marker['bundled_ptx'];source=json.loads(files[prefix+pw.gpu_plugins.BASELINE_MANIFEST])['source_commit']
    payload=dict(source_commit=source,bundled_ptx=dict(descriptor,root=prefix[:-1],source_commit=source),
                 binding_origin={name:dict(origin='ptx-fallback-bundle',**descriptor) for name in hashes})
    with zipfile.ZipFile(wheel) as archive:
        assert release_gate.inspect_bundled_ptx(archive,payload,hashes)['extension_hashes']==hashes
        bad=copy.deepcopy(payload)
        for row in bad['binding_origin'].values():row['origin']='qualified-ptx-bundle'
        with pytest.raises(ValueError):release_gate.inspect_bundled_ptx(archive,bad,hashes)
        bad=copy.deepcopy(payload);bad['bundled_ptx']['admission_sha256']='0'*64
        with pytest.raises((ValueError,KeyError)):release_gate.inspect_bundled_ptx(archive,bad,hashes)
        bad=copy.deepcopy(payload);bad['source_commit']='0'*40
        with pytest.raises(ValueError):release_gate.inspect_bundled_ptx(archive,bad,hashes)
    # An admitted wheel whose inventory drops the admission is not a fallback bundle.
    admitted=pack(tmp_path/'admitted',dirs,admission)
    with zipfile.ZipFile(admitted) as archive:
        with pytest.raises(ValueError):release_gate.inspect_bundled_ptx(archive,payload,hashes)


def test_release_profile_records_fallback_origin_for_a_manifest_only_bundle(tmp_path, monkeypatch):
    dirs,baseline,admission=inputs(tmp_path, include_byte_lm=True)
    source=json.loads((baseline/pw.gpu_plugins.BASELINE_MANIFEST).read_text())['source_commit']
    seen={}
    def native_proof(sets, proofs, version, **kwargs):
        assert {(s.vendor,s.arch) for s in sets} == {('cuda','sm_89'),('cuda','sm_90a')}
        return dict(source_commit=source, extensions={}, binding_origin={}, sets={},
                    reuse=dict(built=0, reused=0, from_release=None))
    class Captured(Exception):
        pass
    def capture(out, kinds, entries, generated, dist, proj, version, tag, inventory, sets, readme, bundled_ptx=None):
        seen.update(inventory=inventory, bundled_ptx=bundled_ptx)
        raise Captured()
    monkeypatch.setattr(pw,'release_inventory',native_proof)
    monkeypatch.setattr(pw,'write_split',capture)
    # The tree-at-commit gate is not under test; it would tie this case to a clean checkout.
    monkeypatch.setattr(pw,'require_shipped_python_at_commit',lambda entries,commit,**kw:None)
    args=[arg for directory in dirs for arg in ('--set',directory)]
    with pytest.raises(Captured):
        pw.main(args+['--profile','release-split','--wheels','nvidia','--bundle-ptx','--out',str(tmp_path/'out')],_gates=False)
    assert set(seen['bundled_ptx'])=={'manifest_sha256'}
    assert seen['inventory']['bundled_ptx']==dict(seen['bundled_ptx'],root=pw.gpu_plugins.BUNDLED_PTX_ROOT,source_commit=source)
    origins={row['origin'] for name,row in seen['inventory']['binding_origin'].items() if '/cuda_ptx/' in name}
    assert origins=={'ptx-fallback-bundle'}
