"""THE PTX SLOT OF mojolearn-nvidia (Andrew 2026-10-10: PTX is a normal target; no flag).

The cuda/sm_80 set is the PTX set. It packs into the NVIDIA vendor wheel at
cuda_ptx/sm_80 beside the native sets, the vendor marker binds its manifest by
SHA256, and a release-split pack proves it with its own build proof like any
set. Synthetic inert bytes only.
"""
import hashlib
import json
import zipfile

import pytest
from test_split_wheels import make_sets, members, pw, write_ptx_manifest, wheel_api_audit


def inputs(root, include_byte_lm=False):
    dirs = make_sets(root, sets=(("cuda", "sm_89"), ("cuda", "sm_80")), include_byte_lm=include_byte_lm)
    return dirs, root / 'sets/cuda/sm_80'


def pack(root, dirs, wheels='nvidia', extra=()):
    args = [arg for directory in dirs for arg in ('--set', directory)]
    args += ['--wheels', wheels, '--out', str(root/'out'), *extra]
    assert pw.main(args, _gates=False) == 0
    return next((root/'out').glob('mojolearn_nvidia-*.whl'))


def rewrite(path, files):
    with zipfile.ZipFile(path,'w') as archive:
        for member,data in files.items():archive.writestr(member,data)


def test_ptx_set_lands_at_cuda_ptx_with_its_bytes_and_a_native_marker(tmp_path):
    dirs,ptx=inputs(tmp_path)
    wheel=pack(tmp_path,dirs)
    files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    assert prefix=='mojolearn/cuda_ptx/sm_80/'
    manifest=(ptx/pw.gpu_plugins.BASELINE_MANIFEST).read_bytes()
    for row in json.loads(manifest)['files']:
        assert files[prefix+row['file']]==(ptx/row['file']).read_bytes()
    assert files[prefix+pw.gpu_plugins.BASELINE_MANIFEST]==manifest
    marker=json.loads(next(data for name,data in files.items() if name.endswith('/gpu_plugin.json')))
    assert marker['arches']==['sm_89']
    assert marker['bundled_ptx']==dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
    assert marker['ptx']==dict(arch='sm_80',directory='cuda_ptx')
    assert not wheel_api_audit.split_audit([wheel])['problems']
    assert [p.name.split('-')[0] for p in (tmp_path/'out').glob('*.whl')]==['mojolearn_nvidia']


def test_no_separate_ptx_payload_or_bundle_flags_remain():
    gp=pw.gpu_plugins
    assert 'nvidia-ptx80' not in gp.PAYLOADS
    assert [r['profile'] for r in gp.distribution_rows(include_experimental=True)]==['nvidia','amd']
    assert gp.owns_member('nvidia','mojolearn/cuda_ptx/sm_80/_mojolearn_x.so')
    assert gp.owns_member('nvidia','mojolearn/cuda/sm_80/identical/_mojolearn_x.so')
    assert gp.installed_member('mojolearn/cuda/sm_80/_mojolearn_x.so')=='mojolearn/cuda_ptx/sm_80/_mojolearn_x.so'
    assert not gp.owns_member('amd','mojolearn/cuda_ptx/sm_80/_mojolearn_x.so')
    assert gp.required_sets('cuda')=={('cuda','sm_89'),('cuda','sm_80')}
    with pytest.raises(SystemExit):
        pw.main(['--set','x','--bundle-ptx'],_gates=False)


def test_descriptor_is_the_manifest_digest_and_nothing_else():
    gp = pw.gpu_plugins
    gp.validate_bundle_descriptor(dict(manifest_sha256='a'*64))
    for bad in ({}, dict(manifest_sha256='a'*64, admission_sha256='b'*64), dict(admission_sha256='b'*64),
                dict(manifest_sha256='A'*64), dict(manifest_sha256='a'*63), dict(manifest_sha256='a'*64, other='b'*64), None):
        with pytest.raises(ValueError):
            gp.validate_bundle_descriptor(bad)


def test_validate_bundled_ptx_binds_manifest_bytes_and_files(tmp_path):
    dirs,ptx=inputs(tmp_path);gp=pw.gpu_plugins
    manifest=(ptx/gp.BASELINE_MANIFEST).read_bytes()
    files={row['file']:row['sha256'] for row in json.loads(manifest)['files']}
    descriptor=dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
    assert gp.validate_bundled_ptx(descriptor,manifest,files)['schema']=='mojolearn.ptx-set.v2'
    with pytest.raises(ValueError,match='differ from the vendor marker'):gp.validate_bundled_ptx(descriptor,manifest+b' ',files)
    with pytest.raises(ValueError,match='does not match'):gp.validate_bundled_ptx(descriptor,manifest,dict(files,extra='0'*64))
    doc=json.loads(manifest)
    for field in ('experimental','identical_qualified','qualification_required'):
        assert field not in doc


def test_dirty_ptx_manifest_refused(tmp_path):
    dirs,ptx=inputs(tmp_path)
    write_ptx_manifest(ptx,dirty=True)
    with pytest.raises(SystemExit,match='clean source manifest'):pack(tmp_path,dirs)


@pytest.mark.parametrize('mutation', ['binary','manifest','extra','marker','marker-claims-admission','marker-other-sha'])
def test_audit_detects_ptx_slot_tampering(tmp_path,mutation):
    dirs,ptx=inputs(tmp_path);wheel=pack(tmp_path,dirs);files=members(wheel);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    name=next(n for n in files if n.endswith('/gpu_plugin.json'));doc=json.loads(files[name])
    if mutation=='binary':files[next(n for n in files if n.startswith(prefix) and n.endswith('.so'))]+=b'tamper'
    elif mutation=='manifest':files[prefix+pw.gpu_plugins.BASELINE_MANIFEST]+=b' '
    elif mutation=='extra':files[prefix+'unexpected.txt']=b'undeclared'
    elif mutation=='marker':del doc['bundled_ptx'];files[name]=json.dumps(doc).encode()
    elif mutation=='marker-claims-admission':doc['bundled_ptx']['admission_sha256']='0'*64;files[name]=json.dumps(doc).encode()
    else:doc['bundled_ptx']['manifest_sha256']='0'*64;files[name]=json.dumps(doc).encode()
    rewrite(wheel,files)
    assert wheel_api_audit.split_audit([wheel])['problems']


def test_audit_requires_the_ptx_slot_in_the_nvidia_wheel(tmp_path):
    dirs,ptx=inputs(tmp_path);wheel=pack(tmp_path,dirs);prefix=pw.gpu_plugins.BUNDLED_PTX_ROOT+'/'
    files={n:b for n,b in members(wheel).items() if not n.startswith(prefix)}
    rewrite(wheel,files)
    assert any('PTX slot' in p for p in wheel_api_audit.split_audit([wheel])['problems'])


def test_release_split_proves_the_ptx_set_like_any_set(tmp_path, monkeypatch):
    dirs,ptx=inputs(tmp_path, include_byte_lm=True)
    class InventoryReached(Exception):
        pass
    def inventory(sets, proofs, version, **kwargs):
        assert {(s.vendor,s.arch) for s in sets} == {('cuda','sm_89'),('cuda','sm_80')}
        assert kwargs['required'] == {('cuda','sm_89'),('cuda','sm_80')}
        assert len(proofs) == 2
        raise InventoryReached()
    monkeypatch.setattr(pw,'release_inventory',inventory)
    args=[arg for directory in dirs for arg in ('--set',directory)]
    with pytest.raises(InventoryReached):
        pw.main(args+['--profile','release-split','--wheels','nvidia','--build-proof','cuda-sm_89.json',
                     '--build-proof','cuda-sm_80.json','--out',str(tmp_path/'out')],_gates=False)


def test_release_split_refuses_without_the_ptx_proof(tmp_path):
    dirs,ptx=inputs(tmp_path, include_byte_lm=True)
    args=[arg for directory in dirs for arg in ('--set',directory)]
    with pytest.raises(SystemExit,match='proof'):
        pw.main(args+['--profile','release-split','--wheels','nvidia','--build-proof',str(tmp_path/'cuda-sm_89.json'),
                     '--out',str(tmp_path/'out')],_gates=False)


def test_release_split_refuses_a_ptx_set_of_another_commit(tmp_path, monkeypatch):
    dirs,ptx=inputs(tmp_path, include_byte_lm=True)
    write_ptx_manifest(ptx, source='c'*40)
    monkeypatch.setattr(pw,'release_inventory',lambda sets, proofs, version, **kw: dict(
        source_commit='d'*40, extensions={}, binding_origin={}, sets={}, reuse=dict(built=0, reused=0, from_release=None)))
    args=[arg for directory in dirs for arg in ('--set',directory)]
    with pytest.raises(SystemExit,match='different commits'):
        pw.main(args+['--profile','release-split','--wheels','nvidia','--out',str(tmp_path/'out')],_gates=False)
