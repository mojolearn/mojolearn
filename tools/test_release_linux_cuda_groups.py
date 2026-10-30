import pytest
import release_linux_cuda_groups as groups


def env(leg):
    arch, fmt = groups.LEGS[leg]
    return dict(MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_CUDA_CODE_FORMAT=fmt,
                MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_TARGET_COLUMN='nvidia',
                MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_PACKAGE_BYTE_LM='1',
                MOJOLEARN_COMPILE_JOBS='16')


@pytest.mark.parametrize('leg', groups.LEGS)
def test_exact_existing_target_recipe(leg):
    groups.environment_policy(env(leg), leg)


@pytest.mark.parametrize('leg', groups.LEGS)
@pytest.mark.parametrize('key,value', [
    ('MOJOLEARN_GPU_ARCHS', 'sm_86'), ('MOJOLEARN_TARGET_COLUMN', 'cpu'),
    ('MOJOLEARN_NUMERIC_MODE', 'fast'), ('MOJOLEARN_LINUX_CPU', 'native'),
    ('MOJOLEARN_PACKAGE_BYTE_LM', '0'), ('MOJOLEARN_EXTRA_FLAGS', '-O0'),
    ('MOJOLEARN_CUDA_CODE_FORMAT', 'unknown')])
def test_changed_recipe_refused(leg, key, value):
    actual = env(leg); actual[key] = value
    with pytest.raises(ValueError): groups.environment_policy(actual, leg)


def test_native_execution_never_substitutes_for_ptx_recipe():
    with pytest.raises(ValueError): groups.environment_policy(env('cuda/sm_89'), 'cuda/sm_80')


def test_incomplete_individual_group_set_refused(tmp_path):
    with pytest.raises(ValueError, match='coverage'):
        groups.device_groups([], tmp_path, 'a'*40, tmp_path, 'b'*64, 'patchelf', None, 'cuda/sm_89')


@pytest.fixture
def byte_proof(tmp_path):
    import hashlib
    (tmp_path/'host').mkdir(); (tmp_path/'identical').mkdir(); (tmp_path/'.libs').mkdir()
    files = [('host/h.so', b'host'), ('identical/g.so', b'gpu'), ('.libs/r.so', b'runtime')]
    for name, raw in files: (tmp_path/name).write_bytes(raw)
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    proof = dict(producer_inputs={'leg':'cuda/sm_80'},host_extension={'mojolearn/cuda/sm_80/host/h.so':digest(b'host')},
                 extensions={'mojolearn/cuda/sm_80/identical/g.so':digest(b'gpu')},
                 runtime_libraries={'r.so':digest(b'runtime')},merged_indexes={},adapters={})
    return proof,tmp_path


def test_ptx_no_adapters_by_design_byte_boundary(byte_proof):
    proof,root=byte_proof
    assert groups.validate_bytes(proof,root)


@pytest.mark.parametrize('change',['raw','prefix','index','runtime','adapter','leg'])
def test_packed_changed_or_fabricated_bytes_refused(byte_proof,change):
    proof,root=byte_proof
    if change=='raw': (root/'identical/g.so').write_bytes(b'changed')
    elif change=='prefix': proof['extensions']={'mojolearn/cuda/sm_89/identical/g.so':next(iter(proof['extensions'].values()))}
    elif change=='index': proof['merged_indexes']={'fabricated-index':{}}
    elif change=='runtime': (root/'.libs/r.so').unlink()
    elif change=='adapter':
        (root/'_adapters').mkdir(); (root/'_adapters/adapters.json').write_bytes(b'{}')
    else: proof['producer_inputs']['leg']='cuda/sm_86'
    with pytest.raises(ValueError):groups.validate_bytes(proof,root)


def test_cuda_project_dispatch_does_not_fall_back_to_amd(monkeypatch):
    import release_linux_group_composition as composition
    calls=[]
    monkeypatch.setattr(groups,'project',lambda inputs,*args: calls.append(inputs['leg']) or {'checked':True})
    for leg in groups.LEGS:assert composition.project({'leg':leg},'/source','patchelf')=={'checked':True}
    assert calls==list(groups.LEGS)


def test_raw_development_images_do_not_imply_native_postcompile():
    from release_linux_group_composition import Evidence
    with pytest.raises(ValueError,match='postcompile evidence is required'):
        groups.native_postcompile({},'a'*40,Evidence())
