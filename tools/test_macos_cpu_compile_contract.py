"""Synthetic Mach-O metadata and CPU/GPU scope tests, no product execution."""
import copy
import struct

import pytest

import macos_cpu_compile_contract as cpu
import release_requirement_dependencies as deps


def image_bytes(*, sdk=(26<<16)|(5<<8), minimum=11<<16, tool=3):
    build=struct.pack('<8I',0x32,32,1,minimum,sdk,1,tool,1267<<16)
    name=b'/usr/lib/libSystem.B.dylib\0'
    length=(24+len(name)+7)//8*8
    dependency=struct.pack('<6I',0xc,length,24,0,1356<<16,1<<16)+name
    dependency+=b'\0'*(length-len(dependency))
    name=b'/fixture/toolchain/lib\0'
    length=(12+len(name)+7)//8*8
    rpath=struct.pack('<3I',0x8000001c,length,12)+name
    rpath+=b'\0'*(length-len(rpath))
    body=build+dependency+rpath
    return struct.pack('<8I',0xfeedfacf,0x100000c,0,6,3,len(body),0,0)+body


def witness(tmp_path):
    path=tmp_path/'original.so'
    path.write_bytes(image_bytes())
    return cpu.contract('a'*64,'b'*64,cpu.read_image(path))


def test_reads_original_version_metadata_without_current_toolchain(tmp_path):
    value=witness(tmp_path)
    assert value['image']['build']==dict(platform=1,minimum_os=[11,0,0],sdk=[26,5,0],
                                       tools=[dict(tool=3,version=[1267,0,0])])
    assert value['image']['dependencies'][0]['current']==[1356,0,0]
    assert 'xcode' not in value and 'metal' not in value


@pytest.mark.parametrize('data,match', [
    (b'not a native binary','truncated'),
    (image_bytes()[:-1],'truncated'),
    (image_bytes(tool=1),'linker'),
    (image_bytes(minimum=12<<16),'minimum'),
    (image_bytes(sdk=0),'SDK'),
    (struct.pack('<I',0xcafebabe)+image_bytes()[4:],'ARM64'),
])
def test_rejects_missing_or_wrong_original_metadata(tmp_path,data,match):
    path=tmp_path/'original.so';path.write_bytes(data)
    with pytest.raises(ValueError,match=match):cpu.read_image(path)


def test_command_extent_cannot_escape_original_image(tmp_path):
    data=bytearray(image_bytes());struct.pack_into('<I',data,36,0xfffffff8)
    path=tmp_path/'original.so';path.write_bytes(data)
    with pytest.raises(ValueError,match='command size'):cpu.read_image(path)


def snapshot():
    cpu_key='osx-arm64/host/-/host/_mojolearn_core_host'
    gpu_key='osx-arm64/metal/apple/identical/_mojolearn'
    return {'bindings':[
        dict(key=cpu_key,identity=dict(target='osx-arm64',vendor='host',tier='host',
             flags={'target_column':'cpu'},toolchain={'pixi_lock_sha256':'b'*64},host_toolchain=None),
             unresolved=['no record of the Apple toolchain','some other missing dependency']),
        dict(key=gpu_key,identity=dict(target='osx-arm64',vendor='metal',tier='identical',
             flags={},toolchain={'pixi_lock_sha256':'b'*64},host_toolchain=None),
             unresolved=['no record of the Apple toolchain'])]}


def test_cpu_selection_keeps_gpu_and_other_missing_dependencies(tmp_path):
    original=snapshot();key=original['bindings'][0]['key'];value=witness(tmp_path)
    result=deps._selected_macos_cpu_toolchains(original,{'dependency_context':{'selected_macos_cpu_toolchains':{key:value}}})
    assert result['bindings'][0]['identity']['host_toolchain']==value
    assert result['bindings'][0]['unresolved']==['some other missing dependency']
    assert result['bindings'][1]==original['bindings'][1]
    assert original['bindings'][0]['identity']['host_toolchain'] is None


def test_cpu_witness_cannot_stand_in_for_metal_compiler(tmp_path):
    original=snapshot();key=original['bindings'][1]['key']
    with pytest.raises(ValueError,match='GPU/foreign'):
        deps._selected_macos_cpu_toolchains(original,{'dependency_context':{'selected_macos_cpu_toolchains':{key:witness(tmp_path)}}})


def test_cpu_lock_mismatch_refused(tmp_path):
    original=snapshot();key=original['bindings'][0]['key'];value=witness(tmp_path);value['pixi_lock_sha256']='c'*64
    with pytest.raises(ValueError,match='package lock'):
        deps._selected_macos_cpu_toolchains(original,{'dependency_context':{'selected_macos_cpu_toolchains':{key:value}}})


def test_cpu_unknown_binding_refused(tmp_path):
    with pytest.raises(ValueError,match='unknown bindings'):
        deps._selected_macos_cpu_toolchains(snapshot(),{'dependency_context':{'selected_macos_cpu_toolchains':{'unknown':witness(tmp_path)}}})


@pytest.mark.parametrize('field,value', [('cpu',True),('subtype',False),('filetype',True)])
def test_integer_image_fields_reject_booleans(tmp_path,field,value):
    result=witness(tmp_path);result['image'][field]=value
    with pytest.raises(ValueError):cpu.validate_contract(result)


@pytest.mark.parametrize('field,value', [('platform',True),('tools',[False]),
    ('tools',[{'tool':True,'version':[1267,0,0]}]),('minimum_os',[11,False,0])])
def test_build_fields_reject_malformed_types(tmp_path,field,value):
    result=witness(tmp_path);result['image']['build'][field]=value
    with pytest.raises(ValueError):cpu.validate_contract(result)


def test_dependency_command_rejects_non_integer(tmp_path):
    result=witness(tmp_path);result['image']['dependencies'][0]['command']=12.0
    with pytest.raises(ValueError):cpu.validate_contract(result)
