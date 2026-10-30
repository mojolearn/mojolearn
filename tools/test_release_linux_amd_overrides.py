"""Strict composition policy fixtures; not GPU execution results."""
import copy
import json
from pathlib import Path
import sys
import pytest
import release_linux_amd_overrides as amd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging/linux'))
import merged_layout as layout


def test_effective_index_preserves_unmodified_ownership():
    old={'merged.so':['a','b'],'other.so':['c','d']}
    result=amd.effective_plan(old,['a','d'])
    assert result=={'merged.so':['b'],'other.so':['c'],'a.so':['a'],'d.so':['d']}
    assert old=={'merged.so':['a','b'],'other.so':['c','d']}


@pytest.mark.parametrize('names',[['missing'],['a','a'],['a','b']])
def test_unknown_duplicate_or_completely_unused_group_refused(names):
    with pytest.raises(ValueError):amd.effective_plan({'merged.so':['a','b']},names)


def test_only_explicit_plan_can_describe_own_file_override(tmp_path,monkeypatch):
    base={'merged.so':['a','b'],'other.so':['c']}
    monkeypatch.setattr(layout,'plan',lambda *args:base)
    index=amd.effective_plan(base,['a'])
    (tmp_path/layout.INDEX).write_text(json.dumps(dict(format=1,files=index)))
    for name in index:(tmp_path/name).write_bytes(b'fixture, not executable')
    assert layout.layout(tmp_path,['a','b','c'],glob='*.so')[0]=='merged'
    with pytest.raises(layout.LayoutError):layout.check_plan(tmp_path,'identical',True)
    layout.check_plan(tmp_path,'identical',True,overrides=['a'])
    assert layout.member_file(tmp_path,'a')==tmp_path/'a.so'
    assert layout.member_file(tmp_path,'b')==tmp_path/'merged.so'
    for options in ({'overrides':['b']},{'overrides':['unknown']},{'overrides':['a'],'platform':'macos'}):
        with pytest.raises(layout.LayoutError):layout.check_plan(tmp_path,'identical',True,**options)


ENV=dict(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_TARGET_COLUMN='amd',MOJOLEARN_GPU_ARCHS='gfx942',MOJOLEARN_LINUX_CPU='x86-64-v3',MOJOLEARN_PACKAGE_BYTE_LM='1')

def test_exact_amd_environment():amd.environment_policy(ENV)


@pytest.mark.parametrize('key,value',[
    ('MOJOLEARN_NUMERIC_MODE','fast'),('MOJOLEARN_TARGET_COLUMN','nvidia'),
    ('MOJOLEARN_GPU_ARCHS','gfx950'),('MOJOLEARN_LINUX_CPU','native'),
    ('MOJOLEARN_PACKAGE_BYTE_LM','0'),('MOJOLEARN_EXTRA_FLAGS','-O0')])
def test_different_amd_compilation_refused(key,value):
    with pytest.raises(ValueError):amd.environment_policy(dict(ENV,**{key:value}))


def test_actual_loader_uses_current_image_and_never_falls_back_to_old_export(tmp_path):
    import importlib.util
    source=Path(__file__).resolve().parents[1]/'python/mojolearn/_merged.py'
    spec=importlib.util.spec_from_file_location('retained_loader_boundary',source)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    record=amd.effective_plan({'_mojolearn_native.so':['_mojolearn','_mojolearn_other']},['_mojolearn'])
    (tmp_path/layout.INDEX).write_text(json.dumps(dict(format=1,files=record)))
    old=tmp_path/'_mojolearn_native.so';old.write_bytes(b'inert original merged image')
    current=tmp_path/'_mojolearn.so';current.write_bytes(b'inert current own image')
    assert loader.binding_path(str(tmp_path),'_mojolearn')==str(current)
    assert loader.binding_path(str(tmp_path),'_mojolearn_other')==str(old)
    current.unlink()
    assert loader.merged_file(str(tmp_path),'_mojolearn') is None
    assert loader.binding_path(str(tmp_path),'_mojolearn')==str(current)
    assert not loader.binding_exists(str(tmp_path),'_mojolearn')
