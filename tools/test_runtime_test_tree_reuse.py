"""Input-closure tooling fixtures, not product execution evidence."""
from pathlib import Path
import pytest
import runtime_test_tree_reuse as reuse


def setup(tmp_path, files):
    for name, body in files.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
    return {n:reuse.digest(tmp_path/n) for n in files}


def test_unselected_test_change_does_not_change_selected_inputs(tmp_path):
    before=setup(tmp_path,{'test_one.py':'from .helper import value\nassert value\n','helper.py':'value=1\n','test_other.py':'assert True\n'})
    (tmp_path/'test_other.py').write_text('assert False\n')
    result=reuse.prove(before,tmp_path,['test_one.py'])
    assert result['status']=='proved'
    assert set(result['closure'])=={'test_one.py','helper.py'}


@pytest.mark.parametrize('changed',['helper.py','__init__.py','conftest.py','test_one.py','data.txt'])
def test_relevant_inputs_must_be_unchanged(tmp_path,changed):
    files={'test_one.py':'from . import helper\n','helper.py':'value=1\n','__init__.py':'','conftest.py':'','data.txt':'data'}
    before=setup(tmp_path,files);(tmp_path/changed).write_text(files[changed]+'\n')
    assert reuse.prove(before,tmp_path,['test_one.py'])['status']=='blocked'


@pytest.mark.parametrize('body',[
    'import importlib\nimportlib.import_module(name)',
    'from importlib import import_module as load\nload(name)',
    'import importlib as imp\nf=imp.import_module\nf(name)',
    'getattr(importlib,"import_module")(name)',
    'exec(code)', 'eval(code)',
    'from subprocess import run\nrun(command)',
    'from pathlib import Path\nPath(name).read_text()',
    'pytest_plugins = plugin_names',
])
def test_unresolved_dynamic_inputs_refused(tmp_path,body):
    before=setup(tmp_path,{'test_one.py':body,'test_other.py':'pass'})
    (tmp_path/'test_other.py').write_text('assert True')
    result=reuse.prove(before,tmp_path,['test_one.py'])
    assert result['status']=='blocked',result


def test_literal_dynamic_helper_is_followed(tmp_path):
    before=setup(tmp_path,{'test_one.py':'import importlib\nimportlib.import_module("mojolearn.tests.helper")','helper.py':'pass','test_other.py':'pass'})
    (tmp_path/'helper.py').write_text('value=2')
    assert reuse.prove(before,tmp_path,['test_one.py'])['status']=='blocked'


def test_new_package_cannot_shadow_old_module(tmp_path):
    before=setup(tmp_path,{'test_one.py':'import helper','helper.py':'pass'})
    setup(tmp_path,{'helper/__init__.py':'pass'})
    result=reuse.prove(before,tmp_path,['test_one.py'])
    assert result['status']=='blocked'
    assert any('resolution changed' in r['reason'] for r in result['unresolved'])


def test_relative_parent_helper(tmp_path):
    before=setup(tmp_path,{'nested/test_one.py':'from ..helper import value','helper.py':'value=1','test_other.py':'pass'})
    (tmp_path/'helper.py').write_text('value=2')
    assert reuse.prove(before,tmp_path,['nested/test_one.py'])['status']=='blocked'


def test_byte_identical_tree_needs_no_static_import_assumption(tmp_path):
    before=setup(tmp_path,{'test_one.py':'exec(dynamic)'})
    assert reuse.prove(before,tmp_path,['test_one.py'])['status']=='proved'


def test_forged_after_hashes_do_not_hide_selected_change(tmp_path):
    before=setup(tmp_path,{'test_one.py':'pass','test_other.py':'pass'})
    (tmp_path/'test_one.py').write_text('assert False')
    assert reuse.prove(before,tmp_path,['test_one.py'],after=before)['status']=='blocked'


def test_whole_file_guard_records_parameters_and_refuses_selection_gaps(tmp_path):
    from types import SimpleNamespace
    from qualify_native_runtime_installed import WholeFileCollectionGuard
    items=[SimpleNamespace(path=tmp_path/'test_one.py',nodeid='test_one.py::test_a[1]',user_properties=[]),
           SimpleNamespace(path=tmp_path/'test_one.py',nodeid='test_one.py::test_a[2]',user_properties=[])]
    guard=WholeFileCollectionGuard(tmp_path,['test_one.py'])
    guard.pytest_collection_finish(SimpleNamespace(items=items))
    assert guard.collected==['test_one.py::test_a[1]','test_one.py::test_a[2]']
    assert items[1].user_properties==[('installed_exact_node','test_one.py::test_a[2]')]
    for files, selected in [(['test_one.py'],[]),(['test_one.py'],items+items),
                            (['test_one.py','test_two.py'],items),(['test_two.py'],items)]:
        with pytest.raises(ValueError,match='whole-file collection'):
            WholeFileCollectionGuard(tmp_path,files).pytest_collection_finish(SimpleNamespace(items=selected))


@pytest.mark.parametrize('body',[
    'import importlib\nimportlib.__dict__[name](module)',
    '__import__("mojolearn.tests",fromlist=[name])',
    'import os\nos.system(command)',
    'import io\nio.FileIO(path)',
    'import pickle\npickle.loads(serialized)',
])
def test_indirect_loading_requires_separate_proof(tmp_path,body):
    before=setup(tmp_path,{'test_one.py':body,'test_other.py':'pass'})
    (tmp_path/'test_other.py').write_text('assert True')
    assert reuse.prove(before,tmp_path,['test_one.py'])['status']=='blocked'


def test_whole_file_collection_refuses_pytest_deselection(tmp_path):
    from types import SimpleNamespace
    import qualify_native_runtime_installed as q
    item=SimpleNamespace(path=tmp_path/'test_one.py',nodeid='test_one.py::test_a',user_properties=[])
    guard=q.WholeFileCollectionGuard(tmp_path,['test_one.py'])
    guard.pytest_deselected([SimpleNamespace(nodeid='test_one.py::test_b')])
    with pytest.raises(ValueError,match='whole-file collection'):
        guard.pytest_collection_finish(SimpleNamespace(items=[item]))
    env=q.test_environment({'PYTEST_ADDOPTS':'-k hidden','PYTEST_PLUGINS':'external'},'cpu','identical')
    assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
