"""Native archive dependencies must survive staging without host-library help.

Synthetic dependency readers exercise packaging graphs; these are not native
codec executions or installed-wheel qualification.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location('archive_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['linux', 'macos'])
def platform_stager(request, monkeypatch):
    family = request.param
    module = load('packaging/' + family + ('/stage_libs.py' if family == 'linux' else '/stage_dylibs.py'))
    monkeypatch.setattr(module, 'workers', lambda count: max(1, min(count, 2)))
    return family, module


def dependency_reader(family, module, monkeypatch, graph):
    def read(path):
        dependencies = graph.get(Path(path).name, [])
        return (dependencies, '$ORIGIN', '') if family == 'linux' else dependencies
    monkeypatch.setattr(module, 'elf_dynamic' if family == 'linux' else 'rpath_deps', read)


def present(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(('native-library:' + name).encode())


def test_dlopen_roots_and_their_transitive_dependencies_are_retained(platform_stager, tmp_path, monkeypatch):
    family, module = platform_stager
    codec = module.ARCHIVE_CODEC_LIBRARIES[-1]
    helper = 'libcodec_helper.so.1' if family == 'linux' else 'libcodec_helper.dylib'
    env_lib = tmp_path / 'env'
    present(env_lib, [*module.ARCHIVE_CODEC_LIBRARIES, helper])
    dependency_reader(family, module, monkeypatch, {
        'extension.so': [codec], codec: [helper], helper: [codec],
    })
    result = module.closure([tmp_path / 'extension.so'], env_lib, module.ARCHIVE_CODEC_LIBRARIES)
    names = result[0] if family == 'linux' else result
    assert set(names) == {*module.ARCHIVE_CODEC_LIBRARIES, helper}
    assert len(names) == len(set(names))
    if family == 'linux':
        assert {edge['needs'] for edge in result[1] if edge['from'] == 'native archive dlopen'} == set(module.ARCHIVE_CODEC_LIBRARIES)


def test_missing_codec_refuses_staging_even_if_no_static_dependency_names_it(platform_stager, tmp_path, monkeypatch):
    family, module = platform_stager
    env_lib = tmp_path / 'env'
    present(env_lib, module.ARCHIVE_CODEC_LIBRARIES[:-1])
    dependency_reader(family, module, monkeypatch, {})
    with pytest.raises(SystemExit, match='native archive codec .* missing'):
        module.closure([tmp_path / 'extension.so'], env_lib, module.ARCHIVE_CODEC_LIBRARIES)


def test_staged_closure_requires_codec_bytes_even_when_build_environment_has_them(platform_stager, tmp_path, monkeypatch):
    family, module = platform_stager
    env_lib, staged = tmp_path / 'env', tmp_path / 'staged'
    present(env_lib, module.ARCHIVE_CODEC_LIBRARIES)
    present(staged, module.ARCHIVE_CODEC_LIBRARIES[:-1])
    dependency_reader(family, module, monkeypatch, {})
    def check():
        if family == 'linux':
            module.verify_closed([tmp_path / 'extension.so'], staged, env_lib, module.ARCHIVE_CODEC_LIBRARIES)
        else:
            module.verify_closed(tmp_path / 'extension.so', staged, module.ARCHIVE_CODEC_LIBRARIES)
    with pytest.raises(SystemExit):
        check()
    present(staged, module.ARCHIVE_CODEC_LIBRARIES[-1:])
    check()
