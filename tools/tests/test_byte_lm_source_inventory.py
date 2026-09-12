"""Authored host-only provenance fixtures; root executes, never subagents."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / 'byte_lm_real_text_capture.py'
_spec = importlib.util.spec_from_file_location('byte_capture_inventory', _path)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


def fixture_source(root):
    names = [name + '/__init__.mojo' for name in capture.SOURCE_DIRECTORIES]
    names += list(capture.REQUIRED_MAMBA_SOURCES)
    names += ['bindings/_mojolearn_byte_lm.mojo', 'bindings/build_byte_lm.sh',
              'python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
              'tools/byte_lm_real_text_capture.py', 'tools/byte_lm_gradient_oracle.py',
              # DEVIATION 2682: the shape module decides what the capture even
              # means, so it belongs in the inventory the run is pinned to.
              'tools/byte_lm_shape.py',
              # The surface digests this one and the inventory did not carry it,
              # which made every capture taken after 9bf5115a unadmittable.
              'python/mojolearn/_byte_lm_config.py',
              'pixi.toml', 'pixi.lock']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture source\n')


def test_inventory_includes_nested_mamba_and_tracks_changes(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    fixture_source(root)
    monkeypatch.setattr(capture, 'ROOT', root)
    extra = root / 'mamba/impl/other/new_dependency.mojo'
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b'before')
    first = capture.source_inventory()
    assert all(name in first for name in capture.REQUIRED_MAMBA_SOURCES)
    assert first['mamba/impl/other/new_dependency.mojo'] == capture.sha(b'before')
    extra.write_bytes(b'after')
    second = capture.source_inventory()
    assert second['mamba/impl/other/new_dependency.mojo'] == capture.sha(b'after')
    assert first != second


@pytest.mark.parametrize('missing', capture.REQUIRED_MAMBA_SOURCES)
def test_required_transitive_representative_missing_refuses(tmp_path, monkeypatch, missing):
    root = tmp_path.resolve()
    fixture_source(root)
    monkeypatch.setattr(capture, 'ROOT', root)
    (root / missing).unlink()
    with pytest.raises(ValueError, match='Missing required transitive'):
        capture.source_inventory()


def test_trimmed_transport_missing_whole_mamba_tree_refuses(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(capture, 'ROOT', root)
    with pytest.raises(ValueError, match='Missing required transitive'):
        capture.source_inventory()


def test_other_empty_source_directory_refuses(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    fixture_source(root)
    monkeypatch.setattr(capture, 'ROOT', root)
    (root / 'embedding/__init__.mojo').unlink()
    with pytest.raises(ValueError, match='Missing or empty.*embedding'):
        capture.source_inventory()
