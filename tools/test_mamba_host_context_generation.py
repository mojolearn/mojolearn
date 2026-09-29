"""Device context reuse must not leak into generated CPU backward passes."""
from pathlib import Path

import pytest

import mamba_host_gen as gen


@pytest.mark.parametrize('source', gen._CONTEXT_SOURCES)
def test_context_translation_preserves_qualified_host_bytes(source):
    expected = Path(gen.ROOT, gen.OUT_DIR, gen.SOURCES[source] + '.mojo').read_text()
    actual = gen.generate(source)
    assert actual == expected
    assert 'core.neural_context import' not in actual
    assert 'process_ctx[' not in actual
    assert 'ctx = DeviceContext()' in actual


@pytest.mark.parametrize('source', gen._CONTEXT_SOURCES)
def test_changed_device_context_contract_refuses_generation(source, tmp_path, monkeypatch):
    text = Path(gen.ROOT, source).read_text()
    assert 'process_ctx[_DEVCTX_SLOT]()' in text
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text(text.replace('process_ctx[_DEVCTX_SLOT]()', 'other_process_ctx[_DEVCTX_SLOT]()'))
    monkeypatch.setattr(gen, 'ROOT', str(tmp_path))
    with pytest.raises(gen.GenError):
        gen.generate(source)
