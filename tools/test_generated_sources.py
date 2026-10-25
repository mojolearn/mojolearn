"""Generated tokenizer table: exact regeneration/verification and its use in the compile evidence checks."""
import hashlib
import json
from pathlib import Path

import pytest

import generated_sources as gs
import release_compile_subunit_evidence as sub
import release_retained_compile_evidence as compiled

REAL = json.loads((Path(__file__).resolve().parent / 'test_release_host_build_tokenizer_0857.json').read_text())
TABLE = REAL['generated'][gs.UNICODE_TABLE].encode()
INPUTS = {k: v.encode() for k, v in REAL['generator_inputs'].items()}
SHA = hashlib.sha256(TABLE).hexdigest()


@pytest.fixture(autouse=True)
def fresh_cache():
    gs._EXPECTED.clear()
    yield
    gs._EXPECTED.clear()


class FakeTree:
    def __init__(self, files):
        self.data = files
        self.files = {k: k for k in files}

    def sha256(self, rel):
        return hashlib.sha256(self.data[rel]).hexdigest() if rel in self.data else None

    def text(self, rel):
        return self.data[rel].decode()


def test_expected_sha_regenerates_or_verifies_the_checkout_copy(tmp_path, monkeypatch):
    import unicodedata
    if unicodedata.unidata_version == gs._pins(INPUTS[gs.UNICODE_PIN])[0]:
        assert gs.regenerate(gs.UNICODE_TABLE, INPUTS) == TABLE  # the droplet's file, byte for byte
        assert gs.expected_sha(gs.UNICODE_TABLE, INPUTS) == SHA
        gs._EXPECTED.clear()
    monkeypatch.setattr(gs, 'regenerate', lambda rel, inputs: None)
    with pytest.raises(ValueError, match='no generated copy'):
        gs.expected_sha(gs.UNICODE_TABLE, INPUTS, [tmp_path / 'absent.mojo'])
    copy = tmp_path / 'unicode_table_generated.mojo'
    copy.write_bytes(TABLE.replace(b'L\\t0041', b'L\\t0042', 1))
    with pytest.raises(ValueError, match='pinned sha256'):
        gs.expected_sha(gs.UNICODE_TABLE, INPUTS, [copy])
    copy.write_bytes(TABLE)
    assert gs.expected_sha(gs.UNICODE_TABLE, INPUTS, [copy]) == SHA


def test_apple_pack_tree_key_may_include_only_the_exact_generated_table(tmp_path, monkeypatch):
    files = {'a/x.mojo': b'fn x(): pass\n', 'bindings/merged_build.sh': b'echo\n', **INPUTS}
    tree = FakeTree(files)
    plain = sub._tree_digest(tree, 'bindings/merged_build.sh')
    (tmp_path / gs.UNICODE_TABLE).parent.mkdir(parents=True)
    (tmp_path / gs.UNICODE_TABLE).write_bytes(TABLE)
    generated = sub._generated_shas(tree, tmp_path)
    assert generated == {gs.UNICODE_TABLE: SHA}
    walked = sub._tree_digest(tree, 'bindings/merged_build.sh', generated)
    assert walked['files'] == plain['files'] + 1 and walked['digest'] != plain['digest']
    # the same digest bincache's tree walk computes over a checkout holding the table
    rows = sorted([[r, hashlib.sha256(files[r]).hexdigest()] for r in files if r.endswith('.mojo')]
                  + [['bindings/merged_build.sh', hashlib.sha256(files['bindings/merged_build.sh']).hexdigest()],
                     [gs.UNICODE_TABLE, SHA]], key=lambda r: Path(r[0]).parts)
    assert gs.closure_hash(rows) == walked['digest']
    with pytest.raises(ValueError, match='tracked'):
        sub._tree_digest(FakeTree(dict(files, **{gs.UNICODE_TABLE: TABLE})), 'bindings/merged_build.sh', generated)
    assert sub._generated_shas(FakeTree({'a/x.mojo': b''}), tmp_path) == {}


def test_development_generated_source_is_checked_not_skipped(tmp_path):
    tree = FakeTree(dict(INPUTS))
    (tmp_path / gs.UNICODE_TABLE).parent.mkdir(parents=True)
    (tmp_path / gs.UNICODE_TABLE).write_bytes(TABLE)
    assert compiled.generated_sha(tree, gs.UNICODE_TABLE, tmp_path) == SHA
    with pytest.raises(ValueError, match='no tracked generator'):
        compiled.generated_sha(FakeTree({}), gs.UNICODE_TABLE, tmp_path)
