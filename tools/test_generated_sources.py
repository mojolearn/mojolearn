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


def archive_fixture():
    """One pinned record exercises exact rendering without network/UCD downloads."""
    import gen_archive_unicode as generator
    names = 'SPACE;000020;1F\n'
    payload = (1, names, f'{0:08X}{len(names):08X}', '00003001F', '', '')
    digest = gs.sha256(''.join(payload[1:]).encode())
    inputs = {
        gs.ARCHIVE_GENERATOR: Path(generator.__file__).read_bytes(),
        gs.ARCHIVE_PIN: f'comptime ARCHIVE_UNICODE_TABLE_SHA256_PINNED = "{digest}"\n'.encode(),
    }
    return generator.render_payload(*payload).encode(), inputs


def write_sources(root, files):
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_archive_exact_generated_bytes_and_retained_release_proof(tmp_path):
    table, inputs = archive_fixture()
    expected = gs.sha256(table)
    write_sources(tmp_path, dict(inputs, **{gs.ARCHIVE_TABLE: table}))
    assert gs.verify(gs.ARCHIVE_TABLE, table, inputs) == expected
    assert compiled.generated_sha(FakeTree(inputs), gs.ARCHIVE_TABLE, tmp_path) == expected
    assert sub._generated_shas(FakeTree(inputs), tmp_path) == {gs.ARCHIVE_TABLE: expected}
    assert gs.regenerate(gs.ARCHIVE_TABLE, inputs) is None
    # A table from a different tracked pin must not reuse the cached verdict.
    changed = dict(inputs)
    changed[gs.ARCHIVE_PIN] = b'comptime ARCHIVE_UNICODE_TABLE_SHA256_PINNED = "' + b'0' * 64 + b'"\n'
    with pytest.raises(ValueError, match='pinned sha256'):
        gs.expected_sha(gs.ARCHIVE_TABLE, changed, [tmp_path / gs.ARCHIVE_TABLE])


@pytest.mark.parametrize('before,after,reason', [
    (b'SPACE;', b'SPARE;', 'pinned sha256'),
    (b'NAME_COUNT = 1', b'NAME_COUNT = 2', 'count'),
    (b'Do not edit.', b'Changed header.', 'tracked generator'),
])
def test_archive_rejects_mutated_payload_metadata_and_rendering(before, after, reason):
    table, inputs = archive_fixture()
    with pytest.raises(ValueError, match=reason):
        gs.verify(gs.ARCHIVE_TABLE, table.replace(before, after), inputs)


def test_prepare_replaces_stale_bytes_retains_them_and_reuses_verified_output(tmp_path):
    table, inputs = archive_fixture()
    write_sources(tmp_path, dict(inputs, **{gs.ARCHIVE_TABLE: b'stale bytes'}))
    calls = []
    def generate(command, root, env, log):
        calls.append(command)
        log.write_text('generated fixture\n')
        (root / gs.ARCHIVE_TABLE).write_bytes(table)
        return 0
    evidence = tmp_path / 'evidence'
    rows = gs.prepare_sources(tmp_path, evidence, [gs.ARCHIVE_TABLE], runner=generate)
    assert rows[gs.ARCHIVE_TABLE]['state'] == 'generated'
    assert (evidence / 'replaced-generated' / gs.ARCHIVE_TABLE).read_bytes() == b'stale bytes'
    assert (evidence / 'generated' / gs.ARCHIVE_TABLE).read_bytes() == table
    assert (evidence / 'generated-inputs' / gs.ARCHIVE_GENERATOR).read_bytes() == inputs[gs.ARCHIVE_GENERATOR]
    rows = gs.prepare_sources(tmp_path, evidence / 'reuse', [gs.ARCHIVE_TABLE], runner=generate)
    assert rows[gs.ARCHIVE_TABLE]['state'] == 'reused' and len(calls) == 1


@pytest.mark.parametrize('exit_code', [0, 19])
def test_prepare_missing_output_or_failed_generator_never_qualifies(tmp_path, exit_code):
    _, inputs = archive_fixture()
    write_sources(tmp_path, inputs)
    evidence = tmp_path / 'evidence'
    with pytest.raises(gs.GeneratedSourceError, match='preparation failed'):
        gs.prepare_sources(tmp_path, evidence, [gs.ARCHIVE_TABLE], runner=lambda *_: exit_code)
    row = json.loads((evidence / 'generated-sources.json').read_text())['sources'][gs.ARCHIVE_TABLE]
    assert not row['verified'] and row['state'] == 'failed' and row['exit_code'] == exit_code
    assert row['retained']['state'] == 'missing'
    assert (evidence / 'generated-inputs' / gs.ARCHIVE_PIN).read_bytes() == inputs[gs.ARCHIVE_PIN]


def test_prepare_runs_independent_outputs_concurrently(tmp_path, monkeypatch):
    from threading import Barrier
    archive, archive_inputs = archive_fixture()
    write_sources(tmp_path, dict(INPUTS, **archive_inputs))
    tables = {gs.UNICODE_TABLE: TABLE, gs.ARCHIVE_TABLE: archive}
    barrier = Barrier(2, timeout=10)
    monkeypatch.setattr(gs, 'generation_workers', lambda _: 2)
    def generate(command, root, env, log):
        rel = next(rel for rel, expected in gs.COMMANDS.items() if command == expected)
        barrier.wait()
        (root / rel).write_bytes(tables[rel])
        log.write_text('fixture output\n')
        return 0
    rows = gs.prepare_sources(tmp_path, tmp_path / 'evidence', runner=generate)
    assert set(rows) == set(gs.GENERATED)
    assert all(row['verified'] and row['state'] == 'generated' for row in rows.values())


def test_archive_release_identity_requires_its_single_generator_file(tmp_path):
    import release_host_build as host
    import release_reuse as reuse
    table, inputs = archive_fixture()
    write_sources(tmp_path, inputs)
    inventory = sorted((rel, gs.sha256(data)) for rel, data in inputs.items())
    assert reuse.wanted(gs.ARCHIVE_GENERATOR) and reuse.wanted("tools/generated_sources.py")
    generator = host.generator_dir_digest(inventory, gs.ARCHIVE_GENERATOR)
    assert generator == reuse.dir_digest(tmp_path, gs.ARCHIVE_GENERATOR)
    committed = [[gs.ARCHIVE_PIN, gs.sha256(inputs[gs.ARCHIVE_PIN])]]
    rows = sorted(committed + [[gs.ARCHIVE_TABLE, gs.sha256(table)]])
    identity = {'script': 'build_fixture_host.sh', 'closure': {'digest': gs.closure_hash(committed)},
                'generators': None, 'generated_inputs': {gs.ARCHIVE_TABLE: generator}}
    stamp = {'scope': 'closure', 'script': identity['script'], 'digest': gs.closure_hash(rows)}
    host.check_member_stamp('fixture', stamp, identity, {'fixture': rows},
                            {gs.ARCHIVE_TABLE: gs.sha256(table)}, inventory)
    identity['generated_inputs'][gs.ARCHIVE_TABLE] = '0' * 64
    with pytest.raises(ValueError, match='generator inputs differ'):
        host.check_member_stamp('fixture', stamp, identity, {'fixture': rows},
                                {gs.ARCHIVE_TABLE: gs.sha256(table)}, inventory)
