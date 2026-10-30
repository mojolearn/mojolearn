"""Test-only source selection must never relabel or mutate product inputs."""
import hashlib
from pathlib import Path
import subprocess

import pytest

import qualify_native_runtime_installed as qualifier


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), '-c', 'user.name=fixture',
                                    '-c', 'user.email=fixture@example.invalid', *args], text=True).strip()


@pytest.fixture
def revisions(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    tests = root / 'python/mojolearn/tests'
    tests.mkdir(parents=True)
    (root / 'product.mojo').write_text('unchanged product\n')
    (root / '.gitattributes').write_text('python/mojolearn/tests export-ignore\n')
    (tests / 'test_one.py').write_text('original committed test\n')
    (tests / 'helper.py').write_text('shared helper\n')
    (tests / 'data').mkdir()
    (tests / 'data/binary').write_bytes(bytes(range(256)))
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'product and original tests')
    product = git(root, 'rev-parse', 'HEAD')
    (tests / 'test_one.py').write_text('corrected committed fixture\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'test-only correction')
    corrected = git(root, 'rev-parse', 'HEAD')
    (tests / 'test_one.py').write_text('uncommitted working copy must not execute\n')
    return root, product, corrected


def test_freeze_uses_exact_commit_even_with_dirty_checkout_and_export_ignore(revisions, tmp_path):
    root, product, corrected = revisions
    before = (root / 'product.mojo').read_bytes()
    for revision, expected in ((product, b'original committed test\n'),
                               (corrected, b'corrected committed fixture\n')):
        dest = tmp_path / revision
        hashes = qualifier.freeze_committed_tests(root, revision, dest)
        assert (dest / 'test_one.py').read_bytes() == expected
        assert hashes['test_one.py'] == hashlib.sha256(expected).hexdigest()
        assert (dest / 'data/binary').read_bytes() == bytes(range(256))
        assert set(hashes) == {'test_one.py', 'helper.py', 'data/binary'}
        assert not (dest / 'product.mojo').exists()
    assert (root / 'product.mojo').read_bytes() == before
    assert (root / 'python/mojolearn/tests/test_one.py').read_text().startswith('uncommitted')


@pytest.mark.parametrize('revision', ['HEAD', 'a' * 12, '../outside', 'G' * 40])
def test_only_full_commit_identifiers_are_accepted(revisions, tmp_path, revision):
    with pytest.raises(ValueError, match='full commit SHA'):
        qualifier.freeze_committed_tests(revisions[0], revision, tmp_path / 'frozen')


def test_symlink_cannot_escape_frozen_test_tree(revisions, tmp_path):
    root, _, _ = revisions
    (root / 'python/mojolearn/tests/escape.py').symlink_to('../../../product.mojo')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'symlink fixture')
    with pytest.raises(ValueError, match='regular files'):
        qualifier.freeze_committed_tests(root, git(root, 'rev-parse', 'HEAD'), tmp_path / 'frozen')


def test_existing_frozen_directory_is_not_overwritten(revisions, tmp_path):
    dest = tmp_path / 'frozen'
    dest.mkdir()
    (dest / 'original.log').write_text('retained\n')
    with pytest.raises(FileExistsError):
        qualifier.freeze_committed_tests(revisions[0], revisions[1], dest)
    assert (dest / 'original.log').read_text() == 'retained\n'


def test_worker_interpreter_identity_binds_actual_executable():
    import sys
    row = qualifier.interpreter_identity()
    assert row['version'] == sys.version
    assert row['executable_sha256'] == qualifier.digest(Path(sys.executable).resolve())
    assert row['implementation'] == sys.implementation.name
    assert row['soabi'] and row['platform']
