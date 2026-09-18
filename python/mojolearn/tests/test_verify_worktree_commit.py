"""Source identity must survive a linked worktree's .git indirection."""
from mojolearn import _verify


def test_linked_worktree_commit_is_found(tmp_path, monkeypatch):
    (tmp_path / '.git').write_text('gitdir: /external/repo/.git/worktrees/test\n')
    package = tmp_path / 'python/mojolearn'
    package.mkdir(parents=True)
    seen = []
    def command(args):
        seen.append(args)
        return 'a' * 40 if args[-2:] == ['rev-parse', 'HEAD'] else ''
    monkeypatch.setattr(_verify, '_cmd', command)
    assert _verify._git_commit(str(package)) == 'a' * 40
    assert seen[0][2] == str(tmp_path)


def test_worktree_dirty_state_is_not_hidden(tmp_path, monkeypatch):
    (tmp_path / '.git').write_text('gitdir: /external/repo/.git/worktrees/test\n')
    monkeypatch.setattr(_verify, '_cmd', lambda args:
        'b' * 40 if args[-2:] == ['rev-parse', 'HEAD'] else ' M changed.py')
    assert _verify._git_commit(str(tmp_path)) == 'b' * 40 + ' (WORKING TREE DIRTY)'
