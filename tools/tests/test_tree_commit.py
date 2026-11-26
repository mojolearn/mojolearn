"""Tool tests for tools/ops/tree_commit.sh and the box jobs that refuse a wheel of another commit (2026-11-26).

tools/dev_pod.sh sync leaves a box tree's HEAD at the merge base with main and records the synced commit in
.git/devpod_synced. The jobs must compare a staged wheel's +<sha> with that commit, not with HEAD.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "tools/ops/tree_commit.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "tree"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x\n")
    _git(repo, "add", "f")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "c")
    return repo, _git(repo, "rev-parse", "HEAD")


def _sh(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'. "{LIB}"\n{script}'], capture_output=True, text=True, cwd=cwd)


def _record(repo: Path, commit: str, base: str) -> None:
    (repo / ".git/devpod_synced").write_text(f"commit={commit}\nworktree_dirty=0\nbase={base}\npatch_sha256=x\n")


def test_a_plain_checkout_holds_its_head(tmp_path):
    repo, head = _repo(tmp_path)
    assert _sh(f'tree_commit "{repo}"', tmp_path).stdout.strip() == head


def test_a_synced_tree_holds_the_synced_commit_and_a_stale_record_is_ignored(tmp_path):
    repo, head = _repo(tmp_path)
    synced = "ab" * 20
    _record(repo, synced, head)
    assert _sh(f'tree_commit "{repo}"', tmp_path).stdout.strip() == synced
    _record(repo, synced, "cd" * 20)  # the tree was checked out after that sync
    assert _sh(f'tree_commit "{repo}"', tmp_path).stdout.strip() == head
    assert _sh(f'tree_commit "{tmp_path / "nowhere"}"', tmp_path).stdout.strip() == ""


def test_wheels_match_only_the_trees_commit(tmp_path):
    repo, head = _repo(tmp_path)
    synced = "ab" * 20
    _record(repo, synced, head)
    good = f"mojolearn_core-0.8.67.dev0+{synced[:12]}-cp312-cp312-linux_x86_64.whl"
    other = f"mojolearn_nvidia-0.8.67.dev0+{head[:12]}-py3-none-linux_x86_64.whl"
    plain = "mojolearn_core-0.8.67-cp312-cp312-linux_x86_64.whl"
    ok = _sh(f'wheels_match_tree "{repo}" {good}', tmp_path)
    assert ok.returncode == 0 and ok.stdout.startswith("MATCH ")
    bad = _sh(f'wheels_match_tree "{repo}" {good} {other} {plain}', tmp_path)
    assert bad.returncode == 1
    lines = bad.stdout.splitlines()
    assert lines[0].startswith("MATCH") and "is commit" in lines[1] and "no +<sha>" in lines[2]
    assert _sh(f'wheels_match_tree "{repo}"', tmp_path).returncode == 1  # no wheel at all
    assert _sh(f'wheel_commit /x/{good}', tmp_path).stdout.strip() == synced[:12]


def test_the_proving_job_refuses_a_wheel_of_another_commit_before_installing(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "mojolearn_core-0.8.67.dev0+0123456789ab-cp312-cp312-linux_x86_64.whl").write_bytes(b"")
    env = dict(os.environ, MOJOLEARN_RESULTS_DIR=str(tmp_path / "res"), PV_WHEEL_DIR=str(wheels))
    run = subprocess.run(["bash", str(ROOT / "tools/ops/proving_job.sh")], capture_output=True, text=True, env=env,
                         timeout=120)
    assert run.returncode == 4, run.stdout + run.stderr
    assert "PROVING setup FAIL the wheels are not this tree's commit" in run.stdout
    assert not (tmp_path / "res/venv").exists()  # nothing was installed


def test_the_gpu_smoke_preflight_compares_wheels_with_the_trees_commit(tmp_path):
    head = _git(ROOT, "rev-parse", "HEAD")
    for sha, want in ((head[:12], "PREFLIGHT ok wheel mojolearn_"), ("0123456789ab", "PREFLIGHT FAIL mojolearn_")):
        wheels = tmp_path / sha
        wheels.mkdir()
        for name in (f"mojolearn_core-0.8.67.dev0+{sha}-cp312-cp312-linux_x86_64.whl",
                     f"mojolearn_nvidia-0.8.67.dev0+{sha}-py3-none-linux_x86_64.whl"):
            (wheels / name).write_bytes(b"")
        env = dict(os.environ, GS_WHEEL_DIR=str(wheels), MOJOLEARN_RESULTS_DIR=str(tmp_path / "res"))
        run = subprocess.run(["bash", str(ROOT / "tools/ops/gpu_smoke_job.sh"), "--preflight"], capture_output=True,
                             text=True, env=env, timeout=120)
        wheel_lines = [ln for ln in run.stdout.splitlines()
                       if ln.startswith(("PREFLIGHT ok wheel mojolearn_", "PREFLIGHT FAIL mojolearn_"))]
        assert len(wheel_lines) == 2 and all(ln.startswith(want) for ln in wheel_lines), run.stdout


def test_the_full_pod_sync_records_the_synced_commit():
    text = (ROOT / "tools/dev_pod.sh").read_text()
    full = text[text.index("# FULL SYNC"):]
    full = full[:full.index('\n    ;;\n')]
    assert "devpod_synced" in full and "commit=%s" in full


def test_the_identity_record_job_preflight_reads_the_synced_commit(tmp_path):
    """rel-fix-1 (2026-11-26): the job's preflight compares the wheels with the tree's commit, so a patch-synced tree
    whose HEAD is the merge base still admits the build's wheels and refuses another commit's."""
    repo, head = _repo(tmp_path)
    synced = "ab" * 20
    _record(repo, synced, head)
    wheels = tmp_path / "w"
    wheels.mkdir()
    for name in (f"mojolearn_core-0.8.67.dev0+{synced[:12]}-py3-none-manylinux_2_35_x86_64.whl",
                 f"mojolearn_nvidia-0.8.67.dev0+{synced[:12]}-py3-none-manylinux_2_35_x86_64.whl"):
        (wheels / name).write_text("")
    job = tmp_path / "tree/tools/ops"
    job.mkdir(parents=True)
    for f in ("identity_record_job.sh", "tree_commit.sh"):
        (job / f).write_text((ROOT / "tools/ops" / f).read_text())
    (tmp_path / "tree/tools/identity_wheel_record.py").write_text("")
    env = dict(os.environ, IR_WHEEL_DIR=str(wheels), MOJOLEARN_RESULTS_DIR=str(tmp_path / "res"))
    p = subprocess.run(["bash", str(job / "identity_record_job.sh"), "--preflight"], capture_output=True, text=True,
                       env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("PREFLIGHT ok MATCH") == 2
    _record(repo, "cd" * 20, head)
    p = subprocess.run(["bash", str(job / "identity_record_job.sh"), "--preflight"], capture_output=True, text=True,
                       env=env)
    assert p.returncode == 4 and "PREFLIGHT FAIL MISMATCH" in p.stdout


def test_the_recorder_names_the_trees_commit_not_head():
    text = (ROOT / "tools/record_identity_column.sh").read_text()
    assert '. "$ROOT/tools/ops/tree_commit.sh"' in text and 'COMMIT=$(tree_commit "$ROOT")' in text
    assert "export MOJOLEARN_COMMIT=$COMMIT" in text
    job = (ROOT / "tools/ops/identity_record_job.sh").read_text()
    # the job's harness is its own tree, fixed at the start, and the chunk logs land in the results dir
    assert 'worktree add -q --detach "$TREE"' in job and '--out "$RES/identity"' in job


def _wheel(path: Path, commit: str | None) -> Path:
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mojolearn/__init__.py", "")
        if commit is not None:
            z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
    return path


def test_every_wheel_version_form_names_its_commit(tmp_path):
    """rel-fix-1 (review item 3): the measurement build's `+<sha12>`, the define build's `+<sha12>.d<hex>`
    (tools/release_candidate.py candidate_version), any label, and a published wheel by its recorded commit."""
    repo, head = _repo(tmp_path)
    forms = [f"mojolearn_core-0.8.67.dev0+{head[:12]}-py3-none-manylinux_2_35_x86_64.whl",
             f"mojolearn_nvidia-0.8.67.dev0+{head[:12]}.d1a2b3c4d-py3-none-manylinux_2_35_x86_64.whl",
             f"mojolearn_core-0.8.67.dev0+{head[:12]}.fast.arm2-cp312-cp312-linux_x86_64.whl"]
    for name in forms:
        assert _sh(f'wheel_commit /x/{name}', tmp_path).stdout.strip() == head[:12], name
    ok = _sh(f'wheels_match_tree "{repo}" ' + " ".join(forms), tmp_path)
    assert ok.returncode == 0 and ok.stdout.count("MATCH ") == 3, ok.stdout
    published = _wheel(tmp_path / "mojolearn_core-0.8.67-py3-none-manylinux_2_35_x86_64.whl", head)
    assert _sh(f'wheel_commit "{published}"', tmp_path).stdout.strip() == head
    assert _sh(f'wheels_match_tree "{repo}" "{published}"', tmp_path).returncode == 0
    other = _wheel(tmp_path / "mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl", "cd" * 20)
    assert _sh(f'wheels_match_tree "{repo}" "{other}"', tmp_path).returncode == 1


def test_a_published_wheel_without_a_record_needs_a_stated_commit(tmp_path):
    repo, head = _repo(tmp_path)
    bare = _wheel(tmp_path / "mojolearn_nvidia-0.8.67-py3-none-manylinux_2_35_x86_64.whl", None)
    refused = _sh(f'wheels_match_tree "{repo}" "{bare}"', tmp_path)
    assert refused.returncode == 1 and "BOX_WHEEL_COMMIT" in refused.stdout
    assert _sh(f'BOX_WHEEL_COMMIT={head} wheels_match_tree "{repo}" "{bare}"', tmp_path).returncode == 0
    assert _sh(f'BOX_WHEEL_COMMIT={"cd" * 20} wheels_match_tree "{repo}" "{bare}"', tmp_path).returncode == 1


def test_a_dirty_or_unrecorded_full_sync_is_named_and_refused_unless_expected(tmp_path):
    """rel-fix-1 (review item 4): worktree_dirty=1, and an older dev_pod.sh full sync (a .devpod_manifest with no
    full-sync record), are named in the output and refused unless BOX_EXPECT_DIRTY_SYNC=1."""
    repo, head = _repo(tmp_path)
    synced = "ab" * 20
    wheel = f"mojolearn_core-0.8.67.dev0+{synced[:12]}-py3-none-linux_x86_64.whl"
    (repo / ".git/devpod_synced").write_text(f"commit={synced}\nworktree_dirty=1\nbase={head}\npatch_sha256=x\n")
    assert _sh(f'tree_sync_state "{repo}"', tmp_path).stdout.strip() == "dirty"
    refused = _sh(f'wheels_match_tree "{repo}" {wheel}', tmp_path)
    assert refused.returncode == 1 and refused.stdout.startswith("SYNC dirty ")
    accepted = _sh(f'BOX_EXPECT_DIRTY_SYNC=1 wheels_match_tree "{repo}" {wheel}', tmp_path)
    assert accepted.returncode == 0 and accepted.stdout.startswith("MATCH sync dirty ")
    _record(repo, synced, head)                       # a clean patch sync
    assert _sh(f'tree_sync_state "{repo}"', tmp_path).stdout.strip() == "clean"
    (repo / ".devpod_manifest").write_text("f\n")    # then an older full sync over it, which recorded nothing
    assert _sh(f'tree_sync_state "{repo}"', tmp_path).stdout.strip() == "full-sync-unrecorded"
    assert _sh(f'wheels_match_tree "{repo}" {wheel}', tmp_path).returncode == 1
    (repo / ".git/devpod_synced").write_text(f"commit={synced}\nworktree_dirty=0\nbase={head}\n"
                                             "patch_sha256=none-full-sync\n")  # today's full sync records itself
    assert _sh(f'tree_sync_state "{repo}"', tmp_path).stdout.strip() == "clean"


def test_an_untracked_only_sync_is_named_and_accepted_but_modified_tracked_files_are_refused(tmp_path):
    """w3-integrate-16: the orchestrator's night-main tree carries untracked box-local/ scripts that a job runs by name.
    Those leave every tracked file exactly the commit, so the sync records worktree_dirty=0 with worktree_untracked=N
    and the jobs accept it on a MATCH line; a modified tracked file is still worktree_dirty=1 and refused."""
    repo, head = _repo(tmp_path)
    synced = "cd" * 20
    wheel = f"mojolearn_core-0.8.67.dev0+{synced[:12]}-py3-none-linux_x86_64.whl"
    (repo / ".git/devpod_synced").write_text(f"commit={synced}\nworktree_dirty=0\nworktree_untracked=2\nbase={head}\n"
                                             "patch_sha256=x\n")
    assert _sh(f'tree_sync_state "{repo}"', tmp_path).stdout.strip() == "untracked"
    ok = _sh(f'wheels_match_tree "{repo}" {wheel}', tmp_path)
    assert ok.returncode == 0 and ok.stdout.startswith("MATCH sync untracked "), ok.stdout
    (repo / ".git/devpod_synced").write_text(f"commit={synced}\nworktree_dirty=1\nworktree_untracked=2\nbase={head}\n"
                                             "patch_sha256=x\n")
    assert _sh(f'wheels_match_tree "{repo}" {wheel}', tmp_path).returncode == 1


def test_dev_pod_counts_untracked_files_apart_from_modified_tracked_ones(tmp_path):
    """The two lines tools/dev_pod.sh records the sync with, run on a tree: untracked files (but not .so build outputs)
    only raise worktree_untracked; a modified tracked file raises worktree_dirty."""
    import re
    text = (ROOT / "tools/dev_pod.sh").read_text()
    dirty = re.findall(r"^\s*(_dirty=0; .*)$", text, re.M)
    untracked = re.findall(r"^\s*(_untracked=.*)$", text, re.M)
    assert len(dirty) == 2 and len(untracked) == 2  # the patch sync and the full sync
    repo, _ = _repo(tmp_path)

    def state():
        out = subprocess.run(["bash", "-c", f'wt="{repo}"; {dirty[0]}; {untracked[0]}; echo "$_dirty $_untracked"'],
                             capture_output=True, text=True, check=True).stdout.split()
        return out[0], out[1]
    assert state() == ("0", "0")
    (repo / "box-local").mkdir()
    (repo / "box-local/run.sh").write_text("echo hi\n")
    (repo / "built.so").write_bytes(b"\x7fELF")
    assert state() == ("0", "1")
    (repo / "f").write_text("changed\n")
    assert state() == ("1", "1")
