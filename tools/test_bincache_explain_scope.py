# SPDX-License-Identifier: Apache-2.0
"""tools/bincache.py explain: "<file> invalidates these N groups" per changed file, and the legacy key's
widening reason ("unresolved import <x> in <file> widened the closure")."""
import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bincache as bc  # noqa: E402
from test_bincache import make_repo  # noqa: E402


def explain(argv, env):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bc.cmd_explain(argv, environ=env)
    return rc, buf.getvalue().strip().splitlines()


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOJOLEARN_")}
    env.update(MOJOLEARN_BINCACHE_MAP="/nonexistent/map.tsv", **extra)
    return env


def test_the_legacy_key_names_the_import_that_widened_it(tmp_path):
    repo = make_repo(tmp_path / "repo").resolve()
    (repo / "core/a.mojo").write_text("from nowhere.at_all import thing\n")
    rc, lines = explain([str(repo / "bindings/build_fake.sh")],
                        clean_env(MOJOLEARN_BINCACHE_KEY_RULE="v1", MOJOLEARN_BINCACHE_DIR=str(tmp_path / "c")))
    legacy = next(l for l in lines if l.startswith("legacy key:"))
    assert "scope=tree" in legacy and "unresolved import 'nowhere.at_all' in core/a.mojo widened the closure" in legacy
    assert rc == 3 and any("unkeyable" in l and "nowhere.at_all" in l for l in lines)   # closure-v2 fails loudly


def test_a_resolved_legacy_key_reports_its_closure_scope(tmp_path):
    repo = make_repo(tmp_path / "repo").resolve()
    rc, lines = explain([str(repo / "bindings/build_fake.sh")],
                        clean_env(MOJOLEARN_BINCACHE_KEY_RULE="v1", MOJOLEARN_BINCACHE_DIR=str(tmp_path / "c"),
                                  MOJOLEARN_BINCACHE_OUTPUTS="python/mojolearn/identical/_mojolearn_fake.so"))
    legacy = next(l for l in lines if l.startswith("legacy key:"))
    assert "scope=closure" in legacy and "widened" not in legacy


def test_since_prints_the_groups_each_changed_file_invalidates(tmp_path):
    repo = make_repo(tmp_path / "repo").resolve()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], cwd=repo, check=True)
    (repo / "core/a.mojo").write_text("fn f(): return\n")
    (repo / "core/unrelated.mojo").write_text("fn u(): return\n")

    def inv(_repo, changed, platforms=("darwin", "linux")):
        return {f: (["linux/identical/_mojolearn_native", "darwin/identical/_mojolearn_native"]
                    if f == "core/a.mojo" else []) for f in changed}
    with mock.patch.object(bc, "invalidations", inv):
        rc, lines = explain(["--since", "HEAD", str(repo / "bindings/build_fake.sh")],
                            clean_env(MOJOLEARN_BINCACHE_DIR=str(tmp_path / "c"),
                                      MOJOLEARN_BINCACHE_OUTPUTS="python/mojolearn/identical/_mojolearn_fake.so"))
    assert rc == 0
    assert "core/a.mojo invalidates these 2 groups: linux/identical/_mojolearn_native, " \
           "darwin/identical/_mojolearn_native" in lines
    assert "core/unrelated.mojo invalidates no merged group" in lines


def test_invalidations_walk_the_group_closures():
    groups = [("identical", "_g1", "default", ["_m1"]), ("host", "_g2", "host", ["_m2"])]
    closures = {"_m1": {"core/a.mojo": "x", "core/b.mojo": "y"}, "_m2": {"core/b.mojo": "y"}}
    import compile_closures as cc
    with mock.patch.object(cc, "group_rows", lambda tree, plat: groups), \
            mock.patch.object(cc, "closure_of", lambda tree, roots: closures[roots[0].stem]):
        out = bc.invalidations(tempfile.gettempdir(), ["core/a.mojo", "core/b.mojo", "core/z.mojo"],
                               platforms=("linux",))
    assert out == {"core/a.mojo": ["linux/identical/_g1"], "core/b.mojo": ["linux/identical/_g1", "linux/host/_g2"],
                   "core/z.mojo": []}
    assert bc.invalidation_lines(out)[2] == "core/z.mojo invalidates no merged group"
