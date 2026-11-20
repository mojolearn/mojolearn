"""tools/compile_closures.py: the static plan (hit / compile: <reason> per merged
group between two commits) and the closure-overlap regrouping. No Mojo build.

    python3 -m pytest -q tools/test_compile_closures.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compile_closures as cc  # noqa: E402
from test_bincache import make_merged_repo  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, message):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.hooksPath=/dev/null",
        "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def test_plan_names_the_one_group_an_edit_reaches(tmp_path):
    repo = make_merged_repo(tmp_path / "repo")
    git(repo, "init", "-q")
    old = commit(repo, "base")
    (repo / "kernels" / "kb.mojo").write_text("fn kb(): return\n")
    (repo / "tools" / "merged_bindings.py").write_text((repo / "tools" / "merged_bindings.py").read_text() + "\n# x\n")
    new = commit(repo, "edit kb and the generator tool")
    lines = cc.group_plan(repo, old, new, "linux")
    assert "linux identical/_mojolearn_native_a: hit (closure unchanged since %s)" % old[:12] in lines
    assert "linux identical/_mojolearn_native_t: hit (closure unchanged since %s)" % old[:12] in lines
    assert any(l.startswith("linux identical/_mojolearn_native_b: compile: closure changed: kernels/kb.mojo")
               for l in lines), lines


def test_plan_reports_membership_and_new_groups(tmp_path):
    import json
    repo = make_merged_repo(tmp_path / "repo")
    git(repo, "init", "-q")
    old = commit(repo, "base")
    m = json.loads((repo / "bindings/merged_groups.json").read_text())
    g = m["dirs"]["identical"]["groups"]
    g["_mojolearn_native_b"] = ["_mojolearn_b1", "_mojolearn_t"]
    del g["_mojolearn_native_t"]
    g["_mojolearn_native_c"] = []
    del g["_mojolearn_native_c"]
    (repo / "bindings/merged_groups.json").write_text(json.dumps(m))
    new = commit(repo, "regroup")
    lines = cc.group_plan(repo, old, new, "linux")
    assert "linux identical/_mojolearn_native_b: compile: membership changed" in lines
    assert any(l.startswith("linux identical/_mojolearn_native_a: hit") for l in lines)


def test_regroup_keeps_the_full_build_budget_and_lowers_incremental_scope():
    shared = {"core/%d.mojo" % i: "s" for i in range(10)}
    def member(tag, n):
        return dict(shared, **{"%s/%d.mojo" % (tag, i): "x" for i in range(n)})
    mc = {"a1": member("a", 30), "a2": member("a", 30), "b1": member("b", 30), "b2": member("b", 30)}
    # today: a1 with b1, a2 with b2 (unrelated members share a compile)
    current = {"g1": set(mc["a1"]) | set(mc["b1"]), "g2": set(mc["a2"]) | set(mc["b2"])}
    prop = cc.regroup(mc, current)
    assert sum(len(v) for v in prop.values()) <= sum(len(v) for v in current.values())
    assert sorted(prop) == [("a1", "a2"), ("b1", "b2")]
    weights = {"a/1.mojo": 1}
    assert cc.expected_scope(prop, weights) < cc.expected_scope(current, weights)
    assert cc.scope(prop, {"a/1.mojo"}) == (1, 40)
    assert cc.scope(current, {"a/1.mojo"}) == (2, 140)
