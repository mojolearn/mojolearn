"""Rebuild scope keeps real import dependencies after route-checker retirement."""
import importlib.util
from pathlib import Path
import subprocess
import sys


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rebuild_scope_follows_real_imports_only(tmp_path, monkeypatch):
    files = {
        "bindings/build_alpha.sh": "mojo build bindings/_mojolearn_alpha.mojo\n",
        "bindings/build_beta.sh": "mojo build bindings/_mojolearn_beta.mojo\n",
        "bindings/_mojolearn_alpha.mojo": (
            '\"\"\"Example only:\nfrom core.fake import bogus\n\"\"\"\n'
            "# ''' is not a docstring here\n"
            "from core import (\n    middle as m,\n)\n"
            "import core.direct as d\n"
        ),
        "bindings/_mojolearn_beta.mojo": "from core.leaf import value\n",
        "core/__init__.mojo": "",
        "core/middle.mojo": "from .leaf import value\n",
        "core/leaf.mojo": "",
        "core/direct.mojo": "'''import core.fake'''\n",
        "core/fake.mojo": "",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "add", "."], check=True)
    # write-tree exercises real Git blobs without creating a commit or hooks.
    tree_id = subprocess.check_output(["git", "write-tree"], text=True).strip()
    hooks = Path(__file__).resolve().parent
    graph = _module("mojo_import_graph_test", hooks / "mojo_import_graph.py")
    tree = graph.Tree(tree_id)
    assert tree.edges["bindings/_mojolearn_alpha.mojo"] == {
        "core/__init__.mojo", "core/middle.mojo", "core/direct.mojo"
    }
    assert tree.edges["core/middle.mojo"] == {"core/leaf.mojo", "core/__init__.mojo"}
    assert tree.edges["core/direct.mojo"] == set()
    fence = _module("rebuild_fence_test", hooks / "rebuild_fence.py")
    assert fence.reach_counts(tree_id, {"core/leaf.mojo", "core/direct.mojo", "core/fake.mojo"}) == {
        "core/leaf.mojo": 2, "core/direct.mojo": 1, "core/fake.mojo": 0
    }
