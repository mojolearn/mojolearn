"""No directory named `checks` may exist under a Mojo import root unless it is
a real Mojo package. A `<area>/checks/` holding only sabotage patches shadows the
top-level `checks` package for every `from checks.numerics import ...` in
`<area>/*.mojo` once the group is compiled outside the development cache
(`bindings/build_preprocessing.sh`, 2026-10-24: "unable to locate module
'numerics'"). Fault patches live in `<area>/sabotage/`. Static: reads the index."""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tracked():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_every_nested_checks_directory_is_a_mojo_package():
    files = _tracked()
    dirs = {}
    for path in files:
        parts = path.split("/")
        for i, part in enumerate(parts[:-1]):
            if part == "checks" and i > 0:
                d = "/".join(parts[:i + 1])
                dirs.setdefault(d, []).append(path)
    # only a checks directory beside Mojo sources can shadow: bench/results/release_evidence/checks holds release
    # JSON evidence, with no Mojo file in its parent, and no compile resolves imports there
    beside_mojo = {os.path.dirname(p) for p in files if p.endswith(".mojo")}
    bad = sorted(d for d, members in dirs.items()
                 if os.path.dirname(d) in beside_mojo
                 and not any(m.endswith(".mojo") and m.count("/") == d.count("/") + 1 for m in members))
    assert not bad, f"checks directories with no Mojo module of their own (they shadow `checks`): {bad}"


def test_a_checks_dir_of_patches_beside_mojo_is_still_caught(monkeypatch):
    import sys
    fake = ["area/a.mojo", "area/checks/fault.patch", "bench/results/x/checks/r.json", "core/checks/ok.mojo"]
    monkeypatch.setattr(sys.modules[__name__], "_tracked", lambda: fake)
    try:
        test_every_nested_checks_directory_is_a_mojo_package()
    except AssertionError as error:
        assert "area/checks" in str(error) and "bench/results" not in str(error)
    else:
        raise AssertionError("area/checks holds no Mojo module and was not caught")
