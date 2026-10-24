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
    bad = sorted(d for d, members in dirs.items()
                 if not any(m.endswith(".mojo") and m.count("/") == d.count("/") + 1 for m in members))
    assert not bad, f"checks directories with no Mojo module of their own (they shadow `checks`): {bad}"
