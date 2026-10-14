#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a local build recipe or an immutable recipe for modular-community."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def render(destination, local=False, revision="HEAD"):
    if local:
        source = "  path: " + json.dumps(str(ROOT))
        version_text = (ROOT / "python/mojolearn/_version.py").read_text()
        template = (ROOT / "packaging/native/recipe.yaml.in").read_text()
    else:
        sha = subprocess.check_output(["git", "rev-parse", revision + "^{commit}"], cwd=ROOT, text=True).strip()
        def read(path):
            return subprocess.check_output(["git", "show", sha + ":" + path], cwd=ROOT, text=True)
        # Fail if this revision predates the packager. Read version/template
        # from the same commit as the source, never from another working tree.
        read("packaging/native/build.sh")
        version_text = read("python/mojolearn/_version.py")
        template = read("packaging/native/recipe.yaml.in")
        source = "  git: https://github.com/mojolearn/mojolearn.git\n  rev: " + sha
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', version_text)[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("@VERSION@", version).replace("@SOURCE@", source))
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--local", action="store_true", help="Build this checkout instead of published Git source")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    render(args.destination, args.local, args.revision)
