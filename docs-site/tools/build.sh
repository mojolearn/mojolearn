#!/usr/bin/env bash
# Build docs.mojolearn.org for the installed mojolearn release into docs-site/_site (wrangler.jsonc deploys it).
# mojolearn's tools/publish_sites.sh calls it in its own build worktree of origin/main; it also works alone.
#
#   docs-site/tools/build.sh [python]   (python: an interpreter with docs-site/requirements.txt and the
#                                         mojolearn release to document installed; default python3)
#
# One release is published, at the site root; older releases are not kept. The output carries no build
# internals: doctrees go to _build/doctrees (outside _site), and _sources/ (html_copy_source = False),
# .buildinfo and any environment.pickle are left out.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python3}"
VERSION="$("$PY" -I -c 'import mojolearn; print(mojolearn.__version__)')"
echo "== docs build: mojolearn $VERSION"
# The released code with this repo's docstrings (tools/docstring_overlay.py), imported ahead of site-packages.
"$PY" -I tools/docstring_overlay.py .. _build/pkg
export PYTHONPATH="$PWD/_build/pkg"
"$PY" tools/gen_api.py
rm -rf _site _build/doctrees api/generated   # stubs regenerate; stale ones (excluded names) must not publish
mkdir -p _build
"$PY" -m sphinx -b html . _site -d _build/doctrees -q -j 4 -w _build/warnings.log
rm -rf _site/.doctrees _site/_sources
rm -f _site/.buildinfo
find _site -name 'environment.pickle' -delete
cp site_root/404.html site_root/robots.txt _site/
echo "== docs build: $VERSION -> _site ($(find _site -type f | wc -l | tr -d ' ') files, $(wc -l < _build/warnings.log | tr -d ' ') Sphinx warnings in docs-site/_build/warnings.log)"
