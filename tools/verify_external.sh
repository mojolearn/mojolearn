#!/bin/sh
# Reproduce a retained identity record from PyPI, a local wheelhouse, or an
# installed package. See --help and docs/VERIFY_EXTERNALLY.md.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$here/verify_external.py" "$@"
