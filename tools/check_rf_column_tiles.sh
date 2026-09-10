#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec tools/with_build_lock.sh pixi run python ensemble/checks/rf_column_tiles_gate.py "$@"
