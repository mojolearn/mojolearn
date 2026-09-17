#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility entry point. Scope is explicit; see --help and --list.
exec "${MOJOLEARN_SLOT_PYTHON:-python3}" "$(dirname "$0")/check_python_gates.py" "$@"
