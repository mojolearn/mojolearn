#!/bin/bash
# Moved to tools/compile_slot.sh (2026-10-28); this path forwards so old references keep working.
exec bash "$(cd "$(dirname "$0")/.." && pwd)/compile_slot.sh" "$@"
