#!/usr/bin/env bash
# Keep the shell interface used by local jobs; implementation is versioned.
exec "${MOJOLEARN_SLOT_PYTHON:-python3}" "$(dirname "$0")/mac_slot.py" "$@"
