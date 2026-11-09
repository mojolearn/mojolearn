# SPDX-License-Identifier: Apache-2.0
"""Verification metadata for explicitly authorized Python compatibility work.

This module does not implement or select a product fallback. The verifier uses
it to report an existing callback without calling that operation Python-free.

The list is frozen (AGENTS.md "No Python in the runtime"): nothing is added
without Andrew's explicit word. It is empty. Its one entry,
`array-length-metadata` (the Python `Array.__len__` frame that native
Voting/Stacking/MultiOutput/OneVsRest/Calibrated paths reached through
len(Array)), was removed on lane/p2-labels-lists-arrays: `Array.__len__` and
the other Array protocol methods are now C properties over the native entry
(`_array._bound`), so len(Array) from any caller enters no Python frame.
"""

FALLBACKS = {}


def approved_metadata_callbacks(operation):
    """Return exact callback code identities approved for `operation`: none."""
    return {}
