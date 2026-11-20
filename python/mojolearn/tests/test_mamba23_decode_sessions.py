# SPDX-License-Identifier: Apache-2.0
"""The Mamba-2 and Mamba-3 resident decode sessions are public.

The byte-for-byte relation of the session step to the per-call step is the
native check checks/native_tests/mamba_decode_sessions.mojo (the typed
session entry mamba/mamba_decode_session_run.mojo against the host state
body); this file keeps the public-surface check.
"""
import mojolearn as ml
from mojolearn import mamba


def test_the_sessions_are_public():
    for name in ("Mamba2DecodeSession", "Mamba3DecodeSession"):
        assert name in mamba.__all__ and hasattr(mamba, name), name
    assert callable(ml.Mamba2Block.decode_session)
    assert callable(ml.Mamba3Block.decode_session)
    assert not hasattr(ml.Mamba2Block, "_decode_session")
    assert not hasattr(ml.Mamba3Block, "_decode_session")
