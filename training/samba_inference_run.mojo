# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba inference door (lane rs-neural, wave 8 run split,
from bindings/samba_inference_native.mojo): the Mamba-3 carried-state piece
shapes and names, and the host route's primitive entry. Typed Mojo only, no
Python import (tools/python_free_dirs.sh). Same bodies."""
from core.entry_table import EntryRef
from core.samba_entry_table import samba_ref, SambaEntryKey, SAMBA_TABLE_TRAINING, SAMBA_TABLE_NEURAL_HOST


def _m3_shapes(b: Int, h: Int) -> List[List[Int]]:
    var result = List[List[Int]]()
    result.append([b, h, 32])
    result.append([b, h, 64, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 64])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, h, 128])
    result.append([b, h, 64])
    return result^


def _m3_names() -> List[String]:
    return [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"),
            String("buffer_v"), String("buffer_dt"), String("buffer_sig"), String("buffer_adt"),
            String("pending_k"), String("pending_v")]


def _samba_host_ref(cpu_inference: Bool, key: SambaEntryKey) raises -> EntryRef:
    """The host route's primitive entry, resolved once per forward: the CPU
    inference model's neural_host image or the stack's connected training
    image, through the Samba table."""
    if cpu_inference:
        return samba_ref[SAMBA_TABLE_NEURAL_HOST](key)
    return samba_ref[SAMBA_TABLE_TRAINING](key)
