# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the byte-LM shape admission shared by the device door
(bindings/_mojolearn_byte_lm.mojo), the host door
(bindings/_mojolearn_byte_lm_host.mojo) and the host trainer compatibility
provider (providers/python/byte_lm_host_compat_pyapi.mojo): the typed
completion and validation of the shape words after the door has read them
from Python. Host-safe (nothing from training/byte_lm.mojo, which is device
code); typed Mojo values only, no Python import (tools/python_free_dirs.sh).
Lane rs-embedding-bytelm, 2026-11-12: the bodies moved unchanged."""
from training.byte_lm_config import ByteConfig


def byte_config_words(var values: List[Int], who: String) raises -> ByteConfig:
    """The ByteConfig of 7, 9 or 10 shape words (the door checked the count
    and read them): 7 words take the default layers (2) and vocabulary
    (256); a tenth word selects the chunked head. `who` prefixes the
    refusal ("byte LM" or "byte LM host")."""
    if len(values) == 7:
        values.append(2)
        values.append(256)
    var chunked = False
    if len(values) == 10:
        if values[9] != 0 and values[9] != 1:
            raise Error(who + ": chunked head selector must be 0 or 1")
        chunked = values[9] == 1
    var cfg = ByteConfig(values[0], values[1], values[2], values[3],
                         values[4], values[5], values[6], values[7], values[8], chunked)
    cfg.validate()
    return cfg^


def _config(p: List[Int]) raises -> ByteConfig:
    if p[9] != 0 and p[9] != 1:
        raise Error("byte LM: chunked head selector must be 0 or 1")
    var cfg = ByteConfig(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9] == 1)
    cfg.validate()
    return cfg^
