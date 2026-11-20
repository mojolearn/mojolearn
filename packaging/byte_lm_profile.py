# SPDX-License-Identifier: Apache-2.0
"""The byte-LM profile a built `_mojolearn_byte_lm` binary may report, for the build and wheel read-backs.

A byte-LM binary reports `ByteConfig().profile()`: the base profile below plus
`byte_lm_arithmetic_suffix()` (training/byte_lm_config.mojo), the arithmetic tags of the compile-time switches it
was built with (`.nn-gemm-v2-l<leaf>-c<chains>`, `.nn-clip-v2-leaf128`, `-gemm-leaf<N>`). The runtime admits
exactly base + that binary's own suffix (bindings/byte_lm_trainer_native_door.mojo, `_PROFILE_REFUSAL`). The
read-backs used to demand the bare base string, so every arm build whose define adds a tag (the owed B arm
`gemm-leaf512`, MOJOLEARN_GEMM_FAST_LEAF512 -> CONTRACT_K_LEAF_MIN 512 -> `-gemm-leaf512`) was refused at
assembly with "Wrong byte LM profile" after every shard compiled (GitHub run 44525867606, 2026-11-19).

`check(profile)` accepts the base followed by zero or more arithmetic tags, each starting with `.` or `-`. A
different base (another shape, a different version number such as `fp32.v10`) is still refused.
"""
import re
import sys

BASE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
# One tag: a `.` or `-` separator, then lower-case letters, digits and `-`.
_TAGS = re.compile(r'(?:[.-][a-z0-9][a-z0-9-]*)*')


def admitted(profile):
    """True when `profile` is BASE followed only by arithmetic tags."""
    if not isinstance(profile, str) or not profile.startswith(BASE):
        return False
    return _TAGS.fullmatch(profile[len(BASE):]) is not None


def check(profile):
    """Raise AssertionError('Wrong byte LM profile') unless `profile` is admitted."""
    if not admitted(profile):
        raise AssertionError('Wrong byte LM profile: %r (want %s plus arithmetic tags)' % (profile, BASE))
    return profile


if __name__ == '__main__':
    check(sys.argv[1])
