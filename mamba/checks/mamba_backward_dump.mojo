# SPDX-License-Identifier: Apache-2.0
"""Mamba-1 gradient interchange runner.

This runner exports the complete host-oracle gradient set for an exact corpus
fixture in the format consumed by ``tools/mamba_gradient_oracle.py``. It is a
host-reference gate, not the still-owed device whole-pass composition.

    MOJOLEARN_MAMBA_GRADIENT_DUMP=/tmp/m1-grad \
      mojo run -I . mamba/checks/mamba_backward_dump.mojo
"""

from std.os import getenv

from mamba.checks.mamba_check import dump_mamba1_host_gradients, env_int


def main() raises:
    var out = String(getenv("MOJOLEARN_MAMBA_GRADIENT_DUMP"))
    if out == "":
        raise Error(
            "mamba_backward_dump: MOJOLEARN_MAMBA_GRADIENT_DUMP must name an"
            " existing output directory"
        )
    var k = env_int("MOJOLEARN_MAMBA_GRADIENT_CASE", 1)
    dump_mamba1_host_gradients(k, out)
