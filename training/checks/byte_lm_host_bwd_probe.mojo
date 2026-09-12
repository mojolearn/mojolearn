# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the host backward oracle compile into a CPU-only build? (DEVIATION 2680)

A PROBE, NOT A CHECK. It asserts nothing about arithmetic. It exists to answer
one question that was being argued from a source comment rather than measured:
`transformer/checks/transformer_backward_oracle.mojo` computes on the host and
imports no device type of its own, but it takes three names from
`gemm/checks/gemm_backward.mojo`, which imports `max.gpu.host` at its own line
113, and one name from `core/identity_trace.mojo`, which does the same at 84.

`bindings/_mojolearn_byte_lm_host.mojo` says "HOST ONLY. No DeviceContext, no
kernel, no GPU, and nothing imported from the GPU side." If that is enforced by
the compiler or the link, this file fails to build with no accelerator target
and the error names what has to move. If it builds, the rule is a policy the
toolchain does not enforce here, which is a smaller problem and a different fix.

Delete this file once the question is settled; it is a measurement, not a gate.
"""

from transformer.checks.transformer_backward_oracle import (
    transformer_block_backward_oracle,
)


def main() raises:
    """Compiling is the test, not running.

    The import above forces the compiler to analyze
    `transformer_backward_oracle` and everything IT imports, which is where
    the GPU would enter. So a successful build answers the question whether or
    not this body ever executes, and the body is deliberately trivial rather
    than a fixture that could fail for unrelated reasons."""
    print("host backward oracle compiled into a CPU-only build")
