# SPDX-License-Identifier: Apache-2.0
"""Host generation policy, shared by every accelerator and the CPU.

The sampler lives in core/shell_impl/generation_policy_impl.mojo; this module
re-exports its names so existing imports keep working.
"""
from core.shell_impl.generation_policy_impl import generation_random, generation_pick
