# SPDX-License-Identifier: Apache-2.0
"""Typed archive-read refusal wording (RUN)."""


def _archive_missing(name: String) -> String:
    return "mojolearn: model file is missing field '" + name + "'"
