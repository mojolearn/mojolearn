# SPDX-License-Identifier: Apache-2.0
"""Typed IDNA code-point predicates (RUN): the RFC 3454 ignored and prohibited tables used by
`bindings/native_idna_door.mojo` after it reads each code point and its Unicode category."""


def _idna_ignored(cp: Int) -> Bool:
    return cp == 0xAD or cp == 0x34F or cp == 0x1806 or (0x180B <= cp <= 0x180D) or (
        0x200B <= cp <= 0x200D) or cp == 0x2060 or (0xFE00 <= cp <= 0xFE0F) or cp == 0xFEFF


def _idna_prohibited(cp: Int, category: String) -> Bool:
    return (category == "Zs" and cp != 32) or (cp >= 128 and category == "Cc") or (
        category == "Co" or category == "Cs") or (0xFDD0 <= cp < 0xFDF0) or (
        (cp & 0xFFFF) == 0xFFFE or (cp & 0xFFFF) == 0xFFFF) or (
        0xFFF9 <= cp <= 0xFFFD) or (0x2FF0 <= cp <= 0x2FFB) or (
        cp == 0x340 or cp == 0x341 or cp == 0x200E or cp == 0x200F or 0x202A <= cp <= 0x202E or
        0x206A <= cp <= 0x206F) or (cp == 0xE0001 or 0xE0020 <= cp <= 0xE007F) or (
        cp == 0x6DD or cp == 0x70F or cp == 0x180E or cp == 0x200C or cp == 0x200D or
        cp == 0x2028 or cp == 0x2029 or cp == 0xFEFF or 0x2060 <= cp <= 0x2063 or
        0x1D173 <= cp <= 0x1D17A)
