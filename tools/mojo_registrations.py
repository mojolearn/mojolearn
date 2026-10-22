# SPDX-License-Identifier: Apache-2.0
"""Read explicit Mojo extension registrations shared by coverage and selection."""
import re
from typing import NamedTuple


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_IMPL = rf"(?P<impl>{_IDENT})\s*(?:\[(?P<params>[^\[\]]*)\])?"
_NAME = rf"(?P<quote>['\"])(?P<name>{_IDENT})(?P=quote)"
_BRACKET = re.compile(rf"\bdef_(?:py_c_|py_)?function\[\s*{_IMPL}\s*\]\s*\(\s*{_NAME}")
_CALL = re.compile(rf"\bdef_py_c_function\(\s*{_IMPL}\s*,\s*{_NAME}")
_LEXICAL = re.compile(r'''\#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' ''', re.VERBOSE)


class Registration(NamedTuple):
    name: str
    impl: str
    params: tuple[str, ...]
    start: int
    end: int


def _lexical_context(text):
    """Mask comments/docstrings; retain argument strings and their locations."""
    chunks = list(text)
    strings = []
    for match in _LEXICAL.finditer(text):
        token = match.group()
        if token.startswith('#') or token.startswith(('"""', "'''")):
            chunks[match.start():match.end()] = ['\n' if char == '\n' else ' ' for char in token]
        else:
            strings.append((match.start(), match.end()))
    return ''.join(chunks), strings


def registration_calls(text):
    """Explicit export names, local implementation aliases and generic params.

    Supports def_function/def_py_function/def_py_c_function[impl]("name")
    and the C-entry overload def_py_c_function(impl, "name"). Imported aliases
    remain local names so the caller's import graph resolves the defining file.
    """
    code, strings = _lexical_context(text or '')
    found = []
    for pattern in (_BRACKET, _CALL):
        for match in pattern.finditer(code):
            if any(start <= match.start() < end for start, end in strings):
                continue
            found.append(Registration(match['name'], match['impl'],
                                      tuple(re.findall(_IDENT, match['params'] or '')),
                                      match.start(), match.end()))
    return sorted(found, key=lambda registration: registration.start)


def registered_functions(text):
    return {entry.name: (entry.impl, list(entry.params)) for entry in registration_calls(text)}


def sole_registration(text):
    """A whole simple registration statement, for safe additive diff handling."""
    code, _ = _lexical_context(text)
    entries = registration_calls(text)
    if len(entries) != 1:
        return None
    entry = entries[0]
    if not re.fullmatch(rf"\s*{_IDENT}\.\s*", code[:entry.start]):
        return None
    return entry if re.fullmatch(r"\s*\)\s*", code[entry.end:]) else None
