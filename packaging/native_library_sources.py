"""Build-time library provenance across the compiler and packaging environments.

Runtime discovery still uses the wheel's existing rpaths. This resolver only
selects input bytes while staging; a shared SONAME cannot silently select two
different libraries from the two pinned environments.
"""
import hashlib
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_source(name, preferred, compiler_lib, codec_lib, selected, label='native dependency'):
    preferred, compiler_lib, codec_lib = map(Path, (preferred, compiler_lib, codec_lib))
    directories = [preferred] + [path for path in (compiler_lib, codec_lib) if path != preferred]
    candidate = next((path / name for path in directories if (path / name).is_file()), None)
    if candidate is None:
        raise SystemExit(f'{label} {name} is missing from declared library roots: {directories}')
    previous = selected.get(name)
    if previous is not None and previous.resolve() != candidate.resolve():
        if file_sha(previous) != file_sha(candidate):
            raise SystemExit(f'conflicting native library {name}: {previous} and {candidate} have different bytes')
        # A compiler-linked dependency keeps SDK provenance even when the
        # codec environment carries byte-identical input under the same name.
        if candidate.parent != compiler_lib:
            candidate = previous
    selected[name] = candidate
    return candidate
