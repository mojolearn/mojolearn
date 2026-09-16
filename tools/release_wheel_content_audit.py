#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Audit what a release wheel carries, file by file (0.8.6).

    python3 tools/release_wheel_content_audit.py <wheel> [<wheel> ...] [--notice NOTICE]

For every wheel, each check prints one line and any failure exits 1:

  notice          the wheel's *.dist-info/licenses/NOTICE is byte for byte the
                  NOTICE given (default: this checkout's), and that NOTICE has the
                  copyright, the Apache line, the Modular trademark sentence and
                  the Modular components section
  no-gpt2-data    no GPT-2 vocabulary table, fixture or vocabulary file
                  (a basename containing gpt2 with a data suffix, or vocab.json,
                  merges.txt, encoder.json)
  no-vendored-env no vendored Python environment (a .venv, venv, .pixi,
                  site-packages or __pypackages__ path part, a pyvenv.cfg, or a
                  bin/python)
  host-surface    mojolearn/host_surface.py is present, imports on its own and
                  declares at least one wheel host binding

Read-only: the wheel is opened as a zip; nothing is installed or imported from
the package except host_surface.py, which imports only the standard library.
"""
import argparse
import importlib.util
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

REPO = Path(__file__).resolve().parent.parent
NOTICE_PARTS = (
    ('copyright', 'Copyright 2026 Andrew Hendel'),
    ('apache', 'Licensed under the Apache License, Version 2.0'),
    ('modular trademark', 'are trademarks of Modular, Inc.'),
    ('modular components section', 'Modular components'),
    ('modular runtime license', 'Modular MAX Community License'),
    ('modular source license', 'Apache License v2.0 with LLVM Exceptions'),
)
GPT2_DATA_SUFFIXES = ('.tsv', '.json', '.txt', '.bin', '.tiktoken', '.bpe', '.model')
VOCAB_BASENAMES = frozenset({'vocab.json', 'merges.txt', 'encoder.json', 'vocab.bpe'})
ENV_PARTS = frozenset({'.venv', 'venv', '.pixi', 'site-packages', '__pypackages__'})


#: Phrases the NOTICE must not carry. "used under license" claimed a trademark
#: license the Modular MAX Community License does not grant outside hosted
#: commercial MAX services (main f2806635b).
NOTICE_FORBIDDEN = ('used under license',)


def notice_problems(notice_text):
    problems = [f'NOTICE lacks the {label} ({needle!r})' for label, needle in NOTICE_PARTS if needle not in notice_text]
    problems += [f'NOTICE carries {phrase!r}' for phrase in NOTICE_FORBIDDEN if phrase in notice_text]
    return problems


def audit(wheel, notice_path):
    wheel = Path(wheel)
    results = []

    def check(name, problems):
        results.append((name, list(problems)))

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        expected_notice = Path(notice_path).read_bytes()
        notices = [n for n in names if PurePosixPath(n).match('*.dist-info/licenses/NOTICE')]
        problems = notice_problems(expected_notice.decode('utf-8', 'replace'))
        if len(notices) != 1:
            problems.append(f'expected exactly one *.dist-info/licenses/NOTICE, found {notices}')
        elif archive.read(notices[0]) != expected_notice:
            problems.append(f'{notices[0]} differs from {notice_path}')
        check('notice', problems)

        gpt2 = []
        for n in names:
            base = PurePosixPath(n).name.lower()
            if ('gpt2' in base and base.endswith(GPT2_DATA_SUFFIXES)) or base in VOCAB_BASENAMES:
                gpt2.append(n)
        check('no-gpt2-data', [f'carries {n}' for n in gpt2])

        env = []
        for n in names:
            parts = PurePosixPath(n).parts
            if ENV_PARTS & set(parts) or parts[-1] == 'pyvenv.cfg' or '/bin/python' in '/' + n:
                env.append(n)
        check('no-vendored-env', [f'carries {n}' for n in env])

        problems = []
        member = 'mojolearn/host_surface.py'
        if member not in names:
            problems.append(f'no {member}')
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'host_surface.py'
                path.write_bytes(archive.read(member))
                try:
                    spec = importlib.util.spec_from_file_location('release_audit_host_surface', path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    bindings = list(module.wheel_bindings())
                    if not bindings:
                        problems.append(f'{member} declares no wheel host binding')
                except Exception as exc:
                    problems.append(f'{member} does not import: {type(exc).__name__}: {exc}')
        check('host-surface', problems)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('wheels', nargs='+')
    parser.add_argument('--notice', default=str(REPO / 'NOTICE'))
    args = parser.parse_args(argv)
    failed = 0
    for wheel in args.wheels:
        for name, problems in audit(wheel, args.notice):
            if problems:
                failed += 1
                for p in problems:
                    print(f'FAIL {Path(wheel).name} {name}: {p}')
            else:
                print(f'OK   {Path(wheel).name} {name}')
    print('release wheel content audit: ' + ('PASSED' if not failed else f'FAILED ({failed} check(s))'))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
