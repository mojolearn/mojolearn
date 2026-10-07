#!/usr/bin/env python3
"""Tests for bindings/build_defines.sh: the uniform MOJOLEARN_BUILD_DEFINES passthrough.

No build script has a dry-run mode, and lanes never compile, so this checks (1) the sourced snippet
itself under /bin/sh, by sourcing it in a throwaway shell and printing the flags it produces, and
(2) by reading every binding build script, that it sources the snippet before anything else and
carries $MOJOLEARN_BUILD_DEFINE_FLAGS on its one real `pixi run mojo build` line, next to the
variables it already had. Nothing is built or run besides /bin/sh on the snippet."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / 'bindings'
SNIPPET = BINDINGS / 'build_defines.sh'
SOURCE_LINE = '. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"'
EXPANSION = '${MOJOLEARN_MOJO_BUILD_FLAGS:-} ${MOJOLEARN_BUILD_DEFINE_FLAGS:-}'


def flags(value, unset=False):
    env = {'PATH': '/usr/bin:/bin'}
    if not unset:
        env['MOJOLEARN_BUILD_DEFINES'] = value
    script = 'set -u; . "%s"; printf "[%%s]" "$MOJOLEARN_BUILD_DEFINE_FLAGS"' % SNIPPET
    r = subprocess.run(['/bin/sh', '-c', script], env=env, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def device_scripts():
    return sorted(p for p in BINDINGS.glob('build*.sh')
                  if not p.name.endswith('_host.sh') and p.name != 'build_defines.sh')


class SnippetTests(unittest.TestCase):
    def test_unset_and_empty_give_nothing(self):
        self.assertEqual(flags('', unset=True)[:2], (0, '[]'))
        self.assertEqual(flags('')[:2], (0, '[]'))

    def test_expansion(self):
        self.assertEqual(flags('NAME=1')[:2], (0, '[-D NAME=1]'))
        self.assertEqual(flags('NAME=1,NAME2=3')[:2], (0, '[-D NAME=1 -D NAME2=3]'))
        self.assertEqual(flags('MOJOLEARN_X,MOJOLEARN_Y=-2,')[:2], (0, '[-D MOJOLEARN_X -D MOJOLEARN_Y=-2]'))
        self.assertEqual(flags('A=1,,B=0.5')[:2], (0, '[-D A=1 -D B=0.5]'))

    def test_refuses_malformed_entries(self):
        for bad in ('-D A=1', '-DA=1', 'A=1 B=2', '=1', 'A=$(x)', 'A;B'):
            rc, out, err = flags(bad)
            self.assertEqual(rc, 2, bad)
            self.assertIn('MOJOLEARN_BUILD_DEFINES', err)


class ScriptTests(unittest.TestCase):
    def test_every_build_script_uses_the_snippet(self):
        scripts = device_scripts()
        names = {p.name for p in scripts}
        self.assertIn('build.sh', names)
        self.assertIn('build_host_family.sh', names)
        self.assertGreaterEqual(len(scripts), 34)
        for path in scripts:
            lines = path.read_text().splitlines()
            self.assertTrue(lines[1].startswith(SOURCE_LINE), path.name + ': snippet not sourced on line 2')
            real = [l for l in lines if 'pixi run mojo build' in l and not l.lstrip().startswith('#')]
            self.assertEqual(len(real), 1, path.name)
            self.assertIn(EXPANSION, real[0], path.name)
            self.assertEqual(real[0].count('MOJOLEARN_BUILD_DEFINE_FLAGS'), 1, path.name)

    def test_host_wrappers_go_through_the_family(self):
        for path in sorted(BINDINGS.glob('build_*_host.sh')):
            self.assertIn('build_host_family.sh', path.read_text(), path.name)

    def test_older_variables_still_present(self):
        extra = [p.name for p in device_scripts() if re.search(r'\$\{MOJOLEARN_EXTRA_DEFINES:-\}', p.read_text())]
        build_extra = [p.name for p in device_scripts() if re.search(r'\$\{MOJOLEARN_BUILD_EXTRA_DEFINES:-\}', p.read_text())]
        self.assertTrue(extra and build_extra)
        self.assertIn('build_host_family.sh', build_extra)

    def test_outdir_lines_do_not_read_the_defines(self):
        for path in device_scripts():
            for line in path.read_text().splitlines():
                if re.match(r'\s*(OUTDIR|outdir|out_dir|OUT)=', line):
                    self.assertNotIn('BUILD_DEFINE', line, path.name)


if __name__ == '__main__':
    unittest.main()
