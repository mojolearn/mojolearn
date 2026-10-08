#!/usr/bin/env python3
"""Unit tests for tools/six_lane_grid_prebuild.py and tools/six_lane_grid_install_prebuilt.sh: a synthetic source
tree (fake build scripts that call `pixi run mojo build ... -o`), a fake compiler, fake .so files. Nothing is
compiled, raced or sent to a box."""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import six_lane_grid_prebuild as P  # noqa: E402
import six_lane_grid_lq as G  # noqa: E402

INSTALL = TOOLS / 'six_lane_grid_install_prebuilt.sh'

FAKE_BUILD = r'''#!/bin/sh
. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"
set -eu
cd "$(dirname "$0")/.."
mode_flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1"; outdir=python/mojolearn/identical
target_flags="--target-cpu x86-64-v3 --target-accelerator ${MOJOLEARN_GPU_ARCHS}"
[ "$(uname)" = Darwin ] && target_flags="--target-cpu apple-m1 --target-accelerator metal:1"
column_flags="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/fake.XXXXXX"); trap 'rm -rf "$tmpdir"' EXIT INT TERM
out=$tmpdir/__NAME__.so
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib ${MOJOLEARN_BUILD_DEFINE_FLAGS:-} \
    $target_flags $mode_flags $column_flags -I . -I bindings bindings/__NAME__.mojo -o "$out"
mkdir -p "$outdir"; mv "$out" "$outdir/__NAME__.so"; echo "built $outdir/__NAME__.so"
'''

FAKE_MOJO = r'''#!/bin/sh
# fake compiler: --version, or `build ... -o OUT`: OUT = the -D flags (so bits differ per define set)
[ "${1:-}" = --version ] && { echo "Mojo 9.9.9 (fake)"; exit 0; }
prev=; out=; defs=
for a in "$@"; do [ "$prev" = -o ] && out=$a; [ "$prev" = -D ] && defs="$defs $a"; prev=$a; done
printf 'SO%s\n' "$defs" > "$out"
'''


def chmodx(p):
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def make_tree(d):
    """A git repo with bindings/build.sh, build_x_a.sh, build_x_b.sh (+ the real build_defines.sh) and tiny .mojo roots."""
    d = Path(d)
    (d / 'bindings').mkdir(parents=True)
    (d / 'tools' / 'hooks').mkdir(parents=True)
    shutil.copy(TOOLS.parent / 'bindings' / 'build_defines.sh', d / 'bindings' / 'build_defines.sh')
    for script, name in (('build', '_mojolearn'), ('build_x_a', '_mojolearn_x_a'), ('build_x_b', '_mojolearn_x_b')):
        p = d / 'bindings' / (script + '.sh')
        p.write_text(FAKE_BUILD.replace('__NAME__', name))
        chmodx(p)
        (d / 'bindings' / (name + '.mojo')).write_text('from core.k import run\n' if name != '_mojolearn_x_b' else 'fn main(): pass\n')
    (d / 'core').mkdir()
    (d / 'core' / 'k.mojo').write_text('fn run(): is_defined["MOJOLEARN_A"]()\n')
    subprocess.run(['git', 'init', '-q', str(d)], check=True)
    subprocess.run(['git', '-C', str(d), '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(d), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-m', 'x'], check=True)
    return subprocess.check_output(['git', '-C', str(d), 'rev-parse', 'HEAD'], text=True).strip()


LINES = """lq add nv RACE main ridge-cv taxi MOJOLEARN_GRID_TAG=g.P001 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_A=1,MOJOLEARN_B=2 BUILDS=build,build_x_a
lq add nv RACE main lasso-cv taxi MOJOLEARN_GRID_TAG=g.P002 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_B=2,MOJOLEARN_A=1 BUILDS=build,build_x_b
lq add nv RACE main ridge-cv istella MOJOLEARN_GRID_TAG=g.P003 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_A=1 BUILDS=build,build_x_a,build_x_a_host
lq add nv RACE main ridge-cv taxi MOJOLEARN_GRID_TAG=g.B001r1 BUILDS=build,build_x_a
lq add nv CMD main g.P004.bb MOJOLEARN_GRID_TAG=g.P004.bb MOJOLEARN_BUILD_DEFINES=MOJOLEARN_C=1 .pixi/envs/default/bin/python tools/six_lane_grid_bb.py --tag g.P004.bb --vendor nvidia --race trees:rf:taxi BUILDS=build,build_x_b
lq add nv ID main ridge-cv taxi
"""
REACH = {'MOJOLEARN_A': ['_mojolearn', '_mojolearn_x_a'], 'MOJOLEARN_B': ['_mojolearn_x_b'], 'MOJOLEARN_C': []}


class PlanTests(unittest.TestCase):
    def test_parse_lines(self):
        specs = P.parse_lines(LINES)
        self.assertEqual([s['kind'] for s in specs], ['RACE', 'RACE', 'RACE', 'RACE', 'CMD'])  # ID lines dropped
        self.assertEqual(specs[1]['defines'], ['MOJOLEARN_A=1', 'MOJOLEARN_B=2'])  # sorted
        self.assertEqual(specs[3]['defines'], [])
        self.assertEqual(specs[4]['builds'], ['build', 'build_x_b'])

    def test_full_sha_matches_the_install_script_pipeline(self):
        for csv in ('MOJOLEARN_B=2,MOJOLEARN_A=1,MOJOLEARN_A=1', ''):
            want = P.full_sha(csv.split(',') if csv else [])
            sh = ("{ [ -n \"$1\" ] && printf '%s' \"$1\" | tr ',' '\\n' | sed '/^$/d' | LC_ALL=C sort -u; } | "
                  "(command -v sha256sum > /dev/null 2>&1 && sha256sum || shasum -a 256) | cut -d' ' -f1")
            got = subprocess.check_output(['bash', '-c', sh, 'x', csv], text=True).strip()
            self.assertEqual(got, want, csv)
        self.assertEqual(P.full_sha([]), P.EMPTY_SHA)

    def test_dedup_by_reach(self):
        specs = P.parse_lines(LINES)
        arts, sets, stats = P.plan_artifacts(specs, REACH)
        # 5 lines; 4 distinct full sets ({A,B}, {A}, {}, {C}); host script skipped
        self.assertEqual(len(sets), 4)
        self.assertEqual(stats['skipped_scripts'], {'build_x_a_host': 1})
        self.assertEqual(stats['pairs_binding_x_fullset'], 9)  # (build,x_a)x{A,B} + (build,x_b)x{A,B} shared -> 3; {A}:2; {}:2; {C}:2
        by = {(a['binding'], tuple(a['defines'])): a for a in arts}
        # _mojolearn: {A,B}->{A}, {A}->{A}, {}->{}, {C}->{} : two artifacts
        self.assertEqual(sorted(k for k in by if k[0] == '_mojolearn'), [('_mojolearn', ()), ('_mojolearn', ('MOJOLEARN_A=1',))])
        self.assertEqual(sorted(by[('_mojolearn', ())]['serves']), sorted([P.full_sha([]), P.full_sha(['MOJOLEARN_C=1'])]))
        self.assertEqual(by[('_mojolearn', ('MOJOLEARN_A=1',))]['lines'], 3)
        # _mojolearn_x_a: {A,B} and {A} both narrow to {A}; {} stays {}
        self.assertEqual(sorted(k for k in by if k[0] == '_mojolearn_x_a'), [('_mojolearn_x_a', ()), ('_mojolearn_x_a', ('MOJOLEARN_A=1',))])
        # _mojolearn_x_b: {A,B}->{B}; {C}->{}
        self.assertEqual(sorted(k for k in by if k[0] == '_mojolearn_x_b'), [('_mojolearn_x_b', ()), ('_mojolearn_x_b', ('MOJOLEARN_B=2',))])
        self.assertEqual(len(arts), 6)
        self.assertEqual(arts[0]['lines'], max(a['lines'] for a in arts))  # most-used first

    def test_script_binding(self):
        self.assertEqual(P.script_binding('build'), '_mojolearn')
        self.assertEqual(P.script_binding('build_x_linear'), '_mojolearn_x_linear')
        self.assertIsNone(P.script_binding('build_x_linear_host'))
        self.assertIsNone(P.script_binding('build_host_family'))
        self.assertIsNone(P.script_binding('pixi'))

    def test_render_prebuilt_token(self):
        line = G.lq_line('nv', 'main', [('ridge-cv', 'taxi')], [('MOJOLEARN_GRID_TAG', 'g.P1')], ['build_x_linear'], '/root/grid-prebuilt')
        self.assertTrue(line.endswith(' BUILDS=build,build_x_linear PREBUILT=/root/grid-prebuilt'), line)
        self.assertNotIn('PREBUILT', G.lq_line('nv', 'main', [('ridge-cv', 'taxi')], [], ['build_x_linear']))
        cmd = G.cmd_line('nv', 'nvidia', 'main', 'g.P1.bb', [('trees', 'rf', 'taxi')], [], ['build'], '/root/grid-prebuilt')
        self.assertTrue(cmd.endswith(' BUILDS=build PREBUILT=/root/grid-prebuilt'), cmd)
        self.assertEqual(P.parse_lines(line + '\n')[0]['prebuilt'], '/root/grid-prebuilt')


class BuildAndInstallTests(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix='prebuild-test-'))
        self.root = self.td / 'tree'
        self.head = make_tree(self.root)
        self.out = self.td / 'store'
        self.mojo = self.td / 'mojo'
        self.mojo.write_text(FAKE_MOJO)
        chmodx(self.mojo)
        lines = self.td / 'lines.txt'
        lines.write_text(LINES)
        reach = self.td / 'reach.json'
        reach.write_text(json.dumps(REACH))
        rc = P.main(['plan', '--lines', str(lines), '--vendor', 'nvidia', '--out', str(self.out), '--root', str(self.root),
                     '--reach-json', str(reach)])
        self.assertEqual(rc, 0)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def build(self, *extra):
        return P.main(['build', '--vendor', 'nvidia', '--out', str(self.out), '--root', str(self.root), '--compiler', str(self.mojo),
                       '--jobs', '2', '--no-closure', '--emulate-linux', '--force-box-usable', '--no-smoke-host-cpu', *extra])

    def test_build_receipts_manifest_lookup_and_resume(self):
        plan = json.loads((self.out / 'nvidia' / 'plan.json').read_text())
        self.assertEqual(plan['source_sha'], self.head)
        self.assertEqual(plan['counts']['artifacts'], 6)
        self.assertEqual(self.build(), 0)
        man = json.loads((self.out / 'nvidia' / 'manifest.json').read_text())
        self.assertEqual(man['counts']['compiled'], 6)
        self.assertEqual(man['counts']['failed'], 0)
        self.assertEqual(man['compiler_version'], 'Mojo 9.9.9 (fake)')
        a = next(x for x in man['artifacts'] if x['binding'] == '_mojolearn_x_a' and x['defines'] == ['MOJOLEARN_A=1'])
        rec = json.loads((self.out / 'nvidia' / Path(a['artifact']).parent / 'receipt.json').read_text())
        self.assertEqual(rec['status'], 'COMPILED')
        self.assertEqual(rec['argv_recorded'][:4], ['pixi', 'run', 'mojo', 'build'])
        self.assertEqual(rec['argv'][0], str(self.mojo))
        # the box's flags, from the real script under the Linux shim: identical mode, vendor column, sm_89, the effective define
        for tok in ('--target-accelerator', 'sm_89', '--target-cpu', 'x86-64-v3', 'MOJOLEARN_NUMERIC_IDENTICAL=1', 'MOJOLEARN_COLUMN_NVIDIA',
                    'MOJOLEARN_A=1', '--emit', 'shared-lib'):
            self.assertIn(tok, rec['argv'], tok)
            self.assertIn(tok, rec['argv_recorded'], tok)
        self.assertEqual(rec['argv_substitutions'], [])
        self.assertNotIn('MOJOLEARN_B=2', rec['argv'])  # B does not reach x_a
        self.assertEqual(rec['argv'][rec['argv'].index('-j') + 1], '1')
        self.assertEqual(rec['dest'], 'python/mojolearn/identical/_mojolearn_x_a.so')
        self.assertEqual(rec['dest_source'], 'dry-run')
        self.assertEqual(rec['source_sha'], self.head)
        self.assertTrue(rec['script_sha256'] and rec['build_defines_sha256'])
        so = self.out / 'nvidia' / a['artifact']
        self.assertEqual(hashlib.sha256(so.read_bytes()).hexdigest(), rec['artifact_sha256'])
        self.assertIn('MOJOLEARN_A=1', so.read_text())
        # the dry run left nothing in the tree
        self.assertFalse((self.root / 'python' / 'mojolearn' / 'identical' / '_mojolearn_x_a.so').exists())
        self.assertEqual(subprocess.check_output(['git', '-C', str(self.root), 'status', '--porcelain'], text=True), '')
        # lookup rows: one per (binding, full set served); 4 sets x their bindings: {A,B}:3, {A}:2, {}:2, {C}:2 = 9
        rows = [ln.split('\t') for ln in (self.out / 'nvidia' / 'lookup.tsv').read_text().splitlines() if not ln.startswith('#')]
        self.assertEqual(len(rows), 9)
        self.assertTrue((self.out / 'nvidia' / 'lookup.tsv').read_text().startswith('# schema=%s vendor=nvidia' % P.SCHEMA))
        self.assertIn('source_sha=%s' % self.head, (self.out / 'nvidia' / 'lookup.tsv').read_text().splitlines()[0])
        # resume: a second build compiles nothing new
        self.assertEqual(self.build(), 0)
        man2 = json.loads((self.out / 'nvidia' / 'manifest.json').read_text())
        self.assertEqual(man2['counts']['resumed'], 6)
        self.assertEqual(man2['counts']['compiled'], 6)

    def test_build_refuses_a_moved_tree(self):
        (self.root / 'core' / 'k.mojo').write_text('fn run(): pass\n')
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-am', 'y'], check=True)
        with self.assertRaises(SystemExit) as cm:
            self.build()
        self.assertIn('re-run plan', str(cm.exception))

    def test_install_script(self):
        self.assertEqual(self.build(), 0)
        store = self.out
        tree = self.td / 'worktree'
        subprocess.run(['git', 'clone', '-q', str(self.root), str(tree)], check=True)
        env = dict(os.environ, LQ_PREBUILT=str(store))

        def run(defs, builds, vendor='nvidia'):
            r = subprocess.run(['bash', str(INSTALL), str(tree), vendor, defs, builds], env=env, capture_output=True, text=True)
            return r.returncode, r.stdout
        # arm A line: both bindings from the store, bits of the narrowed sets
        rc, out = run('MOJOLEARN_B=2,MOJOLEARN_A=1', 'build,build_x_a')
        self.assertEqual(rc, 0, out)
        self.assertIn('PREBUILT ok nvidia n=2', out)
        so = tree / 'python' / 'mojolearn' / 'identical' / '_mojolearn_x_a.so'
        self.assertIn('MOJOLEARN_A=1', so.read_text())
        self.assertNotIn('MOJOLEARN_B=2', so.read_text())
        self.assertNotIn('MOJOLEARN_B=2', (tree / 'python' / 'mojolearn' / 'identical' / '_mojolearn.so').read_text())
        # arm B line (no defines) and "-"
        self.assertEqual(run('', 'build,build_x_a')[0], 0)
        self.assertEqual(run('', 'build,build_x_b')[0], 2)  # no rendered line pairs x_b with the empty set
        self.assertEqual(run('-', 'build')[0], 0)
        # unknown define set -> 2 (box builds)
        rc, out = run('MOJOLEARN_Z=1', 'build')
        self.assertEqual(rc, 2, out)
        self.assertIn('no artifact', out)
        # host binding -> 2; unknown vendor store -> 2
        self.assertEqual(run('MOJOLEARN_A=1', 'build,build_x_a_host')[0], 2)
        self.assertEqual(run('MOJOLEARN_A=1', 'build', 'amd')[0], 2)
        # corrupt artifact -> 2 and nothing installed from it
        man = json.loads((store / 'nvidia' / 'manifest.json').read_text())
        a = next(x for x in man['artifacts'] if x['binding'] == '_mojolearn_x_b' and x['defines'] == [])
        (store / 'nvidia' / a['artifact']).write_text('garbage')
        rc, out = run('MOJOLEARN_C=1', 'build_x_b')  # {C} narrows to the empty set for x_b
        self.assertEqual(rc, 2, out)
        self.assertIn('sha mismatch', out)
        self.assertFalse(list((tree / 'python' / 'mojolearn' / 'identical').glob('*.tmp')))
        # a tree at another commit -> 2
        (tree / 'core' / 'k.mojo').write_text('fn run(): pass\n')
        subprocess.run(['git', '-C', str(tree), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-q', '-am', 'z'], check=True)
        rc, out = run('MOJOLEARN_A=1', 'build')
        self.assertEqual(rc, 2, out)
        self.assertIn('store source=', out)

    def test_pack_and_box_script(self):
        self.assertEqual(self.build(), 0)
        rc = P.main(['pack', '--out', str(self.out), '--vendor', 'nvidia'])
        self.assertEqual(rc, 0)
        tars = list(self.out.glob('grid-prebuilt-nvidia-*.tar.gz'))
        self.assertEqual(len(tars), 1)
        side = json.loads(Path(str(tars[0]) + '.json').read_text())
        self.assertEqual(side['artifacts'], 6)
        self.assertEqual(side['sha256'], hashlib.sha256(tars[0].read_bytes()).hexdigest())
        self.assertTrue(side['key'].startswith('grid-prebuilt/%s/nvidia-' % self.head))
        names = subprocess.check_output(['tar', 'tzf', str(tars[0])], text=True).split()
        self.assertIn('nvidia/lookup.tsv', names)
        self.assertIn('nvidia/manifest.json', names)
        self.assertEqual(sum(1 for n in names if n.endswith('.so')), 6)
        script = P.box_script('nvidia', side, 'https://example.invalid/x', '/root/grid-prebuilt')
        self.assertIn(side['sha256'], script)
        self.assertIn('D=/root/grid-prebuilt;', script)
        self.assertIn('"$D/nvidia"', script)
        self.assertIn("sha256sum -c", script)
        # the script's unpack + verify steps work locally (sha256sum or shasum)
        if shutil.which('sha256sum'):
            dest = self.td / 'boxdest'
            local = script.replace("'https://example.invalid/x'", "'file://%s'" % tars[0]).replace('D=/root/grid-prebuilt', 'D=%s' % dest)
            r = subprocess.run(['bash', '-c', local], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((dest / 'nvidia' / 'lookup.tsv').is_file())


if __name__ == '__main__':
    unittest.main()
