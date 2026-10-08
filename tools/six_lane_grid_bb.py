#!/usr/bin/env python3
"""One grid job's tools/bench_board.py races, run ON an lq box by an `lq add <box> CMD` line
(tools/six_lane_grid_lq.py render writes those lines; tools/six_lane_grid_lq.md).

box_job.sh runs a CMD in a detached worktree of the branch (the cwd), after building the BUILDS= bindings into
python/mojolearn with every MOJOLEARN_* token of the line exported (so MOJOLEARN_BUILD_DEFINES reached them).
This script:
  1. points the interpreter at the tree's python/ package with a .pth file in its own site-packages
     (bench_board's child_env drops PYTHONPATH so nothing shadows an installed wheel; the .pth is the
     installation here), and installs scikit-learn into it if missing (the pixi default env has none);
  2. runs tools/bench_board.py --modes identical, ours only, full rows, --no-infer, into <job dir>/bb
     (<job dir> = the parent of the worktree, /root/lq/out/<id>/, which outlives the worktree), once per
     (family, dataset set) so it races exactly the requested (family, lane, dataset) cells;
  3. prints one `GRIDBB ...` line per requested race from <job dir>/bb/board.json (our cell: median_ms,
     status, the output hash, quality) and a last `GRIDBB-DONE ...` line (box_job.sh copies the last line
     into results.txt).

Nothing here times anything itself; bench_board and its drivers do. Exit 0 when every requested race has
an ok cell of ours, else 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
BOARD_DATASETS = ('taxi', 'istella')
FAMILIES = ('trees', 'classical', 'classical2', 'neural', 'algos')


def parse_race(text):
    fam, lane, ds = text.split(':', 2)
    if fam not in FAMILIES:
        raise argparse.ArgumentTypeError('unknown bench_board family %r' % fam)
    return fam, lane, ds


def groups(races):
    """[(family, lanes, datasets)] so that each bench_board call plans only requested races: per family,
    lanes with the same set of board datasets share one call; a lane on its own data (neural, a synthetic
    classical2/algos lane) races that data whatever --datasets says."""
    per = {}
    for fam, lane, ds in races:
        per.setdefault((fam, lane), set()).add(ds)
    out = {}
    for (fam, lane), dss in sorted(per.items()):
        board = tuple(sorted(d for d in dss if d in BOARD_DATASETS)) or BOARD_DATASETS[:1]
        out.setdefault((fam, board), []).append(lane)
    return [(fam, sorted(lanes), list(board)) for (fam, board), lanes in sorted(out.items())]


def ensure_env(py_extra):
    site = Path(sysconfig.get_paths()['purelib'])
    site.mkdir(parents=True, exist_ok=True)
    (site / 'mojolearn_grid_tree.pth').write_text(str(TREE / 'python') + '\n')
    missing = []
    for mod, spec in py_extra:
        try:
            __import__(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        if subprocess.call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            subprocess.call([sys.executable, '-m', 'ensurepip', '--upgrade'])
        rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        print('GRIDBB-PIP %s rc=%d' % (','.join(missing), rc), flush=True)


# The Mojo runtime libraries every compiled binding links (NEEDED libKGENCompilerRTShared.so, ...). A binding built
# on the box finds them through the RUNPATH its build tree's pixi env wrote; a PREBUILT binding
# (tools/six_lane_grid_install_prebuilt.sh) carries the build box's tree path instead, which does not exist here,
# so its import failed: grid ge123e6f9, 150 of 150 PREBUILT jobs refused every race before racing ("our IDENTICAL GPU
# set cannot load ... ImportError: libKGENCompilerRTShared.so: cannot open shared object file"), every built job ran.
RUNTIME_LIBS = ('libKGENCompilerRTShared.so',)


def runtime_lib_dirs():
    """The interpreter env's lib directory (the tree's .pixi/envs/default/lib, the same frozen toolchain the
    prebuild compiled with) when it holds the Mojo runtime libraries; [] otherwise."""
    d = Path(sys.prefix) / 'lib'
    return [str(d)] if all((d / n).exists() for n in RUNTIME_LIBS) else []


def ensure_runtime_path():
    """Prepend runtime_lib_dirs() to LD_LIBRARY_PATH for every bench_board child (child_env copies os.environ; the
    loader searches LD_LIBRARY_PATH before a binding's RUNPATH, so a built binding resolves the same files)."""
    dirs = runtime_lib_dirs()
    have = [x for x in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if x]
    add = [d for d in dirs if d not in have]
    if add:
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(add + have)
    print('GRIDBB-LIBPATH %s' % (','.join(dirs) or 'none: %s/lib holds no %s' % (sys.prefix, ','.join(RUNTIME_LIBS))),
          flush=True)


def run_logged_stderr(cmd):
    """subprocess.call with stderr also kept: (rc, last non-empty stderr line). bench_board's refusals
    (SystemExit before any race, so no record) are on stderr; the GRIDBB-ERR line carries the reason home."""
    p = subprocess.Popen(cmd, cwd=str(TREE), stderr=subprocess.PIPE, text=True, errors='replace')
    last = ''
    for line in p.stderr:
        sys.stderr.write(line)
        if line.strip():
            last = line.strip()
    return p.wait(), last


def head():
    try:
        return subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def race_ids(fam, lane, ds):
    """bench_board race ids (tools/bench_board.py race_id) a requested cell may have."""
    if fam == 'neural':
        return ['neural/%s/%s/shape=full' % (lane, ds)]
    return ['%s/%s/%s/rows=full' % (fam, lane, ds)]


def summarize(board, races, tag, vendor, h):
    recs = (board or {}).get('races') or {}
    ok = 0
    for fam, lane, ds in races:
        rec = next((recs[r] for r in race_ids(fam, lane, ds) if r in recs), None)
        if rec is None:  # the record id names the board's dataset; neural names its own data
            rec = next((v for v in recs.values() if v.get('family') == fam and v.get('lane') == lane
                        and (fam == 'neural' or v.get('dataset') == ds)), None)
        cell = next((c for c in (rec or {}).get('cells') or [] if c.get('arm') == 'ours'), None)
        status = (cell or {}).get('status') or ('NO-RECORD' if rec is None else 'NO-OURS-CELL')
        ok += status == 'ok'
        print('GRIDBB tag=%s vendor=%s head=%s family=%s lane=%s dataset=%s status=%s median_ms=%s hash=%s quality=%s'
              % (tag, vendor, h, fam, lane, ds, str(status).replace(' ', '_'), (cell or {}).get('median_ms'),
                 ((cell or {}).get('hash') or 'none')[:16],
                 json.dumps((cell or {}).get('quality') or {}, sort_keys=True, default=str)), flush=True)
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
    p.add_argument('--race', type=parse_race, action='append', required=True, help='family:lane:dataset (repeatable)')
    p.add_argument('--out', help='bench_board --out (default <worktree parent>/bb)')
    p.add_argument('--cache', default='/root/board-0833/cache', help='bench_board --cache: the box\'s prepped blocks')
    p.add_argument('--data-root', help='bench_board --data-root (default: its own, GBM_BENCH_DATA or ~/datasets/gbm-bench)')
    p.add_argument('--pip', default='sklearn=scikit-learn==1.7.2', help='module=spec,... installed when missing')
    p.add_argument('--dry-run', action='store_true', help='print the bench_board commands only')
    args = p.parse_args(argv)
    out = Path(args.out or TREE.parent / 'bb')
    extra = [tuple(x.split('=', 1)) for x in args.pip.split(',') if x]
    cmds = []
    for fam, lanes, dss in groups(args.race):
        cmd = [sys.executable, str(TREE / 'tools' / 'bench_board.py'), '--out', str(out), '--vendor', args.vendor,
               '--modes', 'identical', '--families', fam, '--lanes', ','.join(lanes), '--datasets', ','.join(dss),
               '--rows', 'full', '--neural-shape', 'full', '--rounds', '1', '--python-env', sys.executable,
               '--skip-install', '--cache', args.cache, '--no-smoke-gate', '--no-infer']
        if args.data_root:
            cmd += ['--data-root', args.data_root]
        cmds.append(cmd)
    if args.dry_run:
        for cmd in cmds:
            print(' '.join(cmd))
        return 0
    ensure_env(extra)
    ensure_runtime_path()
    rcs = []
    for i, cmd in enumerate(cmds):
        print('GRIDBB-RUN ' + ' '.join(cmd[2:]), flush=True)
        rc, last = run_logged_stderr(cmd)
        rcs.append(rc)
        if rc:
            print('GRIDBB-ERR tag=%s call=%d rc=%d last=%s' % (args.tag, i, rc, last[:300].replace(' ', '_')), flush=True)
    try:
        board = json.loads((out / 'board.json').read_text())
    except (OSError, ValueError):
        board = None
    ok = summarize(board, args.race, args.tag, args.vendor, head())
    print('GRIDBB-DONE tag=%s races=%d ok=%d bench_board_rc=%s out=%s'
          % (args.tag, len(args.race), ok, ','.join(map(str, rcs)), out), flush=True)
    return 0 if ok == len(args.race) else 1


if __name__ == '__main__':
    raise SystemExit(main())
