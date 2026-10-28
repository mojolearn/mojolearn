#!/usr/bin/env python3
"""One grid job's tools/bench_board.py races, run ON an lq box by an `lq add <box> CMD` line
(tools/six_lane_grid_lq.py render writes those lines; tools/six_lane_grid_lq.md).

box_job.sh runs a CMD in a detached worktree of the branch (the cwd), after building the BUILDS= bindings into
python/mojolearn with every MOJOLEARN_* token of the line exported (so MOJOLEARN_BUILD_DEFINES reached them).
This script:
  1. points the interpreter at the tree's python/ package with a .pth file in its own site-packages
     (bench_board's child_env drops PYTHONPATH so nothing shadows an installed wheel; the .pth is the
     installation here), and installs scikit-learn into it if missing (the pixi default env has none), plus torch
     (CPU wheels) when a Mamba lane is requested (its conductor's input tables import it; pip_extra);
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
# bench_board --neural-shape values a neural race may ask for with `@shape` (default full); target is the 162 M LM
# shape and exists for LM lanes only (bench_board_neural.LM_ONLY_SHAPES)
NEURAL_RACE_SHAPES = ('full', 'target')
LM_ONLY_SHAPES = ('target',)


def parse_race(text):
    """family:lane:dataset -> (family, lane, dataset); a neural race may name its own shape,
    family:lane:dataset@shape -> (family, lane, dataset, shape), e.g. neural:lm-train-step:bytes@target."""
    fam, lane, ds = text.split(':', 2)
    if fam not in FAMILIES:
        raise argparse.ArgumentTypeError('unknown bench_board family %r' % fam)
    if '@' not in ds:
        return fam, lane, ds
    ds, shape = ds.rsplit('@', 1)
    if fam != 'neural':
        raise argparse.ArgumentTypeError('only a neural race takes a @shape (%r)' % text)
    if shape not in NEURAL_RACE_SHAPES:
        raise argparse.ArgumentTypeError('unknown neural shape %r (one of %s)' % (shape, ', '.join(NEURAL_RACE_SHAPES)))
    return (fam, lane, ds) if shape == 'full' else (fam, lane, ds, shape)


def race_shape(race):
    """A requested race's bench_board --neural-shape: its @shape, else full."""
    return race[3] if len(race) > 3 else 'full'


def shaped_groups(races):
    """[(family, lanes, datasets, neural shape)] so that each bench_board call plans only requested races: per
    family, lanes with the same set of board datasets share one call; a lane on its own data (neural, a synthetic
    classical2/algos lane) races that data whatever --datasets says. A neural race at another shape
    (lm-train-step:bytes@target) is its own call with that --neural-shape; every other race stays at full."""
    per = {}
    for race in races:
        fam, lane, ds = race[:3]
        per.setdefault((fam, lane, race_shape(race)), set()).add(ds)
    out = {}
    for (fam, lane, shape), dss in sorted(per.items()):
        board = tuple(sorted(d for d in dss if d in BOARD_DATASETS)) or BOARD_DATASETS[:1]
        # A neural lane races its own data: bench_board plans one neural race per
        # call (several lanes under one --datasets planned only the first), so
        # each neural lane (and each of its shapes) gets its own call.
        out.setdefault((fam, board, lane if fam == 'neural' else '', shape), []).append(lane)
    return [(fam, sorted(lanes), list(board), shape) for (fam, board, _, shape), lanes in sorted(out.items())]


def groups(races):
    """shaped_groups without the shape: [(family, lanes, datasets)]."""
    return [(fam, lanes, dss) for fam, lanes, dss, _ in shaped_groups(races)]


DEFAULT_PIP = 'sklearn=scikit-learn==1.7.2'
# The Mamba lanes' conductor builds its inputs from mamba/corpus/gen_corpus.py's shape and range tables
# (bench_board_neural.mamba_weight_spec), and gen_corpus imports torch at module top. The box's pixi default env has
# no torch, so in grid ge123e6f9 every mamba1/2/3-forward race died in the conductor before writing its race JSON
# (status UNKNOWN(no race json, rc 1), all vendors, arms A and B). torch only shapes the inputs here: ours runs alone.
TORCH_MODELS = ('mamba1', 'mamba2', 'mamba3')
TORCH_PIP = ('torch', 'torch')
# torch from the CPU wheel index (no CUDA/ROCm runtime wheels: the conductor never runs torch on a device); PyPI
# is the fallback when that index cannot serve the interpreter.
PIP_INDEX = {'torch': 'https://download.pytorch.org/whl/cpu'}


def neural_models():
    """bench_board_neural.MODEL_OF ({lane: model}); its import is standard library only."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('gridbb_bench_board_neural', TREE / 'tools' / 'bench_board_neural.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.MODEL_OF)


def bench_board_module():
    """tools/bench_board.py as a module (its family rosters and tree task tables; standard library imports)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('gridbb_bench_board', TREE / 'tools' / 'bench_board.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unplanned_reason(BB, fam, lane, ds, shape='full', models=None):
    """None when bench_board plans the (family, lane, dataset) race, else why it never will.

    bench_board races only its own rosters: --families neural --lanes conv2d plans nothing (conv2d, moe and
    resnet-block are algos lanes), and a tree task lane races only the board datasets it has a task for
    (bench_board.tree_task_datasets: gbdt-categorical is taxi only). Such a request used to come back as a GRIDBB
    NO-RECORD line, which the main board listed as a FAILED cell with no ok cell (nv n0570 / amd a0885 asked
    neural:conv2d|moe|resnet-block:synthetic; nv n0320 / amd a0499 asked trees:gbdt-categorical:istella)."""
    if lane not in BB.family_lanes(fam):
        owner = [f for f in BB.FAMILIES if lane in BB.family_lanes(f)]
        return 'lane_%s_is_not_in_bench_board_family_%s%s' % (lane, fam, ('_(it_is_%s)' % ','.join(owner)) if owner else '')
    if fam == 'trees' and not BB.tree_task_datasets(lane, [ds]):
        return 'bench_board_plans_no_%s_race_for_%s_(TREE_TASK_DATASETS)' % (ds, lane)
    if fam == 'neural' and shape in LM_ONLY_SHAPES:
        models = models if models is not None else neural_models()
        if models.get(lane) != 'lm':
            return 'shape_%s_is_an_LM_shape_and_%s_is_not_an_LM_lane' % (shape, lane)
    return None


def pip_extra(races, models=None):
    """[(module, spec)] the requested races need beyond DEFAULT_PIP (torch for a Mamba lane)."""
    neural = [race[1] for race in races if race[0] == 'neural']
    if not neural:
        return []
    models = models if models is not None else neural_models()
    return [TORCH_PIP] if any(models.get(lane) in TORCH_MODELS for lane in neural) else []


def pip_arg(races, models=None):
    """The --pip value for these races: DEFAULT_PIP plus pip_extra, comma form."""
    return ','.join([DEFAULT_PIP] + ['%s=%s' % kv for kv in pip_extra(races, models)])


def ensure_env(py_extra):
    site = Path(sysconfig.get_paths()['purelib'])
    site.mkdir(parents=True, exist_ok=True)
    (site / 'mojolearn_grid_tree.pth').write_text(str(TREE / 'python') + '\n')
    missing = []
    for mod, spec in py_extra:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, spec))
    if missing:
        if subprocess.call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            subprocess.call([sys.executable, '-m', 'ensurepip', '--upgrade'])
        plain = [spec for mod, spec in missing if mod not in PIP_INDEX]
        if plain:
            rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-q'] + plain)
            print('GRIDBB-PIP %s rc=%d' % (','.join(plain), rc), flush=True)
        for mod, spec in missing:
            if mod in PIP_INDEX:
                rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', '--index-url', PIP_INDEX[mod], spec])
                if rc:
                    rc = subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', spec])
                print('GRIDBB-PIP %s rc=%d' % (spec, rc), flush=True)


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


def box_persisted(out, env=None):
    """MOJOLEARN_BOX_PERSIST=1 and <out> inside MOJOLEARN_RESULTS_DIR (bench_board.box_persisted_out's rule)."""
    env = os.environ if env is None else env
    res = env.get('MOJOLEARN_RESULTS_DIR')
    if env.get('MOJOLEARN_BOX_PERSIST') != '1' or not res:
        return False
    res, out = os.path.realpath(res), os.path.realpath(str(out))
    return out == res or out.startswith(res.rstrip(os.sep) + os.sep)


def race_ids(fam, lane, ds, shape='full'):
    """bench_board race ids (tools/bench_board.py race_id) a requested cell may have."""
    if fam == 'neural':
        return ['neural/%s/%s/shape=%s' % (lane, ds, shape)]
    return ['%s/%s/%s/rows=full' % (fam, lane, ds)]


def summarize(board, races, tag, vendor, h):
    recs = (board or {}).get('races') or {}
    ok = 0
    for race in races:
        fam, lane, ds = race[:3]
        shape = race_shape(race)
        rec = next((recs[r] for r in race_ids(fam, lane, ds, shape) if r in recs), None)
        if rec is None:  # the record id names the board's dataset; neural names its own data (and its shape)
            rec = next((v for v in recs.values() if v.get('family') == fam and v.get('lane') == lane
                        and (v.get('shape', 'full') == shape if fam == 'neural' else v.get('dataset') == ds)), None)
        cell = next((c for c in (rec or {}).get('cells') or [] if c.get('arm') == 'ours'), None)
        status = (cell or {}).get('status') or ('NO-RECORD' if rec is None else 'NO-OURS-CELL')
        ok += status == 'ok'
        # a non-full neural shape is named after the dataset (dataset=bytes shape=target); full lines are unchanged
        print('GRIDBB tag=%s vendor=%s head=%s family=%s lane=%s dataset=%s%s status=%s median_ms=%s hash=%s quality=%s'
              % (tag, vendor, h, fam, lane, ds, '' if shape == 'full' else ' shape=%s' % shape,
                 str(status).replace(' ', '_'), (cell or {}).get('median_ms'),
                 ((cell or {}).get('hash') or 'none')[:16],
                 json.dumps((cell or {}).get('quality') or {}, sort_keys=True, default=str)), flush=True)
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
    p.add_argument('--race', type=parse_race, action='append', required=True,
                   help='family:lane:dataset[@shape] (repeatable; @target = the 162 M LM shape, LM lanes only)')
    p.add_argument('--out', help='bench_board --out (default <worktree parent>/bb)')
    p.add_argument('--cache', default='/root/board-0833/cache', help='bench_board --cache: the box\'s prepped blocks')
    p.add_argument('--data-root', help='bench_board --data-root (default: its own, GBM_BENCH_DATA or ~/datasets/gbm-bench)')
    p.add_argument('--pip', default=DEFAULT_PIP, help='module=spec,... installed when missing (pip_extra adds torch for a '
                   'Mamba lane whatever this says)')
    p.add_argument('--dry-run', action='store_true', help='print the bench_board commands only')
    args = p.parse_args(argv)
    # A requested race bench_board never plans is refused here, by name, and never printed as a GRIDBB line
    # (a GRIDBB NO-RECORD reads as a failed board cell; tools/main_board_ingest.py skips GRIDBB-UNPLANNED).
    BB = bench_board_module()
    planned = []
    for race in args.race:
        fam, lane, ds = race[:3]
        why = unplanned_reason(BB, fam, lane, ds, race_shape(race))
        if why:
            print('GRIDBB-UNPLANNED tag=%s vendor=%s family=%s lane=%s dataset=%s reason=%s'
                  % (args.tag, args.vendor, fam, lane, ds, why), flush=True)
        else:
            planned.append(race)
    if not planned:
        print('GRIDBB-DONE tag=%s races=0 ok=0 bench_board_rc=none out=none unplanned=%d'
              % (args.tag, len(args.race)), flush=True)
        return 2
    args.race = planned
    out = Path(args.out or TREE.parent / 'bb')
    extra = [tuple(x.split('=', 1)) for x in args.pip.split(',') if x]
    extra += [kv for kv in pip_extra(args.race) if kv[0] not in {m for m, _ in extra}]
    cmds = []
    for fam, lanes, dss, shape in shaped_groups(args.race):
        cmd = [sys.executable, str(TREE / 'tools' / 'bench_board.py'), '--out', str(out), '--vendor', args.vendor,
               '--modes', 'identical', '--families', fam, '--lanes', ','.join(lanes), '--datasets', ','.join(dss),
               '--rows', 'full', '--neural-shape', shape, '--rounds', '1', '--python-env', sys.executable,
               '--skip-install', '--cache', args.cache, '--no-smoke-gate', '--no-infer']
        if args.data_root:
            cmd += ['--data-root', args.data_root]
        if box_persisted(out):
            # the results dir holding <out> reaches R2 from the box (tools/box_persist_run.sh): keep the evidence
            # archives local there instead of uploading them with an R2 credential the box does not hold
            cmd += ['--evidence-local-only']
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
