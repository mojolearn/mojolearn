#!/usr/bin/env python3
"""Run the IDENTICAL switch grid over the lq box queue (tools/six_lane_grid_lq.md).

  render   grid plan -> one `lq add <nv|amd> RACE ...` line per build pack (arm A, the pack's define set
           through MOJOLEARN_BUILD_DEFINES) with the incumbent (arm B, no defines) interleaved so every
           workload's B is repeated --b-repeats times spread through the file (the noise floor).
  redo-not-ready  RACE lines whose job came back status=not_ready (infrastructure) -> the same lines under the
           redo tag <run>.<base>n<k>; collect maps it to <base> where earlier reps were not_ready or absent.
  collect  lq results.txt lines (+ the race logs for lines lq cut at 600 characters) -> grid-verdicts.json
           (six_lane_timing verdicts schema), summary.json (six_lane_compare_results schema: NVIDIA vs AMD
           output digests per arm) and quality.json (quality-review rows), which
           tools/six_lane_grid_decide.py reads unchanged.

Metadata only: never builds, runs, times or connects to a box. lq is the only way to a box.

Two routes per workload id, both one build per line:
  RACE  `expanded:lane@dataset=ds[@input=v]` -> `lq add <box> RACE ... lane ds` (tools/bench_board_algos.py
        race, through box_job.sh / overlay_race_job2.sh); prints ALGOS lines with digest=.
  CMD   every other family -> `lq add <box> CMD <branch> <tag> ... tools/six_lane_grid_bb.py --race
        family:lane:dataset ...`, which runs tools/bench_board.py (families trees, classical, classical2,
        neural) at full rows, IDENTICAL, ours only, and prints GRIDBB lines (median_ms, output hash, quality).
        classical:L@dataset=D -> classical/L/D; more:L@dataset=D -> classical2/L/D; neural:L -> neural/L/<its
        data>; tree L:D -> trees/L/<board dataset> (task lanes map their driver dataset back through
        bench_board TREE_TASK_DATASETS).
Lane and dataset names are checked against bench_board_algos LANES (RACE) and the races bench_board.plan_races
plans on the vendor (CMD). Anything else (a lane variant `lane@variant@...`, a tree driver dataset with no board
dataset such as rf:year or iforest:anomaly) is refused by name, never substituted. `@input=<variant>` is accepted
and recorded: the board races its own full-row block, not the registered full-input variant, so A and B share
one input but it is not the saved recipe's input.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import six_lane_timing as T  # noqa: E402  (reuse: PHASES, judge, _spread, VOTERS)
import six_lane_grid_bb as BBW  # noqa: E402  (pip_arg: what the box runner installs for a line's races)

BOX = {'nvidia': 'nv', 'amd': 'amd'}
VENDOR_OF_BOX = {v: k for k, v in BOX.items()}
PAIR_SECONDS = {'nvidia': 38.0, 'amd': 24.0}  # median scored-pair seconds (tools/six_lane_grid.py PAIR_SECONDS)
REGIME_RANK = {'factorial': 0, 'pairwise': 1}
TAG_ENV = 'MOJOLEARN_GRID_TAG'
DEFINES_ENV = 'MOJOLEARN_BUILD_DEFINES'
PREBUILT_TOKEN = 'PREBUILT='  # + store path on the box (tools/six_lane_grid_prebuild.py); render --prebuilt
IDENTITY_COLUMN = {'nvidia': 'nvidia-native', 'amd': 'amd'}  # six_lane_compare_results REQUIRED_IDENTICAL
DIGEST_CHARS = 16  # box_job.sh keeps the first 16 hex characters of the race digest
QUALITY_REL = 1e-3  # relative: a bits-changing IDENTICAL arm legitimately moves a metric in the 6th digit; 0.1% is the material gate (CLAUDE.md "no material drop")
TIMING_SOURCE = ('median_ms of our arm: lq RACE ALGOS lines (tools/bench_board_algos.py) or lq CMD GRIDBB lines '
                 '(tools/bench_board.py cells); one run per arm')

# ------------------------------------------------------------------ plan + lanes


def load_plan(plan_dir):
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'grid-plan.json').read_text())
    with gzip.open(plan_dir / 'grid-matrix.json.gz', 'rt') as f:
        matrix = json.load(f)
    return plan, matrix


def run_id_of(plan_dir):
    """Short id of the plan files: tags carry it, so collect ignores lines of another campaign."""
    h = hashlib.sha256()
    for name in ('grid-plan.json', 'grid-matrix.json.gz'):
        h.update((Path(plan_dir) / name).read_bytes())
    return 'g' + h.hexdigest()[:8]


def load_lanes(path=None):
    """{lane: [datasets]} from tools/bench_board_algos.py LANES (or a JSON file of the same shape)."""
    if path:
        return {k: list(v) for k, v in json.loads(Path(path).read_text()).items()}
    import bench_board_algos as B  # metadata only: the registry, nothing is raced
    return {k: list(v.get('datasets') or []) for k, v in B.LANES.items()}


def map_workload(wid, lanes):
    """workload id -> (lane, dataset, note) or raise ValueError(reason)."""
    if not wid or ':' not in wid:
        raise ValueError('not a family:lane workload id')
    family, rest = wid.split(':', 1)
    parts = rest.split('@')
    lane, quals = parts[0], parts[1:]
    kv = {}
    for q in quals:
        if '=' not in q:
            raise ValueError('lane variant @%s has no bench_board_algos lane' % q)
        k, v = q.split('=', 1)
        kv[k] = v
    extra = sorted(set(kv) - {'dataset', 'input'})
    if extra:
        raise ValueError('workload qualifier(s) %s have no lq RACE form' % ','.join(extra))
    if 'dataset' not in kv:
        raise ValueError('no @dataset=: %s is not a bench_board_algos workload (family %s)' % (rest, family))
    if lane not in lanes:
        raise ValueError('lane %r is not a tools/bench_board_algos.py lane (family %s)' % (lane, family))
    ds = kv['dataset']
    if lanes[lane] and ds not in lanes[lane]:
        raise ValueError('lane %r races datasets %s, not %r' % (lane, ','.join(lanes[lane]), ds))
    note = None
    if 'input' in kv:
        note = 'input=%s not reproduced: lq races the rows-full board block' % kv['input']
    return lane, ds, note


def binding_builds(bindings):
    """_mojolearn -> build, _mojolearn_x -> build_x (device scripts only; BUILDS= for box_job.sh)."""
    out = []
    for b in bindings or []:
        name = 'build' if b == '_mojolearn' else 'build_' + b[len('_mojolearn_'):] if b.startswith('_mojolearn_') else None
        if name and (ROOT / 'bindings' / (name + '.sh')).exists() and name not in out:
            out.append(name)
    return out


# Bindings a bench_board neural lane's public class loads beyond the ones its plan entry names (the plan's
# workload_bindings list the bindings the CONTROLS reach). A worktree on the box holds only the BUILDS= bindings (or
# the prebuilt ones), so a missing one refuses the race: SmallMLPTrainer._binding() calls
# _linalg_impl.require_identical() in IDENTICAL (python/mojolearn/_mlp_impl.py), and grid ge123e6f9 refused every
# mlp-train-step race with "numeric_mode='identical' needs .../_mojolearn_linalg.so, which is not built".
NEURAL_RUNTIME_BINDINGS = {'mlp-train-step': ('_mojolearn_training', '_mojolearn_linalg')}

# Runtime environment a define set needs: a binary built with an experimental numerical profile refuses to load
# unless the process names that exact profile (python/mojolearn/_linalg_impl.py _EXPECTED_PROFILE_ENV; the profile
# name is gemm/contract.mojo GEMM_NUMERICAL_PROFILE). Grid ge123e6f9 refused the MOJOLEARN_IDN_GEMM_LEAF=3 (all256)
# gemm cells by that check. B lines (no defines) never get these.
DEFINE_ENVS = {'MOJOLEARN_IDN_GEMM_LEAF=3': ('MOJOLEARN_EXPERIMENT_GEMM_PROFILE', 'mojolearn.identical.gemm.fp32.ni08-leaf256')}


def define_envs(defines):
    """[(name, value)] runtime environment the define list needs (DEFINE_ENVS), in define order."""
    if 'MOJOLEARN_IDN_ALL_OFF' in {d.split('=', 1)[0] for d in defines or []}:
        return []  # gemm/contract.mojo: _NEURAL_GEMM_PROFILE_ARM is off under IDN_ALL_OFF, the binary stays v1
    out = []
    for d in defines or []:
        kv = DEFINE_ENVS.get(d)
        if kv and kv not in out:
            out.append(kv)
    return out


def workload_selected(wid, only_workloads):
    """only_workloads: None (all) or a list of workload ids / prefixes (an entry ending in ':' is a family prefix)."""
    if not only_workloads:
        return True
    return any(wid == w or (w.endswith(':') and wid.startswith(w)) or wid.startswith(w + '@') for w in only_workloads)


CMD_PY = '.pixi/envs/default/bin/python'  # the branch tree's pixi interpreter (box_job.sh CMD cwd = the tree)
CMD_SCRIPT = 'tools/six_lane_grid_bb.py'
FAMILY_OF_PREFIX = {'classical': 'classical', 'more': 'classical2', 'neural': 'neural'}


def load_board(vendor):
    """What tools/bench_board.py races on this vendor in IDENTICAL: {races: {(family, lane, dataset)},
    neural_data: {lane: data}, tree_task: {lane: {board ds: driver ds}}, tree_lanes: [...]}."""
    import bench_board as BB  # metadata only: plan_races runs nothing
    fams = [f for f in BB.FAMILIES if f != 'algos']
    races = BB.plan_races(vendor, ['identical'], fams, None, list(BB.DATASETS))
    return dict(races={(r['family'], r['lane'], r['dataset']) for r in races}, neural_data=dict(BB.NEURAL_DATA),
                tree_task={k: dict(v) for k, v in BB.TREE_TASK_DATASETS.items()},
                tree_lanes=list(BB.family_lanes('trees')), board_datasets=list(BB.DATASETS))


def map_board(wid, board):
    """workload id -> (family, lane, dataset, note) for a tools/bench_board.py race, or raise ValueError."""
    prefix, _, rest = wid.partition(':')
    note = None
    if prefix in ('classical', 'more'):
        parts = rest.split('@')
        lane, kv = parts[0], {}
        for q in parts[1:]:
            if '=' not in q:
                raise ValueError('lane variant @%s has no bench_board race' % q)
            k, v = q.split('=', 1)
            kv[k] = v
        extra = sorted(set(kv) - {'dataset', 'input'})
        if extra or 'dataset' not in kv:
            raise ValueError('workload qualifier(s) %s have no bench_board race' % (','.join(extra) or 'no @dataset='))
        fam, ds = FAMILY_OF_PREFIX[prefix], kv['dataset']
        if 'input' in kv:
            note = 'input=%s not reproduced: bench_board races its own full-row block' % kv['input']
    elif prefix == 'neural':
        fam, lane = 'neural', rest
        if '@' in lane or lane not in board['neural_data']:
            raise ValueError('neural lane %r is not a tools/bench_board.py neural lane' % lane)
        ds = board['neural_data'][lane]
    elif prefix in board['tree_lanes'] and '@' not in rest:
        fam, lane, drv = 'trees', prefix, rest
        if lane in board['tree_task']:
            back = {v: k for k, v in board['tree_task'][lane].items()}
            if drv not in back:
                raise ValueError('tree lane %s driver dataset %r has no board dataset (bench_board TREE_TASK_DATASETS %s)'
                                 % (lane, drv, ','.join(sorted(back))))
            ds = back[drv]
        elif drv in board['board_datasets']:
            ds = drv
        else:
            raise ValueError('tree lane %s races %s on the board, not driver dataset %r'
                             % (lane, '/'.join(board['board_datasets']), drv))
    else:
        raise ValueError('family %r has no bench_board route' % prefix)
    if (fam, lane, ds) not in board['races']:
        raise ValueError('bench_board plans no %s/%s/%s race in IDENTICAL on this vendor' % (fam, lane, ds))
    return fam, lane, ds, note


def route(wid, lanes, board):
    """-> dict(kind race|cmd, key (family, lane, dataset), lane, dataset, family, note) or raise ValueError."""
    if wid and wid.startswith('expanded:'):
        lane, ds, note = map_workload(wid, lanes)  # its reason stands: expanded lanes are bench_board_algos lanes
        return dict(kind='race', key=('algos', lane, ds), family='algos', lane=lane, dataset=ds, note=note)
    if board is None:
        raise ValueError('no bench_board registry for the CMD route')
    fam, lane, ds, note = map_board(wid or '', board)
    extra = NEURAL_RUNTIME_BINDINGS.get(lane, ()) if fam == 'neural' else ()
    return dict(kind='cmd', key=(fam, lane, ds), family=fam, lane=lane, dataset=ds, note=note, bindings=list(extra))


def cmd_line(box, vendor, branch, tag, races, envs, builds, prebuilt=None, models=None):
    toks = ['lq', 'add', box, 'CMD', branch, tag] + ['%s=%s' % kv for kv in envs] + [
        CMD_PY, CMD_SCRIPT, '--tag', tag, '--vendor', vendor]
    for fam, lane, ds in sorted(set(races)):
        toks += ['--race', '%s:%s:%s' % (fam, lane, ds)]
    pip = BBW.pip_arg(sorted(set(races)), models)
    if pip != BBW.DEFAULT_PIP:  # explicit on the line, so a frozen branch's runner (no pip_extra yet) installs it too
        toks += ['--pip', pip]
    if builds:
        toks.append('BUILDS=' + ','.join(builds))
    if prebuilt:  # box_job.sh installs the store's prebuilt bindings instead of building (falls back on exit 2)
        toks.append(PREBUILT_TOKEN + prebuilt)
    bad = [x for x in toks if x != CMD_PY and not re.fullmatch(r'[A-Za-z0-9_.,=@:+/-]+', x)]
    if bad:
        raise ValueError('lq CMD token(s) %r are not plain words' % bad)
    return ' '.join(toks)


def _pack_num(pid):
    m = re.search(r'\d+', pid or '')
    return int(m.group()) if m else 0


# ------------------------------------------------------------------ render


def spec_text(pairs):
    """[(lane, ds)] -> (lanes, datasets): the cartesian comma form when exact, else lane@ds,... PAIRS."""
    lanes = sorted({l for l, _ in pairs})
    dss = sorted({d for _, d in pairs})
    if len(lanes) * len(dss) == len(set(pairs)):
        return ','.join(lanes), ','.join(dss)
    return ','.join('%s@%s' % p for p in sorted(set(pairs))), 'PAIRS'


def lq_line(box, branch, pairs, envs, builds, prebuilt=None):
    lanes, dss = spec_text(pairs)
    toks = ['lq', 'add', box, 'RACE', branch, lanes, dss] + ['%s=%s' % kv for kv in envs]
    # The base binding (_mojolearn, script `build`) is rebuilt on every line: the Python layer refuses a
    # stale base .so ("the base binding has no all_finite_f32", first smoke 2026-10-07), and lq's build
    # step only rebuilds what BUILDS= names.
    builds = ['build'] + [b for b in (builds or []) if b != 'build']
    toks.append('BUILDS=' + ','.join(builds))
    if prebuilt:  # tools/six_lane_grid_prebuild.py store on the box; box_job.sh installs instead of building
        toks.append(PREBUILT_TOKEN + prebuilt)
    line = ' '.join(toks)
    bad = [t for t in toks if not re.fullmatch(r'[A-Za-z0-9_.,=@:+/-]+', t)]
    if bad:  # lq refuses shell metacharacters in RACE lines; box_job.sh word-splits the line
        raise ValueError('lq line token(s) %r are not plain words: %s' % (bad, line))
    return line


def plan_jobs(plan, matrix, vendor, lanes, only_lanes=None, phase=None, board=None, only_cells=None, only_workloads=None):
    """A jobs (one per pack, with the kept member cells, RACE and CMD), refused cells, per-workload B info.
    only_cells: a set of (configuration, workload_id) to keep (render --rerun-undecided); None keeps every cell."""
    packs = {p['id']: p for p in plan['builds']['packs']}
    algos = plan.get('algorithms', {})
    configs = {c['id']: c for c in matrix['configurations']}
    cells = {}
    for cell in matrix['cells']:
        if cell.get('vendor') == vendor:
            cells.setdefault(cell['configuration'], []).append(cell['workload_id'])
    refused, jobs, workload_bindings = [], {}, {}
    for cid, wids in cells.items():
        c = configs[cid]
        g = c.get('grid') or {}
        algo = g.get('algorithm')
        regime = (algos.get(algo) or {}).get('regime') or 'factorial'
        if phase and regime != phase:
            continue
        for wid in wids:
            if only_cells is not None and (cid, wid) not in only_cells:
                continue
            if not workload_selected(wid, only_workloads):
                continue
            try:
                r = route(wid, lanes, board)
            except ValueError as exc:
                refused.append(dict(configuration=cid, workload_id=wid, algorithm=algo, reason=str(exc)))
                continue
            lane, ds, note = r['lane'], r['dataset'], r['note']
            if only_lanes and lane not in only_lanes:
                continue
            job = jobs.setdefault(g['pack'], dict(pack=g['pack'], regime_rank=REGIME_RANK.get(regime, 9),
                                                  priority=c.get('priority') or 9, configs=set(), cells=[],
                                                  defines=list(packs[g['pack']]['defines']),
                                                  bindings=set(packs[g['pack']].get('bindings') or [])))
            job['regime_rank'] = min(job['regime_rank'], REGIME_RANK.get(regime, 9))
            job['priority'] = min(job['priority'], c.get('priority') or 9)
            job['configs'].add(cid)
            job['bindings'].update((algos.get(algo) or {}).get('workload_bindings') or [])
            job['bindings'].update(r.get('bindings') or [])
            job['cells'].append(dict(configuration=cid, workload_id=wid, lane=lane, dataset=ds, note=note,
                                     kind=r['kind'], family=r['family']))
            workload_bindings.setdefault(wid, dict(lane=lane, dataset=ds, algorithm=algo, kind=r['kind'],
                                                   family=r['family'], bindings=set()))['bindings'].update(
                list((algos.get(algo) or {}).get('workload_bindings') or []) + list(r.get('bindings') or []))
    for job in jobs.values():
        seen = {}
        for cell in job['cells']:
            key = (cell['family'], cell['lane'], cell['dataset'])
            if key in seen and seen[key] != (cell['configuration'], cell['workload_id']):
                raise ValueError('pack %s races %s for two cells: %s and %s' % (job['pack'], '/'.join(key), seen[key], (cell['configuration'], cell['workload_id'])))
            seen[key] = (cell['configuration'], cell['workload_id'])
    order = sorted(jobs.values(), key=lambda j: (j['regime_rank'], j['priority'], _pack_num(j['pack']), j['pack']))
    return order, refused, workload_bindings


def b_groups(workloads, info, group_size):
    """Group incumbent workloads by binding set (one build covers them), chunks of at most group_size."""
    by_bind = {}
    for wid in sorted(workloads):
        key = (info[wid]['kind'], tuple(sorted(info[wid]['bindings'])))
        by_bind.setdefault(key, []).append(wid)
    out = []
    for (kind, key), wids in sorted(by_bind.items()):
        for i in range(0, len(wids), group_size):
            out.append(dict(kind=kind, bindings=list(key), workloads=wids[i:i + group_size]))
    return out


def render(plan_dir, vendor, branch='main', b_repeats=3, only_lanes=None, phase=None, budget_hours=None,
           lanes=None, b_group_size=8, run_id=None, board=None, prebuilt=None, only_cells=None, race_seconds=None,
           only_workloads=None, models=None):
    plan, matrix = load_plan(plan_dir)
    lanes = lanes if lanes is not None else load_lanes()
    board = board if board is not None else load_board(vendor)
    run_id = run_id or run_id_of(plan_dir)
    box = BOX[vendor]
    per_race = PAIR_SECONDS[vendor] / 2.0
    jobs, refused, info = plan_jobs(plan, matrix, vendor, lanes, only_lanes, phase, board, only_cells, only_workloads)
    # budget: keep A jobs in order while A races + b_repeats x (new workloads) fit
    kept, covered, races = [], set(), 0
    cut = 0
    for job in jobs:
        new = {c['workload_id'] for c in job['cells']} - covered
        cost = len(job['cells']) + b_repeats * len(new)
        if budget_hours is not None and (races + cost) * per_race > budget_hours * 3600.0:
            cut = len(jobs) - len(kept)
            break
        kept.append(job)
        covered |= new
        races += cost
    groups = b_groups(covered, info, b_group_size) if b_repeats > 0 else []
    # spread: B group g, repeat r sits at A position floor(N * (r + (g + 0.5) / G) / R)
    slots = {}
    for r in range(b_repeats):
        for gi, grp in enumerate(groups):
            pos = int(len(kept) * (r + (gi + 0.5) / len(groups)) / b_repeats)
            slots.setdefault(pos, []).append((gi, r, grp))
    lines, manifest = [], []

    def emit(arm, tag, kind, cells, envs, bindings, **extra):
        if kind == 'race':
            line = lq_line(box, branch, [(c['lane'], c['dataset']) for c in cells], envs, binding_builds(bindings), prebuilt)
        else:  # bench_board needs the base binding beside the ones the configuration reaches
            line = cmd_line(box, vendor, branch, tag, [(c['family'], c['lane'], c['dataset']) for c in cells], envs,
                            binding_builds(['_mojolearn'] + sorted(set(bindings) - {'_mojolearn'})), prebuilt, models)
        lines.append(line)
        manifest.append(dict(extra, line=len(lines), tag=tag, arm=arm, kind=kind, cells=cells))

    def emit_b(gi, r, grp):
        tag = '%s.%s%03dr%d' % (run_id, 'B' if grp['kind'] == 'race' else 'C', gi + 1, r + 1)
        cells = [dict(workload_id=w, lane=info[w]['lane'], dataset=info[w]['dataset'], family=info[w]['family'])
                 for w in grp['workloads']]
        emit('B', tag, grp['kind'], cells, [(TAG_ENV, tag)], grp['bindings'], repeat=r + 1, workloads=grp['workloads'])

    for i, job in enumerate(kept):
        for gi, r, grp in slots.get(i, []):
            emit_b(gi, r, grp)
        regime = [k for k, v in REGIME_RANK.items() if v == job['regime_rank']][0]
        for kind in ('race', 'cmd'):
            cells = [c for c in job['cells'] if c['kind'] == kind]
            if not cells:
                continue
            tag = '%s.%s%s' % (run_id, job['pack'], '' if kind == 'race' else '.bb')
            envs = [(TAG_ENV, tag), (DEFINES_ENV, ','.join(job['defines']))] + define_envs(job['defines'])
            emit('A', tag, kind, cells, envs, sorted(job['bindings']),
                 pack=job['pack'], defines=job['defines'], regime=regime,
                 configurations=sorted({c['configuration'] for c in cells}))
    for pos in sorted(p for p in slots if p >= len(kept)):
        for gi, r, grp in slots[pos]:
            emit_b(gi, r, grp)
    A = [m for m in manifest if m['arm'] == 'A']
    Bm = [m for m in manifest if m['arm'] == 'B']
    cfgs = {c for m in A for c in m['configurations']}
    regimes, by_kind = {}, {}
    for m in A:
        r = regimes.setdefault(m['regime'], dict(lines=0, cells=0))
        r['lines'] += 1
        r['cells'] += len(m['cells'])
    for m in manifest:
        k = by_kind.setdefault('%s_%s' % (m['arm'], m['kind']), dict(lines=0, races=0))
        k['lines'] += 1
        k['races'] += len(m['cells'])
    by_family = {}
    for m in A:
        for c in m['cells']:
            by_family[c['family']] = by_family.get(c['family'], 0) + 1
    a_races = sum(len(m['cells']) for m in A)
    b_races = sum(len(m['cells']) for m in Bm)
    noted = sorted({c['workload_id'] for m in A for c in m['cells'] if c['note']})
    totals = dict(vendor=vendor, box=box, branch=branch, run_id=run_id, lines=len(lines), builds=len(lines), prebuilt=prebuilt,
                  a_lines=len(A), b_lines=len(Bm), b_repeats=b_repeats, b_groups=len(groups),
                  a_races=a_races, b_races=b_races, by_kind=by_kind, a_cells_by_family=by_family,
                  configurations=len(cfgs), workloads=len(covered), regimes=regimes,
                  plan_cells=sum(1 for c in matrix['cells'] if c.get('vendor') == vendor),
                  refused_cells=len(refused), refused_workloads=len({r['workload_id'] for r in refused}),
                  cut_a_lines_by_budget=cut, budget_hours=budget_hours, only_workloads=only_workloads or None,
                  projected_race_hours=round((a_races + b_races) * per_race / 3600.0, 2),
                  projected_note='races x pair_seconds/2 (%g s, the plan median pair); binding build time per line '
                                 'is not included, and tree and neural races run longer than the median' % per_race,
                  input_variant_workloads=noted)
    if race_seconds is not None:
        # First-pass measured medians (render --rerun-undecided --first-pass DIR): each A cell at its own first-pass
        # candidate median, each B race at the workload's first-pass incumbent median, the plan default where the first
        # pass has no number. These are the timed medians the racer reports (one run per arm), not the job wall time
        # (data load, warm-up and the prebuilt install are not in them), so the figure is a lower bound.
        sec, hit = 0.0, 0
        for m in manifest:
            for c in m['cells']:
                key = (m['arm'], c.get('configuration'), c['workload_id'])
                v = race_seconds.get(key) if m['arm'] == 'A' else race_seconds.get(('B', None, c['workload_id']))
                hit += v is not None
                sec += v if v is not None else per_race
        totals.update(measured_race_hours=round(sec / 3600.0, 3), measured_races=hit,
                      measured_note='first-pass timed medians where measured (%d of %d races), plan default %g s '
                                    'elsewhere; timed medians only, so a lower bound on the box time' % (hit, a_races + b_races, per_race))
    return lines, dict(schema='mojolearn.six-lane-grid-lq-render/2', plan_dir=str(plan_dir), totals=totals,
                       lines=manifest, refused=refused,
                       refused_by_reason=_by_reason(refused))


# ------------------------------------------------------------------ render --rerun-undecided (second pass)

# A control whose roll-up is final has been acted on (flipped on, deleted, held for identity or quality): its
# configurations are not re-run. HOLD_BROKEN is not final: the arm awaits a fix, and its configurations that ran without
# a failure are still undecided (their own BROKEN cells are skipped cell by cell); --skip-broken-controls skips them too.
FINAL_CONTROL = ('PROMOTE', 'SPLIT', 'DELETE', 'HOLD_IDENTITY', 'HOLD_QUALITY')
DECIDED_TIMING = ('FASTER', 'SLOWER')  # a cell beyond the floor on both voting vendors (timeouts count as SLOWER)
UNDECIDED_TIMING = ('NO_VERDICT', 'INCOMPLETE')
RERUN_SUFFIX = 'r2'


def undecided_cells(decisions, skip_controls=(), skip_broken_controls=False):
    """grid-decisions.json -> (set of (configuration, workload_id) to re-run, report).

    A decision row's `timing` is the case verdict over both voting vendors: FASTER/SLOWER only when the cell is
    beyond its floor on both and they agree, so NO_VERDICT/INCOMPLETE means the cell is undecided on at least one
    vendor. A configuration is re-run when every measured cell is undecided; it is skipped when any cell is decided
    (FASTER/SLOWER, BROKEN = the candidate race failed, quality WORSE, identity MISMATCH) or when its assignment
    touches a control that is already final (flipped, deleted or held) or named in skip_controls. Only the
    configuration's NO_VERDICT/INCOMPLETE cells are re-run: an UNMEASURED cell is first-pass work still owed (or a
    refused route), not a second data point."""
    final = FINAL_CONTROL + (('HOLD_BROKEN',) if skip_broken_controls else ())
    flipped = {k for k, c in (decisions.get('controls') or {}).items() if c.get('recommendation') in final}
    flipped |= set(skip_controls or ())
    keep, skipped, unmeasured_cells = set(), {}, 0
    selected = []
    for cid, ev in sorted((decisions.get('configurations') or {}).items()):
        rows = ev.get('rows') or []
        assign = ev.get('assignment') or {}
        if any(k in flipped for k in assign):
            why = 'flipped_switch'
        elif any(r.get('timing') in DECIDED_TIMING for r in rows):
            why = 'decided_cell'
        elif any(r.get('quality') == 'FAIL' for r in rows):
            why = 'broken_cell'
        elif any(str(r.get('quality')).upper() in ('WORSE', 'REGRESSED') for r in rows):
            why = 'quality_worse'
        elif any(r.get('identity') == 'MISMATCH' for r in rows):
            why = 'identity_mismatch'
        elif not any(r.get('timing') in UNDECIDED_TIMING for r in rows):
            why = 'unmeasured'
        else:
            why = None
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        cells = [r['workload_id'] for r in rows if r.get('timing') in UNDECIDED_TIMING]
        unmeasured_cells += len(rows) - len(cells)
        keep.update((cid, w) for w in cells)
        selected.append(cid)
    report = dict(configurations=len(selected), cells=len(keep), skipped_configurations=dict(sorted(skipped.items())),
                  unmeasured_cells_not_rerun=unmeasured_cells, final_controls=sorted(flipped),
                  rule='re-run the NO_VERDICT/INCOMPLETE cells of configurations with no decided cell (FASTER/SLOWER, '
                       'BROKEN, WORSE, MISMATCH) and no final control (%s) in their assignment' % '/'.join(final))
    return keep, sorted(selected), report


def first_pass_seconds(collect_dir, vendor):
    """First-pass collect output -> {('A', cid, wid): s, ('B', None, wid): s} timed medians on this vendor (or None)."""
    path = Path(collect_dir) / 'grid-verdicts.json'
    if not path.exists():
        return None
    out, b = {}, {}
    for case in json.loads(path.read_text()).get('cases', []):
        row = (case.get('vendors') or {}).get(vendor) or {}
        if row.get('a_ms'):
            out[('A', case['configuration'], case['workload_id'])] = row['a_ms'] / 1000.0
        if row.get('b_ms'):
            b.setdefault(case['workload_id'], []).append(row['b_ms'] / 1000.0)
    for wid, xs in b.items():
        out[('B', None, wid)] = statistics.median(xs)
    return out


def render_rerun(plan_dir, vendor, decide_dir, branch='main', b_repeats=1, rerun_id=None, first_pass=None,
                 skip_controls=(), skip_broken_controls=False, **kw):
    """Second pass over the undecided configurations: render() restricted to their undecided cells, under a new run
    id (same pack ids in the tags), with b_repeats incumbent lines per workload group they touch."""
    dec_path = Path(decide_dir)
    dec_path = dec_path / 'grid-decisions.json' if dec_path.is_dir() else dec_path
    decisions = json.loads(dec_path.read_text())
    cells, selected, sel = undecided_cells(decisions, skip_controls, skip_broken_controls)
    rerun_id = rerun_id or (kw.pop('run_id', None) or run_id_of(plan_dir)) + RERUN_SUFFIX
    kw.pop('run_id', None)
    if '.' in rerun_id or not re.fullmatch(r'[A-Za-z0-9_-]+', rerun_id):
        raise ValueError('rerun id %r must be a plain word without a dot (tags are <run>.<pack>)' % rerun_id)
    secs = first_pass_seconds(first_pass, vendor) if first_pass else None
    lines, manifest = render(plan_dir, vendor, branch, b_repeats, run_id=rerun_id, only_cells=cells, race_seconds=secs, **kw)
    t = manifest['totals']
    t.update(rerun=dict(sel, decisions=str(dec_path), selected=selected, first_pass=str(first_pass) if first_pass else None,
                        rendered_configurations=t['configurations']))
    if secs is None:
        t['measured_note'] = ('no first-pass collect output (--first-pass DIR with grid-verdicts.json): projected_race_hours '
                              'uses the plan default per race')
    manifest['schema'] = 'mojolearn.six-lane-grid-lq-render/2'
    return lines, manifest


# ------------------------------------------------------------------ redo-not-ready (infrastructure failures)
# status=not_ready is INFRASTRUCTURE: the racer refused before the race (ALGOS-REFUSED stage=ready, e.g. the binding's
# ImportError when the box job ran without LD_LIBRARY_PATH=<tree>/.pixi/envs/default/lib, Oct 7 20:10Z). It says nothing
# about the switch: the collector drops such samples (never FAIL, never BROKEN), and `redo-not-ready` writes the RACE
# lines again under a redo tag.
#
# Tag rule: <run>.<base>n<k>, k = 1, 2, ... (base = the original pack id P875 or incumbent tag B022r1/C004r1), e.g.
# ge123e6f9.P875n1, ge123e6f9.B022r1n1. The collector maps <base>n<k> to the same configuration (or incumbent repeat)
# as <base>. Per cell, vendor and pass, the earliest rep (original = 0) with any non-not_ready sample is used: a later
# rep supersedes only a not_ready or absent cell, never an ok (or error) one.
INFRA_STATUS = ('not_ready',)
REDO_TAG_RE = re.compile(r'^(?P<base>.*\d)n(?P<rep>\d+)$')
LINE_TAG_RE = re.compile(r'(?<=\s)' + TAG_ENV + r'=(?P<run>[A-Za-z0-9_-]+)\.(?P<rest>[A-Za-z0-9_.-]+)(?=\s|$)')
INCUMBENT_TAG_RE = re.compile(r'^[BC]\d+r\d+$')


def is_infra(o):
    return str(o.get('status') or '') in INFRA_STATUS


def split_redo(rest):
    """Tag tail after '<run>.' -> (base, rep): 'P875n1' -> ('P875', 1); 'B022r1n2' -> ('B022r1', 2); 'P875' -> ('P875', 0)."""
    m = REDO_TAG_RE.match(rest or '')
    if m:
        return m.group('base'), int(m.group('rep'))
    return rest, 0


def settle_infra(groups):
    """groups {key: [obs with redo_rep, pass, tag_base]} -> mutate in place: drop not_ready samples, keep per (pass,
    tag_base) only the earliest rep that has a sample left, delete keys left empty. -> (infra_keys, dropped, superseded):
    infra_keys = keys that had samples but none past the infra filter."""
    infra_keys, dropped, superseded = [], 0, 0
    for key in list(groups):
        samples = groups[key]
        real = [o for o in samples if not is_infra(o)]
        dropped += len(samples) - len(real)
        first = {}
        for o in real:
            g = (o.get('pass'), o.get('tag_base'))
            first[g] = min(first.get(g, o.get('redo_rep', 0)), o.get('redo_rep', 0))
        keep = [o for o in real if o.get('redo_rep', 0) == first[(o.get('pass'), o.get('tag_base'))]]
        superseded += len(real) - len(keep)
        if keep:
            groups[key] = keep
        else:
            del groups[key]
            infra_keys.append(key)
    return infra_keys, dropped, superseded


def not_ready_cells(results, vendor, run_id):
    """results.txt copies -> {base: {(lane, dataset): set(statuses)}} over every rep of run_id on vendor."""
    jobs, obs, _ = parse_results(results)
    cells = {}
    for o in obs:
        if o['vendor'] != vendor or o['arm'] != 'ours':
            continue
        tag = (jobs.get((o['vendor'], o['id'])) or {}).get('tag') or ''
        rid, _, rest = tag.partition('.')
        if rid != run_id or rest.endswith('.bb'):
            continue
        base, _ = split_redo(rest)
        cells.setdefault(base, {}).setdefault((o['lane'], o['dataset']), set()).add(str(o['status']))
    return cells


def render_not_ready_redo(lines_paths, results, vendor, run_id, rep=1):
    """RACE lines of run_id whose job left a lane x dataset with only not_ready samples (over every rep so far) ->
    (the same lines with tag <base>n<rep>, manifest). The line is copied token for token (same branch, lanes,
    datasets, defines, BUILDS and PREBUILT): only the tag changes."""
    if rep < 1:
        raise ValueError('redo rep must be 1 or more')
    box = BOX[vendor]
    cells = not_ready_cells(results, vendor, run_id)
    need = {}
    for base, per in cells.items():
        bad = sorted(k for k, st in per.items() if st and all(x in INFRA_STATUS for x in st))
        if bad:
            need[base] = bad
    out, seen, by_lane = [], set(), {}
    skipped = dict(other_box=0, other_run=0, not_race=0, duplicate=0, no_not_ready=0)
    for path in lines_paths:
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            toks = line.split()
            if len(toks) < 4 or toks[:2] != ['lq', 'add']:
                continue
            if toks[2] != box:
                skipped['other_box'] += 1
                continue
            if toks[3] != 'RACE':
                skipped['not_race'] += 1
                continue
            m = LINE_TAG_RE.search(line)
            if not m or m.group('run') != run_id:
                skipped['other_run'] += 1
                continue
            base, _ = split_redo(m.group('rest'))
            if base in seen:
                skipped['duplicate'] += 1
                continue
            seen.add(base)
            if base not in need:
                skipped['no_not_ready'] += 1
                continue
            new = '%s=%s.%sn%d' % (TAG_ENV, run_id, base, rep)
            out.append(line[:m.start()] + new + line[m.end():])
            for lane, ds in need[base]:
                by_lane[lane] = by_lane.get(lane, 0) + 1
    done = [b for b in need if b in seen]
    manifest = dict(schema='mojolearn.six-lane-grid-lq-redo/1', vendor=vendor, box=box, run_id=run_id, rep=rep,
                    lines=len(out), a_lines=sum(1 for b in done if not INCUMBENT_TAG_RE.match(b)),
                    b_lines=sum(1 for b in done if INCUMBENT_TAG_RE.match(b)),
                    not_ready_cells=sum(len(need[b]) for b in done), not_ready_cells_by_lane=dict(sorted(by_lane.items())),
                    lanes=sorted(by_lane), tags_without_line=sorted(set(need) - seen), skipped=skipped,
                    lines_files=[str(x) for x in lines_paths], results=[str(x) for x in results],
                    rule='tag <run>.<base>n<rep>; collect maps it to <base> and uses it only where every earlier rep of '
                         'the cell was not_ready or absent')
    return out, manifest


def add_redo_parser(s):
    d = s.add_parser('redo-not-ready', help='RACE lines whose job came back status=not_ready -> the same lines, redo tag')
    d.add_argument('--vendor', choices=sorted(BOX), required=True)
    d.add_argument('--lines', nargs='+', type=Path, required=True,
                   help='rendered RACE lines files of the run (e.g. <vendor>.race.pb.lines); the first line per tag is used')
    d.add_argument('--results', nargs='+', type=Path, required=True, help="the box's results.txt copy (with any earlier redo results)")
    d.add_argument('--run-id', required=True, help='the run whose tags are redone (e.g. ge123e6f9)')
    d.add_argument('--rep', type=int, default=1, help='redo rep k in the tag <base>n<k> (default 1; 2 for a redo of a redo)')
    d.add_argument('--out', type=Path, required=True, help='lines file; <out>.json gets the manifest')
    return d


def redo_main(args):
    lines, manifest = render_not_ready_redo(args.lines, args.results, args.vendor, args.run_id, args.rep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(line + '\n' for line in lines))
    Path(str(args.out) + '.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: manifest[k] for k in ('vendor', 'run_id', 'rep', 'lines', 'a_lines', 'b_lines', 'not_ready_cells',
                                                'not_ready_cells_by_lane', 'tags_without_line')}))
    return 0


def _by_reason(refused):
    out = {}
    for r in refused:
        key = re.sub(r"'[^']*'", "'...'", r['reason'])
        e = out.setdefault(key, dict(cells=0, workloads=set()))
        e['cells'] += 1
        e['workloads'].add(r['workload_id'])
    return {k: dict(cells=v['cells'], workloads=sorted(v['workloads'])) for k, v in sorted(out.items())}


# ------------------------------------------------------------------ collect: parse

RESULT_RE = re.compile(r'^(?P<id>[A-Za-z]\d+) (?P<vendor>nvidia|amd) (?P<branch>\S+?)@(?P<head>[0-9a-f]+)(?P<rest>.*)$')
ALGOS_RE = re.compile(r'(?:^|\s)ALGOS lane=(?P<lane>\S+) dataset=(?P<ds>\S+) arm=(?P<arm>\S+) status=(?P<status>\S+) '
                      r'median_ms=(?P<ms>\S+)(?: quality=(?P<q>.*))?$')
TAG_RE = re.compile(TAG_ENV + r'=([A-Za-z0-9_.-]+)')
DIGEST_TAIL_RE = re.compile(r' digest=([0-9a-f]+|none)$')
GRIDBB_RE = re.compile(r'^GRIDBB tag=(?P<tag>\S+) vendor=(?P<vendor>\S+) head=(?P<head>\S+) family=(?P<family>\S+) '
                       r'lane=(?P<lane>\S+) dataset=(?P<ds>\S+) status=(?P<status>\S+) median_ms=(?P<ms>\S+) '
                       r'hash=(?P<hash>\S+) quality=(?P<q>.*)$')
ROUND_RE = re.compile(r'^ALGOS-ROUND lane=(?P<lane>\S+) dataset=(?P<ds>\S+) arm=(?P<arm>\S+) .*digest=(?P<digest>[0-9a-f]+)')


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_algos(text):
    """ALGOS payload -> dict(lane, dataset, arm, status, median_ms, quality|None, digest|None, truncated)."""
    m = ALGOS_RE.search(text)
    if not m:
        return None
    q, digest, truncated, tail = m.group('q'), None, False, False
    if q is not None:
        d = DIGEST_TAIL_RE.search(q)
        if d:
            tail = True
            digest = None if d.group(1) == 'none' else d.group(1)[:DIGEST_CHARS]
            q = q[:d.start()]
        try:
            q = json.loads(q)
        except ValueError:
            q, truncated = None, True
    return dict(lane=m.group('lane'), dataset=m.group('ds'), arm=m.group('arm'), status=m.group('status'),
                median_ms=_num(m.group('ms')), quality=q, digest=digest, truncated=truncated or not tail)


def parse_results(paths):
    """results.txt lines -> (jobs {(vendor, id): {tag, head, branch}}, observations [..], no_result [..])."""
    jobs, obs, nores = {}, [], []
    for path in paths:
        for raw in Path(path).read_text(errors='replace').splitlines():
            m = RESULT_RE.match(raw.strip())
            if not m:
                continue
            key = (m.group('vendor'), m.group('id'))
            rest = m.group('rest')
            tag = TAG_RE.search(rest)
            job = jobs.setdefault(key, dict(tag=None, head=m.group('head'), branch=m.group('branch')))
            if tag:
                job['tag'] = tag.group(1)
            cmd = re.match(r' CMD (\S+) rc=(-?\d+)', rest)
            if cmd:
                job.update(cmd_tag=cmd.group(1), cmd_rc=int(cmd.group(2)))
                continue
            if ' NO-RESULT' in rest:
                nores.append(dict(vendor=key[0], id=key[1], line=raw[:300], evidence=str(path)))
                continue
            a = parse_algos(rest)
            if a:
                obs.append(dict(a, vendor=key[0], id=key[1], head=m.group('head'), evidence=str(path)))
    return jobs, obs, nores


def _log_id(path):
    """/root/lq/out/<id>/race-*.log -> <id>."""
    return Path(path).parent.name


def parse_gridbb(line, evidence, jid):
    m = GRIDBB_RE.match(line.strip())
    if not m:
        return None
    try:
        q = json.loads(m.group('q'))
    except ValueError:
        q = None
    h = m.group('hash')
    return dict(kind='cmd', key=(m.group('family'), m.group('lane'), m.group('ds')), family=m.group('family'),
                lane=m.group('lane'), dataset=m.group('ds'), arm='ours', status=m.group('status'),
                median_ms=_num(m.group('ms')), quality=q, digest=None if h in ('none', 'None', '') else h[:DIGEST_CHARS],
                vendor=m.group('vendor'), head=m.group('head'), tag=m.group('tag'), id=jid, evidence=evidence,
                truncated=q is None)


def parse_logs(paths, gridbb=None):
    """Race logs and CMD logs (directories walked for *.log under <id>/, or `grep -H` dumps of them) ->
    {(id, lane, dataset): dict(algos=<last ALGOS arm ours>, digest=<last round digest>)}; GRIDBB lines
    (tools/six_lane_grid_bb.py) are appended to `gridbb`, the last per (job, family, lane, dataset)."""
    per_file = {}
    bb = {}

    def feed(file_key, line):
        line = line.rstrip('\n')
        if line.startswith('GRIDBB '):
            o = parse_gridbb(line, file_key, _log_id(file_key))
            if o:
                bb[(o['id'], o['vendor']) + o['key']] = o
            return
        rec = per_file.setdefault(file_key, dict(algos=None, digest=None, lane=None, ds=None))
        r = ROUND_RE.match(line)
        if r:
            if r.group('arm') == 'ours':
                rec['digest'] = r.group('digest')[:DIGEST_CHARS]
                rec['lane'], rec['ds'] = r.group('lane'), r.group('ds')
            return
        if line.startswith('ALGOS '):
            a = parse_algos(line)
            if a and a['arm'] == 'ours':
                rec['algos'] = a
                rec['lane'], rec['ds'] = a['lane'], a['dataset']
            return
        d = re.search(r'digest=([0-9a-f]+)', line)
        if d:
            rec['digest'] = d.group(1)[:DIGEST_CHARS]

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.glob('**/*.log')):
                for line in f.read_text(errors='replace').splitlines():
                    feed(str(f), line)
        else:
            for line in p.read_text(errors='replace').splitlines():
                if ':' not in line:
                    continue
                fname, _, rest = line.partition(':')
                if not fname.endswith('.log'):
                    continue
                feed(fname, rest)
    out = {}
    for fname, rec in per_file.items():
        if rec['lane'] is None:
            continue
        out[(_log_id(fname), rec['lane'], rec['ds'])] = dict(rec, evidence=fname)
    if gridbb is not None:
        gridbb.extend(bb.values())
    return out


def merge_observations(jobs, obs, logs):
    """Fill digests/quality that lq's 600-character cut dropped from the race logs; add log-only rows."""
    seen = set()
    for o in obs:
        key = (o['id'], o['lane'], o['dataset'])
        seen.add((o['vendor'],) + key)
        rec = logs.get(key)
        if rec:
            a = rec.get('algos') or {}
            if o['digest'] is None and rec.get('digest'):
                o['digest'] = rec['digest']
            if o['quality'] is None and a.get('quality') is not None:
                o['quality'] = a['quality']
            o['truncated'] = o['quality'] is None or (o['digest'] is None and o['status'] == 'ok')
            o['log_evidence'] = rec['evidence']
    for (jid, lane, ds), rec in logs.items():
        vendors = [v for (v, i) in jobs if i == jid]
        if len(vendors) != 1 or (vendors[0], jid, lane, ds) in seen or not rec.get('algos'):
            continue
        a = rec['algos']
        job = jobs[(vendors[0], jid)]
        obs.append(dict(a, digest=a.get('digest') or rec.get('digest'), vendor=vendors[0], id=jid, head=job['head'],
                        evidence=rec['evidence'], log_evidence=rec['evidence']))
    for o in obs:
        o['tag'] = (jobs.get((o['vendor'], o['id'])) or {}).get('tag')
    return obs


# ------------------------------------------------------------------ collect: judge

FLOOR_MIN = math.log(1.05)  # the smallest credible same-build spread: Oct 6 receipts put the median at 1.07x, p90 1.25x
TIMEOUT_RATIO = 4.0  # a candidate race the racer refused for timeout counts as at least this many times the incumbent
LOWER = re.compile(r'rmse|logloss|log_loss|inertia|error|residual|distortion|perplexity|stress|diff|shift|l1_vs')
HIGHER = re.compile(r'^(accuracy|auc|roc_auc|r2|mean_r2|silhouette|modularity|explained_variance_fraction|'
                    r'mean_canonical_corr|mean_log_likelihood|mean_llf|precision|f1)$|trustworthiness|recall|jaccard|'
                    r'subspace_cos|count_agreement')


def direction(metric):
    if HIGHER.search(metric):
        return 'higher'
    if LOWER.search(metric):
        return 'lower'
    return None


def quality_verdict(qa, qb):
    """Board quality json of A vs B -> (SAME|WORSE|BETTER|PENDING, per-metric detail, unjudged keys)."""
    if not isinstance(qa, dict) or not isinstance(qb, dict):
        return 'PENDING', {}, []
    detail, unjudged, worse, better = {}, [], False, False
    for k in sorted(set(qa) | set(qb)):
        a, b = qa.get(k), qb.get(k)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)):
            if k in qa and k in qb and a != b:
                unjudged.append(k)
            continue
        d = direction(k)
        if d is None:
            if a != b:
                unjudged.append(k)
            continue
        rel = (a - b) / max(abs(b), 1e-12)
        bad = rel > QUALITY_REL if d == 'lower' else rel < -QUALITY_REL
        good = rel < -QUALITY_REL if d == 'lower' else rel > QUALITY_REL
        v = 'WORSE' if bad else 'BETTER' if good else 'SAME'
        worse |= bad
        better |= good
        detail[k] = dict(a=a, b=b, rel=rel, direction=d, verdict=v)
    if not detail:
        return 'PENDING', detail, unjudged
    return ('WORSE' if worse else 'BETTER' if better else 'SAME'), detail, unjudged


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_ids_of(run_id, plan_dir):
    """--run-id value(s) -> ordered list of passes (a str, a comma list or a list; default the plan hash)."""
    if not run_id:
        return [run_id_of(plan_dir)]
    items = [run_id] if isinstance(run_id, str) else list(run_id)
    out = []
    for item in items:
        for r in str(item).split(','):
            if r and r not in out:
                out.append(r)
    return out


def merge_quality(verdicts):
    """Per-pass quality verdicts of one cell -> one: WORSE in any pass wins, then FAIL, TIMEOUT, SAME, BETTER, PENDING."""
    for v in ('WORSE', 'FAIL', 'TIMEOUT', 'SAME', 'BETTER'):
        if v in verdicts:
            return v
    return 'PENDING'


def collect(results, logs, plan_dir, out_dir, lanes=None, run_id=None, min_samples=2, board=None, repair_runs=()):
    """run_id: one run or several passes (list or comma list). Each pass' A is judged against the same pass' B
    median; a cell's log ratio is the mean over its passes; floors pool the incumbent repeats of every pass.
    repair_runs: passes that re-ran cells whose earlier race failed for a tooling reason (render --only-workloads, e.g.
    the neural redo of ge123e6f9: missing torch / binding / profile env on the box). Appended to the passes when absent.
    Where a repair pass measured a cell (its candidate race ok), the earlier passes' FAIL verdicts for that cell are
    superseded: kept in pass_verdicts and listed as superseded_fail_passes, not merged (FAIL would otherwise win).
    A FAIL in the repair pass itself, and any WORSE, still stand."""
    plan, matrix = load_plan(plan_dir)
    lanes = lanes if lanes is not None else load_lanes()
    board = board if board is not None else load_board('nvidia')
    run_ids = run_ids_of(run_id, plan_dir)
    repair_runs = run_ids_of(list(repair_runs), plan_dir) if repair_runs else []
    run_ids += [r for r in repair_runs if r not in run_ids]
    run_id = run_ids[0] if len(run_ids) == 1 else ','.join(run_ids)
    packs = {p['id']: p for p in plan['builds']['packs']}
    configs = {c['id']: c for c in matrix['configurations']}
    jobs, obs, nores = parse_results(results)
    bb_obs = []
    obs = merge_observations(jobs, obs, parse_logs(logs or [], bb_obs))
    for o in obs:
        o.setdefault('kind', 'race')
        o['key'] = ('algos', o['lane'], o['dataset'])
    obs += bb_obs
    key_of = {}

    def wkey(wid):
        if wid not in key_of:
            try:
                key_of[wid] = route(wid, lanes, board)['key']
            except ValueError:
                key_of[wid] = None
        return key_of[wid]

    # (family, lane, dataset) -> workload ids (for the incumbent)
    lane_to_wids = {}
    for c in configs.values():
        for wid in c.get('workloads') or []:
            if wkey(wid):
                lane_to_wids.setdefault(wkey(wid), set()).add(wid)
    A, B, ignored = {}, {}, dict(other_campaign=0, untagged=0, not_ours=0, unknown_cell=0)
    for o in obs:
        tag = o.get('tag')
        if o['arm'] != 'ours':
            ignored['not_ours'] += 1
            continue
        if not tag:
            ignored['untagged'] += 1
            continue
        rid, _, rest = tag.partition('.')
        if rid not in run_ids:
            ignored['other_campaign'] += 1
            continue
        o['pass'] = rid
        rest, o['redo_rep'] = split_redo(rest)  # <base>n<k> redo tags map to <base> (redo-not-ready)
        o['tag_base'] = rest
        if re.match(r'^[BC]\d+r\d+$', rest):
            for wid in lane_to_wids.get(o['key'], ()):
                B.setdefault((wid, o['vendor']), []).append(o)
            continue
        pack = packs.get(rest[:-3] if rest.endswith('.bb') else rest)
        hit = None
        for cid in (pack or {}).get('members', []):
            for wid in configs[cid].get('workloads') or []:
                if wkey(wid) == o['key']:
                    hit = (cid, wid)
        if hit is None:
            ignored['unknown_cell'] += 1
            continue
        A.setdefault(hit + (o['vendor'],), []).append(o)

    # not_ready = infrastructure (redo-not-ready): dropped before judging, the earliest real rep wins per cell
    a_infra, a_dropped, a_superseded = settle_infra(A)
    b_infra, b_dropped, b_superseded = settle_infra(B)
    ignored.update(not_ready=a_dropped + b_dropped, superseded_redo=a_superseded + b_superseded)
    a_infra_set = set(a_infra)

    def ok(o):
        return o['status'] == 'ok' and o['median_ms'] is not None and o['median_ms'] > 0

    # floors (incumbent repeats, same build per head) -> six_lane_timing floors schema
    floors = {}
    for (wid, vendor), samples in sorted(B.items()):
        by_head = {}
        for s in samples:
            if ok(s):
                by_head.setdefault(s['head'], []).append(s)
        if not by_head:
            continue
        head, best = max(by_head.items(), key=lambda kv: len(kv[1]))
        ms = [s['median_ms'] / 1000.0 for s in best]
        f = T._spread(ms) if len(ms) >= min_samples else None
        if f is not None:
            f = max(f, FLOOR_MIN)  # two or three repeats that happen to agree are not a 0.1% floor
        floors[wid + '|' + vendor] = dict(workload_id=wid, vendor=vendor, samples=len(ms), builds_seen=len(by_head),
                                          b_head=head, timing_source=TIMING_SOURCE,
                                          passes=sorted({s['pass'] for s in best}, key=run_ids.index),
                                          evidence=sorted({'%s:%s' % (s['evidence'], s['id']) for s in best}),
                                          floor={p: (f if p == 'scored' else None) for p in T.PHASES} if f is not None else None)
    # Provisional floor for workloads with fewer than min_samples incumbent repeats: the median of the measured floors
    # of this run (5+ of them), else log(1.25) = the Oct 6 same-build p90 spread. Marked floor_source=provisional so
    # the decision stays PARTIAL until the real floor replaces it (six_lane_grid_decide gates flips on completeness).
    measured_floors = sorted(v['floor']['scored'] for v in floors.values() if v['floor'] and v['floor'].get('scored') is not None)
    provisional = measured_floors[len(measured_floors) // 2] if len(measured_floors) >= 5 else math.log(1.25)
    for k, v in floors.items():
        if v['floor'] is None:
            v['floor'] = {p: (provisional if p == 'scored' else None) for p in T.PHASES}
            v['floor_source'] = 'provisional (%s)' % ('run median of %d floors' % len(measured_floors) if len(measured_floors) >= 5 else 'log 1.25 default')
        else:
            v['floor_source'] = 'incumbent repeats'
    floor_doc = dict(schema='mojolearn.six-lane-aa-floors/1',
                     floors={k: v for k, v in floors.items() if v['floor'] is not None},
                     rejected=[dict(key=k, samples=v['samples'], reason='fewer than %d incumbent repeats: provisional floor used' % min_samples)
                               for k, v in floors.items() if v.get('floor_source', '').startswith('provisional')],
                     provisional_floor=provisional,
                     policy='Floor from the incumbent arm B repeated through the lq grid file (same head, repeats of '
                            'every pass pooled): log(p90/p10) of median_ms with 4+ samples, log(max/min) below that '
                            '(six_lane_timing._spread).', runs=run_ids)

    def b_ref(wid, vendor, head, run=None):
        """Incumbent samples for one A head: the same pass when run is given (else every pass), same head preferred.
        -> (samples, head_differs, from_other_pass)."""
        samples = [s for s in B.get((wid, vendor), []) if ok(s)]
        other = False
        if run is not None:
            mine = [s for s in samples if s['pass'] == run]
            other = bool(samples) and not mine  # the pass has no incumbent of its own: fall back to the other passes
            samples = mine or samples
        same = [s for s in samples if s['head'] == head]
        use = same or samples
        return use, bool(samples) and not same, other

    cases, quality_rows, identity_cases = {}, [], {}
    for (cid, wid, vendor), samples in sorted(A.items()):
        c = configs[cid]
        case = cases.setdefault((cid, wid), dict(configuration=cid, workload_id=wid, mode=c.get('mode', 'identical'), vendors={}))
        per_pass = []
        for rid in run_ids:
            mine = [s for s in samples if s['pass'] == rid]
            if not mine:
                continue
            good = [s for s in mine if ok(s)]
            a = good[-1] if good else mine[-1]
            bs, head_differs, other = b_ref(wid, vendor, a['head'], rid)
            timed_out = not good and any('timeout' in str(s.get('status') or '').lower() for s in mine)
            pp = dict(run=rid, a=a, good=good, bs=bs, head_differs=head_differs, b_other_pass=other, timed_out=False,
                      r=None, a_ms=None, b_ms=_median([s['median_ms'] for s in bs]) if bs else None)
            if timed_out and bs:
                # The candidate did not finish inside the racer's cap while the incumbent did: that is a loss, not a
                # hole. Recorded as TIMEOUT_RATIO x the incumbent (beyond any floor) with timed_out=True, so the
                # verdict is SLOWER on this vendor; the true ratio is unknown and at least this large.
                pp.update(timed_out=True, r=math.log(TIMEOUT_RATIO))
            elif good and bs:
                pp['a_ms'] = _median([s['median_ms'] for s in good])
                pp['r'] = math.log(pp['a_ms'] / pp['b_ms'])
            # quality: A vs the incumbent sample of the same head (first in queue order) of the same pass
            if timed_out and bs:
                pp['quality'] = ('TIMEOUT', {}, [])  # judged by timing (SLOWER), not by quality
            elif not good:
                pp['quality'] = ('FAIL', {}, [])
            elif not bs:
                pp['quality'] = ('PENDING', {}, [])
            else:
                pp['quality'] = quality_verdict(a.get('quality'), bs[0].get('quality'))
            per_pass.append(pp)
        last = per_pass[-1]
        a = last['a']
        timed = [pp for pp in per_pass if pp['r'] is not None]
        if timed:
            # The cell's log ratio is the mean of the per-pass log ratios (each pass' A over the SAME pass' B median).
            # a_ms / b_ms are the geometric means of the per-pass medians (one pass: exactly the medians).
            ref = timed[-1]
            r = sum(pp['r'] for pp in timed) / len(timed)
            a_ok = [pp['a_ms'] for pp in timed if pp['a_ms']]
            gm = (lambda xs: None if not xs else xs[0] if len(xs) == 1 else math.exp(sum(math.log(x) for x in xs) / len(xs)))
            fl = floors.get(wid + '|' + vendor) or {}
            any_timeout = any(pp['timed_out'] for pp in timed)
            row = dict(
                evidence='%s:%s' % (ref['a']['evidence'], ref['a']['id']), a_ms=gm(a_ok), b_ms=gm([pp['b_ms'] for pp in timed]),
                b_samples=sum(len(pp['bs']) for pp in timed),
                log_ratio={p: (r if p == 'scored' else None) for p in T.PHASES},
                candidate_over_baseline={p: (math.exp(r) if p == 'scored' else None) for p in T.PHASES},
                timing_source=TIMING_SOURCE + (' (candidate timed out: ratio is a lower bound)' if any_timeout else ''),
                floor=fl.get('floor'), floor_evidence=fl.get('evidence'), floor_source=fl.get('floor_source'),
                a_head=ref['a']['head'], b_head_differs=any(pp['head_differs'] for pp in timed),
                passes=len(timed), pass_runs=[pp['run'] for pp in timed], pass_log_ratios=[pp['r'] for pp in timed])
            if any_timeout:
                row.update(timed_out=True, a_status=ref['a']['status'], timed_out_passes=sum(pp['timed_out'] for pp in timed))
            if any(pp['b_other_pass'] for pp in timed):
                row['b_from_other_pass'] = [pp['run'] for pp in timed if pp['b_other_pass']]
            case['vendors'][vendor] = row
        repaired = any(pp['run'] in repair_runs and pp['good'] for pp in per_pass)
        superseded = [pp['run'] for pp in per_pass
                      if repaired and pp['run'] not in repair_runs and pp['quality'][0] == 'FAIL']
        merged = [pp for pp in per_pass if pp['run'] not in superseded] or per_pass
        verdicts_p = [pp['quality'][0] for pp in merged]
        verdict = merge_quality(verdicts_p)
        judged = [pp for pp in merged if pp['quality'][0] == verdict] or merged
        _, detail, unjudged = judged[-1]['quality']
        quality_rows.append(dict(configuration=cid, workload_id=wid, lane=a['lane'], dataset=a['dataset'], vendor=vendor,
                                 family=a['key'][0], route=a.get('kind'), passes=len(per_pass),
                                 pass_verdicts={pp['run']: pp['quality'][0] for pp in per_pass},
                                 **(dict(superseded_fail_passes=superseded) if superseded else {}),
                                 candidate_vs_baseline=dict(verdict=verdict, metrics=detail, unjudged=unjudged,
                                                            a_status=a['status'], rel_tolerance=QUALITY_REL),
                                 evidence='%s:%s' % (a['evidence'], a['id'])))
        identity_cases.setdefault((cid, wid), {})[vendor] = dict(a=a, passes=per_pass)
    verdict_cases = []
    for case in cases.values():
        voters = T.VOTERS.get(case['mode'], T.VOTERS['identical'])
        case['voters'] = list(voters)
        case['phases'] = {p: T.judge(case['vendors'], voters, p) for p in T.PHASES}
        case['verdict'] = case['phases']['scored']['verdict']
        verdict_cases.append(case)
    verdicts = dict(schema='mojolearn.six-lane-timing-verdicts/1',
                    cases=sorted(verdict_cases, key=lambda c: (c['configuration'], c['workload_id'])),
                    rule='Verdict only when |log(A/B)| exceeds the workload floor (incumbent repeats) on every voting vendor '
                         'and all agree in direction. A = candidate (pack defines), B = incumbent (no defines); '
                         'FASTER means A faster. Scored phase only: lq reports one median_ms per race.',
                    source='tools/six_lane_grid_lq.py collect', accepted=False, promoted=False)

    # identity: NVIDIA vs AMD digest per arm
    def arm_status(dn, da, hn, ha, unstable):
        if unstable:
            return 'MISMATCH', 'incumbent digests differ run to run on ' + ','.join(unstable)
        if not dn or not da:
            return 'INCOMPLETE', 'digest missing on ' + ','.join(v for v, d in (('nvidia', dn), ('amd', da)) if not d)
        if dn == da:
            return 'MATCH', None
        if hn != ha:
            return 'INCOMPLETE', 'digests differ but the vendors raced different heads (%s vs %s)' % (hn, ha)
        return 'MISMATCH', None

    out_cases, mismatch_run = [], []
    for (cid, wid), per_v in sorted(identity_cases.items()):
        c = configs[cid]
        per = {v: x['a'] for v, x in per_v.items()}  # the last pass' candidate sample per vendor
        arms, columns = {}, {}
        a_n, a_a = per.get('nvidia'), per.get('amd')
        dn = a_n['digest'] if a_n and ok(a_n) else None
        da = a_a['digest'] if a_a and ok(a_a) else None
        # MISMATCH-RUN: the candidate digest changed between passes on one vendor at one head (a run-to-run bits bug,
        # like the RidgeCV warm/cold one). Reported apart from the NVIDIA-vs-AMD MISMATCH; the cell is not MATCH.
        run_unstable = {}
        for vendor, x in sorted(per_v.items()):
            by_head = {}
            for pp in x['passes']:
                if ok(pp['a']) and pp['a'].get('digest'):
                    by_head.setdefault(pp['a']['head'], {})[pp['run']] = pp['a']['digest']
            for head, digs in by_head.items():
                if len(set(digs.values())) > 1:
                    run_unstable[vendor] = dict(head=head, digests=digs)
        if run_unstable:
            st, why = 'MISMATCH_RUN', 'candidate digest changed between passes on ' + ','.join(sorted(run_unstable))
            for vendor, u in sorted(run_unstable.items()):
                mismatch_run.append(dict(configuration=cid, workload_id=wid, arm='A', vendor=vendor, **u))
        else:
            st, why = arm_status(dn, da, a_n and a_n['head'], a_a and a_a['head'], [])
        arms['A'] = dict(status=st, decided_by=list(IDENTITY_COLUMN.values()), nvidia=dn, amd=da, issue=why)
        bd, bh, unstable = {}, {}, []
        for vendor in ('nvidia', 'amd'):
            head = (per.get(vendor) or {}).get('head')
            samples = b_ref(wid, vendor, head)[0]
            ds = {s['digest'] for s in samples if s.get('digest')}
            if len(ds) > 1 and len({s['head'] for s in samples}) == 1:
                unstable.append(vendor)
                if len({s['pass'] for s in samples}) > 1:
                    mismatch_run.append(dict(configuration=cid, workload_id=wid, arm='B', vendor=vendor, head=samples[0]['head'],
                                             digests={r: sorted({s['digest'] for s in samples if s['pass'] == r and s.get('digest')})
                                                      for r in sorted({s['pass'] for s in samples}, key=run_ids.index)}))
            bd[vendor] = sorted(ds)[0] if ds else None
            bh[vendor] = samples[0]['head'] if samples else None
        st, why = arm_status(bd['nvidia'], bd['amd'], bh['nvidia'], bh['amd'], unstable)
        arms['B'] = dict(status=st, decided_by=list(IDENTITY_COLUMN.values()), nvidia=bd['nvidia'], amd=bd['amd'], issue=why)
        for vendor in ('nvidia', 'amd'):
            a = per.get(vendor)
            columns[IDENTITY_COLUMN[vendor]] = dict(
                status='READY' if a and ok(a) and a.get('digest') else 'INCOMPLETE',
                arms=dict(A=dict(output_sha256=a and a.get('digest'), head=a and a['head'], status=a and a['status']),
                          B=dict(output_sha256=bd[vendor], head=bh[vendor])))
        states = {arms['A']['status'], arms['B']['status']}
        status = 'MISMATCH' if 'MISMATCH' in states else 'MISMATCH_RUN' if 'MISMATCH_RUN' in states else \
            'MATCH' if states == {'MATCH'} else 'INCOMPLETE'
        out_cases.append(dict(id='%s|%s' % (cid, wid), configuration_id=cid, workload_id=wid,
                              implementation_ids=(c.get('members') or []), mode=c.get('mode', 'identical'),
                              status=status, columns=columns, arms=arms,
                              output_sha256={v: (per.get(v) or {}).get('digest') for v in ('nvidia', 'amd')},
                              accepted=False, promoted=False))
    counts = {s: sum(c['status'] == s for c in out_cases) for s in ('MATCH', 'MISMATCH', 'INCOMPLETE', 'NOT_REQUIRED')}
    if len(run_ids) > 1:
        counts['MISMATCH_RUN'] = sum(c['status'] == 'MISMATCH_RUN' for c in out_cases)
    summary = dict(schema='mojolearn.six-lane-comparison/1', cases=out_cases, counts=counts,
                   required_identical_columns=list(IDENTITY_COLUMN.values()), promotion_voters=['nvidia', 'amd'],
                   digest='bench_board_algos race output digest (first %d hex, as box_job.sh records it)' % DIGEST_CHARS,
                   apple_timing_votes=False, accepted=False, promoted=False, source='tools/six_lane_grid_lq.py collect')
    quality = dict(schema='mojolearn.six-lane-grid-lq-quality/1', rows=quality_rows,
                   rule='A vs the incumbent B on the board quality json, per metric: lower-is-better (rmse, logloss, '
                        'inertia, error, residual, ...) or higher-is-better (accuracy, auc, r2, trustworthiness, ...); '
                        'WORSE beyond %g relative. A race that failed is FAIL. Unknown keys are listed, not judged.' % QUALITY_REL)
    with_obs = {(o['vendor'], o['id']) for o in obs}  # a job lq cut before its ALGOS text needs its race logs
    # coverage per vendor over the plan's cells
    coverage = {}
    for cell in matrix['cells']:
        v = cell['vendor']
        key = (cell['configuration'], cell['workload_id'], v)
        try:
            route(cell['workload_id'], lanes, board)
            state = 'MEASURED' if key in A and any(ok(s) for s in A[key]) else 'FAILED' if key in A else \
                'NOT_READY' if key in a_infra_set else 'MISSING'
        except ValueError:
            state = 'REFUSED'
        coverage.setdefault(v, {}).setdefault(state, 0)
        coverage[v][state] += 1
    passes_hist = {}
    for case in verdict_cases:
        for vendor, row in case['vendors'].items():
            k = '%s:%d' % (vendor, row.get('passes', 1))
            passes_hist[k] = passes_hist.get(k, 0) + 1
    report = dict(schema='mojolearn.six-lane-grid-lq-collect/1', run_id=run_id, run_ids=run_ids, plan_dir=str(plan_dir),
                  repair_runs=repair_runs,
                  passes_per_cell=dict(sorted(passes_hist.items())), mismatch_run=mismatch_run,
                  results=[str(p) for p in results], logs=[str(p) for p in (logs or [])],
                  observations=len(obs), a_cells=len(A), b_workload_vendor=len(B), ignored=ignored,
                  no_result=nores, coverage=coverage,
                  not_ready=dict(a_cells=len(a_infra), b_workload_vendor=len(b_infra), samples=a_dropped + b_dropped,
                                 superseded_redo_samples=a_superseded + b_superseded,
                                 a_cells_by_lane=_infra_by_lane(a_infra),
                                 rule='status=not_ready is infrastructure: never FAIL/BROKEN; redo with redo-not-ready'), verdict_counts=_count(verdict_cases, 'verdict'),
                  identity_counts=counts, quality_counts=_count(quality_rows, None),
                  truncated_without_log=sum(1 for o in obs if o.get('truncated') and not o.get('log_evidence')),
                  cmd_jobs_without_gridbb=sorted('%s:%s' % k for k, j in jobs.items()
                                                 if (j.get('cmd_tag') or '').partition('.')[0] in run_ids
                                                 and (k[0], k[1]) not in {(o['vendor'], o['id']) for o in bb_obs}),
                  jobs_without_algos=sorted('%s:%s' % k for k, j in jobs.items()
                                            if (j.get('tag') or '').partition('.')[0] in run_ids and k not in with_obs
                                            and k not in {(n['vendor'], n['id']) for n in nores}))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, doc in (('floors.json', floor_doc), ('grid-verdicts.json', verdicts), ('summary.json', summary),
                      ('quality.json', quality), ('collect-report.json', report)):
        (out_dir / name).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    return report


def _infra_by_lane(keys):
    out = {}
    for _cid, wid, vendor in keys:
        k = '%s:%s' % (vendor, wid.split('@', 1)[0].split(':', 1)[-1])
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _count(rows, key):
    out = {}
    for r in rows:
        v = r[key] if key else r['candidate_vs_baseline']['verdict']
        out[v] = out.get(v, 0) + 1
    return out


# ------------------------------------------------------------------ CLI


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('render', help='grid plan -> lq add lines for one box')
    r.add_argument('--plan-dir', type=Path, required=True, help='dir with grid-plan.json + grid-matrix.json.gz')
    r.add_argument('--vendor', choices=sorted(BOX), required=True)
    r.add_argument('--branch', default='main', help='pushed branch both arms build (freeze one commit per round)')
    r.add_argument('--b-repeats', type=int, help='incumbent repeats per workload group (default 3; 1 with --rerun-undecided)')
    r.add_argument('--b-group-size', type=int, default=8, help='workloads per incumbent line (one build covers them)')
    r.add_argument('--only-lanes', help='comma list of bench_board_algos lanes')
    r.add_argument('--phase', choices=sorted(REGIME_RANK))
    r.add_argument('--budget-hours', type=float, help='cut A lines (in order) at this many race hours incl. their B repeats')
    r.add_argument('--run-id', help='tag prefix (default: hash of the plan files)')
    r.add_argument('--lanes-json', help='lane registry JSON {lane: [datasets]} instead of importing bench_board_algos')
    r.add_argument('--board-json', help='bench_board registry JSON (load_board shape, races as [family, lane, dataset] lists)')
    r.add_argument('--out', type=Path, required=True, help='lines file; <out>.json gets the manifest')
    r.add_argument('--prebuilt', metavar='STORE', help='add PREBUILT=STORE to every line (e.g. /root/grid-prebuilt): box_job.sh '
                   'installs tools/six_lane_grid_prebuild.py artifacts instead of building, and builds when none fit')
    r.add_argument('--rerun-undecided', metavar='DECIDE_DIR', type=Path,
                   help='second pass: only the undecided cells of configurations with no decided cell, from DECIDE_DIR/grid-decisions.json')
    r.add_argument('--rerun-id', help='run id of the second pass (default: <plan run id>%s); tags keep the pack ids' % RERUN_SUFFIX)
    r.add_argument('--first-pass', metavar='COLLECT_DIR', type=Path,
                   help='first-pass collect output (grid-verdicts.json): measured medians for the projected hours')
    r.add_argument('--skip-controls', help='comma list of controls already flipped by hand (final controls are skipped anyway)')
    r.add_argument('--skip-broken-controls', action='store_true', help='also skip every configuration of a HOLD_BROKEN control')
    add_redo_parser(s)
    r.add_argument('--only-workloads', help='comma list of workload ids or family prefixes ending in ":" (e.g. neural:): '
                   'render only those cells (a one-shot redo; give it its own --run-id so fed tags do not collide)')
    c = s.add_parser('collect', help='lq results + race logs -> verdicts, identity summary, quality rows')
    c.add_argument('--results', nargs='+', required=True, help='results.txt copies (one per box)')
    c.add_argument('--logs', nargs='*', default=[], help='lq out dirs (race-*.log under <id>/) or `grep -H` dumps of them')
    c.add_argument('--plan-dir', type=Path, required=True)
    c.add_argument('--run-id', nargs='+', help='one run id, or several passes (first pass first) merged per cell')
    c.add_argument('--runs', help='comma list of passes (same as several --run-id values)')
    c.add_argument('--repair-runs', help='comma list of repair passes (e.g. <run>r1n from render --only-workloads): a cell '
                   'they measured drops the FAIL verdicts of earlier passes (tooling failures), kept as evidence')
    c.add_argument('--min-samples', type=int, default=2)
    c.add_argument('--lanes-json')
    c.add_argument('--board-json')
    c.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == 'redo-not-ready':
        return redo_main(args)
    lanes = load_lanes(args.lanes_json)
    board = None
    if args.board_json:
        board = json.loads(Path(args.board_json).read_text())
        board['races'] = {tuple(r) for r in board['races']}
    if args.cmd == 'render':
        only = {x for x in (args.only_lanes or '').split(',') if x} or None
        only_wl = [x for x in (args.only_workloads or '').split(',') if x] or None
        if only:
            unknown = sorted(only - set(lanes) - {r[1] for r in (board or load_board(args.vendor))['races']})
            if unknown:
                p.error('unknown lane(s): ' + ','.join(unknown))
        if args.rerun_undecided:
            lines, manifest = render_rerun(args.plan_dir, args.vendor, args.rerun_undecided, args.branch,
                                           1 if args.b_repeats is None else args.b_repeats, args.rerun_id, args.first_pass,
                                           [x for x in (args.skip_controls or '').split(',') if x], args.skip_broken_controls,
                                           only_lanes=only, only_workloads=only_wl,
                                           phase=args.phase, budget_hours=args.budget_hours, lanes=lanes,
                                           b_group_size=args.b_group_size, run_id=args.run_id, board=board, prebuilt=args.prebuilt)
        else:
            if args.rerun_id or args.first_pass or args.skip_controls or args.skip_broken_controls:
                p.error('--rerun-id, --first-pass, --skip-controls and --skip-broken-controls need --rerun-undecided')
            lines, manifest = render(args.plan_dir, args.vendor, args.branch, 3 if args.b_repeats is None else args.b_repeats,
                                     only, args.phase, args.budget_hours, lanes, args.b_group_size, args.run_id, board, args.prebuilt,
                                     only_workloads=only_wl)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(''.join(line + '\n' for line in lines))
        Path(str(args.out) + '.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
        t = manifest['totals']
        print(json.dumps({k: t[k] for k in ('vendor', 'run_id', 'lines', 'a_lines', 'b_lines', 'b_repeats', 'a_races', 'b_races',
                                            'by_kind', 'plan_cells', 'configurations', 'workloads', 'refused_cells',
                                            'refused_workloads', 'cut_a_lines_by_budget', 'projected_race_hours')}))
        if 'rerun' in t:
            rr = t['rerun']
            print(json.dumps(dict(rerun_configurations=rr['configurations'], rerun_cells=rr['cells'],
                                  rendered_configurations=rr['rendered_configurations'], lines=t['lines'],
                                  races=t['a_races'] + t['b_races'], skipped_configurations=rr['skipped_configurations'],
                                  unmeasured_cells_not_rerun=rr['unmeasured_cells_not_rerun'],
                                  projected_race_hours=t['projected_race_hours'],
                                  measured_race_hours=t.get('measured_race_hours'), measured_note=t.get('measured_note'))))
        return 0
    runs = list(args.run_id or []) + [x for x in (args.runs or '').split(',') if x]
    report = collect(args.results, args.logs, args.plan_dir, args.out, lanes, runs or None, args.min_samples, board,
                     [x for x in (args.repair_runs or '').split(',') if x])
    print(json.dumps({k: report[k] for k in ('run_id', 'passes_per_cell', 'observations', 'a_cells', 'coverage', 'verdict_counts',
                                             'identity_counts', 'quality_counts', 'ignored', 'truncated_without_log',
                                             'jobs_without_algos', 'cmd_jobs_without_gridbb')}))
    if report['mismatch_run']:
        print('MISMATCH-RUN %d cell arm(s): digest changed between passes on one vendor (see collect-report.json mismatch_run)'
              % len(report['mismatch_run']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
