#!/usr/bin/env python3
"""Per-algorithm permutation grid of IDENTICAL performance switches (planning only).

Inputs (never the stale catalog.json/matrix.json.gz define lists):
  experiments/six_lane_integration/grid_controls/*.json  (mojolearn.grid-controls/1)
  core/six_lane_experiment_guards.mojo                    (guard exclusions, evaluated here)
Workload ids come from the harness's saved recipes: the existing matrix's workload
inventory (ids only), the registered full-input variants (classification-full-v1,
tsvd-full-v1, mlp-full-v1) and the neural board lanes.

Outputs (experiments/six_lane_integration/grid/):
  grid-plan.json        algorithms, configs, deferred, excluded, A/A pairs, builds, verdict rule
  grid-matrix.json.gz   mojolearn.six-lane-matrix/1, accepted by six_lane_ab.py queue/materialize
  grid-build-plan.json  mojolearn.six-lane-build-plan/1, accepted by six_lane_ab.py compile
  GRID.md               per-family tables, totals, owed special A/Bs and the exact queue commands

Planning only: this never imports an estimator, compiles, queues or measures.
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

CONTROLS_DIR = ROOT / 'experiments/six_lane_integration/grid_controls'
GUARDS = ROOT / 'core/six_lane_experiment_guards.mojo'
OUT_DIR = ROOT / 'experiments/six_lane_integration/grid'
PHASE2_DIR = ROOT / 'experiments/six_lane_integration/grid-phase2'
CAP = 16
FULL_BUDGET = 128  # --crosses auto: full factorial when its raw size is at most this
# Median scored-pair seconds per box from the Oct 6 receipts (projection only; override with --pair-seconds).
PAIR_SECONDS = {'nvidia': 38.0, 'amd': 24.0}
MODE = 'identical'
IDENTICAL_DEFINE = 'MOJOLEARN_NUMERIC_IDENTICAL=1'
# Owner 2026-10-07: IDENTICAL is decided by NVIDIA and AMD; Apple does not vote.
VENDORS = {
    'nvidia': dict(box='nv (RunPod L40S)', target_track='nvidia-native', compile_flags='--vendor nvidia --nvidia-target native --nvidia-arch sm_89'),
    'amd': dict(box='amd (DO MI325X)', target_track='amd-gfx942', compile_flags='--vendor amd --accelerator gfx942'),
}
STANDARD_BLOCKER = ('Full dataset/version/hash, dimensions, settings, cap audit and accepted artifacts must be '
                    'supplied from the frozen saved recipe.')
DEFER_LIST_LIMIT = 32  # deferred crosses are reproducible from (members, rule); list a sample + exact count

# ---------------------------------------------------------------- reach rules
# Control-level exclusions read from the authored notes/tags. Each match is quoted
# into the EXCLUDED list; nothing is dropped silently.
EXCLUDE_PATTERNS = [
    (r'\bUNREACHED\b', 'unreached'),
    (r'NOT reached by the board', 'unreached'),
    (r'never engages on the board', 'unreached'),
    (r'no board \w+ lane .*reaches it', 'unreached'),
    (r'not on the board lane list', 'off-board'),
    (r'\(peripheral\)', 'peripheral-only'),
    (r'\bvariants? only\b', 'peripheral-only (variant lanes only)'),
]
EXCLUDE_TAGS = {'off_board': 'off-board', 'blocked_quality': 'blocked-quality'}

# Dependencies the notes state in prose ("only read with X on", "scope of X").
PARENT_PATTERNS = [r'only (?:read|meaningful) with (\w+)', r'[Cc]aller-class scope of (\w+)']
# (nn20_split_kv_leaves -> nn20_split_kv was the one entry; both controls were deleted 2026-10-08, lane grid-act-3.)
EXPLICIT_PARENTS = {}
# Arm-level reach the notes state; scope None = every algorithm.
ARM_EXCLUSIONS = [
    ('neural_gemm_epilogue', 'cnn', None, 'note: "cnn arm is non-board (x_cnn)"'),
    ('T11_LEVELS', '1', 'trees:rf', 'note: "RF K1/K2/K8 already showed no combined gain (forest-final-decisions 2026-10-05): RF grid needs only 16"'),
    ('T11_LEVELS', '2', 'trees:rf', 'note: "RF K1/K2/K8 already showed no combined gain (forest-final-decisions 2026-10-05): RF grid needs only 16"'),
    ('T11_LEVELS', '8', 'trees:rf', 'note: "RF K1/K2/K8 already showed no combined gain (forest-final-decisions 2026-10-05): RF grid needs only 16"'),
    # T17_BATCH '32' was excluded as identical to the incumbent until lane/grid-flips-1 (2026-10-08) made 128 the default.
]
# Arms whose notes declare reach beyond the declared algorithm lists. A config
# carrying one is never packed with another algorithm's config.
GLOBAL_REACH = {
    ('gemm_contract_leaf', 'all256'): 'note: "NI08 contract leaf 256 for every GEMM caller in the build (gemm lane and classical too)"',
    ('gemm_tile_min_blocks', '*'): 'note: "Reaches every gemm_identical tuned-tile caller in the build (classical included)"',
    ('gemm_split_min_leaves', '*'): 'note: "the split-plan floor of every gemm_identical caller (classical included)"',
    ('gemm_kpack_rpt4', '*'): 'note: "Reaches classical callers"',
    ('neural_gemm_ozaki_slices', '*'): 'other_reach: "non-board callers of gemm/neural_dispatch.identical_gemm_into also take the switch"',
}
# Notes that name a shipped default equal to an arm on one vendor (kept in the
# grid so both vendors run the same config; the result is expected neutral there).
VENDOR_DEFAULT_EQUIVALENT = {
    ('RF_SAMPLE', 'fused'): ('nvidia', 'note: "vendor_default = fused on NVIDIA"'),
    ('RF_SAMPLE', 'two_launch'): ('amd', 'note: "two_launch on AMD"'),
}

# ------------------------------------------------------------ workload mapping
# grid_controls ids that name a board lane under another harness id.
ALIASES = {
    'gemm:gemm': ('neural:gemm', 'board lane "gemm" in tools/bench_board_neural.py LANES'),
    'neural:cnn-clf': ('expanded:cnn-clf', 'CNNClassifier board lane is _add("cnn-clf") in tools/bench_board_algos.py'),
}
UNMAPPED_REASONS = {
    'neural:mamba-forward': 'ambiguous: no harness lane "mamba-forward"; neural:mamba1-forward, mamba2-forward and mamba3-forward are separate saved lanes. Not substituted; the owning lane must name the lanes.',
}


def sha_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_define(d):
    return d if '=' in d else d + '=1'


def unique(items):
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


# ---------------------------------------------------------------------- guards
class Guards:
    """Evaluates core/six_lane_experiment_guards.mojo for a define set.

    Every `comptime NAME = expr` and `comptime assert cond, "msg"` line is
    translated to Python (is_defined -> _d, get_defined_int -> _i). Anything the
    translation does not understand raises, so a new guard shape cannot pass
    silently.
    """

    def __init__(self, text):
        self.steps = []
        body = text.split('def _check_configuration', 1)[1].split('return True', 1)[0]
        for raw in body.splitlines():
            line = raw.strip()
            if not line.startswith('comptime '):
                continue
            expr_line = re.sub(r'is_defined\["(\w+)"\]\(\)', r'_d("\1")', line)
            expr_line = re.sub(r'get_defined_int\["(\w+)",\s*(-?\d+)\]\(\)', r'_i("\1",\2)', expr_line)
            m = re.fullmatch(r'comptime assert (.*), "((?:[^"\\]|\\.)*)"', expr_line)
            if m:
                cond, msg = m.group(1), m.group(2)
                self._check(cond, raw)
                self.steps.append(('assert', compile(cond, 'guard', 'eval'), msg))
                continue
            m = re.fullmatch(r'comptime (\w+) = (.*)', expr_line)
            if m:
                self._check(m.group(2), raw)
                self.steps.append(('let', m.group(1), compile(m.group(2), 'guard', 'eval')))
                continue
            raise ValueError('Unrecognized guard line: ' + line)
        if not any(s[0] == 'assert' for s in self.steps):
            raise ValueError('No guard asserts parsed')
        self.pairs = []

    def add_pair(self, a, b, reason):
        """A source-level exclusion outside the guard file (a binding's own comptime assert)."""
        if (a, b, reason) not in self.pairs:
            self.pairs.append((a, b, reason))

    @staticmethod
    def _check(expr, raw):
        if '[' in expr or 'is_defined' in expr or 'get_defined' in expr:
            raise ValueError('Untranslated guard expression: ' + raw.strip())

    @property
    def assert_count(self):
        return sum(1 for s in self.steps if s[0] == 'assert')

    def problems(self, defines):
        values = {}
        for d in defines:
            k, _, v = norm_define(d).partition('=')
            values[k] = v

        def _d(name):
            return name in values

        def _i(name, default):
            if name not in values:
                return default
            try:
                return int(values[name])
            except ValueError:
                raise ValueError('get_defined_int on a non-integer define ' + name + '=' + values[name])

        env = {'_d': _d, '_i': _i, '__builtins__': {}}
        out = []
        for step in self.steps:
            if step[0] == 'let':
                env[step[1]] = eval(step[2], env)
            elif not eval(step[1], env):
                out.append(step[2])
        for a, b, reason in self.pairs:
            if a in values and b in values:
                out.append('source exclusion ' + a + ' / ' + b + ' (' + reason + ')')
        return out


def harness_problems(defines):
    """The existing harness's combine() checks (conflicts, retired, _OFF pairs)."""
    from six_lane_ab import combine
    _, problems = combine([dict(defines=list(defines), environment={}, runtime={})])
    return problems


# ---------------------------------------------------------------- load inputs
def load_controls(directory=CONTROLS_DIR):
    controls, algorithms, removed, owed, files = {}, {}, {}, [], {}
    for path in sorted(glob.glob(str(Path(directory) / '*.json'))):
        doc = json.loads(Path(path).read_text())
        name = Path(path).stem
        if doc.get('schema') != 'mojolearn.grid-controls/1':
            raise ValueError(path + ': unexpected schema ' + str(doc.get('schema')))
        files[name] = dict(path=str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else path,
                           sha256=file_sha(path), lane=doc.get('lane'), branch=doc.get('branch'), head=doc.get('head'))
        for key, c in doc['controls'].items():
            if key in controls:
                raise ValueError('Duplicate control key ' + key)
            controls[key] = dict(c, key=key, file=name,
                                 arms={a: [norm_define(x) for x in ds] for a, ds in c['arms'].items()})
        for wid, a in doc['algorithms'].items():
            algorithms.setdefault(normalize_algorithm_id(wid), []).append(dict(a, file=name, raw_id=wid))
        rem = doc.get('removed', [])
        removed[name] = len(rem)
        for item in doc.get('owed_removal_ab', []):
            owed.append(dict(item, file=name))
    return controls, algorithms, removed, owed, files


def normalize_algorithm_id(wid):
    return 'trees:' + wid[len('trees/'):] if wid.startswith('trees/') else wid


def bindings_of(text, workload=False):
    if isinstance(text, list):
        text = ' '.join(text)
    if workload:
        text = text.split('(')[0]
    return sorted({b for b in re.findall(r'_mojolearn\w*', text) if not b.endswith('_host')})


def binding_path(name):
    return 'bindings/' + ('_mojolearn.mojo' if name == '_mojolearn' else name + '.mojo')


def control_text(c):
    return ' '.join(str(c.get(k, '')) for k in ('note', 'notes', 'reach'))


def token_index(controls):
    idx = {}
    for key, c in controls.items():
        names = {key, key.lower(), c['define'], c['define'].removeprefix('MOJOLEARN_')}
        for m in re.findall(r'split_from:([\w+,/ ]+)', str(c.get('status', ''))):
            for part in re.split(r'[+,/ ]', m):
                if part:
                    names.add(part)
                    names.add(part.removeprefix('MOJOLEARN_'))
        for part in c.get('merged_from', []) or []:
            names.add(str(part))
        for n in names:
            idx.setdefault(n, key)
    return idx


def resolve_token(token, controls, idx):
    if token in controls:
        return token
    for cand in [token.split('(')[0].strip(), token.split(':', 1)[-1].split('(')[0].strip()]:
        cand = cand.split(' ')[0]
        for form in (cand, cand.removeprefix('MOJOLEARN_'), cand.lower()):
            if form in idx:
                return idx[form]
    m = re.search(r'MOJOLEARN_\w+', token)
    if m:
        for form in (m.group(0), m.group(0).removeprefix('MOJOLEARN_')):
            if form in idx:
                return idx[form]
    return None


# ------------------------------------------------------------------ inventory
def workload_inventory():
    inv = {}
    # The six-lane planning matrix was retired on 2026-10-08; its workload ids live on in workload_ids.json.
    for wid in json.loads((ROOT / 'experiments/six_lane_integration/workload_ids.json').read_text())['workload_ids']:
        inv.setdefault(wid, 'existing six-lane matrix workload inventory (ids only; its define catalog is stale)')
    for path, field in (('experiments/six_lane_integration/classification_full_contracts.json', 'variant_workload_id'),
                        ('experiments/six_lane_integration/mlp_full_contracts.json', 'variant_workload_id')):
        for row in json.loads((ROOT / path).read_text())['rows']:
            if row.get(field):
                inv[row[field]] = 'registered full-input variant (' + path + ')'
    from six_lane_full_variants import LANES as TSVD_LANES, SUFFIX as TSVD_SUFFIX
    for lane in TSVD_LANES:
        base = 'more:tsvd' if lane == 'tsvd' else 'expanded:' + lane
        for ds in ('taxi', 'istella'):
            inv[base + '@dataset=' + ds + TSVD_SUFFIX] = 'registered full-input variant (tools/six_lane_full_variants.py)'
    tree = ast.parse((ROOT / 'tools/bench_board_neural.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LANES' for t in node.targets):
            for lane in ast.literal_eval(node.value):
                inv.setdefault('neural:' + lane, 'neural board lane (tools/bench_board_neural.py LANES)')
    tree = ast.parse((ROOT / 'tools/bench_board_algos.py').read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_add'
                and node.args and isinstance(node.args[0], ast.Constant)):
            for kw in node.keywords:
                if kw.arg == 'datasets':
                    try:
                        datasets = ast.literal_eval(kw.value)
                    except ValueError:
                        continue  # computed dataset tuples: those lanes are in the matrix inventory
                    for ds in datasets:
                        inv.setdefault('expanded:' + node.args[0].value + '@dataset=' + ds,
                                       'board lane _add("' + node.args[0].value + '") datasets in tools/bench_board_algos.py')
    return inv


_ALGOS_DATASETS = None


def algos_lane_datasets():
    """{lane: datasets} of tools/bench_board_algos.py LANES, the lanes `expanded:` workloads race (lq RACE and
    tools/bench_board.py's algos family). Imported, not parsed: several lanes compute their dataset tuple."""
    global _ALGOS_DATASETS
    if _ALGOS_DATASETS is None:
        sys.path.insert(0, str(ROOT / 'tools'))
        import bench_board_algos
        _ALGOS_DATASETS = {k: tuple(v.get('datasets') or ()) for k, v in bench_board_algos.LANES.items()}
    return _ALGOS_DATASETS


def board_lane_datasets(target):
    """Datasets the board races for an expanded: (bench_board_algos) or more: (bench_board_more) lane."""
    lane = target.split(':', 1)[1].split('@', 1)[0]
    if target.startswith('expanded:'):
        return algos_lane_datasets().get(lane)
    sys.path.insert(0, str(ROOT / 'tools'))
    import bench_board_more
    return tuple(bench_board_more.datasets_of(lane)) if lane in bench_board_more.LANES else None


def map_workloads(algo_id, inv):
    """-> (workload ids, notes) or (None, reason). Never substitutes smaller data.

    expanded: and more: workloads keep only the datasets their board lane races (2026-10-07: the expanded
    time-series lanes race taxi-hourly and synthetic, lu-solve and more:arima/ets synthetic only; the stale
    matrix inventory named taxi and istella for them, which no board race can run). The lane's own datasets
    are added when missing."""
    if algo_id in UNMAPPED_REASONS:
        return None, UNMAPPED_REASONS[algo_id]
    target, notes = algo_id, []
    if algo_id in ALIASES:
        target, why = ALIASES[algo_id]
        notes.append('alias ' + algo_id + ' -> ' + target + ': ' + why)
    if target.startswith('neural:'):
        return ([target], notes) if target in inv else (None, 'no saved neural lane ' + target)
    if target.startswith('trees:'):
        lane = target[len('trees:'):]
        ids = sorted(w for w in inv if w.split(':', 1)[0] == lane and w.count(':') == 1)
        return (ids, notes) if ids else (None, 'no saved tree lane ' + lane + ':<dataset>')
    if target.split(':', 1)[0] in ('classical', 'more', 'expanded'):
        out, capped = [], []
        datasets = sorted({w.split('@dataset=', 1)[1].split('@', 1)[0] for w in inv
                           if w.startswith(target + '@dataset=')})
        if target.startswith(('expanded:', 'more:')):
            own = board_lane_datasets(target)
            if own:
                dropped = [d for d in datasets if d not in own]
                datasets = sorted(set(d for d in datasets if d in own) |
                                  ({d for d in own if '@' not in target} if dropped else set()))
                if dropped:
                    notes.append('datasets ' + ','.join(dropped) + ' are not raced by board lane ' + target +
                                 ' (it races ' + ','.join(own) + ')')
                    for d in own:
                        inv.setdefault(target + '@dataset=' + d, 'board lane datasets (tools/bench_board_algos.py LANES)')
        for ds in datasets:
            base = target + '@dataset=' + ds
            variants = sorted(w for w in inv if w.startswith(base + '@input='))
            if variants:
                out.extend(variants)
                notes.append(base + ' is a capped original; full registered input ' + ','.join(v.split('@input=')[1] for v in variants) + ' used')
                continue
            lane = target.split('@', 1)[0]
            if '@' in target and any(w.startswith(lane + '@dataset=' + ds + '@input=') for w in inv):
                capped.append(base)
                continue
            if base in inv:
                out.append(base)
        if capped:
            notes.append('capped originals without a registered full input (not queued): ' + ', '.join(capped))
        if out:
            return out, notes
        if capped:
            return None, 'only capped originals exist (' + ', '.join(capped) + '); the base lane has a registered full input but this operation suffix does not'
        return None, 'no saved workload ' + target + '@dataset=<taxi|istella>'
    return None, 'unknown workload family'


# ------------------------------------------------------------ algorithm model
def build_model(controls, algorithms, inv):
    idx = token_index(controls)
    reach = {}
    model = {}
    for algo_id, entries in sorted(algorithms.items()):
        ctrl, groups, unresolved, notes, not_reached = [], [], [], [], set()
        wb = set()
        for e in entries:
            wb.update(bindings_of(e['binding'], workload=True))
            ctrl.extend(e['controls'])
            for n in ('note', 'notes'):
                if e.get(n):
                    notes.append(e['file'] + ': ' + str(e[n]))
            for item in e.get('not_reached', []) or []:
                tok = resolve_token(str(item), controls, idx)
                if tok:
                    not_reached.add(tok)
        ctrl = unique(ctrl)
        for e in entries:
            for g in e['interaction_groups']:
                members, missing = [], []
                for tok in g:
                    key = resolve_token(tok, controls, idx)
                    if key and key in ctrl:
                        members.append(key)
                    else:
                        missing.append(tok if not key else tok + ' (resolves to ' + key + ', not declared for this algorithm)')
                if missing:
                    unresolved.append(dict(group=g, members_outside_grid=missing, file=e['file']))
                members = unique(members)
                if len(members) >= 2 and members not in groups:
                    groups.append(members)
        for c in ctrl:
            reach.setdefault(c, set()).add(algo_id)
        restrict = {}
        for n in notes:
            m = re.search(r'Only (.+?) reach(?:es)? this lane', n)
            if m:
                for k, a in re.findall(r'(\w+)=(\w+)', m.group(1)):
                    restrict.setdefault(k, set()).add(a)
        workloads, map_note = map_workloads(algo_id, inv)
        model[algo_id] = dict(id=algo_id, files=unique(e['file'] for e in entries), raw_ids=unique(e['raw_id'] for e in entries),
                              workload_bindings=sorted(wb), controls=ctrl, groups=groups, unresolved_interactions=unresolved,
                              notes=notes, not_reached=sorted(not_reached), arm_restrict={k: sorted(v) for k, v in restrict.items()},
                              workloads=workloads, mapping=map_note if workloads is None else None,
                              mapping_notes=map_note if workloads is not None else [])
    return model, reach


def source_exclusions(controls):
    """Exclusions the controls declare outside the guard file."""
    out = []
    for key, c in sorted(controls.items()):
        own = c['define']
        for other in c.get('exclusive_with') or []:
            out.append((own, other.split('=')[0], key + '.exclusive_with'))
        for m in re.finditer(r'Refused \(comptime assert\) together with (MOJOLEARN_\w+)', control_text(c)):
            out.append((own, m.group(1), key + ' note: "' + m.group(0) + '"'))
    return out


def control_exclusion(c, reach):
    reasons = []
    text = control_text(c)
    for pat, kind in EXCLUDE_PATTERNS:
        m = re.search(pat, text)
        if m:
            start = max(0, m.start() - 60)
            reasons.append(kind + ': "' + text[start:m.end() + 60].strip() + '"')
            break
    for tag, kind in EXCLUDE_TAGS.items():
        if tag in (c.get('tags') or []):
            reasons.append(kind + ' (tag ' + tag + (', board ' + str(c.get('board')) if c.get('board') else '') + ')')
    if c.get('grid') == 'exclude':
        reasons.append('grid: exclude (authored)')
    if c.get('owed_removal'):
        reasons.append('owed legacy-rule removal A/B (special, see OWED): ' + c['owed_removal'].get('reason', ''))
    if not reach.get(c['key']):
        reasons.append('not declared by any board algorithm workload')
    return reasons


def parent_of(c, controls):
    if c['key'] in EXPLICIT_PARENTS:
        return EXPLICIT_PARENTS[c['key']]
    text = control_text(c)
    for pat in PARENT_PATTERNS:
        m = re.search(pat, text)
        if m and m.group(1) in controls and m.group(1) != c['key']:
            return m.group(1), 'note: "' + m.group(0) + '"'
    return None


def arm_exclusions(c, algo_id):
    out = {}
    for key, arm, scope, why in ARM_EXCLUSIONS:
        if key == c['key'] and (scope is None or scope == algo_id) and arm in c['arms']:
            out[arm] = why
    for arm, reqs in (c.get('arm_requirements') or {}).items():
        unresolved = [r for r in reqs if '<' in r and not r.startswith('optional')]
        if unresolved and arm in c['arms']:
            out[arm] = 'requires unresolved compile parameter(s): ' + '; '.join(unresolved)
    return out


def global_reach(key, arm):
    return GLOBAL_REACH.get((key, arm)) or GLOBAL_REACH.get((key, '*'))


# --------------------------------------------------------------- configurations
def assignment_defines(assign, controls):
    ds = []
    for key, arm in assign:
        ds.extend(controls[key]['arms'][arm])
    return unique(sorted(norm_define(d) for d in ds))


def define_conflicts(defines):
    seen, out = {}, []
    for d in defines:
        k, _, v = d.partition('=')
        if k in seen and seen[k] != v:
            out.append('Conflicting define values ' + k + '=' + seen[k] + ' / ' + v)
        seen[k] = v
    return out


def check(defines, guards):
    return define_conflicts(defines) + guards.problems(defines) + harness_problems(defines)


def label(assign):
    return '+'.join(k + '=' + a for k, a in assign)


def plan_algorithm(algo, controls, reach, guards, cap=CAP, crosses='full', phase1=None, full_budget=FULL_BUDGET):
    """Per-algorithm configurations in priority order under the cap (cap 0 = no cap).

    crosses: 'full' (default, the committed grids) = within-group products admitted group-atomically;
    'pairwise' = every pair of group members x every non-default arm combination, admitted one config at a
    time (the cap cuts configs, not whole groups); 'pairwise+triples' = pairs, then triples; 'auto' = the
    whole factorial over the eligible controls when its raw size (product of (arms + 1) - 1) is at most
    full_budget (regime 'factorial', no cap), else singles + all-on + pairwise crosses (regime 'pairwise').
    phase1: {frozenset(assignment items): phase-1 decision row} for this algorithm (--survivors); the plan
    is then the phase-2 grid (see plan_phase2).
    """
    excluded, eligible, arm_info, invalid = [], [], {}, []
    for key in algo['controls']:
        c = controls[key]
        reasons = control_exclusion(c, reach)
        if key in algo['not_reached']:
            reasons.append('algorithm not_reached list')
        if reasons:
            excluded.append(dict(control=key, define=c['define'], reasons=reasons))
            continue
        eligible.append(key)
    parents = {}
    for key in list(eligible):
        p = parent_of(controls[key], controls)
        if p:
            if p[0] not in eligible:
                excluded.append(dict(control=key, define=controls[key]['define'],
                                     reasons=['parent ' + p[0] + ' not eligible for this algorithm (' + p[1] + ')']))
                eligible.remove(key)
                continue
            parents[key] = p
    for key in eligible:
        c = controls[key]
        skip = arm_exclusions(c, algo['id'])
        allowed = algo['arm_restrict'].get(key)
        arms, dropped = [], []
        for arm, ds in c['arms'].items():
            if arm == c['default'] or not ds:
                continue
            if arm in skip:
                dropped.append(dict(arm=arm, reason=skip[arm]))
            elif allowed is not None and arm not in allowed:
                dropped.append(dict(arm=arm, reason='algorithm note restricts reach to ' + key + '=' + '|'.join(allowed)))
            else:
                arms.append(arm)
        if not arms:
            excluded.append(dict(control=key, define=c['define'], reasons=['no reachable non-default arm'] + [d['reason'] for d in dropped]))
        arm_info[key] = dict(arms=arms, dropped_arms=dropped)
    eligible = [k for k in eligible if arm_info[k]['arms']]
    for key in list(eligible):
        if key in parents and parents[key][0] not in eligible:
            excluded.append(dict(control=key, define=controls[key]['define'], reasons=['parent ' + parents[key][0] + ' has no reachable arm']))
            eligible.remove(key)
    rep = {k: arm_info[k]['arms'][0] for k in eligible}

    def with_parent(assign):
        # Transitive: a grandchild (child -> parent -> grandparent) carries
        # every ancestor's representative arm.
        ordered = list(assign)
        while True:
            keys = {k for k, _ in ordered}
            extra = [(parents[k][0], rep[parents[k][0]]) for k, _ in ordered if k in parents and parents[k][0] not in keys]
            if not extra:
                break
            ordered = unique(extra + ordered)
        return tuple(sorted(ordered, key=lambda ka: eligible.index(ka[0])))

    candidates, seen, reused = [], {(): 'B'}, []

    def add(tier, assign, origin):
        assign = with_parent(assign)
        defines = assignment_defines(assign, controls)
        key = tuple(defines)
        if not defines:
            return None
        if key in seen:
            return dict(tier=tier, assignment=label(assign), duplicate_of=seen[key])
        if phase1 is not None:
            prev = phase1.get(frozenset(assign))
            if prev is not None:  # measured in phase 1 with the same assignment: reused, never re-emitted
                seen[key] = prev['configuration']
                reused.append(dict(tier=tier, assignment={k: a for k, a in assign}, phase1_configuration=prev['configuration'],
                                   phase1_verdict=prev['verdict']))
                return dict(tier=tier, assignment=label(assign), reused=prev['configuration'])
        problems = check(defines, guards)
        if problems:
            invalid.append(dict(tier=tier, assignment=label(assign), problems=problems))
            return None
        cid = 'G.' + algo['id'] + '.' + (tier if tier in ('all_on', 'all_survivors') else label(assign))
        seen[key] = cid
        cfg = dict(id=cid, algorithm=algo['id'], tier=tier, origin=origin, assignment={k: a for k, a in assign},
                   defines=defines, controls=[k for k, _ in assign],
                   global_reach=sorted({global_reach(k, a) for k, a in assign if global_reach(k, a)}),
                   vendor_default_equivalent=sorted({VENDOR_DEFAULT_EQUIVALENT[(k, a)][0] + ': ' + VENDOR_DEFAULT_EQUIVALENT[(k, a)][1]
                                                     for k, a in assign if (k, a) in VENDOR_DEFAULT_EQUIVALENT}))
        candidates.append(cfg)
        return cfg

    unlimited = cap == 0
    factorial_product = 1
    for key in eligible:
        factorial_product *= len(arm_info[key]['arms']) + 1
    factorial_product -= 1
    regime = crosses
    if crosses == 'auto':
        regime = 'factorial' if factorial_product <= full_budget else 'pairwise'
        crosses = 'full' if regime == 'factorial' else 'pairwise'
    if phase1 is not None:
        return plan_phase2(algo, controls, guards, cap, crosses, phase1, eligible, excluded, arm_info, parents, rep,
                           with_parent, add, candidates, reused, invalid)
    for key in eligible:
        for arm in arm_info[key]['arms']:
            add('single', ((key, arm),), 'one control alone')
    singles = list(candidates)
    all_on, dropped_from_all_on = None, []
    if len(eligible) >= 2:
        assign = []
        for key in eligible:
            trial = with_parent(tuple(assign + [(key, rep[key])]))
            problems = check(assignment_defines(trial, controls), guards)
            if problems:
                dropped_from_all_on.append(dict(control=key, arm=rep[key], problems=problems))
            else:
                assign = list(trial)
        if len({k for k, _ in assign}) >= 2:
            before = len(candidates)
            res = add('all_on', tuple(assign), 'one representative arm per control, guard-compatible')
            all_on = candidates[-1] if len(candidates) > before else res
    if regime == 'factorial':
        # Every combination of the eligible controls (singles and all-on above are part of it, emitted once).
        combos = [[(k, a) for k, a in zip(eligible, combo) if a is not None]
                  for combo in itertools.product(*[[None] + arm_info[k]['arms'] for k in eligible])]
        for assign in sorted(combos, key=len):  # stable: lower orders first
            keys = {k for k, _ in assign}
            if len(assign) < 2 or any(k in parents and parents[k][0] not in keys for k in keys):
                continue  # singles are above; a child without its parent set is inert (with_parent adds it)
            add('factorial', tuple(assign), 'full factorial over ' + str(len(eligible)) + ' eligible controls (' +
                str(factorial_product) + ' <= budget ' + str(full_budget) + ')')
        for i, c in enumerate(candidates):
            c['priority'] = i + 1
        return dict(eligible=eligible, excluded=excluded, arm_info=arm_info, parents={k: v[0] for k, v in parents.items()},
                    representative_arms=rep, dropped_from_all_on=dropped_from_all_on,
                    all_on=None if not all_on else (all_on.get('id') or 'duplicate of ' + all_on['duplicate_of']),
                    configs=list(candidates), deferred=[], deferred_groups=[], invalid=invalid, cross_groups=[],
                    regime='factorial', factorial_product=factorial_product)
    cross_groups = []
    for g in (algo['groups'] if crosses == 'full' else []):
        members = [k for k in g if k in eligible]
        if len(members) < 2:
            cross_groups.append(dict(group=g, members=members, configs=[], note='fewer than two eligible members'))
            continue
        options = [[None] + arm_info[k]['arms'] for k in members]
        product = 1
        for o in options:
            product *= len(o)
        configs, dup = [], 0
        mark = len(candidates)
        for combo in itertools.product(*options):
            assign = [(k, a) for k, a in zip(members, combo) if a is not None]
            if len(assign) < 2:
                continue
            keys = {k for k, _ in assign}
            if any(k in parents and parents[k][0] in members and parents[k][0] not in keys for k in keys):
                continue  # child set while its in-group parent is at default: inert
            res = add('cross', tuple(assign), 'interaction group ' + '/'.join(members))
            if res is None:
                continue
            if 'duplicate_of' in res:
                dup += 1
            else:
                configs.append(res)
        del candidates[mark:]  # crosses are admitted below, group-atomically
        cross_groups.append(dict(group=g, members=members, product=product, configs=configs, duplicates=dup))
    ordered = singles + ([all_on] if all_on and 'duplicate_of' not in all_on else [])
    chosen = list(ordered) if unlimited else ordered[:cap]
    deferred = [] if unlimited else [dict(id=c['id'], tier=c['tier'], assignment=c['assignment'], reason='cap ' + str(cap) + ' reached in priority order') for c in ordered[cap:]]
    deferred_groups = []
    if crosses != 'full':
        units = []
        for size, tier in ((2, 'cross'), (3, 'triple')):
            if size == 3 and crosses != 'pairwise+triples':
                continue
            for g in algo['groups']:
                members = [k for k in g if k in eligible]
                cg = subset_crosses(members, {k: arm_info[k]['arms'] for k in members}, size, tier,
                                    'interaction group ' + '/'.join(members) + (' (pair)' if size == 2 else ' (triple)'), add, candidates)
                cross_groups.append(dict(group=g, **cg))
                units.append(cg)
        deferred_groups = admit(units, chosen, cap)
    for cg in (cross_groups if crosses == 'full' else []):
        new = [c for c in cg['configs'] if c['id'] not in {x['id'] for x in chosen}]
        if not new:
            continue
        if unlimited or len(chosen) + len(new) <= cap:
            chosen.extend(new)
        else:
            deferred_groups.append(dict(members=cg['members'], configs=len(new), product=cg['product'],
                                        reason='group cross (' + str(len(new)) + ' configs) does not fit the remaining cap ('
                                        + str(cap - len(chosen)) + '); generate after one-at-a-time results',
                                        listed=[dict(id=c['id'], assignment=c['assignment']) for c in new[:DEFER_LIST_LIMIT]],
                                        listed_truncated=len(new) > DEFER_LIST_LIMIT))
    for i, c in enumerate(chosen):
        c['priority'] = i + 1
    return dict(eligible=eligible, excluded=excluded, arm_info=arm_info, parents={k: v[0] for k, v in parents.items()},
                representative_arms=rep, dropped_from_all_on=dropped_from_all_on,
                all_on=None if not all_on else (all_on.get('id') or 'duplicate of ' + all_on['duplicate_of']),
                configs=chosen, deferred=deferred, deferred_groups=deferred_groups, invalid=invalid,
                cross_groups=[dict(members=cg['members'], product=cg.get('product'), new_configs=len(cg['configs']),
                                   duplicates=cg.get('duplicates', 0), note=cg.get('note'),
                                   **({'kind': cg['kind']} if 'kind' in cg else {})) for cg in cross_groups],
                **({} if regime == 'full' else dict(regime=regime, factorial_product=factorial_product)))


def subset_crosses(members, arms_of, size, tier, origin, add, candidates):
    """Every `size`-subset of members x every non-default arm combination (guards applied by add()).

    A child control carries its parent's representative arm (as its single does). Returns the admission unit."""
    configs, dup, reused, product = [], 0, [], 0
    mark = len(candidates)
    for subset in itertools.combinations(members, size):
        combos = list(itertools.product(*[arms_of[k] for k in subset]))
        product += len(combos)
        for combo in combos:
            res = add(tier, tuple(zip(subset, combo)), origin)
            if res is None:
                continue
            if 'reused' in res:
                reused.append(res['reused'])
            elif 'duplicate_of' in res:
                dup += 1
            else:
                configs.append(res)
    del candidates[mark:]  # admitted by admit(), one config at a time
    return dict(members=list(members), kind=tier, product=product, configs=configs, duplicates=dup, reused=reused,
                note=None if len(members) >= size else 'fewer than %d eligible members' % size)


def admit(units, chosen, cap):
    """Per-config admission in unit order (cap 0 = no cap); returns the deferred remainder per unit."""
    deferred = []
    for u in units:
        ids = {x['id'] for x in chosen}
        new = [c for c in u['configs'] if c['id'] not in ids]
        room = len(new) if cap == 0 else max(0, cap - len(chosen))
        chosen.extend(new[:room])
        left = new[room:]
        if left:
            deferred.append(dict(members=u['members'], kind=u['kind'], configs=len(left), product=u['product'],
                                 reason='%s configs cut by the cap %d (admitted one config at a time in priority order)' % (u['kind'], cap),
                                 listed=[dict(id=c['id'], assignment=c['assignment']) for c in left[:DEFER_LIST_LIMIT]],
                                 listed_truncated=len(left) > DEFER_LIST_LIMIT))
    return deferred


SURVIVING = ('FASTER', 'NEUTRAL')


def plan_phase2(algo, controls, guards, cap, crosses, phase1, eligible, excluded, arm_info, parents, rep,
                with_parent, add, candidates, reused, invalid):
    """Phase 2 from phase-1 decisions: crosses among the (control, arm) singles that survived.

    A (control, arm) survives for this algorithm when its phase-1 single (the same assignment, parents carried)
    has verdict FASTER or NEUTRAL. SLOWER and HOLD_* are dropped; UNMEASURED, IDENTITY_INCOMPLETE and a single
    absent from the decision file are not measured and do not survive. Priority: (a) pairwise crosses among
    survivors inside each interaction group, (b) one all-survivors-on, (c) triples inside groups when
    crosses == 'pairwise+triples', (d) pairwise crosses across groups among FASTER survivors only. A config whose
    assignment phase 1 already measured is listed as reused, never re-emitted."""
    survivors, faster, status = {}, {}, []
    for key in eligible:
        for arm in arm_info[key]['arms']:
            prev = phase1.get(frozenset(with_parent(((key, arm),))))
            verdict = prev['verdict'] if prev else 'NOT_IN_DECISIONS'
            if verdict in SURVIVING:
                state = 'survived'
            elif verdict in ('UNMEASURED', 'NOT_MEASURED', 'IDENTITY_INCOMPLETE', 'NOT_IN_DECISIONS'):
                state = 'not_measured'
            else:
                state = 'dropped'
            entry = dict(control=key, arm=arm, verdict=verdict, state=state,
                         phase1_configuration=prev['configuration'] if prev else None,
                         combined_ratio=prev.get('combined_ratio') if prev else None)
            status.append(entry)
            if state == 'survived':
                survivors.setdefault(key, []).append(arm)
                if verdict == 'FASTER':
                    faster.setdefault(key, []).append(arm)
    for key in list(survivors):
        anc, k = [], key
        while k in parents:
            k = parents[k][0]
            anc.append(k)
        lost = [a for a in anc if rep[a] not in survivors.get(a, [])]
        if lost:
            for e in status:
                if e['control'] == key and e['state'] == 'survived':
                    e['state'] = 'dropped'
                    e['verdict'] += ' (parent ' + lost[0] + '=' + rep[lost[0]] + ' did not survive)'
            survivors.pop(key)
            faster.pop(key, None)
    order = [k for k in eligible if k in survivors]
    units = []
    for g in algo['groups']:
        members = [k for k in g if k in survivors]
        units.append(subset_crosses(members, survivors, 2, 'cross', 'phase 2: survivor pair in group ' + '/'.join(g), add, candidates))
    all_surv, dropped_from_all = None, []
    if len(order) >= 2:
        def best_arm(k):
            rows = {e['arm']: e for e in status if e['control'] == k and e['state'] == 'survived'}
            return min(survivors[k], key=lambda a: (rows[a]['verdict'] != 'FASTER',
                                                    rows[a]['combined_ratio'] if rows[a]['combined_ratio'] else float('inf'),
                                                    survivors[k].index(a)))
        assign = []
        for key in order:
            trial = with_parent(tuple(assign + [(key, best_arm(key))]))
            problems = check(assignment_defines(trial, controls), guards)
            if problems:
                dropped_from_all.append(dict(control=key, arm=best_arm(key), problems=problems))
            else:
                assign = list(trial)
        if len({k for k, _ in assign}) >= 2:
            mark = len(candidates)
            res = add('all_survivors', tuple(assign), 'phase 2: best surviving arm per control (FASTER first, then smallest phase-1 ratio), guard-compatible')
            del candidates[mark:]
            all_surv = res
            units.append(dict(members=order, kind='all_survivors', product=1,
                              configs=[res] if res and 'id' in res else [], duplicates=0, reused=[]))
    if crosses == 'pairwise+triples':
        for g in algo['groups']:
            members = [k for k in g if k in survivors]
            units.append(subset_crosses(members, survivors, 3, 'triple', 'phase 2: survivor triple in group ' + '/'.join(g), add, candidates))
    grouped = {frozenset(p) for g in algo['groups'] for p in itertools.combinations(g, 2)}
    fast_keys = [k for k in eligible if k in faster]
    pairs = [(a, b) for a, b in itertools.combinations(fast_keys, 2) if frozenset((a, b)) not in grouped]
    mark = len(candidates)
    across = dict(members=fast_keys, kind='cross_across', product=0, configs=[], duplicates=0, reused=[])
    for a, b in pairs:
        for arm_a in faster[a]:
            for arm_b in faster[b]:
                across['product'] += 1
                res = add('cross_across', ((a, arm_a), (b, arm_b)), 'phase 2: FASTER survivors across groups')
                if res is None:
                    continue
                if 'reused' in res:
                    across['reused'].append(res['reused'])
                elif 'duplicate_of' in res:
                    across['duplicates'] += 1
                else:
                    across['configs'].append(res)
    del candidates[mark:]
    units.append(across)
    chosen = []
    deferred_groups = admit(units, chosen, cap)
    deferred = []
    for d in [d for d in deferred_groups if d['kind'] == 'all_survivors']:
        deferred.append(dict(id=d['listed'][0]['id'], tier='all_survivors', assignment=d['listed'][0]['assignment'],
                             reason='cap ' + str(cap) + ' reached in priority order'))
        deferred_groups.remove(d)
    for i, c in enumerate(chosen):
        c['priority'] = i + 1
    counts = {}
    for e in status:
        counts[e['state']] = counts.get(e['state'], 0) + 1
    return dict(eligible=eligible, excluded=excluded, arm_info=arm_info, parents={k: v[0] for k, v in parents.items()},
                representative_arms=rep, dropped_from_all_on=dropped_from_all,
                all_on=None if not all_surv else (all_surv.get('id') or ('reused ' + all_surv['reused'] if 'reused' in all_surv else 'duplicate of ' + all_surv['duplicate_of'])),
                configs=chosen, deferred=deferred, deferred_groups=deferred_groups, invalid=invalid,
                cross_groups=[dict(members=u['members'], kind=u['kind'], product=u['product'], new_configs=len(u['configs']),
                                   duplicates=u['duplicates'], reused=len(u['reused']), note=u.get('note')) for u in units],
                regime='phase2', survivors=dict(counts=counts, arms=status, surviving={k: survivors[k] for k in order},
                                                faster={k: faster[k] for k in fast_keys}), reused=reused)


def config_bindings(algo, cfg, controls):
    wb = set(algo['workload_bindings'])
    out = set(wb)
    for key in cfg['controls']:
        cb = set(bindings_of(controls[key]['binding']))
        if not cb & wb:
            out |= cb
    return sorted(b for b in out if (ROOT / binding_path(b)).exists())


# ---------------------------------------------------------------------- packing
def pack(configs, reach, guards):
    """First-fit packing of A-arm define sets across algorithms.

    Two configs share a build only when: their algorithms differ, their binding
    sets are equal, neither carries a declared global-reach arm, every control the
    pack assigns that is not the member's own (control, arm) does not reach the
    member's algorithm, and the union define set passes the guards and harness checks.
    """
    packs = []
    for cfg in sorted(configs, key=lambda c: (c['bindings'], c['algorithm'], c['priority'])):
        placed = None
        if not cfg['global_reach']:
            for p in packs:
                if p['bindings'] != cfg['bindings'] or p['is_global'] or cfg['algorithm'] in p['algorithms']:
                    continue
                ok = True
                for other in p['members']:
                    for (a_cfg, b_cfg) in ((cfg, other), (other, cfg)):
                        for k, arm in a_cfg['assignment'].items():
                            if b_cfg['assignment'].get(k) == arm:
                                continue
                            if b_cfg['algorithm'] in reach.get(k, set()):
                                ok = False
                    if not ok:
                        break
                if not ok:
                    continue
                union = unique(sorted(set(p['defines']) | set(cfg['defines'])))
                if check(union, guards):
                    continue
                placed = p
                break
        if placed is None:
            packs.append(dict(members=[cfg], algorithms={cfg['algorithm']}, bindings=cfg['bindings'],
                              defines=list(cfg['defines']), is_global=bool(cfg['global_reach'])))
        else:
            placed['members'].append(cfg)
            placed['algorithms'].add(cfg['algorithm'])
            placed['defines'] = unique(sorted(set(placed['defines']) | set(cfg['defines'])))
    for i, p in enumerate(packs):
        p['id'] = 'P%03d' % (i + 1)
    return packs


def build_key(path, vendor, defines):
    return sha_value([path, vendor, MODE, defines])[:20]


# ------------------------------------------------------------------- generate
def load_phase1(path):
    """grid-decisions.json (tools/six_lane_grid_decide.py) -> {algorithm: {frozenset(assignment items): row}}."""
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != 'mojolearn.six-lane-grid-decisions/1':
        raise ValueError(str(path) + ': unexpected schema ' + str(doc.get('schema')))
    out = {}
    for row in (doc.get('configurations') or {}).values():
        out.setdefault(row['algorithm'], {})[frozenset((row.get('assignment') or {}).items())] = dict(
            configuration=row['configuration'], verdict=row.get('verdict') or 'UNMEASURED', tier=row.get('tier'),
            combined_ratio=row.get('combined_ratio'))
    return out, doc


def hours(cells, seconds):
    return round(cells * seconds / 3600.0, 2)


def generate(controls_dir=CONTROLS_DIR, guards_path=GUARDS, cap=CAP, crosses='full', full_budget=FULL_BUDGET,
             survivors=None, pair_seconds=None):
    # Default arguments reproduce the committed grid byte for byte; anything else is an extended plan
    # (regimes, deferred total and projected box-hours in the summary and GRID.md).
    extended = crosses != 'full' or survivors is not None or pair_seconds is not None or cap == 0
    pair_seconds = dict(pair_seconds or PAIR_SECONDS)
    phase1_all, phase1_doc = (load_phase1(survivors) if survivors is not None else ({}, None))
    guards = Guards(Path(guards_path).read_text())
    controls, algorithms, removed, owed_misc, files = load_controls(controls_dir)
    for a, b, why in source_exclusions(controls):
        guards.add_pair(a, b, why)
    inv = workload_inventory()
    model, reach = build_model(controls, algorithms, inv)
    plans, unmapped, excluded_all = {}, [], {}
    all_configs = []
    for algo_id, algo in model.items():
        if algo['workloads'] is None:
            unmapped.append(dict(algorithm=algo_id, files=algo['files'], reason=algo['mapping'],
                                 controls=algo['controls']))
        plan = plan_algorithm(algo, controls, reach, guards, cap, crosses=crosses, full_budget=full_budget,
                              phase1=None if survivors is None else phase1_all.get(algo_id, {}))
        plans[algo_id] = plan
        for e in plan['excluded']:
            excluded_all.setdefault(e['control'], dict(control=e['control'], define=e['define'],
                                                       file=controls[e['control']]['file'], reasons=e['reasons'], algorithms=[]))
            excluded_all[e['control']]['algorithms'].append(algo_id)
        if algo['workloads'] is None:
            for c in plan['configs']:
                c['unmapped'] = True
            continue
        for c in plan['configs']:
            c['bindings'] = config_bindings(algo, c, controls)
            c['workloads'] = algo['workloads']
            all_configs.append(c)
    listed = {k for a in model.values() for k in a['controls']}
    for key, c in controls.items():
        if key not in listed:
            excluded_all[key] = dict(control=key, define=c['define'], file=c['file'], algorithms=[],
                                     reasons=control_exclusion(c, reach))
    no_binding = [c['id'] for c in all_configs if not c['bindings']]
    if no_binding:
        raise ValueError('Configs without an existing binding: ' + ', '.join(no_binding[:5]))

    # Builds before packing: one per distinct (binding, define set, vendor), plus B.
    binding_set = sorted({b for c in all_configs for b in c['bindings']})
    before = {(b, tuple(c['defines'])) for c in all_configs for b in c['bindings']}
    packs = pack(all_configs, reach, guards)
    after = {(b, tuple(p['defines'])) for p in packs for b in p['bindings']}
    jobs = {}

    def job(binding, vendor, defines, configuration, arm, pack_id):
        path = binding_path(binding)
        full = unique(sorted(set(defines) | {IDENTICAL_DEFINE}))
        key = build_key(path, vendor, full)
        j = jobs.setdefault(key, dict(key=key, binding=path, vendor=vendor, mode=MODE, defines=full, configurations=[],
                                      status='NOT_COMPILED', pack=pack_id, runtime_reach='NOT_VERIFIED',
                                      target=dict(nvidia='native sm_89 (nv box, --nvidia-target native --nvidia-arch sm_89)',
                                                  amd='native gfx942 (amd box, --accelerator gfx942)')[vendor]))
        entry = dict(configuration=configuration, arm=arm)
        if entry not in j['configurations']:
            j['configurations'].append(entry)
        return key

    for p in packs:
        for cfg in p['members']:
            cfg['pack'] = p['id']
            cfg['A_defines'] = list(p['defines'])
            cfg['packed_extra_defines'] = sorted(set(p['defines']) - set(cfg['defines']))
            cfg['builds'] = {}
            for vendor in VENDORS:
                cfg['builds'][vendor] = dict(A=[job(b, vendor, p['defines'], cfg['id'], 'A', p['id']) for b in cfg['bindings']],
                                             B=[job(b, vendor, [], cfg['id'], 'B', 'B') for b in cfg['bindings']])
    b_builds = {(b, v) for c in all_configs for b in c['bindings'] for v in VENDORS}

    # Matrix (six_lane_ab queue / materialize input).
    configurations, cells, aa = [], [], {}
    for cfg in sorted(all_configs, key=lambda c: (c['algorithm'], c['priority'])):
        configurations.append(dict(
            id=cfg['id'], name=cfg['id'], members=[cfg['algorithm'] + '/' + k for k in cfg['controls']], mode=MODE,
            vendors=list(VENDORS), A=dict(defines=cfg['A_defines'], environment={}, runtime={}),
            B=dict(defines=[], environment={}, runtime={}), problems=[], workloads=cfg['workloads'],
            kind='candidate', campaign_role='new_candidate', experiment_kind='grid_per_algorithm_authored_controls',
            priority=cfg['priority'], parameters={}, source_gaps=[], source_selectable=True, alias_of=None,
            rationale='grid ' + cfg['tier'] + ' (' + cfg['origin'] + ') for ' + cfg['algorithm'],
            grid=dict(algorithm=cfg['algorithm'], tier=cfg['tier'], assignment=cfg['assignment'],
                      effective_defines=cfg['defines'], packed_extra_defines=cfg['packed_extra_defines'],
                      pack=cfg['pack'], bindings=cfg['bindings'], builds=cfg['builds'],
                      vendor_default_equivalent=cfg['vendor_default_equivalent'], global_reach=cfg['global_reach'])))
        for vendor in VENDORS:
            for wid in cfg['workloads']:
                cells.append(dict(
                    key=sha_value([cfg['id'], vendor, wid])[:20], configuration=cfg['id'],
                    implementation_ids=[cfg['algorithm'] + '/' + k for k in cfg['controls']], vendor=vendor, mode=MODE,
                    workload=dict(id=wid, source_workload=wid.split('@dataset=')[0], grid_algorithm=cfg['algorithm']),
                    workload_id=wid, status='PENDING_COVERAGE', blockers=[STANDARD_BLOCKER], source_coverage_pending=[],
                    campaign_role='new_candidate', alias_of=None, promotion_vote=True, identity_group='nvidia-amd-output-hash',
                    planned_excluded_warmups=1, planned_scored_samples=1, actual_samples=0, priority=cfg['priority']))
                aa.setdefault((wid, vendor), dict(workload_id=wid, vendor=vendor, algorithm=cfg['algorithm'],
                                                  B_builds=cfg['builds'][vendor]['B'],
                                                  method='six_lane_timing.aa_job: both arms load the B (shipped default) build; byte-identical artifacts required'))
    matrix = dict(schema='mojolearn.six-lane-matrix/1', base_main=None, configurations=configurations, cells=cells,
                  execution='NOT EXECUTED', qualification=dict(status='GRID_PLANNED_NOT_MEASURED',
                                                               not_exhaustive_full_board=True,
                                                               source_inputs={k: v['sha256'] for k, v in files.items()}))
    build_plan = dict(schema='mojolearn.six-lane-build-plan/1', jobs=sorted(jobs.values(), key=lambda j: (j['vendor'], j['binding'], j['key'])),
                      blocked=[], unsupported=[],
                      policy='Grid builds: packed A define sets (disjoint algorithm reach, guard-checked union) plus one B (shipped default) build per binding x vendor. Compile success is not runtime reach; no import, smoke, quality, timing or identity execution.')
    owed = owed_special(controls, owed_misc)
    summary = dict(
        algorithms=len(model), mapped_algorithms=sum(1 for a in model.values() if a['workloads'] is not None),
        unmapped_algorithms=len(unmapped), algorithms_with_configs=len({c['algorithm'] for c in all_configs}),
        configs=len(all_configs), configs_per_vendor=len(all_configs), cells=len(cells),
        workloads=len({w for c in all_configs for w in c['workloads']}), aa_pairs=len(aa),
        deferred_configs=sum(len(p['deferred']) for p in plans.values()),
        deferred_cross_groups=sum(len(p['deferred_groups']) for p in plans.values()),
        deferred_cross_configs=sum(g['configs'] for p in plans.values() for g in p['deferred_groups']),
        excluded_controls=len(excluded_all), controls=len(controls),
        builds_before_packing=len(before) * len(VENDORS) + len(b_builds),
        builds_after_packing=len(after) * len(VENDORS) + len(b_builds),
        a_builds_before_packing=len(before) * len(VENDORS), a_builds_after_packing=len(after) * len(VENDORS),
        b_builds=len(b_builds), packs=len(packs), packed_multi_member=sum(1 for p in packs if len(p['members']) > 1),
        bindings=binding_set, guard_asserts=guards.assert_count, cap=cap)
    if extended:
        tiers, per_regime = {}, {}
        for c in all_configs:
            tiers[c['tier']] = tiers.get(c['tier'], 0) + 1
        for aid, pl in plans.items():
            r = per_regime.setdefault(pl.get('regime', crosses), dict(algorithms=0, configs=0, cells_per_vendor=0, deferred=0))
            r['algorithms'] += 1
            r['configs'] += len(pl['configs']) if model[aid]['workloads'] is not None else 0
            r['cells_per_vendor'] += len(pl['configs']) * len(model[aid]['workloads'] or [])
            r['deferred'] += len(pl['deferred']) + sum(g['configs'] for g in pl['deferred_groups'])
        for r in per_regime.values():
            r['projected_hours'] = {v: hours(r['cells_per_vendor'], pair_seconds[v]) for v in VENDORS}
        cells_v = len(cells) // len(VENDORS)
        summary.update(
            crosses=crosses, full_budget=full_budget, configs_by_tier=dict(sorted(tiers.items())),
            cells_per_vendor=cells_v, deferred_total=summary['deferred_configs'] + summary['deferred_cross_configs'],
            pair_seconds=pair_seconds,
            projected_hours=dict(ab={v: hours(cells_v, pair_seconds[v]) for v in VENDORS},
                                 aa={v: hours(len(aa) // len(VENDORS), pair_seconds[v]) for v in VENDORS}),
            regimes=dict(sorted(per_regime.items())))
        if survivors is not None:
            states = {}
            for pl in plans.values():
                for k, v in pl['survivors']['counts'].items():
                    states[k] = states.get(k, 0) + v
            summary.update(phase='phase2', survivors_file=str(survivors), survivor_arms=dict(sorted(states.items())),
                           reused_configs=sum(len(pl['reused']) for pl in plans.values()),
                           phase1_verdicts=dict(sorted(_count(r['verdict'] for a in phase1_all.values() for r in a.values()).items())))
    if summary['builds_after_packing'] != len(jobs):
        raise ValueError('Build accounting mismatch: %d jobs vs %d planned' % (len(jobs), summary['builds_after_packing']))
    plan = dict(
        schema='mojolearn.six-lane-grid-plan/1', mode=MODE, status='PLANNED_NOT_RUN',
        all_switches='NOT MEASURED until each config has an A/A floor, scored A/B on NVIDIA and AMD, identity and quality evidence',
        inputs=dict(grid_controls=files, guards=dict(path=str(Path(guards_path).relative_to(ROOT)) if Path(guards_path).is_relative_to(ROOT) else str(guards_path),
                                                       sha256=file_sha(guards_path), asserts=guards.assert_count),
                    stale_not_used=['experiments/six_lane_integration/catalog.json (define lists)',
                                    'experiments/six_lane_integration/workload_ids.json (ids of the retired six-lane matrix; its define lists are not used)']),
        vendors=VENDORS, measurement=dict(excluded_warmups=1, scored_samples=1, rule='owner: one run per arm; Apple does not vote on IDENTICAL'),
        generation_rules=dict(
            cap_per_algorithm_per_vendor=cap,
            B='main shipped defaults: every experiment define absent (plus MOJOLEARN_NUMERIC_IDENTICAL=1)',
            priority=['(a) every eligible control alone: each non-default arm / sweep value (child controls carry their parent representative arm)',
                      '(b) all-on: one representative arm (first reachable non-default) per control, guard-compatible; controls whose arm breaks the guards are dropped and listed',
                      '(c) crosses within each declared interaction group, admitted group-atomically only if the whole group fits the remaining cap'],
            deferred='anything cut by the cap is DEFERRED: to be generated after one-at-a-time results, never dropped',
            exclusions='controls marked off-board/unreached/peripheral-only/blocked-quality/grid:exclude/owed-removal, or not declared by any board algorithm, are EXCLUDED with the quoted reason',
            guards='every A define set (single, all-on, cross, packed union) is evaluated against core/six_lane_experiment_guards.mojo and six_lane_ab.combine()',
            packing='configs from different algorithms share one A build when binding sets are equal, no assigned control reaches another member algorithm (reach = algorithms that declare the control), no member carries a declared global-reach arm, and the union passes the guards',
            arm_exclusions=[dict(control=k, arm=a, scope=s or 'all', reason=w) for k, a, s, w in ARM_EXCLUSIONS],
            global_reach=[dict(control=k, arm=a, reason=w) for (k, a), w in GLOBAL_REACH.items()],
            vendor_default_equivalent=[dict(control=k, arm=a, vendor=v[0], reason=v[1]) for (k, a), v in VENDOR_DEFAULT_EQUIVALENT.items()],
            parents=dict(explicit={k: v for k, v in EXPLICIT_PARENTS.items()}, patterns=PARENT_PATTERNS),
            workload_mapping='grid workload -> saved harness ids: classical/more/expanded X -> X@dataset=<taxi|istella> (registered full-input variant replaces a capped original); trees:L -> L:<every saved dataset>; neural:L -> neural:L (bench_board_neural LANES); aliases ' + json.dumps({k: v[0] for k, v in ALIASES.items()}) + '. Unmapped workloads are reported, never substituted.'),
        verdict_rule=dict(
            timing='tools/six_lane_timing.py verdicts: FASTER/SLOWER only when |log(A/B)| of the scored clock exceeds that box\'s A/A floor for the same workload on BOTH NVIDIA and AMD and both agree in direction; otherwise NO_VERDICT',
            identity='NVIDIA output hash == AMD output hash for the same config (owner 2026-10-07: identity across the two GPU vendors only); tools/six_lane_compare_results.py',
            quality='the board quality metric must not drop vs B on either vendor',
            promotion='combined faster, neither vendor materially slower, identity and quality preserved; then flip the switch in source with the evidence beside it',
            default_state='every switch is NOT MEASURED until then'),
        summary=summary,
        algorithms={aid: dict(
            files=model[aid]['files'], raw_ids=model[aid]['raw_ids'], workloads=model[aid]['workloads'],
            mapping_notes=model[aid]['mapping_notes'], unmapped_reason=model[aid]['mapping'],
            workload_bindings=model[aid]['workload_bindings'], declared_controls=model[aid]['controls'],
            eligible_controls=p['eligible'], representative_arms=p['representative_arms'], parents=p['parents'],
            dropped_arms={k: v['dropped_arms'] for k, v in p['arm_info'].items() if v['dropped_arms']},
            excluded=p['excluded'], interaction_groups=model[aid]['groups'],
            unresolved_interactions=model[aid]['unresolved_interactions'], cross_groups=p['cross_groups'],
            all_on=p['all_on'], dropped_from_all_on=p['dropped_from_all_on'],
            configs=[dict(id=c['id'], priority=c['priority'], tier=c['tier'], assignment=c['assignment'], defines=c['defines'],
                          pack=c.get('pack'), bindings=c.get('bindings'), packed_extra_defines=c.get('packed_extra_defines', []),
                          global_reach=c['global_reach'], vendor_default_equivalent=c['vendor_default_equivalent'],
                          queued=not c.get('unmapped')) for c in p['configs']],
            deferred=p['deferred'], deferred_groups=p['deferred_groups'], invalid=summarize_invalid(p['invalid']),
            notes=model[aid]['notes'],
            **({k: p[k] for k in ('regime', 'factorial_product', 'survivors', 'reused') if k in p}))
            for aid, p in plans.items()},
        unmapped=unmapped, excluded=sorted(excluded_all.values(), key=lambda e: (e['file'], e['control'])),
        removed_per_file=removed,
        aa_pairs=sorted(aa.values(), key=lambda a: (a['algorithm'], a['workload_id'], a['vendor'])),
        builds=dict(before_packing=summary['builds_before_packing'], after_packing=summary['builds_after_packing'],
                    packs=[dict(id=p['id'], bindings=p['bindings'], defines=p['defines'], members=[m['id'] for m in p['members']],
                                algorithms=sorted(p['algorithms']), global_reach=p['is_global']) for p in packs]),
        owed_special_ab=owed)
    if extended:
        plan['generation_rules']['crosses'] = dict(
            mode=crosses, full_budget=full_budget, cap='0 = no cap' if cap == 0 else cap,
            full='within-group products admitted group-atomically (the committed grid)',
            pairwise='every pair of interaction-group members x every non-default arm combination; admitted one config at a time, so the cap cuts configs, not whole groups',
            triples='pairwise+triples: all pairs first, then every triple of group members',
            auto='per algorithm: the whole factorial over its eligible controls (tier factorial; singles/all-on emitted once with their own tier) when product(arms + 1) - 1 <= full_budget, no cap; otherwise singles + all-on + pairwise crosses (higher orders come from --survivors phase 2)',
            projection='projected box-hours = cells per vendor x median scored-pair seconds (--pair-seconds)')
        if survivors is not None:
            plan['status'] = 'PHASE2_PLANNED_NOT_RUN'
            plan['inputs']['survivors'] = dict(path=str(survivors), sha256=file_sha(survivors),
                                               schema=phase1_doc.get('schema'), mode=phase1_doc.get('mode'))
            plan['generation_rules']['phase2'] = plan_phase2.__doc__.strip()
    return plan, matrix, build_plan


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def summarize_invalid(invalid, examples=3):
    """Guard-refused combinations, grouped by the first refusal (counts + a few examples)."""
    groups = {}
    for item in invalid:
        g = groups.setdefault(item['problems'][0], dict(problem=item['problems'][0], count=0, tiers=[], examples=[]))
        g['count'] += 1
        if item['tier'] not in g['tiers']:
            g['tiers'].append(item['tier'])
        if len(g['examples']) < examples:
            g['examples'].append(item['assignment'])
    return sorted(groups.values(), key=lambda g: -g['count'])


def owed_special(controls, owed_misc):
    out = []
    for item in owed_misc:
        out.append(dict(define=item['define'], file=item['file'], source=item.get('source'), A=item.get('A'), B=item.get('B'),
                        vendors=item.get('vendors'), shapes=item.get('shapes'), effect=item.get('effect'), status=item.get('status')))
    for key, c in sorted(controls.items()):
        if c.get('owed_removal'):
            vendors = [v for v, tok in (('nvidia', '_NV_'), ('amd', '_AMD_'), ('apple', '_APPLE_')) if tok in c['define'] + '_']
            out.append(dict(define=c['define'], file=c['file'], control=key, source=c.get('source'), vendors=vendors or ['nvidia', 'amd'],
                            A='main (rule removed / define absent)', B=[d for ds in c['arms'].values() for d in ds],
                            reason=c['owed_removal'].get('reason'), ab=c['owed_removal'].get('ab'),
                            status='RUN OWED: neighboring shapes + one non-board dataset; not expressible as a grid cell'))
    return out


# ---------------------------------------------------------------------- GRID.md
FAMILY_ORDER = ['neural', 'gemm', 'classical', 'trees']


def family_of(algo_id, files):
    if algo_id.startswith('trees:'):
        return 'trees'
    if algo_id.startswith('gemm:'):
        return 'gemm'
    if algo_id.startswith('neural:'):
        return 'neural'
    return 'classical'


def render_md(plan):
    s = plan['summary']
    ext = 'crosses' in s
    gdir = plan.get('outputs_dir', 'experiments/six_lane_integration/grid')
    L = ['# IDENTICAL switch grid' + (', phase 2 from phase-1 survivors' if s.get('phase') == 'phase2' else '') + ' (planned, not run)', '',
         'Generated by `tools/six_lane_grid.py` from `experiments/six_lane_integration/grid_controls/*.json` and',
         '`core/six_lane_experiment_guards.mojo`. The stale `catalog.json`/`matrix.json.gz` define lists are not used.',
         '**Every switch is NOT MEASURED** until it has an A/A floor, a scored A/B on NVIDIA and AMD, identity and quality evidence.', '',
         '## Totals', '',
         '| item | count |', '|---|---|',
         '| algorithms (grid_controls workloads, merged across files) | %d |' % s['algorithms'],
         '| algorithms with configs to queue | %d |' % s['algorithms_with_configs'],
         '| unmapped algorithms | %d |' % s['unmapped_algorithms'],
         '| configs per vendor (NVIDIA = AMD) | %d |' % s['configs'],
         '| saved full workloads touched | %d |' % s['workloads'],
         '| A/B cells (configs x workloads x 2 vendors) | %d |' % s['cells'],
         '| A/A pairs (one per workload per vendor) | %d |' % s['aa_pairs'],
         '| deferred configs (singles/all-on cut by the cap) | %d |' % s['deferred_configs'],
         '| deferred interaction-group crosses | %d groups, %d configs |' % (s['deferred_cross_groups'], s['deferred_cross_configs']),
         '| excluded controls | %d of %d |' % (s['excluded_controls'], s['controls']),
         '| builds before packing (A + B) | %d |' % s['builds_before_packing'],
         '| builds after packing (A + B) | %d (%d A packs, %d with >1 member; %d B) |' % (s['builds_after_packing'], s['packs'], s['packed_multi_member'], s['b_builds']),
         ] + (extended_totals(s) if ext else []) + [
         '', ('Cap: none (--cap 0).' if s['cap'] == 0 else 'Cap: %d configs per algorithm per vendor.' % s['cap']) + ' One excluded warmup + one scored sample per arm. Vendors: NVIDIA (native sm_89, nv box) and AMD (gfx942, amd box); Apple does not vote on IDENTICAL.', '',
         '## Verdict rule', '']
    for k in ('timing', 'identity', 'quality', 'promotion', 'default_state'):
        L.append('- **' + k + '**: ' + plan['verdict_rule'][k])
    L += ['', '## Per-family tables', '',
          'configs: priority order (`s` single, `all` all-on, `x` cross); `+N` = deferred singles/all-on; groups = deferred crosses.', '']
    fams = {}
    for aid, a in plan['algorithms'].items():
        fams.setdefault(family_of(aid, a['files']), []).append((aid, a))
    for fam in FAMILY_ORDER:
        if fam not in fams:
            continue
        L += ['### ' + fam, '', '| algorithm | #configs | configs | deferred | builds |', '|---|---|---|---|---|']
        for aid, a in sorted(fams[fam]):
            short = {'single': 's', 'all_on': 'all', 'cross': 'x', 'triple': 'x3', 'factorial': 'f', 'all_survivors': 'all*', 'cross_across': 'xa'}
            shown = a['configs'] if not ext or len(a['configs']) <= MD_CONFIG_LIMIT else a['configs'][:MD_CONFIG_LIMIT]
            cfgs = '<br>'.join(short[c['tier']] + ' ' + (('all-on: ' + ', '.join(k + '=' + v for k, v in c['assignment'].items())) if c['tier'] == 'all_on' else ', '.join(k + '=' + v for k, v in c['assignment'].items())) for c in shown) or '-'
            if len(shown) < len(a['configs']):
                cfgs += '<br>... +%d more (grid-plan.json)' % (len(a['configs']) - len(shown))
            dparts = []
            if a['deferred']:
                dparts.append('+%d: ' % len(a['deferred']) + ', '.join(d['id'].split('.', 2)[-1] for d in a['deferred'][:12]) + (' ...' if len(a['deferred']) > 12 else ''))
            for g in a['deferred_groups'][:12] if ext else a['deferred_groups']:
                dparts.append(g.get('kind', 'cross') + ' ' + '/'.join(g['members']) + ': %d' % g['configs'])
            if ext and len(a['deferred_groups']) > 12:
                dparts.append('... +%d groups' % (len(a['deferred_groups']) - 12))
            if a['unmapped_reason']:
                status = 'UNMAPPED: ' + a['unmapped_reason']
                builds = '-'
            else:
                status = ''
                packs = sorted({c['pack'] for c in a['configs'] if c['pack']})
                builds = ('%d packs x %d bindings x 2 vendors' % (len(packs), len(a['configs'][0]['bindings'])) if a['configs'] else '-')
            n = len(a['configs'])
            L.append('| %s | %d | %s | %s | %s |' % (aid + (' (' + status + ')' if status else ''), n, cfgs,
                                                    '<br>'.join(dparts) or '-', builds))
        L.append('')
    if ext:
        L += regime_section(plan)
    L += ['## Unmapped workloads', '', '| algorithm | reason | controls |', '|---|---|---|']
    for u in plan['unmapped']:
        L.append('| %s | %s | %s |' % (u['algorithm'], u['reason'], ', '.join(u['controls']) or '-'))
    L += ['', '## Excluded controls', '', '| control | define | file | reason |', '|---|---|---|---|']
    for e in plan['excluded']:
        L.append('| %s | `%s` | %s | %s |' % (e['control'], e['define'], e['file'], '; '.join(e['reasons']).replace('|', '/')))
    L += ['', '## Owed special A/Bs the grid cannot express', '',
          'Legacy exact-shape rule removals: B = the old rule, A = main; timed on neighboring shapes plus one non-board dataset (AGENTS.md "No dimension targeting").', '',
          '| define | file | vendors | B | shapes / plan | status |', '|---|---|---|---|---|---|']
    for o in plan['owed_special_ab']:
        B = o['B'] if isinstance(o['B'], str) else ', '.join(o['B'] or [])
        vend = ', '.join(v if v != 'apple' else 'apple (Apple does not vote on IDENTICAL)' for v in (o.get('vendors') or ['nvidia', 'amd']))
        L.append('| `%s` | %s | %s | %s | %s | %s |' % (o['define'], o['file'], vend,
                                                    B, (o.get('shapes') or o.get('ab') or '') + ((' Reason: ' + o['reason']) if o.get('reason') else ''), o.get('status') or ''))
    L += ['', '## How to queue (not run by this lane)', '',
          'Files: `grid-matrix.json.gz` (configurations + cells, `mojolearn.six-lane-matrix/1`), `grid-build-plan.json`',
          '(`mojolearn.six-lane-build-plan/1`), `grid-plan.json` (everything above, machine-readable).', '',
          '1. Freeze one commit; compile once per vendor on cheap build boxes (orchestrator only):', '', '```bash']
    for v, info in VENDORS.items():
        L.append('python3 tools/six_lane_ab.py compile --plan ' + gdir + '/grid-build-plan.json \\')
        L.append('  %s --compiler <mojo> --output <evidence>/grid-build-%s --keep-going' % (info['compile_flags'], v))
    L += ['```', '', '2. Materialize recipes from the saved full-workload facts and deployed receipts (per vendor):', '', '```bash']
    for v, info in VENDORS.items():
        L.append('python3 tools/six_lane_materialize.py --matrix ' + gdir + '/grid-matrix.json.gz \\')
        L.append('  --workloads <saved-workload-facts.json> --deployments <deployments-%s.json> --vendor %s --target-track %s --output <evidence>/grid-recipes-%s' % (v, v, info['target_track'], v))
    L += ['```', '', '   `@input=` variant cells go through the existing full-input transfer (`tools/six_lane_targeted_variants.py`).', '',
          '3. Write the A/B queue and its A/A queue (one pair per workload per box; both arms the B build):', '', '```bash']
    for v in VENDORS:
        L.append(('python3 tools/six_lane_ab.py queue --vendor %s --matrix ' + gdir + '/grid-matrix.json.gz \\') % v)
        L.append('  --recipes <evidence>/grid-recipes-%s/recipes.json --output <evidence>/grid-queue-%s.json' % (v, v))
        L.append('python3 tools/six_lane_ab.py aa-queue --queue <evidence>/grid-queue-%s.json --output <evidence>/grid-aa-%s.json' % (v, v))
    L += ['```', '', '4. Run A/A first, then A/B, serially per box, NVIDIA and AMD in parallel (orchestrator, after recorded authorization):', '', '```bash']
    for v in VENDORS:
        L.append('python3 tools/performance_full_ab_queue.py --config <evidence>/grid-aa-%s.json --output <evidence>/grid-aa-%s-results' % (v, v))
        L.append('python3 tools/performance_full_ab_queue.py --config <evidence>/grid-queue-%s.json --output <evidence>/grid-ab-%s-results' % (v, v))
    L += ['```', '', '5. Floors, verdicts and NVIDIA==AMD identity:', '', '```bash',
          'python3 tools/six_lane_timing.py floors <evidence>/grid-aa-nvidia-results <evidence>/grid-aa-amd-results --out <evidence>/grid-floors.json',
          'python3 tools/six_lane_timing.py verdicts <evidence>/grid-ab-nvidia-results <evidence>/grid-ab-amd-results --floors <evidence>/grid-floors.json --out <evidence>/grid-verdicts.json',
          'python3 tools/six_lane_compare_results.py --manifest <results-manifest.json> --aa-floors <evidence>/grid-floors.json --out <evidence>/grid-compare',
          '```', '',
          '6. One decision per switch arm (PROMOTE / SPLIT per algorithm / DELETE / HOLD / NOT_MEASURED), interactions and the',
          '   recommended configuration per algorithm: `GRID_DECISIONS.md` + `grid-decisions.json` beside this file.', '', '```bash',
          'python3 tools/six_lane_grid_decide.py --verdicts <evidence>/grid-verdicts.json --identity <evidence>/grid-compare/summary.json \\',
          '  --quality <evidence>/grid-quality-review.json',
          '```', '',
          'Regenerate: `python3 tools/six_lane_grid.py` (deterministic; `--check` fails if the committed outputs are stale).', '']
    return '\n'.join(L)


MD_CONFIG_LIMIT = 24  # extended plans only: configs listed per algorithm row (the full list is in grid-plan.json)


def extended_totals(s):
    hv = lambda d: ', '.join('%s %.1f h' % (v, d[v]) for v in VENDORS)
    L = ['| crosses mode | %s%s |' % (s['crosses'], (' (full budget %d)' % s['full_budget']) if s['crosses'] == 'auto' else ''),
         '| configs by tier | %s |' % ', '.join('%s %d' % kv for kv in s['configs_by_tier'].items()),
         '| A/B cells per vendor | %d |' % s['cells_per_vendor'],
         '| deferred by the cap (all kinds, configs) | %d |' % s['deferred_total'],
         '| median scored-pair seconds | %s |' % ', '.join('%s %g s' % (v, s['pair_seconds'][v]) for v in VENDORS),
         '| projected A/B box-hours | %s |' % hv(s['projected_hours']['ab']),
         '| projected A/A box-hours | %s |' % hv(s['projected_hours']['aa'])]
    for name, r in s['regimes'].items():
        L.append('| regime %s | %d algorithms, %d configs, %d cells per vendor, %d deferred; %s |' % (
            name, r['algorithms'], r['configs'], r['cells_per_vendor'], r['deferred'], hv(r['projected_hours'])))
    if s.get('phase') == 'phase2':
        L += ['| phase-1 decisions | `%s` |' % s['survivors_file'],
              '| phase-1 verdicts (configs) | %s |' % (', '.join('%s %d' % kv for kv in s['phase1_verdicts'].items()) or 'none'),
              '| phase-1 single arms | %s |' % (', '.join('%s %d' % kv for kv in s['survivor_arms'].items()) or 'none'),
              '| reused phase-1 configs (not re-emitted) | %d |' % s['reused_configs']]
    return L


def regime_section(plan):
    s = plan['summary']
    ps = s['pair_seconds']
    L = ['## Regimes per algorithm', '',
         'factorial = every combination of the eligible controls (no cap); pairwise = singles + all-on + pairs inside each group '
         '(higher orders from `--survivors`); phase2 = crosses among phase-1 survivors. Hours = cells per vendor x median pair seconds.', '',
         '| algorithm | regime | eligible controls | full factorial | configs | deferred | cells per vendor | ' + ' | '.join(v + ' h' for v in VENDORS) + ' |',
         '|---|---|---|---|---|---|---|' + '---|' * len(VENDORS)]
    for aid, a in sorted(plan['algorithms'].items()):
        n = len(a['configs'])
        cells = n * len(a['workloads'] or [])
        deferred = len(a['deferred']) + sum(g['configs'] for g in a['deferred_groups'])
        L.append('| %s | %s | %d | %s | %d | %d | %d | %s |' % (
            aid + (' (unmapped)' if a['unmapped_reason'] else ''), a.get('regime', s['crosses']), len(a['eligible_controls']),
            a.get('factorial_product', '-'), n, deferred, cells, ' | '.join('%.2f' % hours(cells, ps[v]) for v in VENDORS)))
    L.append('')
    if s.get('phase') == 'phase2':
        L += ['## Phase-1 survivors', '',
              'A (control, arm) survives when its phase-1 single is FASTER or NEUTRAL; SLOWER and HOLD_* are dropped; '
              'UNMEASURED / IDENTITY_INCOMPLETE / absent from the decision file are not measured and do not survive.', '',
              '| algorithm | survived | FASTER (cross-group eligible) | dropped | not measured | reused |', '|---|---|---|---|---|---|']
        for aid, a in sorted(plan['algorithms'].items()):
            sv = a.get('survivors') or {}
            arms = sv.get('arms', [])
            fmt = lambda st: ', '.join('%s=%s (%s)' % (e['control'], e['arm'], e['verdict']) for e in arms if e['state'] == st) or '-'
            nm = [e for e in arms if e['state'] == 'not_measured']
            nm_s = ('%d: ' % len(nm) + ', '.join(sorted({e['verdict'] for e in nm}))) if nm else '-'
            L.append('| %s | %s | %s | %s | %s | %d |' % (
                aid, ', '.join('%s=%s' % (k, '|'.join(v)) for k, v in (sv.get('surviving') or {}).items()) or 'none',
                ', '.join('%s=%s' % (k, '|'.join(v)) for k, v in (sv.get('faster') or {}).items()) or '-',
                fmt('dropped'), nm_s, len(a.get('reused') or [])))
        L.append('')
    return L


def write_outputs(plan, matrix, build_plan, out_dir=OUT_DIR):
    from six_lane_matrix_io import write_matrix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'grid-plan.json').write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    (out_dir / 'grid-build-plan.json').write_text(json.dumps(build_plan, indent=1, sort_keys=True) + '\n')
    write_matrix(out_dir / 'grid-matrix.json.gz', matrix)
    (out_dir / 'GRID.md').write_text(render_md(plan))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=OUT_DIR)
    p.add_argument('--cap', type=int, default=CAP)
    p.add_argument('--check', action='store_true', help='fail if committed outputs differ from a fresh generation')
    p.add_argument('--mode', choices=('identical', 'fast'), default='identical',
                   help='identical (default): NVIDIA + AMD IDENTICAL grid; fast: Apple FAST trees + classical grid (tools/six_lane_grid_fast.py)')
    p.add_argument('--vendor', choices=('apple',), help='FAST vendor (fast mode only; apple = M3 Ultra, Metal)')
    p.add_argument('--queue-branch', help='fast mode: the lane/apple-fast* branch the M3 queue lines name')
    p.add_argument('--crosses', choices=('full', 'pairwise', 'pairwise+triples', 'auto'),
                   help='full (default; the committed grid): group products admitted group-atomically; pairwise: every pair of '
                        'group members x arm combinations, admitted per config; pairwise+triples: pairs then triples; auto: the '
                        'whole factorial when product(arms + 1) - 1 <= --full-budget, else pairwise (default with --survivors: pairwise)')
    p.add_argument('--full-budget', type=int, default=FULL_BUDGET, help='--crosses auto: largest full factorial emitted whole (default %d)' % FULL_BUDGET)
    p.add_argument('--survivors', type=Path, help='phase 2: grid-decisions.json from tools/six_lane_grid_decide.py; crosses among '
                                                   'the FASTER/NEUTRAL phase-1 singles (default --out ' + str(PHASE2_DIR.relative_to(ROOT)) + ')')
    p.add_argument('--pair-seconds', help='median scored-pair seconds per vendor for the projected box-hours, e.g. nvidia=38,amd=24')
    args = p.parse_args(argv)
    if args.mode == 'fast':
        if args.crosses or args.survivors or args.pair_seconds or args.full_budget != FULL_BUDGET:
            p.error('--crosses / --survivors / --pair-seconds / --full-budget apply to the IDENTICAL grid only')
        import six_lane_grid_fast as F
        if args.vendor not in (None, 'apple'):
            p.error('--mode fast supports --vendor apple only')
        branch = args.queue_branch or F.QUEUE_BRANCH
        if not branch.startswith('lane/apple-fast'):
            p.error('only lane/apple-fast* branches may target m3')
        return F.main_fast(out=args.out if args.out != OUT_DIR else F.OUT_DIR, cap=args.cap, check=args.check, branch=branch)
    if args.vendor or args.queue_branch:
        p.error('--vendor / --queue-branch apply to --mode fast only')
    if args.cap < 0:
        p.error('--cap must be >= 0 (0 = no cap)')
    crosses = args.crosses or ('pairwise' if args.survivors else 'full')
    if args.survivors and crosses == 'full':
        p.error('--survivors plans pairwise crosses (optionally pairwise+triples); --crosses full does not apply')
    pair_seconds = None
    if args.pair_seconds:
        pair_seconds = dict(PAIR_SECONDS)
        for part in args.pair_seconds.split(','):
            v, _, sec = part.partition('=')
            if v not in VENDORS:
                p.error('--pair-seconds: unknown vendor ' + v)
            pair_seconds[v] = float(sec)
    if args.survivors and args.out == OUT_DIR:
        args.out = PHASE2_DIR
    plan, matrix, build_plan = generate(cap=args.cap, crosses=crosses, full_budget=args.full_budget,
                                        survivors=args.survivors, pair_seconds=pair_seconds)
    if 'crosses' in plan['summary']:
        out = Path(args.out).resolve()
        plan['outputs_dir'] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(args.out)
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(plan, matrix, build_plan, tmp)
            stale = [n for n in ('grid-plan.json', 'grid-build-plan.json', 'grid-matrix.json.gz', 'GRID.md')
                     if not (args.out / n).exists() or (Path(tmp) / n).read_bytes() != (args.out / n).read_bytes()]
        print(json.dumps(dict(stale=stale)))
        return 1 if stale else 0
    write_outputs(plan, matrix, build_plan, args.out)
    print(json.dumps(plan['summary']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
