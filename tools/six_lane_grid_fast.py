#!/usr/bin/env python3
"""FAST-mode (Apple, M3 Ultra, Metal) switch grid for TREES and CLASSICAL (planning only).

`python3 tools/six_lane_grid.py --mode fast --vendor apple` lands here. The planner core of
tools/six_lane_grid.py (singles, one guard-compatible all-on, interaction-group crosses admitted
group-atomically under the cap, first-fit packing of disjoint-reach define sets) is reused unchanged;
this module only supplies the FAST control source, the Apple vendor, the FAST verdict rule and the
M3 queue lines.

Control source (opt-in FAST candidates only; promoted `_OFF` arms are never gridded: each was A/B'd
on top of the defaults current at its promotion, so the shipped FAST default is the all-on point the
M3 board already measures on every row):
  experiments/apple_fast_trees/{F,G,N,P}.json          48 MOJOLEARN_AFT_* tree cards
  tools/apple_fast_tree_ideas.py INTERACTIONS           tree interaction groups X01-X10 (X11/X12 empty)
  experiments/apple_fast_classical_20261006/            54 MOJOLEARN_AFCL_* classical cards
     ideas.json, lanes/*.json (arms), full_workloads.json (recipes), build_bindings.json (bindings)
  experiments/six_lane_integration/grid-fast/controls/  authored rows: the EXPERIMENTS.md untried
                                                         candidates and their three small groups
Workloads: the lane x dataset rows of the M3 FAST board (docs/apple-fast/BOARD_M3_FAST.md, first
table). A card lane without a board row is reported UNMAPPED, never substituted.

Arms follow the Apple A/B scripts: A = FAST main (plus any prerequisite the card keeps in both arms),
B = the candidate. Builds are FAST builds (MOJOLEARN_NUMERIC_MODE=fast through the scripts); no
MOJOLEARN_NUMERIC_IDENTICAL define anywhere. FAST needs no identical anything: there is no identity
column; the verdict is speed beyond the M3 A/A floor with the board quality metric held.

Outputs (experiments/six_lane_integration/grid-fast/): GRID.md, grid-plan.json, grid-matrix.json.gz,
grid-build-plan.json, grid-fast-queue.txt. Planning only: nothing is imported from an estimator,
compiled, queued or measured.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import six_lane_grid as G  # noqa: E402

MODE = 'fast'
OUT_DIR = ROOT / 'experiments/six_lane_integration/grid-fast'
CONTROLS_DIR = OUT_DIR / 'controls'
AFT_DIR = ROOT / 'experiments/apple_fast_trees'
AFCL_DIR = ROOT / 'experiments/apple_fast_classical_20261006'
BOARD_MD = ROOT / 'docs/apple-fast/BOARD_M3_FAST.md'
QUEUE_BRANCH = 'lane/apple-fast-round2'  # lane R8 runs on the M3; lq refuses m3 for any branch not lane/apple-fast*
OUTPUTS = ('grid-plan.json', 'grid-build-plan.json', 'grid-matrix.json.gz', 'GRID.md', 'grid-fast-queue.txt')
VENDORS = {
    'apple': dict(box='m3 (M3 Ultra, Metal; the Apple FAST peer\'s box)', target_track='apple-metal-fast',
                  compile_flags='MOJOLEARN_NUMERIC_MODE=fast through tools/aft_ab.sh / tools/afc_ab_def.sh on the box (no MOJOLEARN_NUMERIC_IDENTICAL)'),
}

# Lane -> the binding its board route loads. Source: tools/afc_ab_def.sh `auto` (gbdt-* -> gbdt, rf -> rf,
# et -> trees, iforest -> svm) plus the binding of every afc_ab_def.sh line already queued for that lane
# (docs/apple-fast, ~/mojolearn-evidence/lq history, 2026-10-07; `calibrated` used two bindings and is left out).
LANE_BINDING = {
    'rf': 'rf', 'et': 'trees', 'iforest': 'svm',
    'adaboost-clf': 'x_trees', 'adaboost-reg': 'x_trees', 'adafactor': 'x_sequence', 'adagrad': 'x_sequence', 
    'adamax': 'x_sequence', 'additive-chi2': 'x_neighbors', 'affinity-prop': 'x_cluster', 'ard': 'x_linear', 
    'autoarima': 'arima', 'bayesian-gmm': 'x_cluster', 'bayesian-ridge': 'x_linear', 'bisecting-kmeans': 
    'x_cluster', 'cagra': 'x_ann', 'categorical-nb': 'x_prep', 'complement-nb': 'x_prep', 'connected-components': 
    'x_neighbors', 'croston': 'x_sequence', 'damped-ets': 'x_sequence', 'dbscan': 'estimators', 'dict-learning': 
    'x_decomp', 'elasticnet': 'solver', 'elliptic-envelope': 'x_decomp', 'garch': 'x_sequence', 'gmm': 'mixture', 
    'gpc': 'gp', 'gpr': 'gp', 'hdbscan': 'hdbscan', 'isomap': 'x_decomp', 'iterative-imputer': 'x_prep', 'ivf': 
    'ivf', 'ivf-filter': 'x_ann', 'ivf-pq': 'x_ann', 'ivf-rabitq': 'x_ann', 'ivf-refine': 'x_ann', 'ivf-sq': 
    'x_ann', 'kde': 'estimators', 'kernel-pca': 'x_neighbors', 'kmeans': 'base', 'knn': 'base', 'kpss': 'tsa', 
    'label-propagation': 'x_neighbors', 'lars': 'x_linear', 'lasso': 'solver', 'lasso-lars': 'x_linear', 
    'layernorm': 'x_sequence', 'lda': 'x_decomp', 'lda-clf': 'x_prep', 'linearsvc': 'estimators', 'linearsvr': 
    'estimators', 'lof': 'x_neighbors', 'logreg': 'estimators', 'louvain': 'x_neighbors', 'lstm-clf': 'x_sequence', 
    'lstm-reg': 'x_sequence', 'mb-dict-learning': 'x_decomp', 'mb-sparse-pca': 'x_decomp', 'meanshift': 
    'x_cluster', 'minibatch-kmeans': 'x_cluster', 'minmax-scaler': 'preprocessing', 'multinomial-nb': 'x_prep', 
    'multioutput-clf': 'x_trees', 'multioutput-reg': 'estimators', 'nadam': 'x_sequence', 'nearest-centroid': 
    'x_neighbors', 'nystroem': 'kernel_methods', 'ols': 'estimators', 'optics': 'x_cluster', 'ovr': 'x_trees', 
    'pagerank': 'x_neighbors', 'poly-count-sketch': 'x_neighbors', 'power-transformer': 'x_prep', 'qda': 'x_prep', 
    'ridge': 'estimators', 'ridge-clf': 'x_linear', 'ridge-cv': 'x_linear', 'rmsprop': 'x_sequence', 'select-d': 
    'tsa', 'select-f-classif': 'x_prep', 'select-f-regression': 'x_prep', 'select-mutual-info': 'x_prep', 
    'select-mutual-info-reg': 'x_prep', 'select-r-regression': 'x_prep', 'simple-imputer': 'x_prep', 'skewed-chi2': 
    'x_neighbors', 'sparse-pca': 'x_decomp', 'sparse-rp': 'x_neighbors', 'stacking-clf': 'x_trees', 'stacking-reg': 
    'x_trees', 'stl': 'x_sequence', 'svgp': 'x_neighbors', 'theta': 'x_sequence', 'tsne': 'x_ann', 'var': 
    'x_sequence', }
AFT_BINDINGS = ('base', 'rf', 'gbdt', 'trees', 'svm')  # what tools/aft_ab.sh builds
AFC_FAMILY = {'algos': 'algos', 'classical2': 'classical2', 'classical': 'classical', 'trees': 'trees'}
# Workloads the source says cannot be timed at all.
WORKLOAD_EXCLUSIONS = {
    'classical:dbscan@dataset=taxi': 'both arms time out on taxi (EXPERIMENTS.md untried row DBSCAN_FAST_CC_BATCH: "A/B on dbscan istella")',
}
DROP_ENV = {'MOJOLEARN_NUMERIC_MODE'}  # implied: the scripts build FAST and race the ours-fast arm
REGISTRY_FILE = re.compile(r'experiments?\.mojo$')


def binding_name(b):
    b = b.removeprefix('mojolearn.').removeprefix('_mojolearn_')
    return 'base' if b in ('core', '_mojolearn', '') else b


def binding_file(b):
    return ROOT / 'bindings' / ('_mojolearn.mojo' if b == 'base' else '_mojolearn_' + b + '.mojo')


def so_path(b):
    return 'python/mojolearn/' + ('_mojolearn.so' if b == 'base' else '_mojolearn_' + b + '.so')


def build_script(b):
    return 'bindings/build.sh' if b == 'base' else 'bindings/build_' + b + '.sh'


# ------------------------------------------------------------------ source closure
_IMPORT = re.compile(r'^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))', re.MULTILINE)
_closures = {}


def _local_imports(path):
    """Same conservative discovery as experiments/apple_fast_classical_20261006/build_pair.py."""
    found = set()
    for m in _IMPORT.finditer(path.read_text()):
        name = m.group(1) or m.group(2)
        rel = len(name) - len(name.lstrip('.'))
        if rel:
            base = path.parent
            for _ in range(rel - 1):
                base = base.parent
            roots = [base]
        else:
            roots = [ROOT, ROOT / 'bindings']
        parts = name.lstrip('.').split('.') if name.lstrip('.') else []
        for base in roots:
            target = base.joinpath(*parts)
            cands = [target.with_suffix('.mojo'), target / '__init__.mojo']
            cands.extend(base.joinpath(*parts[:i], '__init__.mojo') for i in range(1, len(parts)))
            for c in cands:
                if c.is_file() and c.is_relative_to(ROOT):
                    found.add(c)
    return found


def closure(b):
    """Repo-relative .mojo files the binding's entry imports, transitively (import reach, not route proof)."""
    if b not in _closures:
        start = binding_file(b)
        seen, todo = set(), [start] if start.exists() else []
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            todo.extend(_local_imports(p) - seen)
        _closures[b] = {str(p.relative_to(ROOT)) for p in seen}
    return _closures[b]


def reaches(b, paths):
    paths = [p.split(':')[0] for p in paths]
    specific = [p for p in paths if not REGISTRY_FILE.search(p)] or paths
    return bool(set(specific) & closure(b))


# ------------------------------------------------------------------ board inventory
def board_rows(path=BOARD_MD):
    """(family, lane) -> sorted datasets, from the first table of the M3 FAST board."""
    rows, inside = {}, False
    for line in Path(path).read_text().splitlines():
        if line.startswith('| lane | dataset | family |'):
            inside = True
            continue
        if inside:
            if not line.startswith('|'):
                break
            cells = [c.strip() for c in line.strip('|').split('|')]
            if cells[0].startswith('---'):
                continue
            rows.setdefault((cells[2], cells[0]), set()).add(cells[1])
    return {k: sorted(v) for k, v in rows.items()}


def algo_id(family, lane):
    return family + ':' + lane


def workload_id(aid, ds):
    return aid + '@dataset=' + ds


# ------------------------------------------------------------------ guards (FAST)
class FastGuards:
    """The FAST grid's refusals, with the interface G.plan_algorithm / G.pack expect.

    * card exclusions: a card's `conflicting_defines` / `defines_absent_in_both_arms` with its own define;
    * an env switch set in the candidate arm only, together with a build define: one A/B line cannot
      express it (tools/afc_ab_def.sh races both arms under one environment);
    * controls of one configuration resolving to different bindings: the Apple A/B scripts build one
      binding per line.
    There is no guard file for FAST: core/six_lane_experiment_guards.mojo is the IDENTICAL guard set.
    """

    assert_count = 0

    def __init__(self, pairs=(), env_names=(), binding_of=None):
        self.pairs = list(pairs)
        self.env_names = set(env_names)
        self.binding_of = dict(binding_of or {})

    def add_pair(self, a, b, reason):
        if (a, b, reason) not in self.pairs:
            self.pairs.append((a, b, reason))

    def problems(self, defines):
        names = {G.norm_define(d).split('=')[0] for d in defines}
        out = []
        env, build = sorted(names & self.env_names), sorted(names - self.env_names)
        if env and build:
            out.append('env switch ' + ', '.join(env) + ' with build define(s): one A/B line cannot set an env in the candidate arm only (afc_ab_def.sh races both arms under one environment)')
        bs = sorted({self.binding_of[n] for n in names if n in self.binding_of})
        if len(bs) > 1:
            out.append('spans bindings ' + ', '.join(bs) + ': the Apple A/B scripts build one binding per line')
        for a, b, reason in self.pairs:
            if a in names and b in names:
                out.append('card exclusion ' + a + ' / ' + b + ' (' + reason + ')')
        return out


# ------------------------------------------------------------------ control sources
def _find_key(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return True, v
            hit = _find_key(v, key)
            if hit[0]:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key(v, key)
            if hit[0]:
                return hit
    return False, None


def _rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _ctrl(key, define, on, both, both_env, bindings, paths, lanes, file, source, kind='switch', **extra):
    return dict(key=key, file=file, define=define, kind=kind, env=(kind == 'env'), default='off',
                arms={'on': G.unique(sorted(G.norm_define(d) for d in on))},
                both_defines=G.unique(sorted(G.norm_define(d) for d in both)),
                both_env={k: str(v) for k, v in sorted((both_env or {}).items()) if k not in DROP_ENV},
                bindings=G.unique(binding_name(b) for b in bindings), paths=list(paths), lanes=G.unique(lanes),
                source=source, conflicts=[], exclusion=None, status='opt-in, unmeasured', binding=None, **extra)


def load_aft():
    from apple_fast_tree_ideas import INTERACTIONS
    controls, files = {}, {}
    for lane_letter in 'FGNP':
        path = AFT_DIR / (lane_letter + '.json')
        doc = json.loads(path.read_text())
        files['aft-' + lane_letter] = dict(path=_rel(path), sha256=G.file_sha(path))
        groups = doc.get('integration_groups') or []
        for card in doc['cards']:
            lanes, bindings = [], []
            for g in groups:
                if card['id'] in g['ids']:
                    fam = 'trees' if str(g.get('public_harness', '')).endswith('forest_speed_arm.py') else 'algos'
                    lanes += [(fam, x) for x in g.get('public_harness_lanes') or []]
                    lanes += [('algos', x) for x in g.get('expanded_harness_lanes') or []]
                    bindings += g.get('required_bindings') or []
            for holder in (card, card.get('integration') or {}):
                if holder.get('harness_lanes'):
                    lanes += [('trees', x) for x in holder['harness_lanes']]
                    bindings.append(holder['binding_module'])
            key = 'AFT_' + card['id']
            c = _ctrl(key, 'MOJOLEARN_AFT_' + card['id'], card['candidate_defines'], card.get('baseline_defines') or [], {},
                      bindings, card.get('source_paths') or [], lanes, 'aft-' + lane_letter, card.get('source_paths') or [],
                      card=card['id'], title=card.get('title'))
            c['conflicts'] = [(d.split('=')[0], 'AFT ' + card['id'] + ' conflicting_defines') for d in card.get('conflicting_defines') or []]
            hit, value = _find_key(card, 'board_default_exercises_candidate')
            if hit and value is False:
                _, adj = _find_key(card, 'required_recipe_adjustment')
                c['exclusion'] = ('unreached', 'card: board_default_exercises_candidate=false; ' + str(adj or ''))
            controls[key] = c
    groups = [['AFT_' + m for m in members] for gid, members in sorted(INTERACTIONS.items()) if len(members) >= 2]
    files['aft-interactions'] = dict(path='tools/apple_fast_tree_ideas.py', sha256=G.file_sha(ROOT / 'tools/apple_fast_tree_ideas.py'))
    return controls, groups, files


def load_afcl():
    controls, files = {}, {}
    names = ['ideas.json', 'full_workloads.json', 'build_bindings.json'] + ['lanes/' + n + '.json' for n in ('geometry', 'linear', 'preprocessing', 'trees')]
    for n in names:
        files['afcl-' + n.replace('/', '-')] = dict(path=_rel(AFCL_DIR / n), sha256=G.file_sha(AFCL_DIR / n))
    ideas = {e['id']: e for e in json.loads((AFCL_DIR / 'ideas.json').read_text())['entries']}
    recipes = {e['id']: e for e in json.loads((AFCL_DIR / 'full_workloads.json').read_text())['entries']}
    binds = json.loads((AFCL_DIR / 'build_bindings.json').read_text())['cards']
    arms, pending = {}, []
    for n in ('geometry', 'linear', 'preprocessing', 'trees'):
        doc = json.loads((AFCL_DIR / 'lanes' / (n + '.json')).read_text())
        entries = doc['entries'] if isinstance(doc['entries'], list) else list(doc['entries'].values())
        for e in entries:
            arms[e['id']] = e
        pending += doc.get('pending_combinations') or []
    for cid in sorted(ideas):
        idea, arm = ideas[cid], arms[cid]
        lanes, unsupported = [], []
        for r in recipes[cid]['recipes']:
            adapter, _, lane = r.partition('/')
            if adapter in AFC_FAMILY:
                lanes.append((adapter, lane))
            else:
                unsupported.append(r)
        key = cid.replace('-', '_')
        c = _ctrl(key, idea['define'], arm['candidate_defines'], arm.get('baseline_defines') or [], arm.get('baseline_env') or {},
                  binds[cid]['affected_bindings'], idea.get('implementation_paths') or [], lanes, 'afcl', idea.get('implementation_paths') or [],
                  card=cid, title=idea.get('title'), unsupported_recipes=unsupported)
        c['conflicts'] = [(d.split('=')[0], 'AFCL ' + cid + ' defines_absent_in_both_arms') for d in arm.get('defines_absent_in_both_arms') or []]
        cand_env = {k: str(v) for k, v in (arm.get('candidate_env') or {}).items() if k not in DROP_ENV}
        if cand_env != c['both_env']:
            raise ValueError(cid + ': candidate_env differs from baseline_env; a FAST env candidate needs its own control')
        controls[key] = c
    # README "Interactions need their own future A/B: ... ARIMA differencing+likelihood"; select.py --factorial example P11 P12.
    groups = [[m.replace('-', '_') for m in g] for g in pending] + [['AFCL_P11', 'AFCL_P12']]
    return controls, groups, files


def load_untried(directory=CONTROLS_DIR):
    controls, algos, excluded, files = {}, {}, [], {}
    for path in sorted(Path(directory).glob('*.json')):
        doc = json.loads(path.read_text())
        if doc.get('schema') != 'mojolearn.grid-controls/1' or doc.get('mode') != 'fast':
            raise ValueError(str(path) + ': expected schema mojolearn.grid-controls/1 with mode fast')
        name = path.stem
        files[name] = dict(path=_rel(path), sha256=G.file_sha(path), lane=doc.get('lane'), branch=doc.get('branch'))
        lanes_of = {}
        for aid, a in doc['algorithms'].items():
            for k in a['controls']:
                lanes_of.setdefault(k, []).append(tuple(aid.split(':', 1)))
            algos.setdefault(aid, []).extend(a.get('interaction_groups') or [])
        for key, c in doc['controls'].items():
            if key in controls:
                raise ValueError('Duplicate control ' + key)
            on = [d for d in c['arms']['on']]
            ctl = _ctrl(key, c['define'], on, c.get('both_arms') or [], c.get('both_env') or {}, [c['binding']],
                        c.get('paths') or [], lanes_of.get(key, []), name, c.get('source') or [], kind=c.get('kind', 'switch'),
                        title=c.get('note'), quality=c.get('quality'), origin_branch=c.get('branch'))
            ctl['status'] = c.get('status', 'untried')
            controls[key] = ctl
        for e in doc.get('excluded') or []:
            excluded.append(dict(e, file=name))
    return controls, algos, excluded, files
