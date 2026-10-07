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
# Lane -> the bindings its Python estimator module imports (python/mojolearn, 2026-10-07), used only to
# pick among card bindings that all import the card's source. Evidence per entry is the module named.
_TREES_PY = ('x_trees', 'rf', 'estimators')  # _expansion_trees.py
LANE_PY_BINDINGS = {
    'knn-clf': ('base',), 'knn-reg': ('base',),                        # neighbors.py _BINDING = "_mojolearn"
    'umap': ('metrics',),                                              # _umap_impl.py
    'spectral': ('metrics', 'x_decomp'), 'spectral-embedding': ('metrics', 'x_decomp'),  # _spectral_impl.py
    'agglomerative': ('solver', 'x_neighbors', 'estimators'),          # _hierarchy_impl.py
    'ridge': ('estimators',), 'ols': ('estimators',),                  # linear_model.py
    'pca': ('x_decomp', 'estimators'),                                 # decomposition.py
    'nmf': ('x_decomp',), 'factor-analysis': ('x_decomp',), 'pls': ('x_decomp',), 'pls-canonical': ('x_decomp',),
    'cca': ('x_decomp',), 'als': ('x_decomp',), 'randomized-svd': ('x_decomp',),  # _expansion_decomp.py
    'qr': ('linalg',), 'svd': ('linalg',),                             # _linalg_impl.py (bench lane linalg.qr / linalg.svd)
    'tree-shap': _TREES_PY, 'dart': _TREES_PY, 'dart-reg': _TREES_PY, 'random-trees-embedding': _TREES_PY,
    'decision-tree-clf': _TREES_PY, 'decision-tree-reg': _TREES_PY, 'bagging-clf': _TREES_PY, 'bagging-reg': _TREES_PY,
    'adaboost-clf': _TREES_PY, 'adaboost-reg': _TREES_PY,
}
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

    def __init__(self, pairs=(), env_names=(), binding_of=None, prereqs=None):
        self.pairs = list(pairs)
        self.env_names = set(env_names)
        self.binding_of = dict(binding_of or {})
        self.prereqs = {k: list(v) for k, v in (prereqs or {}).items()}  # own define -> both-arm prerequisite defines

    def add_pair(self, a, b, reason):
        if (a, b, reason) not in self.pairs:
            self.pairs.append((a, b, reason))

    def problems(self, defines):
        names = {G.norm_define(d).split('=')[0] for d in defines}
        own = set(names)
        for n in own:
            names |= {G.norm_define(d).split('=')[0] for d in self.prereqs.get(n, [])}
        out = []
        for n in sorted(names):
            if n.endswith('_OFF') and n[:-4] in names:
                out.append('enabled and disabled together (with prerequisites): ' + n[:-4])
        env, build = sorted(names & self.env_names), sorted(names - self.env_names)
        if env and build:
            out.append('env switch ' + ', '.join(env) + ' with build define(s): one A/B line cannot set an env in the candidate arm only (afc_ab_def.sh races both arms under one environment)')
        bs = sorted({self.binding_of[n] for n in own if n in self.binding_of})
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
    # The arm carries only what the candidate adds; prerequisites kept in both arms live in both_defines
    # (FastGuards expands them for every check; finish_config adds them to both arms).
    both = G.unique(sorted(G.norm_define(d) for d in both))
    on = [d for d in G.unique(sorted(G.norm_define(d) for d in on)) if d not in both]
    if not on:
        raise ValueError(key + ': the candidate adds nothing over its both-arm prerequisites')
    return dict(key=key, file=file, define=define, kind=kind, env=(kind == 'env'), default='off',
                arms={'on': on}, both_defines=both,
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


# ------------------------------------------------------------------ model
def resolve_binding(c, lane):
    """-> (binding, why) or (None, reason). Lane binding first; a card binding when the lane's own does not import it."""
    lb = LANE_BINDING.get(lane)
    lb = binding_name(lb) if lb else None
    if c['env']:
        b = c['bindings'][0] if c['bindings'] else lb
        return b, 'run-time env switch (no build); binding of the lane route, used for its main build and A/A'
    src = ', '.join(c['paths']) or '-'
    if lb and binding_file(lb).exists() and reaches(lb, c['paths']):
        return lb, 'lane binding ' + lb + ' imports ' + src
    reached = [b for b in c['bindings'] if binding_file(b).exists() and reaches(b, c['paths'])]
    py = [b for b in reached if b in LANE_PY_BINDINGS.get(lane, ())]
    if len(reached) > 1 and len(py) == 1:
        return py[0], ('card binding ' + py[0] + ' imports ' + src + ' and is the one of ' + ', '.join(reached)
                       + ' the lane\'s Python module loads (LANE_PY_BINDINGS)')
    if len(reached) == 1:
        return reached[0], 'card binding ' + reached[0] + ' imports ' + src + ('' if not lb else ' (lane binding ' + lb + ' does not)')
    if not reached:
        return None, 'unreached by import scan: no binding of ' + ', '.join(c['bindings'] or ['-']) + ((' or lane binding ' + lb) if lb else '') + ' imports ' + src
    return None, ('ambiguous binding: ' + ', '.join(reached) + ' each import ' + src + '; lane binding ' + (lb or 'unknown')
                  + '; the Apple A/B scripts build one binding per line (add the lane to LANE_BINDING with its source to resolve)')


def build_fast_model(controls, global_groups, untried_groups, board):
    fams_of = {}
    for fam, lane in board:
        fams_of.setdefault(lane, []).append(fam)
    algos, unmapped, excluded, binding_excluded = {}, [], {}, []
    for key, c in controls.items():
        if c['exclusion']:
            excluded[key] = dict(control=key, define=c['define'], file=c['file'], algorithms=[], reasons=[c['exclusion'][0] + ': ' + c['exclusion'][1]])
            continue
        for r in c.get('unsupported_recipes') or []:
            unmapped.append(dict(control=key, lane=r, reason='recipe adapter has no Apple A/B script (experiments/apple_fast_classical_20261006/run_pair.py PublicProgram only)'))
        if not c['lanes']:
            excluded[key] = dict(control=key, define=c['define'], file=c['file'], algorithms=[], reasons=['no board lane recipe the Apple A/B scripts can race'])
            continue
        placed = False
        for fam, lane in c['lanes']:
            note = None
            if (fam, lane) not in board:
                fams = fams_of.get(lane, [])
                if len(fams) != 1:
                    unmapped.append(dict(control=key, lane=fam + ':' + lane, reason='no M3 FAST board row for this lane' if not fams else 'lane name on several board families: ' + ', '.join(fams)))
                    continue
                note = fam + ':' + lane + ' is board family ' + fams[0]
                fam = fams[0]
            aid = algo_id(fam, lane)
            b, why = resolve_binding(c, lane)
            if b is None:
                binding_excluded.append(dict(control=key, define=c['define'], algorithm=aid, reason=why))
                continue
            a = algos.setdefault(aid, dict(id=aid, family=fam, lane=lane, controls=[], binding_of={}, binding_why={}, groups=[],
                                           not_reached=[], arm_restrict={}, notes=[], datasets=board[(fam, lane)]))
            if key not in a['controls']:
                a['controls'].append(key)
                a['binding_of'][key] = b
                a['binding_why'][key] = why
                if note:
                    a['notes'].append(note)
            placed = True
        if not placed and key not in excluded:
            excluded[key] = dict(control=key, define=c['define'], file=c['file'], algorithms=[],
                                 reasons=['no mapped board lane with a resolvable binding (see unmapped / binding exclusions)'])
    for aid, a in algos.items():
        # The planner's all-on is greedy in control order, and a configuration builds one binding: put the
        # controls of the lane's main binding (most controls; ties: the lane binding, then name) first.
        counts = {}
        for k in a['controls']:
            counts[a['binding_of'][k]] = counts.get(a['binding_of'][k], 0) + 1
        lb = binding_name(LANE_BINDING[a['lane']]) if a['lane'] in LANE_BINDING else None
        main = sorted(counts, key=lambda b: (-counts[b], b != lb, b))[0]
        a['main_binding'] = main
        a['controls'] = [k for k in a['controls'] if a['binding_of'][k] == main] + [k for k in a['controls'] if a['binding_of'][k] != main]
        for g in global_groups + untried_groups.get(aid, []):
            members = [k for k in g if k in a['controls']]
            if len(members) >= 2 and members not in a['groups']:
                a['groups'].append(members)
        a['workloads'] = [workload_id(aid, ds) for ds in a['datasets'] if workload_id(aid, ds) not in WORKLOAD_EXCLUSIONS]
        a['excluded_workloads'] = [dict(workload_id=workload_id(aid, ds), reason=WORKLOAD_EXCLUSIONS[workload_id(aid, ds)])
                                   for ds in a['datasets'] if workload_id(aid, ds) in WORKLOAD_EXCLUSIONS]
    return dict(sorted(algos.items())), unmapped, excluded, binding_excluded


def load_all(controls_dir=CONTROLS_DIR):
    aft, aft_groups, f1 = load_aft()
    afcl, afcl_groups, f2 = load_afcl()
    untried, untried_groups, untried_excluded, f3 = load_untried(controls_dir)
    controls = {}
    for part in (aft, afcl, untried):
        for k, c in part.items():
            if k in controls:
                raise ValueError('Duplicate FAST control ' + k)
            controls[k] = c
    files = dict(f1, **f2, **f3)
    files['board'] = dict(path=_rel(BOARD_MD), sha256=G.file_sha(BOARD_MD))
    return controls, aft_groups + afcl_groups, untried_groups, untried_excluded, files


# ------------------------------------------------------------------ generate
def dflags(defines):
    return ' '.join('-D ' + (d[:-2] if d.endswith('=1') else d) for d in defines)


def envstr(env):
    return ' '.join(k + '=' + v for k, v in sorted(env.items()))


def tool_of(cfg):
    if cfg['kind'] == 'env':
        return 'afc_env'
    return 'aft' if cfg['family'] == 'trees' and cfg['binding'] in AFT_BINDINGS else 'afc_def'


def finish_config(cfg, algo, controls, env_names):
    members = cfg['controls']
    own = {controls[k]['define'] for k in members}
    env_defs = [d for d in cfg['defines'] if d.split('=')[0] in env_names]
    build = [d for d in cfg['defines'] if d not in env_defs]
    incumbent = sorted({d for k in members for d in controls[k]['both_defines'] if d.split('=')[0] not in own})
    build = sorted(set(build) | set(incumbent))
    both_env = {}
    for k in members:
        for name, v in controls[k]['both_env'].items():
            if both_env.get(name, v) != v:
                raise ValueError(cfg['id'] + ': conflicting both-arm env ' + name)
            both_env[name] = v
    cand_env = dict(both_env)
    for d in env_defs:
        name, _, v = d.partition('=')
        cand_env[name] = v or '1'
    bindings = sorted({algo['binding_of'][k] for k in members})
    if len(bindings) != 1:
        raise ValueError(cfg['id'] + ': configuration spans bindings ' + ', '.join(bindings))
    cfg.update(kind='env' if env_defs else 'build', family=algo['family'], lane=algo['lane'], binding=bindings[0],
               bindings=bindings, candidate_build_defines=build,
               incumbent_defines=incumbent, incumbent_env=both_env, candidate_env=cand_env, workloads=algo['workloads'])
    cfg['tool'] = tool_of(cfg)
    return cfg


def generate_fast(controls_dir=CONTROLS_DIR, cap=G.CAP, branch=QUEUE_BRANCH):
    controls, global_groups, untried_groups, untried_excluded, files = load_all(controls_dir)
    board = board_rows()
    algos, unmapped, excluded, binding_excluded = build_fast_model(controls, global_groups, untried_groups, board)
    env_names = {c['define'] for c in controls.values() if c['env']}
    pairs = [(c['define'], other, why) for c in controls.values() for other, why in c['conflicts']]
    prereqs = {c['define']: c['both_defines'] for c in controls.values() if c['both_defines']}
    reach = {}
    for aid, a in algos.items():
        for k in a['controls']:
            reach.setdefault(k, set()).add(aid)
    plans, configs = {}, []
    for aid, a in algos.items():
        guards = FastGuards(pairs, env_names, {controls[k]['define']: a['binding_of'][k] for k in a['controls']}, prereqs)
        plan = G.plan_algorithm(a, controls, reach, guards, cap)
        plans[aid] = plan
        for e in plan['excluded']:
            excluded.setdefault(e['control'], dict(control=e['control'], define=e['define'], file=controls[e['control']]['file'],
                                                   reasons=e['reasons'], algorithms=[]))
            excluded[e['control']]['algorithms'].append(aid)
        if not a['workloads']:
            for c in plan['configs']:
                c['unmapped'] = True
            continue
        for c in plan['configs']:
            configs.append(finish_config(c, a, controls, env_names))

    # Packing: define configs with the same tool, binding and incumbent share one candidate build.
    global_guards = FastGuards(pairs, env_names, prereqs=prereqs)
    groups = {}
    for c in configs:
        if c['kind'] == 'build':
            groups.setdefault((c['tool'], c['binding'], tuple(c['incumbent_defines']), tuple(sorted(c['incumbent_env'].items()))), []).append(c)
    packs = []
    for gkey in sorted(groups):
        for p in G.pack(groups[gkey], reach, global_guards):
            p['tool'], p['binding'], p['incumbent_defines'], p['incumbent_env'] = gkey[0], gkey[1], list(gkey[2]), dict(gkey[3])
            p['defines'] = sorted(set(p['defines']) | set(gkey[2]))  # arm B keeps the both-arm prerequisites
            packs.append(p)
    for i, p in enumerate(packs):
        p['id'] = 'F%03d' % (i + 1)
        for m in p['members']:
            m['pack'] = p['id']
            m['B_defines'] = list(p['defines'])
            m['packed_extra_defines'] = sorted(set(p['defines']) - set(m['candidate_build_defines']))
    for c in configs:
        if c['kind'] == 'env':
            c['pack'], c['B_defines'], c['packed_extra_defines'] = None, list(c['incumbent_defines']), []
        c['A_defines'] = list(c['incumbent_defines'])

    run = 'gf' + G.sha_value(sorted([c['id'], c['A_defines'], c['B_defines'], c['candidate_env'], c['workloads']] for c in configs))[:6]
    queue, build_jobs = render_queue(configs, packs, run, branch)
    matrix = fast_matrix(configs, files, queue)
    build_plan = dict(schema='mojolearn.six-lane-build-plan/1', jobs=build_jobs, blocked=[], unsupported=[],
                      policy='FAST grid builds, as the Apple A/B scripts perform them on the M3: tools/aft_ab.sh and tools/afc_ab_def.sh build '
                             'arm A (FAST main + both-arm prerequisites) and arm B (the packed candidate) of one binding per build directory, '
                             'MOJOLEARN_NUMERIC_MODE=fast, no MOJOLEARN_NUMERIC_IDENTICAL. Lanes never compile; nothing here was built.')
    n_build = sum(1 for c in configs if c['kind'] == 'build')
    mains = {(j['tool'], j['binding']) for j in build_jobs if j['role'] == 'main'}
    distinct = {(j['binding'], tuple(j['defines'])) for j in build_jobs}
    summary = dict(
        mode=MODE, vendor='apple', box='m3', algorithms=len(algos), algorithms_with_configs=len({c['algorithm'] for c in configs}),
        unmapped_algorithms=sum(1 for a in algos.values() if not a['workloads']),
        controls=len(controls), excluded_controls=len(excluded), binding_exclusions=len(binding_excluded), unmapped_lane_recipes=len(unmapped),
        configs=len(configs), build_configs=n_build, env_configs=len(configs) - n_build,
        cells=len(matrix['cells']), workloads=len({w for c in configs for w in c['workloads']}),
        aa_pairs=sum(1 for q in queue if q['kind'] == 'aa'), queue_lines=len(queue),
        deferred_configs=sum(len(p['deferred']) for p in plans.values()),
        deferred_cross_groups=sum(len(p['deferred_groups']) for p in plans.values()),
        deferred_cross_configs=sum(g['configs'] for p in plans.values() for g in p['deferred_groups']),
        packs=len(packs), packed_multi_member=sum(1 for p in packs if len(p['members']) > 1),
        builds_before_packing=2 * n_build + 2 * len(mains), builds_after_packing=len(build_jobs),
        main_builds=2 * len(mains), distinct_define_sets=len(distinct),
        bindings=sorted({c['binding'] for c in configs}), cap=cap, run=run, queue_branch=branch)
    plan = dict(
        schema='mojolearn.six-lane-grid-plan/1', mode=MODE, vendor='apple', status='PLANNED_NOT_RUN',
        all_switches='NOT MEASURED until each config has an M3 A/A floor and a scored A/B with the board quality metric',
        arm_convention='A = FAST main (shipped FAST defaults plus any prerequisite the card keeps in both arms); B = the candidate (as tools/aft_ab.sh and tools/afc_ab_def.sh use them)',
        promoted_off_arms='NOT gridded: every promoted FAST default (a *_OFF rollback exists) was A/B\'d on top of the defaults current at its promotion, so the shipped FAST default is the all-on point and the M3 board measures it on every row; promoted arms enter only as arm A',
        inputs=dict(sources=files), vendors=VENDORS,
        measurement=dict(excluded_warmups=0, scored_samples=1, rule='owner: one run per arm (tools/aft_ab.sh pairs 1; tools/afc_ab_def.sh reps 1 rounds 1; AB_MULTI_RUN unset)'),
        generation_rules=dict(
            cap_per_algorithm=cap, algorithm='one M3 FAST board lane (family:lane); its workloads are that lane\'s board datasets',
            priority=['(a) every eligible control alone', '(b) all-on: one arm per control, guard-compatible', '(c) declared interaction-group crosses, admitted group-atomically within the cap'],
            deferred='anything cut by the cap is DEFERRED (generate after one-at-a-time results), never dropped',
            controls='opt-in FAST cards (AFT, AFCL) and the EXPERIMENTS.md untried rows; promoted _OFF arms excluded',
            groups='tools/apple_fast_tree_ideas.py X01-X10; AFCL lanes/trees.json pending_combinations + P11/P12; untried MI pair, iterative-imputer env pair, ARIMA quality trio and STEPWISE/CSS_SEARCH',
            guards='FastGuards: card conflicting_defines / defines_absent_in_both_arms, no candidate-only env with a build define, one binding per line, plus six_lane_ab.combine()',
            binding='per (control, lane): the lane binding when it imports the card source; else the one card binding that imports it (ties broken by the bindings the lane\'s Python module loads); otherwise excluded with the reason',
            packing='define configs of different lanes share one candidate build when tool, binding and arm A are equal, no assigned control reaches another member lane, and the union passes the guards',
            workloads='M3 FAST board rows (docs/apple-fast/BOARD_M3_FAST.md first table); lanes without a row are UNMAPPED',
            env='MOJOLEARN_NUMERIC_MODE from AFCL baseline_env is implied by the FAST builds and the ours-fast arm and is not repeated'),
        verdict_rule=VERDICT_RULE, summary=summary,
        algorithms={aid: dict(
            family=a['family'], lane=a['lane'], workloads=a['workloads'], excluded_workloads=a['excluded_workloads'],
            declared_controls=a['controls'], bindings={k: a['binding_of'][k] for k in a['controls']},
            binding_reasons={k: a['binding_why'][k] for k in a['controls']}, notes=a['notes'],
            eligible_controls=plans[aid]['eligible'], interaction_groups=a['groups'], cross_groups=plans[aid]['cross_groups'],
            all_on=plans[aid]['all_on'], dropped_from_all_on=plans[aid]['dropped_from_all_on'],
            configs=[dict(id=c['id'], priority=c['priority'], tier=c['tier'], assignment=c['assignment'], kind=c.get('kind'),
                          binding=c.get('binding'), tool=c.get('tool'), pack=c.get('pack'), A_defines=c.get('A_defines'),
                          B_defines=c.get('B_defines'), packed_extra_defines=c.get('packed_extra_defines', []),
                          A_env=c.get('incumbent_env'), B_env=c.get('candidate_env'), queued=not c.get('unmapped'))
                     for c in plans[aid]['configs']],
            deferred=plans[aid]['deferred'], deferred_groups=plans[aid]['deferred_groups'], invalid=G.summarize_invalid(plans[aid]['invalid']))
            for aid, a in algos.items()},
        unmapped=unmapped, binding_exclusions=binding_excluded,
        excluded=sorted(excluded.values(), key=lambda e: (e['file'], e['control'])), excluded_untried=untried_excluded,
        builds=dict(before_packing=summary['builds_before_packing'], after_packing=summary['builds_after_packing'],
                    packs=[dict(id=p['id'], tool=p['tool'], binding=p['binding'], A_defines=p['incumbent_defines'], A_env=p['incumbent_env'],
                                B_defines=p['defines'], members=[m['id'] for m in p['members']], algorithms=sorted(p['algorithms'])) for p in packs]),
        queue=[{k: q[k] for k in ('tag', 'kind', 'configuration', 'algorithm', 'workload_id', 'tool', 'binding', 'pack')} for q in queue])
    return plan, matrix, build_plan, queue


VERDICT_RULE = dict(
    timing='per workload on the M3: FASTER / SLOWER only when |log(B/A)| of the scored clock (AFC-DEF-SUMMARY / AFT-MEDIAN median_ms) '
           'exceeds the M3 A/A floor for the same workload (its grid A/A line: both arms the FAST main build); otherwise NO_VERDICT. '
           'One box votes (apple); there is no second vendor',
    identity='NOT_REQUIRED: FAST needs no identical anything (CLAUDE.md): no same bits across vendors, none vs arm A, none run to run',
    quality='the board quality metric of B must not drop vs arm A (FAST main) on any workload: tools/af_quality.py, rel 1e-3 / abs 1e-6, '
            'the gate tools/af_board_apply.py enforces (which also refuses a row worse than the best opponent); a noise-level change, '
            'a different fold order or different bits never hold a candidate',
    promotion='FASTER on at least one workload, SLOWER on none, quality held: the switch becomes the FAST default with a *_OFF '
              'rollback and a code comment citing the A/B numbers, plus one docs/apple-fast/EXPERIMENTS.md row; a loser (slower, '
              'noise, quality drop) is deleted from the code and gets its row',
    default_state='every switch is NOT MEASURED until then')


# ------------------------------------------------------------------ M3 queue
def _dirs(tool, run, name):
    return ('$HOME/aft-ab/' if tool == 'aft' else '$HOME/afc-def/') + run + '-' + name


def render_queue(configs, packs, run, branch):
    """grid-fast-queue.txt lines: A/A first (they also leave the FAST main build of each binding in place),
    then the define A/Bs pack by pack (one build pair per pack), then the env A/Bs on the main build.
    Every define line restores its binding's FAST main .so afterwards: both scripts leave arm B installed."""
    queue, jobs = [], []
    seen_dirs = set()

    def tag():
        return '%s-%03d' % (run, len(queue) + 1)

    def add(kind, cmd, **info):
        t = tag()
        cmd = cmd.replace('<TAG>', t)
        queue.append(dict(tag=t, kind=kind, command=cmd, line="lq add m3 CMD %s %s '%s'" % (branch, t, cmd), **info))

    aa = {}
    for c in configs:
        tm = 'aft' if c['tool'] == 'aft' else 'afc'
        for w in c['workloads']:
            aa.setdefault((tm, c['binding'], w), c)
    mains = []
    for (tm, b, w) in sorted(aa):
        c = aa[(tm, b, w)]
        ds = w.split('@dataset=', 1)[1]
        d = _dirs(tm, run, 'main-' + b)
        if tm == 'aft':
            skip = ' AFT_SKIP_BUILD=1' if d in seen_dirs else ''
            cmd = 'AFT_OUT=%s%s bash tools/aft_ab.sh %s %s %s 1 "" ""' % (d, skip, b, c['lane'], ds)
        else:
            cmd = 'AFC_FAMILY=%s AFC_DEF_BUILD_TAG=%s bash tools/afc_ab_def.sh <TAG> %s %s %s 1 1 "" ""' % (
                AFC_FAMILY[c['family']], run + '-main-' + b, b, c['lane'], ds)
        if d not in seen_dirs:
            seen_dirs.add(d)
            mains.append((tm, b, d))
        add('aa', cmd, configuration=None, algorithm=c['algorithm'], workload_id=w, tool=tm, binding=b, pack=None)
    for tm, b, d in mains:
        for arm in 'AB':
            jobs.append(dict(key=G.sha_value([run, tm, b, 'main', arm])[:20], role='main', arm=arm, tool=tm, binding=G.binding_path(b) if b != 'base' else 'bindings/_mojolearn.mojo',
                             script=build_script(b), build_dir=d, vendor='apple', mode=MODE, defines=[], environment={},
                             status='NOT_COMPILED', configurations=[], target='metal (M3 Ultra), FAST'))
    by_pack = {p['id']: p for p in packs}
    for p in packs:
        d = _dirs(p['tool'], run, p['id'])
        for arm, defs in (('A', p['incumbent_defines']), ('B', p['defines'])):
            jobs.append(dict(key=G.sha_value([run, p['id'], arm])[:20], role='pack', arm=arm, tool=p['tool'], binding=G.binding_path(p['binding']) if p['binding'] != 'base' else 'bindings/_mojolearn.mojo',
                             script=build_script(p['binding']), build_dir=d, vendor='apple', mode=MODE, defines=list(defs),
                             environment=dict(p['incumbent_env']), status='NOT_COMPILED', pack=p['id'],
                             configurations=[m['id'] for m in p['members']], target='metal (M3 Ultra), FAST'))
    builds = sorted((c for c in configs if c['kind'] == 'build'), key=lambda c: (c['pack'], c['algorithm'], c['priority']))
    started = set()
    for c in builds:
        p = by_pack[c['pack']]
        pre = (envstr(c['incumbent_env']) + ' ') if c['incumbent_env'] else ''
        so = so_path(c['binding'])
        for w in c['workloads']:
            ds = w.split('@dataset=', 1)[1]
            if c['tool'] == 'aft':
                d = _dirs('aft', run, p['id'])
                skip = ' AFT_SKIP_BUILD=1' if p['id'] in started else ''
                cmd = '%sAFT_OUT=%s%s bash tools/aft_ab.sh %s %s %s 1 "%s" "%s"; rc=$?; cp %s/A.so %s; exit $rc' % (
                    pre, d, skip, c['binding'], c['lane'], ds, dflags(c['A_defines']), dflags(c['B_defines']),
                    _dirs('aft', run, 'main-' + c['binding']), so)
            else:
                cmd = '%sAFC_FAMILY=%s AFC_DEF_BUILD_TAG=%s bash tools/afc_ab_def.sh <TAG> %s %s %s 1 1 "%s" "%s"; rc=$?; cp %s/A.so %s; exit $rc' % (
                    pre, AFC_FAMILY[c['family']], run + '-' + p['id'], c['binding'], c['lane'], ds, dflags(c['A_defines']),
                    dflags(c['B_defines']), _dirs('afc', run, 'main-' + c['binding']), so)
            started.add(p['id'])
            add('ab', cmd, configuration=c['id'], algorithm=c['algorithm'], workload_id=w, tool=c['tool'], binding=c['binding'], pack=p['id'])
    for c in sorted((c for c in configs if c['kind'] == 'env'), key=lambda c: (c['algorithm'], c['priority'])):
        for w in c['workloads']:
            ds = w.split('@dataset=', 1)[1]
            cmd = 'AFC_FAMILY=%s bash tools/afc_ab.sh <TAG> %s %s 1 1 "%s" "%s"' % (
                AFC_FAMILY[c['family']], c['lane'], ds, envstr(c['incumbent_env']) or '-', envstr(c['candidate_env']))
            add('ab', cmd, configuration=c['id'], algorithm=c['algorithm'], workload_id=w, tool='afc_env', binding=c['binding'], pack=None)
    return queue, jobs


def fast_matrix(configs, files, queue):
    tags = {}
    for q in queue:
        if q['configuration']:
            tags[(q['configuration'], q['workload_id'])] = q['tag']
    configurations, cells = [], []
    for c in sorted(configs, key=lambda c: (c['algorithm'], c['priority'])):
        configurations.append(dict(
            id=c['id'], name=c['id'], members=[c['algorithm'] + '/' + k for k in c['controls']], mode=MODE, vendors=['apple'],
            A=dict(defines=c['A_defines'], environment=c['incumbent_env'], runtime={}),
            B=dict(defines=c['B_defines'], environment=c['candidate_env'], runtime={}),
            candidate_arm='B', problems=[], workloads=c['workloads'], kind='candidate', campaign_role='new_candidate',
            experiment_kind='grid_fast_per_lane_cards', priority=c['priority'],
            rationale='FAST grid ' + c['tier'] + ' (' + c['origin'] + ') for ' + c['algorithm'],
            grid=dict(algorithm=c['algorithm'], tier=c['tier'], assignment=c['assignment'], effective_defines=c['candidate_build_defines'],
                      packed_extra_defines=c['packed_extra_defines'], pack=c['pack'], binding=c['binding'], tool=c['tool'], kind=c['kind'])))
        for w in c['workloads']:
            cells.append(dict(
                key=G.sha_value([c['id'], 'apple', w])[:20], configuration=c['id'], vendor='apple', mode=MODE,
                implementation_ids=[c['algorithm'] + '/' + k for k in c['controls']],
                workload=dict(id=w, lane=c['lane'], family=c['family'], dataset=w.split('@dataset=', 1)[1], grid_algorithm=c['algorithm']),
                workload_id=w, status='PENDING', identity='NOT_REQUIRED', identity_group=None, promotion_vote=True,
                planned_excluded_warmups=0, planned_scored_samples=1, actual_samples=0, priority=c['priority'],
                queue_tag=tags.get((c['id'], w))))
    return dict(schema='mojolearn.six-lane-matrix/1', base_main=None, mode=MODE, configurations=configurations, cells=cells,
                execution='NOT EXECUTED', arm_convention='A = FAST main, B = candidate',
                qualification=dict(status='GRID_PLANNED_NOT_MEASURED', identity='NOT_REQUIRED',
                                   source_inputs={k: v['sha256'] for k, v in sorted(files.items())}))


# ------------------------------------------------------------------ GRID.md
FAMILY_TITLES = [('trees', 'trees (forest_speed_arm.py lanes: tools/aft_ab.sh)'),
                 ('algos', 'classical and tree-wrapper lanes of tools/bench_board_algos.py (AFC_FAMILY=algos)'),
                 ('classical2', 'classical: tools/bench_board_more.py (AFC_FAMILY=classical2)'),
                 ('classical', 'classical: tools/classical_two_datasets.py (AFC_FAMILY=classical)')]


def _assign(c):
    return ', '.join(k + ('' if v == 'on' else '=' + v) for k, v in c['assignment'].items())


def render_md_fast(plan):
    s = plan['summary']
    L = ['# FAST switch grid, Apple M3 (planned, not run)', '',
         'Generated by `tools/six_lane_grid.py --mode fast --vendor apple` (`tools/six_lane_grid_fast.py`) from the opt-in Apple FAST',
         'cards (`experiments/apple_fast_trees/{F,G,N,P}.json`, `experiments/apple_fast_classical_20261006/`), the',
         'EXPERIMENTS.md untried candidates (`grid-fast/controls/`) and the M3 FAST board rows (`docs/apple-fast/BOARD_M3_FAST.md`).',
         '**Every switch is NOT MEASURED** until it has an M3 A/A floor and a scored A/B with the board quality metric.', '',
         '- **Arms**: ' + plan['arm_convention'] + '.',
         '- **Promoted `_OFF` arms are not gridded**: ' + plan['promoted_off_arms'].split(': ', 1)[1] + '.',
         '- **Builds**: FAST builds only (MOJOLEARN_NUMERIC_MODE=fast through the scripts); no `MOJOLEARN_NUMERIC_IDENTICAL` define anywhere.',
         '- **Identity**: none. FAST needs no identical anything (CLAUDE.md); there is no identity column.', '',
         '## Totals', '', '| item | count |', '|---|---|',
         '| controls (48 AFT + 54 AFCL + %d untried rows and their group partners) | %d |' % (s['controls'] - 102, s['controls']),
         '| excluded controls | %d |' % s['excluded_controls'],
         '| (control, lane) pairs excluded for an unresolvable binding | %d |' % s['binding_exclusions'],
         '| card lane recipes without an M3 board row or script | %d |' % s['unmapped_lane_recipes'],
         '| algorithms (board lanes) with configs | %d |' % s['algorithms_with_configs'],
         '| **configs** (build %d, env %d) | **%d** |' % (s['build_configs'], s['env_configs'], s['configs']),
         '| board workloads touched | %d |' % s['workloads'],
         '| **A/B cells** (configs x workloads, one box) | **%d** |' % s['cells'],
         '| A/A lines (one per workload x binding x script) | %d |' % s['aa_pairs'],
         '| queue lines (`grid-fast-queue.txt`) | %d |' % s['queue_lines'],
         '| deferred singles/all-on (cap) | %d |' % s['deferred_configs'],
         '| deferred interaction-group crosses | %d groups, %d configs |' % (s['deferred_cross_groups'], s['deferred_cross_configs']),
         '| builds before packing (2 per build config + main) | %d |' % s['builds_before_packing'],
         '| **builds after packing** (%d packs x 2 + %d main; %d packs with >1 member) | **%d** |' % (s['packs'], s['main_builds'], s['packed_multi_member'], s['builds_after_packing']),
         '| distinct (binding, define set) among them | %d |' % s['distinct_define_sets'],
         '', 'Cap: %d configs per lane. One run per arm (aft_ab.sh pairs 1; afc_ab_def.sh / afc_ab.sh reps 1 rounds 1). Box: m3 only. Run id `%s`.' % (s['cap'], s['run']), '',
         '## Verdict rule', '']
    for k in ('timing', 'identity', 'quality', 'promotion', 'default_state'):
        L.append('- **' + k + '**: ' + plan['verdict_rule'][k])
    L += ['', '## Per-family tables', '',
          'configs in priority order (`s` single, `all` all-on, `x` cross); env configs are marked `(env)`; deferred = crosses cut by the cap.', '']
    fams = {}
    for aid, a in plan['algorithms'].items():
        fams.setdefault(a['family'], []).append((aid, a))
    for fam, title in FAMILY_TITLES:
        if fam not in fams:
            continue
        L += ['### ' + title, '', '| lane | workloads | #configs | configs | deferred | builds |', '|---|---|---|---|---|---|']
        for aid, a in sorted(fams[fam]):
            short = {'single': 's', 'all_on': 'all', 'cross': 'x'}
            cfgs = '<br>'.join(short[c['tier']] + ' ' + _assign(c) + (' (env)' if c['kind'] == 'env' else '') for c in a['configs']) or '-'
            dparts = ['cross ' + '/'.join(g['members']) + ': %d' % g['configs'] for g in a['deferred_groups']]
            if a['deferred']:
                dparts.insert(0, '+%d capped' % len(a['deferred']))
            packs = sorted({c['pack'] for c in a['configs'] if c['pack']})
            binds = sorted({c['binding'] for c in a['configs']})
            L.append('| %s | %s | %d | %s | %s | %s |' % (
                aid, ', '.join(w.split('@dataset=')[1] for w in a['workloads']) + (' (excl. ' + ', '.join(e['workload_id'].split('@dataset=')[1] for e in a['excluded_workloads']) + ')' if a['excluded_workloads'] else ''),
                len(a['configs']), cfgs, '<br>'.join(dparts) or '-', '%d packs, %s' % (len(packs), '/'.join(binds))))
        L.append('')
    L += ['## Excluded controls', '', '| control | define | source | reason |', '|---|---|---|---|']
    for e in plan['excluded']:
        L.append('| %s | `%s` | %s | %s |' % (e['control'], e['define'], e['file'], '; '.join(e['reasons']).replace('|', '/')))
    L += ['', '## Binding exclusions (control on one lane)', '', '| control | lane | reason |', '|---|---|---|']
    for e in plan['binding_exclusions']:
        L.append('| %s | %s | %s |' % (e['control'], e['algorithm'], e['reason'].replace('|', '/')))
    L += ['', '## Card lane recipes not queued', '', '| control | recipe | reason |', '|---|---|---|']
    for u in plan['unmapped']:
        L.append('| %s | %s | %s |' % (u['control'], u['lane'], u['reason']))
    L += ['', '## Untried rows not gridded', '', '| define | kind | reason |', '|---|---|---|']
    for e in plan['excluded_untried']:
        L.append('| `%s` | %s | %s |' % (e['define'], e['kind'], e['reason']))
    L += ['', '## How to queue (not run by this lane)', '',
          '`grid-fast-queue.txt` holds one `lq add m3 CMD %s <tag> \'<command>\'` line per A/A and A/B, in order.' % s['queue_branch'],
          'Only `lane/apple-fast*` branches may target m3 (`~/mojolearn-evidence/lq/lq`); the branch must contain this grid\'s',
          'source (AFT/AFCL cards and the untried defines are on the integration base). Queue the lines in file order:', '',
          '1. A/A lines first: both arms the FAST main build of the binding (they build `<run>-main-<binding>` once per script and',
          '   leave FAST main installed). Their |log(B/A)| is the M3 floor for that workload.',
          '2. Define A/Bs pack by pack: `AFC_DEF_BUILD_TAG=<run>-<pack>` (afc_ab_def.sh stamp reuse) or `AFT_OUT=$HOME/aft-ab/<run>-<pack>`',
          '   with `AFT_SKIP_BUILD=1` after the pack\'s first line, so each pack builds one A/B pair. Each line then copies the',
          '   binding\'s FAST main .so back (both scripts leave arm B installed) and exits with the script\'s status.',
          '3. Env A/Bs (`tools/afc_ab.sh`, no build) run last, on the FAST main build.', '',
          'Then one decision per switch arm (PROMOTE / SPLIT / DELETE / HOLD_QUALITY / NOT_MEASURED), no identity requirement:', '', '```bash',
          'python3 tools/six_lane_grid_decide.py --mode fast --verdicts <evidence>/grid-fast-verdicts.json --quality <evidence>/grid-fast-quality.json',
          '```', '',
          'Regenerate: `python3 tools/six_lane_grid.py --mode fast --vendor apple` (deterministic; `--check` fails if these outputs are stale).', '']
    return '\n'.join(L)


def write_fast_outputs(plan, matrix, build_plan, queue, out_dir=OUT_DIR):
    from six_lane_matrix_io import write_matrix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'grid-plan.json').write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    (out_dir / 'grid-build-plan.json').write_text(json.dumps(build_plan, indent=1, sort_keys=True) + '\n')
    write_matrix(out_dir / 'grid-matrix.json.gz', matrix)
    (out_dir / 'GRID.md').write_text(render_md_fast(plan))
    head = ['# FAST grid queue (planned, NOT queued): run %s, %d lines, box m3 only, branch %s.' % (plan['summary']['run'], len(queue), plan['summary']['queue_branch']),
            '# Generated by tools/six_lane_grid.py --mode fast --vendor apple; queue in file order (A/A, define A/Bs by pack, env A/Bs).']
    (out_dir / 'grid-fast-queue.txt').write_text('\n'.join(head + [q['line'] for q in queue]) + '\n')


def main_fast(out=OUT_DIR, cap=G.CAP, check=False, branch=QUEUE_BRANCH):
    plan, matrix, build_plan, queue = generate_fast(cap=cap, branch=branch)
    if check:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            write_fast_outputs(plan, matrix, build_plan, queue, tmp)
            stale = [n for n in OUTPUTS if not (Path(out) / n).exists() or (Path(tmp) / n).read_bytes() != (Path(out) / n).read_bytes()]
        print(json.dumps(dict(mode=MODE, stale=stale)))
        return 1 if stale else 0
    write_fast_outputs(plan, matrix, build_plan, queue, out)
    print(json.dumps(plan['summary']))
    return 0
