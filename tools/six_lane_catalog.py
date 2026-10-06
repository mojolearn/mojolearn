#!/usr/bin/env python3
"""Pure source catalog adapter for the six-lane integration; imports no estimators."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'experiments/six_lane_integration'
QUALIFICATION = dict(timing='unmeasured', quality='not_assessed', identity='not_assessed', acceptance='pending', promoted=False, runtime_reach='unverified')


def read(path):
    return json.loads((ROOT / path).read_text())


def unique(items):
    return list(dict.fromkeys(x for x in items if isinstance(x, str) and x))


def seq(value):
    return value if isinstance(value, list) else [] if value is None else [value]


def metadata_module(name):
    # These adapters contain only source metadata/discovery functions. Never load
    # workload runners here: some import runtime libraries at module scope.
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(defines=(), env=None, runtime=None):
    return dict(defines=unique(defines), environment=env or {}, runtime=runtime or {})


def config_from(e, arm):
    name = 'candidate' if arm == 'A' else 'baseline'
    nested = e.get(name)
    if isinstance(nested, dict):
        return config(nested.get('compile_defines', []), nested.get('environment', {}), nested.get('runtime', nested.get('runtime_settings', {})))
    return config(e.get(name+'_defines', e.get(arm, [])), e.get(name+'_env', {}), e.get(name+'_settings', {}))


def reference(path, pointer):
    return dict(path=path, pointer=pointer, sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())


def normalize(lane, source, e, path, *, original=None, workloads=None, bindings=None, callers=None, variants=None):
    ident=original or e['id']; key=lane+'.'+ident
    paths=unique(seq(e.get('implementation_paths'))+seq(e.get('source_paths'))+seq(e.get('production_paths'))+seq(e.get('primary_source_paths'))+seq(e.get('shared_source_paths')))
    calls=unique(seq(callers)+seq(e.get('production_callers'))+seq(e.get('caller_paths')))
    gaps=seq(e.get('source_gaps'))+seq(e.get('remaining_gaps'))+seq(e.get('remaining_work'))+seq(e.get('limitations'))+seq(e.get('remaining_integration_limitations'))
    variant_rows=variants
    if variant_rows is None:
        variant_rows=[('default', e)]
        v=e.get('variants', {})
        for name, value in (v.items() if isinstance(v,dict) else ((x['name'],x) for x in v)):
            merged={**e,**value}
            for arm in ('baseline','candidate'):
                if isinstance(e.get(arm),dict) and isinstance(value.get(arm),dict): merged[arm]={**e[arm],**value[arm]}
            variant_rows.append((name,merged))
    arms=[]
    for name,v in variant_rows:
        A=config_from(v,'A'); B=config_from(v,'B')
        # B is the frozen shipped incumbent. Lane-authored comparison controls
        # are retained for attribution, never silently treated as the incumbent.
        incumbent=config([],{}, {})
        changed_reference=bool(B['defines'] or B['environment'] or B['runtime'])
        arms.append(dict(id=key+':'+name,name=name,A=A,B=incumbent,authored_B=B,
                         reference_policy='frozen incumbent defaults; authored reference retained separately',
                         authored_reference_differs=changed_reference,
                         prerequisites=seq(v.get('dependencies'))+seq(v.get('prerequisites')),
                         conflicts=seq(v.get('mutually_exclusive_with'))+seq(v.get('incompatible_defines')),
                         source_selectable=v.get('selectable',True),
                         source_gaps=seq(v.get('source_gaps'))+seq(v.get('blocker')),
                         parameters=v.get('parameters',e.get('required_compile_parameters',{})),
                         workloads=workloads if workloads is not None else v.get('required_workload_keys',e.get('workloads',[]))))
    return dict(id=key,lane=lane,source_id=source,original_id=ident,title=e.get('title',ident),
                source_record=reference(path,ident),implementation_paths=paths,production_callers=calls,
                bindings=bindings or e.get('required_bindings',[]),affected_workloads=workloads if workloads is not None else e.get('workloads',[]),
                affected_estimators=e.get('estimators',e.get('affected_estimators',e.get('intended_models',[]))),
                prerequisites=seq(e.get('dependencies'))+seq(e.get('prerequisites')),
                conflicts=seq(e.get('mutually_exclusive_with'))+seq(e.get('incompatible_defines')),
                gaps=gaps,implementation=dict(idea=True,programmed=bool(paths),production_wired=bool(calls or e.get('caller_integration')),harness_wired='source_adapter',compiled=[]),
                source_status=e.get('source_status',e.get('implementation_status',e.get('status','idea'))),
                qualification=dict(QUALIFICATION),mode='fast' if lane.startswith('AF.') else 'identical',
                vendors=['apple'] if lane.startswith('AF.') else ['nvidia','amd','apple','host'],
                new_defaults_enabled=False,arms=arms)


def discover():
    entries=[]; interactions=[]; aliases=[]
    ci='experiments/classical_identical_ideas/'
    ledger={e['id']:e for e in read(ci+'implementation_ledger.json')['entries']}
    catalog=read(ci+'catalog.json')
    for e in catalog['experiments']:
        n=int(e['id'][1:])
        if 45<=n<=51:continue
        p=ci+e['id']+'/manifest.json';m=read(p);v=[]
        for name,c in m['configurations'].items():
            if name.startswith(('interaction','complete')):continue
            v.append((name,{**m,**c}))
        item=normalize('I.C','identical.classical',{**e,**ledger[e['id']],**m},p,workloads=m.get('required_workload_keys',[]),variants=v or [('pending',m)])
        for arm in item['arms']:arm['workloads']=m['configurations'].get(arm['name'],{}).get('required_workload_keys',item['affected_workloads'])
        entries.append(item)
    t='experiments/trees_identical_20261006/'
    inv=read(t+'NEW_TREE_AB_INVENTORY.json')
    for e in inv['records']:
        path=next((str(p.relative_to(ROOT)) for p in (ROOT/t).glob('*/'+e['id']+'.json') if p.parent.name!='overlaps'),t+'NEW_TREE_AB_INVENTORY.json')
        raw=read(path) if path!=t+'NEW_TREE_AB_INVENTORY.json' else e
        controls=e.get('controls',[])
        workloads=unique([w for c in controls for w in c.get('full_dataset_recipes',[])])
        item=normalize('I.T','identical.trees',{**raw,**e},path,workloads=workloads,
            callers=[p for c in controls for p in c.get('production_callers',[])],variants=[(c['name'],c) for c in controls] or [('pending',{})])
        for arm,c in zip(item['arms'],controls):arm['workloads']=c.get('full_dataset_recipes',workloads)
        entries.append(item)
    for e in inv['classical_overlap_cards']:
        aliases.append(dict(id='I.C.'+e['id'],kind='cross_link',members=['I.T.'+x for x in e['shared_implementations']],reason=e['overlap_explanation']))
    nn='experiments/neural_identical_ab/arms.json'
    for e in read(nn)['experiments']:
        item=normalize('I.N','identical.neural.r3',e,nn)
        # Enumerate the finite, explicitly authored numerical subarms.
        parameter_options={'NN03':('MOJOLEARN_IDN_NEURAL_LEAF',[64,128,256]),'NN04':('MOJOLEARN_IDN_NEURAL_CHAINS',[2,4]),'NN09':('MOJOLEARN_IDN_NEURAL_STAGE_DEPTH',[1,2])}
        if e['id'] in parameter_options:
            key,values=parameter_options[e['id']]
            for value in values:
                arm=json.loads(json.dumps(item['arms'][0]));arm['name']=key.lower()+'-'+str(value);arm['id']=item['id']+':'+arm['name'];arm['A']['defines']=[d for d in arm['A']['defines'] if d.split('=')[0]!=key]+[key+'='+str(value)];item['arms'].append(arm)
        entries.append(item)
    niroot='experiments/neural_identical_20261006/'
    nimap=read(niroot+'integration.json')
    for name in ['gemm_cnn','transformer_training','sequence','neural_aux']:
        p=niroot+name+'.json'
        for e in read(p)['experiments']:
            mapping=nimap['ideas'][e['id']]
            entries.append(normalize('I.N','identical.neural.v2',e,p,workloads=[dict(id=w,**nimap['workloads'][w]) for w in mapping['workloads']],bindings=mapping['binding_families']))
    afc='experiments/apple_fast_classical_20261006/'
    afcw=read(afc+'full_workloads.json');afcb=read(afc+'build_bindings.json')['cards']
    for name in ('geometry','linear','preprocessing','trees'):
        p=afc+'lanes/'+name+'.json'
        for e in read(p)['entries']:
            w=next(w for w in afcw['entries'] if w['id']==e['id'])
            entries.append(normalize('AF.C','apple_fast.classical',e,p,workloads=w['recipes'],bindings=afcb[e['id']]['affected_bindings'],callers=e.get('affected_callers',[])))
    aft=metadata_module('apple_fast_tree_integration')
    for name in 'FGNP':
        p='experiments/apple_fast_trees/'+name+'.json'
        for e in read(p)['cards']:
            route=aft.route(e['id'])
            item=normalize('AF.T','apple_fast.trees',e,p,workloads=route['saved_workloads'],bindings=route['candidate_bindings'])
            item['route']=route;entries.append(item)
    afn=metadata_module('apple_fast_neural_ideas')
    for name in ['attention','training','mamba','cnn_embedding','interactions']:
        p='experiments/apple_fast_neural_20261006/'+name+'.json'
        for e in read(p)['cards']:
            vs=[(v['name'],{**e,**v}) for v in e['variants']]
            routes=[]
            for name2,v in vs: routes+=afn.target_routes(e['id'],name2,v.get('candidate_defines',[]))
            item=normalize('AF.N','apple_fast.neural',e,p,variants=vs,bindings=unique([r['binding'] for r in routes]),workloads=routes)
            if name=='interactions':item['kind']='interaction';interactions.append(item)
            else:entries.append(item)
    # Explicit equivalence after source reconciliation; narrower subarms keep IDs.
    for ni,nn,scope in [('NI24','NN28','owned ByteLM prefill cache omission'),('NI32','NN51','owned validated token admission'),('NI33','NN52','CE weights and logits gradient same cell'),('NI43','NN38','descending suffix seed cache'),('NI53','NN44','stable expert scatter')]:
        aliases.append(dict(id='I.N.'+ni+':default',kind='equivalent_implementation',canonical='I.N.'+nn+':default',scope=scope,legacy_controls_preserved=True))
    # Retained source-specified configurations, not the powerset of switches.
    for e in read(ci+'interactions.json')['interactions']+[dict(read(ci+'C01/manifest.json')['configurations']['complete_classical'], id='complete_classical')]:
        members=e.get('candidate_ids',[]);ident=e.get('id',e.get('configuration','complete'))
        item=normalize('I.C.X','identical.classical',dict(e,id=ident,title=ident),ci+'interactions.json',variants=[(e.get('configuration','combined'),e)],workloads=e.get('required_workload_keys',[]));item.update(kind='interaction',members=['I.C.'+x for x in members],rationale='Authored classical dataflow interaction; individual arms precede combined configuration');interactions.append(item)
    tree_records={e['original_id']:e for e in entries if e['lane']=='I.T'}
    for e in read(t+'interactions.json')['interactions']:
        interactions.append(dict(id='I.T.X.'+e['id'],members=['I.T.'+m for m in e['members']],kind='interaction',rationale=e.get('scope','Authored tree stage interaction'),source_record=reference(t+'interactions.json',e['id']),selection_only=True))
    # Concrete complete-configuration proposals choose one schedule/graph per
    # seam. Alternatives remain individually selectable and visibly incompatible.
    groups=[('I.N.X.gemm-memory',['I.N.NN02','I.N.NN11','I.N.NN12'],'Streaming planes, fold and retained owner share workspace'),
            ('I.N.X.attention-training',['I.N.NN25','I.N.NN26','I.N.NN27','I.N.NN28'],'Norm, activation, rotary and training-cache passes in one block'),
            ('I.N.X.embedding-loss',['I.N.NN49','I.N.NN50','I.N.NN51','I.N.NN52'],'Embedding ownership and CE share token admission'),
            ('I.N.X.optimizer-arena',['I.N.NN55','I.N.NN56','I.N.NN60','I.N.NN62'],'Clipping, optimizer transaction and views share lifetimes'),
            ('I.N.X.cnn-memory',['I.N.NN14','I.N.NN45','I.N.NI10','I.N.NI14'],'Convolution tiles, ReLU and backward tape lifetimes'),
            ('AF.T.X.histogram',['AF.T.F01','AF.T.F02','AF.T.F04','AF.T.F05'],'Histogram workspace and scoring pipeline'),
            ('AF.T.X.quantization',['AF.T.G05','AF.T.G06','AF.T.G09','AF.T.G10'],'Quantization and categorical preprocessing'),
            ('AF.T.X.prediction',['AF.T.P02','AF.T.P04','AF.T.P05','AF.T.P06'],'Prediction traversal and link stages'),
            ('AF.T.X.shap',['AF.T.P07','AF.T.P08','AF.T.P09','AF.T.P10'],'SHAP scratch and row/fold work distribution'),
            ('AF.T.X.dart',['AF.T.P11','AF.T.P12'],'DART residual and score-add stages'),
            ('AF.C.X.prep-linear',['AF.C.AFCL-P01','AF.C.AFCL-L01'],'Preparation and Gram shared full linear callers')]
    for ident,members,why in groups:interactions.append(dict(id=ident,members=members,kind='interaction',rationale=why,selection_only=True))
    for lane in ('I.C','I.T','I.N','AF.C','AF.T','AF.N'):
        members=[e['id'] for e in entries if e['lane']==lane]
        interactions.append(dict(id=lane+'.X.complete-proposed',members=members,kind='complete_proposed',selection_only=True,
            rationale='All new mechanisms with declared alternatives unresolved explicitly; this proposal is blocked when members conflict, lack controls, or alter frozen workload settings. Never enabled by default.'))
    for e in entries+interactions:
        e.setdefault('qualification',dict(QUALIFICATION));e.setdefault('new_defaults_enabled',False)
    return entries,interactions,aliases


def source_graph():
    paths={str(p.relative_to(ROOT)) for p in ROOT.rglob('*.mojo') if '.git' not in p.parts and '.pixi' not in p.parts}
    graph={};texts={}
    for path in paths:
        text=(ROOT/path).read_text();texts[path]=text;imports=[]
        for name in re.findall(r'^\s*from\s+([\w.]+)\s+import',text,re.M):
            rel=name.replace('.','/')
            if rel+'.mojo' in paths:imports.append(rel+'.mojo')
            elif rel+'/__init__.mojo' in paths:imports.append(rel+'/__init__.mojo')
        graph[path]=set(imports)
    closures={}
    for binding in sorted(p for p in paths if p.startswith('bindings/_mojolearn_')):
        seen=set();todo=[binding]
        while todo:
            p=todo.pop()
            if p in seen:continue
            seen.add(p);todo.extend(graph.get(p,()))
        closures[binding]=seen
    return closures,texts


def catalog_document():
    entries,interactions,aliases=discover();closures,texts=source_graph()
    for e in entries:
        paths={p.split(':')[0] for p in e['implementation_paths']+e['production_callers']}
        e['source_binding_reach']=[b for b,reach in closures.items() if paths & reach]
        e['source_binding_reach_policy']='Conservative import closure, not observed runtime reach or template instantiation proof.'
        for arm in e['arms']:
            arm['define_sources']={d.split('=')[0]:[p for p,t in texts.items() if d.split('=')[0] in t and p!='core/six_lane_experiment_guards.mojo'] for d in arm['A']['defines']}
            missing=[d for d,ps in arm['define_sources'].items() if not ps]
            arm['source_gaps'] += ['Define not referenced in retained Mojo source: '+d for d in missing]
    return dict(schema='mojolearn.six-lane-catalog/1',base_main=read('experiments/six_lane_integration/inputs.json')['base_main'],entries=entries,interactions=interactions,aliases=aliases,qualification=dict(QUALIFICATION))
