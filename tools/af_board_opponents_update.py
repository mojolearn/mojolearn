#!/usr/bin/env python3
"""Import captured M3 opponent cells, preserving own measurements and resource provenance."""
import argparse,copy,hashlib,json,math,pathlib
import af_board_render as render
import bench_board as bb

def merge(board, snapshot, resources, digest, evidence):
    assert snapshot['box']['gpu']['name']==board['box']['gpu']['name']=='Apple M3 Ultra'
    assert resources['cpu_count']==28 and all(v is None for v in resources['thread_caps'].values())
    result=copy.deepcopy(board);records={**result.get('races',{}),**result.get('extra_races',{})}
    counts={'imported':0,'successful':0,'failed':0,'stored_skipped':0,'already_current':0}
    for rid, incoming in snapshot['races'].items():
        cells=[c for c in incoming.get('cells',[]) if not c.get('stored')]
        counts['stored_skipped']+=len(incoming.get('cells',[]))-len(cells)
        assert all(not render.is_ours(c) for c in incoming.get('cells',[]))
        if not cells:continue
        target=records.get(rid)
        if target is None:
            target={k:copy.deepcopy(v) for k,v in incoming.items() if k not in ['cells','infer_cells']}
            target['cells']=[];target['infer_cells']=[]
            for mode in ['fast','identical']:target[mode+'_page']={'status':'Opponent full measurement; no matching own cell'}
            result.setdefault('extra_races',{})[rid]=target;records[rid]=target
        own=[c for c in target['cells'] if render.is_ours(c)]
        for cell in cells:
            # Exact race ID is necessary but not sufficient to compare old own rows.
            assert all(c.get('shape') and c['shape']==cell.get('shape') for c in own),('shape mismatch',rid)
            c=copy.deepcopy(cell);old=next((x for x in target['cells'] if x['arm']==c['arm'] and not render.is_ours(x)),None)
            if isinstance((old or {}).get('source'),dict) and old['source'].get('opponent_snapshot_sha256')==digest:
                counts['already_current']+=1;continue
            if c['status']=='ok':
                assert c.get('rounds')==1 and len(c.get('times_ms',[]))==1
                assert math.isfinite(c['median_ms']) and c['median_ms']>0
                assert c.get('warmup_ms') is not None
                counts['successful']+=1
            else:counts['failed']+=1
            if old:target.setdefault('opponent_history',[]).append(copy.deepcopy(old))
            c['source']={'opponent_snapshot_sha256':digest,'evidence':evidence,'finished':incoming.get('finished'),'original_source':c.get('source')}
            c['resource_policy']={'cpu_allocation':'full-machine-uncapped','available_logical_cpus':resources['cpu_count'],'thread_caps':resources['thread_caps'],'effective_utilization':'Library dependent; not a claim that all cores were busy','harness_sha':resources['harness_sha']}
            target['cells']=[x for x in target['cells'] if render.is_ours(x) or x['arm']!=c['arm']]+[c]
            counts['imported']+=1
        arms={c['arm'] for c in cells}
        incoming_infer=[copy.deepcopy(c) for c in incoming.get('infer_cells',[]) if c['arm'] in arms]
        if incoming_infer:
            target['infer_cells']=[c for c in target.get('infer_cells',[]) if c['arm'] not in arms]+incoming_infer
        bb.add_ratios(target['cells'])
    notes=['CPU opponents in the 2026-10-05 fresh sweep received the full 28-core M3 Ultra allocation with no imposed CPU thread cap. Library parallelism varies; all-core availability does not mean all cores were busy.', 'Stored historical opponent cells keep their original resource provenance. Fresh measurements use one excluded warmup and one scored sample; failures remain failures.', 'At the user-directed transition to missing-only opponents, the active LinearSVR/Istella attempt was interrupted without admitting a timing. Completed measurements were preserved; the replacement missing-only sweep remains active.']
    result['opponent_resource_notes']=notes
    for mode in ['fast','identical']:
        p=result.setdefault(mode+'_page',{});p['notes_md']=[x for x in p.get('notes_md',[]) if not x.startswith('- CPU resource policy:')]+['- CPU resource policy: '+x for x in notes]
    result.setdefault('opponent_imports',{})[digest]={'evidence':evidence,'counts':counts,'resource_policy':resources}
    return result,counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--resources',required=True);p.add_argument('--board-dir',default=render.BOARD_DIR);p.add_argument('--docs-dir',default=render.DOCS);a=p.parse_args()
    raw=pathlib.Path(a.snapshot).read_bytes();digest=hashlib.sha256(raw).hexdigest();path=pathlib.Path(a.board_dir)/'board.json'
    board,counts=merge(json.loads(path.read_text()),json.loads(raw),json.loads(pathlib.Path(a.resources).read_text()),digest,a.snapshot)
    path.write_text(json.dumps(board,indent=1)+'\n');render.write_all(a.board_dir,a.docs_dir,board)
    errors=render.check(a.board_dir,a.docs_dir);assert not errors,errors
    print(json.dumps(counts))
if __name__=='__main__':main()
