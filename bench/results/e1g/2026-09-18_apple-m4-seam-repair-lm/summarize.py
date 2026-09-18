import json,glob,os,statistics
for d in sorted(glob.glob('*-*-*/')):
    ev=[json.loads(l) for l in open(d+'events.jsonl')]
    steps=[e for e in ev if e.get('event')=='step_end']
    wit=[e for e in ev if 'witness' in e.get('event','') or 'hashes' in e]
    secs=[round(e['seconds'],2) for e in steps]
    late=[e['seconds'] for e in steps if not e.get('first_call_includes_setup')]
    ws=[]
    for e in ev:
        h=e.get('hashes') or e.get('witness')
        if isinstance(h,dict): ws.append((e.get('step'),h.get('gradients','')[:12],h.get('parameters','')[:12]))
    print(d, 'steps',secs,'median(non-setup)',round(statistics.median(late),2) if late else None)
    print('   witnesses', ws)
