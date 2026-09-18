import json,glob
res={}
for d in sorted(glob.glob('*-*-*/result.json')):
    r=json.load(open(d)); name=d.split('/')[0]
    sw=r.get('step_witnesses')
    res[name]=sw
    fw=r.get('final_witness')
    print(name, 'final', {k:str(v)[:12] for k,v in (fw or {}).items()} if isinstance(fw,dict) else str(fw)[:80])
    if isinstance(sw,list):
        for s in sw: print('   ', {k:(str(v)[:12]) for k,v in s.items()} if isinstance(s,dict) else str(s)[:120])
