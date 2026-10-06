"""Operational regression check; never connects to cloud or runs algorithms."""
import ast
import json
import pathlib
import tempfile
import traceback

source=pathlib.Path(__file__).with_name('stream-repairs.py')
tree=ast.parse(source.read_text())
functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['route_retired','publish_queues']]
with tempfile.TemporaryDirectory() as temp:
    root=pathlib.Path(temp)
    for route in ['specific','default']:
        (root/(route+'-capture')).mkdir()
        (root/(route+'-capture/manager-status.json')).write_text(json.dumps({'status':'MANAGING'}))
        (root/(route+'-owner')).mkdir()
        (root/(route+'-owner/config.json')).write_text(json.dumps({'ssh':[route]}))
        (root/(route+'-repair-queue.json')).write_text('[]')
    calls=[]; normalized=[]
    ns={'E':root,'json':json,'traceback':traceback,'normalize':normalized.append,
        'cmd':lambda argv,**kwargs:calls.append(argv[1]),'queue_install_command':lambda:'publish',
        'notify':lambda *args:None}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
    (root/'specific-capture/manager-status.json').write_text(json.dumps({'status':'TERMINATED_VERIFIED'}))
    assert ns['publish_queues']()=={}
    assert calls==['default'] and normalized==['specific','default']
    (root/'specific-capture/manager-status.json').write_text(json.dumps({'status':'MANAGING'}))
    calls.clear();normalized.clear()
    def publish(argv,**kwargs):
        calls.append(argv[1])
        if argv[1]=='specific':raise RuntimeError('simulated SSH failure')
    ns['cmd']=publish
    errors=ns['publish_queues']()
    assert list(errors)==['specific'] and calls==['specific','default']
    assert normalized==['specific','default']
print('PASS retired route never SSHs, captured normalization retained, route failure does not block other route')
