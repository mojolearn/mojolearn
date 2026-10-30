"""Select complete whole files from raw JUnit; never manufacture a new JUnit."""
from pathlib import Path
import qualify_native_runtime_installed as q


def project(tests, original, selected, collected):
    q.require(selected and len(selected)==len(set(selected)) and set(selected)<=set(original),
              'invalid whole-file projection selection')
    q.require(isinstance(collected,list) and collected and len(collected)==len(set(collected)),
              'whole-file projection requires original worker collection evidence')
    q.require({n.partition('::')[0] for n in collected}==set(original),
              'worker collection differs from original whole-file selection')
    observed=[c.get('exact_node','') for c in tests['cases']]
    q.require(q.exact_node_summary(collected,observed)['complete'],
              'original JUnit is incomplete or differs from worker collection')
    kept=[]
    for case in tests['cases']:
        filename,node=case['exact_node'].split('::',1)
        words=case['classname'].split('.')
        q.require(Path(filename).stem in words and node.split('::')[-1]==case['name'],
                  'JUnit node disagrees with its class/name')
        if filename in selected:kept.append(case)
    keys={(c['classname'],c['name']) for c in kept}
    failures=[r for r in tests['failures'] if (r['classname'],r['name']) in keys]
    skipped=[r for r in tests['skipped'] if (r['classname'],r['name']) in keys]
    return dict(total=len(kept),passed=sum(c['outcome']=='passed' for c in kept),
                failures=failures,skipped=skipped,cases=kept)
