#!/usr/bin/env python3
"""Group raw diagnostic ticks by call; nested phase times are not additive."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
summary={}
for path in sorted(args.directory.glob('*.log')):
    if not path.name.startswith(('saved-binding-','phase-timers-')):continue
    ticks=collections.defaultdict(float);units={};calls=[]
    for line in path.read_text().splitlines():
        tick=re.fullmatch(r'timing (\S+) ([\d.eE+-]+) (ms|count)',line)
        if tick:
            name,value,unit=tick.groups();ticks[name]+=float(value);units[name]=unit
        call=re.fullmatch(r'CALL (\d+) (\S+) ([\d.eE+-]+) ms',line)
        if call:
            n,lane,wall=call.groups()
            calls.append({'call':int(n),'lane':lane,'wall_ms':float(wall),'ticks':dict(ticks)})
            ticks.clear()
    steady=calls[1:]
    names=sorted(set().union(*(c['ticks'] for c in steady))) if steady else []
    summary[path.stem]={'calls':calls,'median_wall_after_first_ms':statistics.median(c['wall_ms'] for c in steady) if steady else None,
        'median_ticks_per_call_after_first':{name:{'value':statistics.median(c['ticks'].get(name,0) for c in steady),'unit':units[name]} for name in names},
        'note':'Repeated names summed within each call; nested phases overlap. Counted source call sites exclude runtime internals and timer-added waits.'}
(args.directory/'tick-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:{'calls':len(value['calls']),'steady_ms':value['median_wall_after_first_ms']} for key,value in summary.items()},indent=2))
