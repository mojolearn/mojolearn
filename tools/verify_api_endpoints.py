#!/usr/bin/env python3
"""Developer capture/negative-control runner for the normal verifier's fixtures."""
import argparse
import json
from pathlib import Path

from retain_run_summary import retain_verification, try_retain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--targets',type=Path,help='retained subset of audit rows')
    parser.add_argument('--family',default='all',choices=('all','metadata','estimators','utilities',
        'optimizers','neural','training','programs','models','misc'))
    parser.add_argument('--public',help='one exact public class')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    from mojolearn._verify_api_runner import capture,registry,TARGETS,differences
    routes=registry()
    targets=json.loads(args.targets.read_text()) if args.targets else TARGETS
    selected=[r['endpoint'] for r in targets if (not args.public or args.public==r['public'])
              and (args.family=='all' or routes.get(r['endpoint'],(None,))[0]==args.family)]
    report=capture(args.out,endpoints=selected)
    failed=[e for e,r in report['endpoints'].items() if r['status']!='captured']
    if args.self_test:
        from mojolearn._verify_api_faults import check_faults
        report['faults']=check_faults(report,retain=lambda name,data:
            args.out.with_name(args.out.stem+'-'+name+'.json').write_text(json.dumps(data,sort_keys=True)+'\n'))
    if args.reference:report['differences']=differences(report,json.loads(args.reference.read_text()))
    args.out.write_text(json.dumps(report,sort_keys=True,indent=1)+'\n')
    print(json.dumps(dict(expected=len(selected),completed=len(report['endpoints']),failed=len(failed),
        pending=len(report['pending']),faults={k:v['status'] for k,v in report.get('faults',{}).items()},
        differences=len(report.get('differences',[])))))
    code = int(bool(failed or report['pending'] or report.get('differences') or
        any(r['status']!='CAUGHT_AND_RESTORED' for r in report.get('faults',{}).values())))
    try_retain(retain_verification, 'api-' + args.family + ('-faults' if args.self_test else ''),
               [args.out], metadata=dict(family=args.family, public=args.public,
                                         self_test=args.self_test, exit_code=code))
    return code


if __name__=='__main__':raise SystemExit(main())
