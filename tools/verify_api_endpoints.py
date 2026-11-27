#!/usr/bin/env python3
"""Developer capture/negative-control runner for the normal verifier's fixtures.

Captures one file per side: the CPU side (device="cpu") and, with a GPU, the GPU side
(device="gpu"), both in this process. The files are the columns verify_api_reference.py admits
(--cpu for the CPU side, --cuda/--hip/--metal for the GPU side)."""
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
    parser.add_argument('--side',default='both',choices=('both','cpu','gpu'),
        help='both (default): the CPU side (device="cpu") and, on an installation with a GPU, the GPU side '
             '(device="gpu"), in this one process; a CPU-only installation records the CPU side only')
    parser.add_argument('--cpu-out',type=Path,help='the CPU side capture when a GPU side also runs '
        '(--side both on an installation with a GPU); --out is then the GPU side')
    args=parser.parse_args()
    from mojolearn._verify_api_runner import capture,registry,TARGETS,differences,sides
    routes=registry()
    targets=json.loads(args.targets.read_text()) if args.targets else TARGETS
    selected=[r['endpoint'] for r in targets if (not args.public or args.public==r['public'])
              and (args.family=='all' or routes.get(r['endpoint'],(None,))[0]==args.family)]
    available=sides()
    run=available if args.side=='both' else [args.side]
    if args.side=='gpu' and 'gpu' not in available:parser.error('no GPU side on this installation')
    if len(run)>1 and not args.cpu_out:parser.error('--side both with a GPU needs --cpu-out for the CPU side')
    # Each side's own file: the last side run is the primary one (the GPU side when one runs).
    paths={side:(args.cpu_out if side=='cpu' and len(run)>1 else args.out) for side in run}
    reference=json.loads(args.reference.read_text()) if args.reference else None
    reports={};summary={};failed_any=False
    for side in run:
        report=capture(paths[side],endpoints=selected,side=side)
        # A device-only endpoint the CPU side records not applicable has no CPU form; it is not a failure.
        failed=[e for e,r in report['endpoints'].items() if r['status'] not in ('captured','not applicable')]
        if args.self_test:
            from mojolearn._verify_api_faults import check_faults
            report['faults']=check_faults(report,retain=lambda name,data:
                paths[side].with_name(paths[side].stem+'-'+name+'.json').write_text(json.dumps(data,sort_keys=True)+'\n'))
        if reference is not None:
            applicable=[e for e in report['expected'] if report['endpoints'].get(e,{}).get('status')!='not applicable']
            report['differences']=differences(dict(report,expected=applicable),
                                              dict(reference,expected=[e for e in reference.get('expected',()) if e in applicable]))
        paths[side].write_text(json.dumps(report,sort_keys=True,indent=1)+'\n')
        reports[side]=report
        summary[side]=dict(out=str(paths[side]),expected=len(selected),completed=len(report['endpoints']),
            failed=len(failed),pending=len(report['pending']),
            not_applicable=sum(r['status']=='not applicable' for r in report['endpoints'].values()),
            faults={k:v['status'] for k,v in report.get('faults',{}).items()},
            differences=len(report.get('differences',[])))
        failed_any=failed_any or bool(failed or report['pending'] or report.get('differences') or
            any(r['status'] not in ('CAUGHT_AND_RESTORED','NOT_APPLICABLE') for r in report.get('faults',{}).values()))
    print(json.dumps(summary,sort_keys=True))
    code=int(failed_any)
    try_retain(retain_verification, 'api-' + args.family + ('-faults' if args.self_test else ''),
               [paths[side] for side in run], metadata=dict(family=args.family, public=args.public,
                                         self_test=args.self_test, sides=run, exit_code=code))
    return code


if __name__=='__main__':raise SystemExit(main())
