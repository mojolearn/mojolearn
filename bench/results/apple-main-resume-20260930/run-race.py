import argparse,sys,pathlib
p=argparse.ArgumentParser(add_help=False);p.add_argument('--race-id',required=True);a,args=p.parse_known_args()
sys.path.insert(0,str(pathlib.Path.cwd()/'tools'))
import bench_board as board
original=board.plan_races
def selected(*pos,**kw):
 races=[r for r in original(*pos,**kw) if r['id']==a.race_id]
 if len(races)!=1:raise SystemExit('Expected exactly one latest-main race: '+a.race_id)
 return races
board.plan_races=selected
raise SystemExit(board.main(args))
