"""Honor the explicitly authorized existing CPU opponent, before GPU preference."""
import os,sys
from pathlib import Path
SOURCE=Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple/source-47301d12')
sys.path.insert(0,str(SOURCE/'tools'))
import bench_board as board
original=board.plan_races
def selected(*args,**kwargs):
 race,arm=os.environ['REPAIR_RACE_ID'],os.environ['REPAIR_ARM']
 assert race in ('classical/ols/taxi/rows=full','classical/pca/taxi/rows=full','classical/kmeans/taxi/rows=full')
 assert arm=='sklearn-cpu'
 # This measurement selector restores the existing supported roster solely
 # for the owner's explicit CPU-opponent request. No estimator dispatch,
 # numerical setting, input, or general board preference changes.
 preference=board.gpu_opponents_first
 try:
  board.gpu_opponents_first=lambda roster:list(roster)
  races=[r for r in original(*args,**kwargs) if r['id']==race]
 finally:board.gpu_opponents_first=preference
 assert len(races)==1 and arm in races[0]['opponents']
 return [dict(races[0],arms=[arm],opponents=[arm],our_arms={})]
board.plan_races=selected
if __name__=='__main__':raise SystemExit(board.main())
