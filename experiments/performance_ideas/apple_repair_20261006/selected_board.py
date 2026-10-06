"""Run one explicitly selected opponent cell through the full board driver."""
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'tools'))
import bench_board as board

original = board.plan_races


def selected(*args, **kwargs):
    race_id, arm = os.environ['REPAIR_RACE_ID'], os.environ['REPAIR_ARM']
    races = [r for r in original(*args, **kwargs) if r['id'] == race_id]
    assert len(races) == 1 and arm in races[0]['opponents']
    assert not arm.startswith('ours')
    return [dict(races[0], arms=[arm], opponents=[arm], our_arms={})]


board.plan_races = selected
if __name__ == '__main__':
    board.main()
