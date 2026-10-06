"""Run the opt-in full LARS direct-X recipe and review held-out quality.

The parent controller supplies the original deadline, machine lock, resources,
and frozen source. This driver never changes the original LARS recipe or receipt.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[3]
RACE = 'algos/lars-direct/istella/rows=full'
ARM = 'sklearn-cpu'
os.environ['MOJOLEARN_BENCH_LARS_DIRECT'] = '1'
sys.path.insert(0, str(SOURCE / 'tools'))
import bench_board as board

original = board.plan_races


def selected(*args, **kwargs):
    if (os.environ.get('REPAIR_RACE_ID') != RACE
            or os.environ.get('REPAIR_ARM') != ARM):
        raise SystemExit('Select the explicit full LARS direct-X opponent recipe')
    races = [race for race in original(*args, **kwargs) if race['id'] == RACE]
    if len(races) != 1 or ARM not in races[0]['opponents']:
        raise SystemExit('Expected exactly one full LARS direct-X race')
    return [dict(races[0], arms=[ARM], opponents=[ARM], our_arms={})]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    args, _ = parser.parse_known_args()
    board.plan_races = selected
    code = board.main()
    if code not in (None, 0):
        return code
    raw = json.loads((args.out / 'board.json').read_text())
    race = raw.get('races', {}).get(RACE, {})
    cells = race.get('cells', [])
    quality = cells[0].get('quality', {}) if len(cells) == 1 else {}
    r2, rmse = quality.get('r2'), quality.get('rmse')
    # R2=0 is the constant held-out-mean predictor, a dimension-independent
    # baseline, not a dataset-tuned tolerance. Finite alone admitted the original
    # catastrophic predictions. This guard qualifies this new recipe only.
    accepted = (race.get('status') == 'done' and len(cells) == 1
                and cells[0].get('arm') == ARM and cells[0].get('status') == 'ok'
                and quality.get('finite') is True and not quality.get('error')
                and isinstance(r2, (int, float)) and math.isfinite(r2) and r2 >= 0
                and isinstance(rmse, (int, float)) and math.isfinite(rmse) and rmse >= 0)
    review = dict(race=RACE, arm=ARM,
                  status='QUALITY_ACCEPTED' if accepted else 'QUALITY_FAILED',
                  acceptance_rule='finite predictions, finite RMSE >= 0, R2 >= 0 '
                                  '(held-out constant-mean baseline)',
                  quality=quality, original_board_receipt='board.json',
                  default_promotion=False)
    (args.out / 'quality-review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, sort_keys=True), flush=True)
    return 0 if accepted else 3


if __name__ == '__main__':
    raise SystemExit(main())
