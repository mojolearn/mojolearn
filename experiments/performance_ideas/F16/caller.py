#!/usr/bin/env python3
"""Full estimator prerequisite fixture; never treats private K3 probe as dispatch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
from arima_task import search_cases
def exercise(args):
    from mojolearn import ARIMA
    binding=ARIMA()._extension()
    before=[int(binding.arima_product_df_count(stage)) for stage in (0,1)]
    result=search_cases(args)
    reached=[int(binding.arima_product_df_count(stage))-before[stage] for stage in (0,1)]
    if args.arm=='B':assert all(count>0 for count in reached),'public AutoARIMA never reached admitted compensated production tail'
    result['compensated_product_hits']=reached
    return result
if __name__=='__main__':capture_main(exercise)
