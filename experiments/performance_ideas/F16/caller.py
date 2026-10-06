#!/usr/bin/env python3
"""Full estimator prerequisite fixture; never treats private K3 probe as dispatch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
from arima_task import search_cases
if __name__=='__main__':capture_main(search_cases)
