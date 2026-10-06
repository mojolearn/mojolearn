#!/usr/bin/env python3
# F17: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Independent stacked gradient/workspace arms on actual panel order search."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
from arima_task import search_cases
if __name__=='__main__':capture_main(search_cases)
