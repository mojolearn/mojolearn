#!/usr/bin/env python3
"""Source integration readiness only; native oracle + actual caller still required."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
paths=['arima/impl/fast_eval_df.mojo','arima/checks/product_df_quality_check.mojo']
missing=[name for name in paths if not (root/name).is_file()]
production=(root/'arima/impl/fast_eval_ws.mojo').read_text()
ready=not missing and 'PRODUCT_DF_ON' in production
print(json.dumps(dict(id='F16',status='source_ready' if ready else 'blocked_prerequisite',missing=missing,
    required=['independent every-component gradient oracle on device','actual AutoARIMA admitted-route reach','selected-order and heldout-forecast quality'],
    timing_authorized=False),indent=2))
sys.exit(0 if ready else 2)
