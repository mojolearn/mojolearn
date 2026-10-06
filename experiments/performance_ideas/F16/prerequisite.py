#!/usr/bin/env python3
"""Fail closed on the still-private K3 product integration prerequisite."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
source=(root/'arima/impl/fast_scalar_df.mojo').read_text()
callers=[str(p.relative_to(root)) for p in (root/'arima').rglob('*.mojo') if p.name!='fast_scalar_df.mojo' and 'k3_df_parts_kernel' in p.read_text()]
packet=dict(id='F16',status='blocked_prerequisite',product_callers=callers,
    reason='Compensated K3 uses supplied AR1 state only; initializer, full covariance and retained-low-word optimizer gradient are not integrated into actual AutoARIMA',
    required=['actual initializer/covariance preservation','per-component independent likelihood/gradient oracle','public selected-order and forecast quality'],
    evidence_paths=['arima/impl/fast_scalar_df.mojo','docs/apple-fast/ARIMA_K3_DF_GPU_PLAN.txt','tools/arima_k3_df_gpu_quality.py'],timing_authorized=False)
print(json.dumps(packet,indent=2));sys.exit(2 if not callers else 1)
