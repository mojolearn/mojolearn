#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/amd-step-time (2026-09-24): mint the presigned GETs a step-time box
needs to replay T3 steps (read only; nothing is written to R2):

    python3 tools/amd_step_time_urls.py <outdir> [ckpt_key ...]

writes <outdir>/tokens.json (the pinned FineWeb-Edu stream's parts and
manifest), <outdir>/ckpt.json ({file name: url}) and <outdir>/recipe.url,
the URL files tools/amd_step_time_session.sh fetch reads from /root/urls/.
Credentials stay on this machine (tools/dataset_store.sh presign)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_segment_leg import presign_get, tokens_urls  # noqa: E402

RUN = "runs/t3/2026-09-22"
TOKENS = "corpus/fineweb-edu-10BT/tokens/mojolearn-bpe-fineweb-edu-50257-v1"


def main(argv):
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    keys = argv[2:] or [RUN + "/A/1/ckpt_00000100.blm"]
    secs = 12 * 3600
    (out / "tokens.json").write_text(json.dumps(tokens_urls(TOKENS, secs)))
    (out / "ckpt.json").write_text(json.dumps({k.rsplit("/", 1)[1]: presign_get(k, secs) for k in keys}))
    (out / "recipe.url").write_text(presign_get(RUN + "/recipe.json", secs))
    print("wrote", out, "checkpoints:", [k.rsplit("/", 1)[1] for k in keys])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
