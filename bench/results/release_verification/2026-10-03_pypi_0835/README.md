# mojolearn 0.8.35

Published macOS arm64, Linux core, NVIDIA and AMD wheels from main at `8897404daf6b4727c81602b707e9119d7686711a`.

The release reuses the completed checks on merged main (validation profile `preverified`): every binding built on NVIDIA (sm_89), AMD (gfx942) and Apple, and the verifier cross-check agreed bit for bit on all 48 compared cell parts across NVIDIA, AMD, Apple and the host column (main b2f5fa7f2; logs in R2 under measurements/2026-10-03/). No fresh installed wheel smoke was run before publication. All four published file hashes match the release manifest; `pip install mojolearn==0.8.35` from PyPI imports and fits on the release Mac.

The Linux sets sm_90a, sm_89 and gfx942 were built on GitHub runners; the macOS wheel on the release Mac with `--build-only`. See CHANGELOG.md for what changed and which outputs change bits from 0.8.34.
