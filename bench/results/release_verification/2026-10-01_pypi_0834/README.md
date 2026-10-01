# mojolearn 0.8.34

Published macOS arm64, Linux core, NVIDIA and AMD wheels from main at `42b06235e1ec35e83eb02f10ee01b7be63a1c4b5`.

The release reuses the completed checks of merged lanes (validation profile `preverified`). No fresh runtime verification or wheel smoke was run before publication; the 0.8.34 boards on the M3 Ultra, an L40S and an MI300X install it from PyPI. All four published file hashes match the release manifest. See release-provenance.json for the workflows and CHANGELOG.md for what changed.

The Linux sets sm_90a, sm_89 and gfx942 were built on GitHub runners with the binding cache (188 rebuilt, 168 reused from 0.8.33); the macOS wheel on the release Mac with `--build-only`. The published Linux wheels are the packed wheels, not the audit-stage copies: their payload files are byte-identical and their RECORDs validate; only RECORD ordering differs.
