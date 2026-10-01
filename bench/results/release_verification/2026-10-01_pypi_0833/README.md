# mojolearn 0.8.33

Published macOS arm64, Linux core, NVIDIA and AMD wheels from main at `4e8caad23ce079b14eb2e083c96c1b86e72d4a3c`.

The release reuses the completed checks of merged lanes at the user's explicit request. No fresh runtime verification or wheel smoke was run. All four published file hashes match the release manifest. See release-provenance.json for the merge records and publication workflow, and CHANGELOG.md for what changed.

Every binding was rebuilt: the Linux sets sm_90a, sm_89 and gfx942 on GitHub runners with the binding cache, the macOS wheel on the release Mac with `--build-only`.
