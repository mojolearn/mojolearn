# mojolearn 0.8.32

Published macOS arm64, Linux core, NVIDIA and AMD wheels from main at `32ad138884fd39d69b8bf50c82ecf262b22e0da6`.

The release reuses the completed checks of merged lanes at the user's explicit request. No fresh runtime verification or wheel smoke was run. All four published file hashes match the release manifest. See release-provenance.json for the merge records and publication workflow.

Every binding was rebuilt (the reuse plan against 0.8.31 found none reusable): the Linux sets sm_90a, sm_89 and gfx942 on GitHub runners with the binding cache, the macOS wheel on the release Mac with `--build-only`.

AMD MI325X checks of the merged changes are in bench/results/amd-verify-20260930. They found three differences that the 0.8.31 wheel has too: SGD with regularization and LARS at 200k rows follow the host CPU model, and IVF at 40k rows differs on the AMD GPU. These are not fixed in this release.
