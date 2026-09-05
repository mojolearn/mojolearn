# UMAP 0.6.0 installation failure recovery

Workflow 33974940904 built source 4e303aed75a94868d305daedbaa7fa4418ea644f.
Its standard interpreter smoke matrix passed, but the additional UMAP gate
failed to install NumPy because the package index could not resolve. The
original downloaded qualification artifact is preserved in original-failure/.
This was an infrastructure failure, not a GPU algorithm failure.

The exact retained wheel SHA256 is
`dab65d03c546169d285a92b2ff3ac2247c77d2288b81495cb9a7b34ba318ea02`.
The revised qualifier installed it in a fresh Python 3.12 environment with
`--wheelhouse /tmp/mojolearn-release-wheelhouse`, without an index. It passed
pip check and all nine installed fit/transform/quality jobs on Apple M4 in
FAST, DETERMINISTIC and IDENTICAL. Dependency wheel hashes and installed
versions are retained. The temporary environment was removed.

The IDENTICAL input and embedding records match the prior exact macOS
candidate for both held-out profiles (comparison.json). This is local
qualification of another artifact, not publication or a new Linux wheel claim.
The workflow itself remains failed; it has not been rerun by this recovery.

An empty-wheelhouse control failed during installation with a named missing
NumPy dependency. It never reached a GPU job. Three workflow shell regression
tests pass, including paths with spaces/literal shell text and refusal of zero
or multiple candidate wheels. The multiple-wheel case exposed the prior
`test ... && test ...` errexit loophole. The 15 existing release artifact
controls and extension inventory agreement also pass.

The repository source commit in results.json is the harness checkout parent;
harness-sha256.json records the changed qualifier and workflow used here.
The wheel's build source remains the workflow revision named above.
