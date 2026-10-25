# Retained Apple production device groups

The two production `identical` device groups compiled at `e087692fd` are
independent of subsequent host-only edits. The existing merged cache keys the
whole Mojo tree, so a host edit can otherwise trigger unnecessary GPU compiles.
`tools/apple_retained_groups.py` admits their original bytes explicitly through
the local cache placement boundary. Production compiler, staging, and packer
scripts are unchanged.

After the final source freeze, run from the same stable production checkout:

```sh
python3 tools/apple_retained_groups.py prepare \
  --run "$HOME/mojolearn-evidence/apple-release-0857-device-20261025/run-20261025T162843Z" \
  --root "$PWD" \
  --output "$HOME/mojolearn-evidence/apple-0857-final/device-admission.json"

python3 tools/apple_retained_groups.py pack \
  --admission "$HOME/mojolearn-evidence/apple-0857-final/device-admission.json" \
  --root "$PWD" \
  --output "$HOME/mojolearn-evidence/apple-0857-final/pack" \
  --cache "$HOME/mojolearn-evidence/bincache-apple-release-0857"
```

The admission and pack output paths must not already exist. The target checkout
must be clean and contain these tools. `prepare` reproduces the original
33 device identities from the retained source archive, verifies original
compiler outputs, successful build/stamp receipts and actual cache keys, then
compares the target compiler executable, Apple toolchain, device closures,
flags, merged builder, generator and manifest. The original source revision,
cache key, raw proof bytes, logs, stamp, source archive, and binary remain
unchanged. A version-only Python package edit does not change these native
dependencies.

The pack wrapper sets `MOJOLEARN_BINCACHE_RETAINED_APPLE` and invokes the existing
Mac packer. Its two GPU invocations place the retained raw outputs and record
`origin=retained-admitted`. A mismatched explicit admission fails without
starting a replacement GPU compile. No archive is created under the new broad
cache key. The independently changed host and adapters still follow the normal
production build path, as do native readbacks, staging, signing and wheel
audits. This is `identical` only.

`pack/pack-proof.json` is retained beside the final wheel and its full pack log.
It embeds the original admission and placement witnesses, binds the wheel's
actual SHA, and distinguishes each original raw compiler output, the staged
package file, and the stripped/signed wheel member. These bytes need not be
equal: the existing packaging transformations own the latter two. The sidecar
records `pack_exit`, not an invented device `build_exit`. Transport this proof
with the final wheel; do not flatten its retained origins into a current-source
build claim.

The wrapper runs the packer's `--build-only` form. Installed-wheel loading,
adapter smoke, and any outstanding public-runtime checks remain separate
requirements. A passing artifact admission is not a runtime result.

Read-only validation of the actual completed subset reproduced both original
source/cache captures and all 1,771 dependency files. At integration `a7b00bd3d`,
none of those device dependency bytes changed. Evidence and source-test logs
are under `~/mojolearn-evidence/release-artifact-reuse-20261025/`.
