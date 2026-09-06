# Final Linux wheel admission

The release workflow accepts a staged Linux wheel only after both GPU
vendors qualify the **same final wheel bytes**. Packing vendor candidates
together or repairing a wheel changes its SHA; qualify the resulting wheel
again on AMD and NVIDIA. Individual vendor candidate passes do not authorize
upload of an assembled wheel.

Stage exactly one repaired manylinux wheel and its SHA256 sidecar under
`MOJOLEARN_LINUX_WHEEL_DIR` (default `~/.mojolearn-linux-wheel`). Beside it,
retain this layout:

```text
mojolearn-<version>-...-manylinux_..._x86_64.whl
mojolearn-<version>-...-manylinux_..._x86_64.whl.sha256
qualification/
  hip/   # complete fetched qualification directory, plus build-provenance.json
  cuda/  # complete fetched qualification directory, plus build-provenance.json
```

Each vendor directory needs the original `qualification.json`, `exit_code`,
`results.tsv`, all 24 jobs' logs and installed records, all retained quality
arrays, dependency evidence, `qualification-sources.json`, and
`wheel-audit.json`. Copy the original vendor `build-provenance.json` alongside
these files; its SHA must match the proof hash already recorded by that
vendor's wheel audit. Original remote absolute paths need not exist.

Run the read-only admission locally before dispatching a release:

```sh
python3 tools/check_linux_release_qualification.py /path/to/staged/final.whl \
  --qualification-root /path/to/staged/qualification --source-root "$PWD"
```

Admission validates the final archive's RECORD and complete HIP/CUDA sets,
the exact wheel SHA on both vendor qualifications, all 24 installed jobs
per vendor, and the current native source inventory. It recomputes the
existing UMAP and OrderedRMSE IDENTICAL comparisons from retained evidence;
a standalone comparator PASS file is insufficient.

The native inventory follows `tools/linux_surface_qualification.sh` exactly:
all included Mojo files, binding/Linux packaging/package Python and shell
sources, Pixi configuration/lock, and the qualification shell driver.
Qualification-only tool and workflow edits do not change that native build
inventory. Both hardware runs must still have the same qualification-source
snapshot, as required by the retained comparators.

Missing or stale evidence fails admission. With no Linux wheel staged, the
existing macOS-only release path remains available. These checks perform no
builds, GPU runs, or publication themselves.

All files under the three installed Mamba corpus directories must also match
current hashes in each `qualification-sources.json`. Mamba2 and Mamba3 are
nested under `mamba/corpus/mamba2` and `mamba/corpus/mamba3`. Older snapshots
which silently omitted those directories are refused for final publication.
