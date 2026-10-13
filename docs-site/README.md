# docs-site

Source for docs.mojolearn.org (Sphinx, PyData theme, MyST Markdown). Distinct from `docs/`,
which holds the project's internal engineering notes.

Publish both sites (this one and mojolearn.org) with the one command, from the mojolearn repo:

```sh
tools/publish_sites.sh --dry-run   # build and record, no deploy
tools/publish_sites.sh             # the same, then deploy both
```

It builds from the latest origin/main in its own worktrees, against the release on PyPI. To build only
the docs by hand, install `requirements.txt` and the release being documented, then:

```sh
docs-site/tools/build.sh <python>   # api/*.rst from mojolearn.__all__, then Sphinx into docs-site/_site
```

The documented release is the only one published, at the site root. Prose never types a release fact:
`{{release}}`, `{{nvidia_arches}}`, `{{amd_arches}}`, `{{nvidia_examples}}` and `{{amd_examples}}` are filled
in by `conf.py` from the installed version and the vendor packages' PyPI summaries.

`tools/gen_api.py` places every public name on one family page; add a new module or name to
its `FAMILIES` table when it fails. `api/*.rst` are generated at every build and not tracked.
Deployed as the `mojolearn-docs` Worker (static assets) on docs.mojolearn.org.
